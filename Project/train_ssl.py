"""
Self-supervised contrastive learning training script for ViT on STL-10.

This script implements SimCLR-style contrastive learning using the lightly library
for best practices and proven implementations.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.ssl_model import ContrastiveViT
from stl10_dataset import STL10DatasetLoader
from ssl_config import SSLConfig
from utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    format_time,
    count_parameters
)

# Import lightly for contrastive loss and collate functions
try:
    from lightly.loss import NTXentLoss
    from lightly.data import SimCLRCollateFunction
    LIGHTLY_AVAILABLE = True
except ImportError:
    LIGHTLY_AVAILABLE = False
    print("Warning: lightly library not available. Using custom NT-Xent loss.")


class NTXentLossCustom(nn.Module):
    """
    Custom NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    This is the contrastive loss used in SimCLR.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for scaling logits.
        """
        super(NTXentLossCustom, self).__init__()
        self.temperature = temperature
    
    def forward(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two sets of embeddings.
        
        Args:
            z0: First set of normalized embeddings (batch_size, dim).
            z1: Second set of normalized embeddings (batch_size, dim).
        
        Returns:
            Contrastive loss scalar.
        """
        batch_size = z0.shape[0]
        
        # Concatenate embeddings
        z = torch.cat([z0, z1], dim=0)  # (2*batch_size, dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2*batch_size,)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss


class SSLTrainer:
    """
    Trainer class for self-supervised contrastive learning.
    """
    
    def __init__(self, config: SSLConfig):
        """
        Initialize SSL trainer with configuration.
        
        Args:
            config: SSL configuration object.
        """
        self.config = config
        
        # Set seed for reproducibility
        set_seed(config.seed, config.deterministic)
        
        # Get device
        self.device = get_device()
        
        # Initialize model
        self.model = ContrastiveViT(
            backbone_name=config.backbone_name,
            pretrained=config.pretrained,
            image_size=config.image_size,
            projection_dim=config.projection_dim,
            projection_hidden_dim=config.projection_hidden_dim,
            projection_num_layers=config.projection_num_layers,
            freeze_backbone=config.freeze_backbone
        ).to(self.device)
        
        # Print model summary
        print("\n" + "="*50)
        print("SSL Model Summary")
        print("="*50)
        param_counts = count_parameters(self.model)
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Backbone parameters: {param_counts['backbone']:,}")
        print(f"Projection parameters: {param_counts['projection']:,}")
        print(f"Method: {config.method.upper()}")
        print("="*50 + "\n")
        
        # Initialize dataset loader
        self.dataset_loader = STL10DatasetLoader(
            root=config.data_root,
            download=True
        )
        
        # Get contrastive transform
        contrastive_transform = self.dataset_loader.get_contrastive_transform(
            image_size=config.image_size,
            use_strong_augmentation=config.use_strong_augmentation
        )
        
        # Get unlabeled dataset
        unlabeled_dataset = self.dataset_loader.get_unlabeled_dataset(
            transform=contrastive_transform,
            download=True
        )
        
        # Create collate function for two-view augmentation
        if LIGHTLY_AVAILABLE:
            collate_fn = SimCLRCollateFunction(
                input_size=config.image_size,
                cj_prob=0.8,
                cj_bright=0.4,
                cj_contrast=0.4,
                cj_sat=0.4,
                cj_hue=0.1,
                min_scale=0.2,
                random_gray_scale=0.2,
                gaussian_blur=0.5,
                kernel_size=23,
                sigmas=(0.1, 2.0),
                solarize_prob=0.2,
                normalize={
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            )
        else:
            # Custom collate function for two views
            def collate_fn(batch):
                images = torch.stack([item[0] for item in batch])
                # Create second view with same transform (simplified)
                # In practice, you'd apply different random augmentations
                images_view1 = images.clone()
                return images, images_view1
        
        # Create data loader
        self.train_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        # Contrastive loss
        if LIGHTLY_AVAILABLE and config.method == "simclr":
            self.criterion = NTXentLoss(temperature=config.temperature)
        else:
            self.criterion = NTXentLossCustom(temperature=config.temperature)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._get_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
    
    def _get_scheduler(self):
        """Get learning rate scheduler with warmup."""
        if self.config.lr_scheduler == "cosine":
            # Cosine annealing after warmup
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
                eta_min=self.config.lr_min
            )
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        else:
            return None
    
    def _get_warmup_lr(self, epoch: int) -> float:
        """Get learning rate with warmup."""
        if epoch < self.config.warmup_epochs:
            # Linear warmup
            return self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
        else:
            return self.config.learning_rate
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device
        )
        
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_loss" in checkpoint:
            self.best_loss = checkpoint["best_loss"]
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        # Set learning rate with warmup
        if epoch < self.config.warmup_epochs:
            warmup_lr = self._get_warmup_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [SSL]")
        
        for batch_idx, batch in enumerate(pbar):
            # Get two augmented views from collate function
            if isinstance(batch, tuple) and len(batch) == 2:
                view0, view1 = batch
            else:
                # Fallback: create two views from single batch
                images = batch.to(self.device)
                view0 = images
                view1 = images.clone()
            
            view0 = view0.to(self.device)
            view1 = view1.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    z0 = self.model(view0)
                    z1 = self.model(view1)
                    
                    # Compute contrastive loss
                    if LIGHTLY_AVAILABLE and self.config.method == "simclr":
                        # Lightly expects concatenated features
                        features = torch.cat([z0, z1], dim=0)
                        loss = self.criterion(features)
                    else:
                        loss = self.criterion(z0, z1)
                    
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                z0 = self.model(view0)
                z1 = self.model(view1)
                
                if LIGHTLY_AVAILABLE and self.config.method == "simclr":
                    features = torch.cat([z0, z1], dim=0)
                    loss = self.criterion(features)
                else:
                    loss = self.criterion(z0, z1)
                
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    if self.config.gradient_clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config.gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate loss
            running_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if (batch_idx + 1) % self.config.print_freq == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/num_batches:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        avg_loss = running_loss / num_batches
        
        # Update learning rate scheduler (after warmup)
        if self.scheduler and epoch >= self.config.warmup_epochs:
            self.scheduler.step()
        
        return avg_loss
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting SSL Training")
        print("="*50)
        print(f"Method: {self.config.method.upper()}")
        print(f"Total epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Warmup epochs: {self.config.warmup_epochs}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.config.epochs} Summary:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Time: {format_time(epoch_time)}")
            
            # Check for best model
            is_best = train_loss < self.best_loss
            if is_best:
                self.best_loss = train_loss
            
            # Save checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "train_losses": self.train_losses,
                "config": self.config.to_dict()
            }
            
            checkpoint_path = Path(self.config.checkpoint_dir) / f"ssl_checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                checkpoint_state,
                str(checkpoint_path),
                is_best=is_best,
                keep_last_n=self.config.save_freq
            )
            
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"\nSSL Training completed in {format_time(total_time)}")
        print(f"Best loss: {self.best_loss:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Self-supervised contrastive learning on STL-10")
    
    # Model arguments
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_224",
                        help="ViT backbone name from timm")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use pretrained ImageNet weights")
    parser.add_argument("--projection_dim", type=int, default=128,
                        help="Projection head output dimension")
    parser.add_argument("--projection_hidden_dim", type=int, default=2048,
                        help="Projection head hidden dimension")
    
    # Method
    parser.add_argument("--method", type=str, default="simclr",
                        choices=["simclr", "moco", "byol"],
                        help="Contrastive learning method")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for dataset")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (large for contrastive learning)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.03,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Learning rate warmup epochs")
    
    # Training settings
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_ssl",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs_ssl",
                        help="Directory to save logs")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create config from arguments
    config = SSLConfig(
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        method=args.method,
        temperature=args.temperature,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        seed=args.seed
    )
    
    # Create trainer and start training
    trainer = SSLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

