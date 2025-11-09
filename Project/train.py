"""
Training script for finetuning ViT on STL-10 dataset.

This script implements a complete training pipeline with best practices including:
- Mixed precision training (AMP)
- Learning rate scheduling
- Early stopping
- Checkpointing
- Comprehensive logging
"""

import argparse
import time
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.vit_model import ViTFinetuner
from stl10_dataset import STL10DatasetLoader
from config import TrainingConfig
from utils import (
    set_seed,
    get_device,
    calculate_accuracy_simple,
    save_checkpoint,
    load_checkpoint,
    get_model_summary,
    format_time,
    count_parameters
)


class Trainer:
    """
    Trainer class for ViT finetuning on STL-10.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration object.
        """
        self.config = config
        
        # Set seed for reproducibility
        set_seed(config.seed, config.deterministic)
        
        # Get device
        self.device = get_device()
        
        # Initialize model
        # Determine if we should use SSL pretrained weights
        ssl_path = config.ssl_pretrained_path if config.use_ssl_pretrained else None
        use_pretrained = config.pretrained and not config.use_ssl_pretrained
        
        self.model = ViTFinetuner(
            model_name=config.model_name,
            num_classes=config.num_classes,
            pretrained=use_pretrained,
            image_size=config.image_size,
            freeze_backbone=config.freeze_backbone,
            ssl_pretrained_path=ssl_path
        ).to(self.device)
        
        # Print model summary
        print("\n" + "="*50)
        print("Model Summary")
        print("="*50)
        param_counts = count_parameters(self.model)
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Frozen parameters: {param_counts['frozen']:,}")
        print("="*50 + "\n")
        
        # Initialize dataset loader
        self.dataset_loader = STL10DatasetLoader(
            root=config.data_root,
            download=True
        )
        
        # Get data loaders
        self.train_loader, self.val_loader = self.dataset_loader.get_vit_data_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            use_augmentation=config.use_augmentation,
            pin_memory=config.pin_memory
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Early stopping
        self.early_stopping_counter = 0
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on config."""
        if self.config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr_min
            )
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.lr_scheduler == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=len(self.train_loader)
            )
        else:
            return None
    
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
        if "best_val_acc" in checkpoint:
            self.best_val_acc = checkpoint["best_val_acc"]
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]
        if "train_accs" in checkpoint:
            self.train_accs = checkpoint["train_accs"]
        if "val_accs" in checkpoint:
            self.val_accs = checkpoint["val_accs"]
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
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
                
                # Update learning rate (for OneCycleLR)
                if self.config.lr_scheduler == "onecycle":
                    self.scheduler.step()
            
            # Calculate metrics
            acc = calculate_accuracy_simple(outputs, labels)
            running_loss += loss.item() * self.config.gradient_accumulation_steps
            running_acc += acc
            num_batches += 1
            
            # Update progress bar
            if (batch_idx + 1) % self.config.print_freq == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/num_batches:.4f}',
                    'acc': f'{running_acc/num_batches:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches
        
        return avg_loss, avg_acc
    
    def validate(self, epoch: int) -> tuple:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Val]")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                acc = calculate_accuracy_simple(outputs, labels)
                running_loss += loss.item()
                running_acc += acc
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/num_batches:.4f}',
                    'acc': f'{running_acc/num_batches:.4f}'
                })
        
        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches
        
        return avg_loss, avg_acc
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Total epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate scheduler (except OneCycleLR)
            if self.scheduler and self.config.lr_scheduler != "onecycle":
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.config.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Time: {format_time(epoch_time)}")
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "best_val_loss": self.best_val_loss,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_accs": self.train_accs,
                "val_accs": self.val_accs,
                "config": self.config.to_dict()
            }
            
            checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                checkpoint_state,
                str(checkpoint_path),
                is_best=is_best,
                keep_last_n=self.config.save_freq
            )
            
            # Early stopping
            if self.config.early_stopping_patience > 0:
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best validation accuracy: {self.best_val_acc:.4f}")
                    break
            
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Finetune ViT on STL-10 dataset")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="ViT model name from timm")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone layers")
    
    # SSL pretraining
    parser.add_argument("--ssl_pretrained_path", type=str, default=None,
                        help="Path to SSL pretrained checkpoint")
    parser.add_argument("--use_ssl_pretrained", action="store_true",
                        help="Use SSL pretrained weights instead of ImageNet")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Learning rate scheduling
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "onecycle", None],
                        help="Learning rate scheduler")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum learning rate")
    
    # Training settings
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Data augmentation
    parser.add_argument("--use_augmentation", action="store_true", default=True,
                        help="Use data augmentation")
    
    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save logs")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic algorithms")
    
    # Logging
    parser.add_argument("--print_freq", type=int, default=10,
                        help="Print frequency")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="Save checkpoint frequency")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create config from arguments
    config = TrainingConfig(
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        ssl_pretrained_path=args.ssl_pretrained_path,
        use_ssl_pretrained=args.use_ssl_pretrained,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        use_augmentation=args.use_augmentation,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        seed=args.seed,
        deterministic=args.deterministic,
        print_freq=args.print_freq,
        save_freq=args.save_freq
    )
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

