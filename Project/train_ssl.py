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
    # Try importing SimCLRTransform as well in case we need it
    try:
        from lightly.transforms import SimCLRTransform
        SIMCLR_TRANSFORM_AVAILABLE = True
    except ImportError:
        SIMCLR_TRANSFORM_AVAILABLE = False
    LIGHTLY_AVAILABLE = True
except ImportError:
    LIGHTLY_AVAILABLE = False
    SIMCLR_TRANSFORM_AVAILABLE = False
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
        
        # Create labels: positive pairs are (i, i+batch_size) for i in [0, batch_size-1]
        # For sample i: positive is at index i+batch_size
        # For sample i+batch_size: positive is at index i
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2*batch_size,)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        # Each row i should predict label labels[i]
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
        print(f"Method: {config.method.upper()}")
        print("="*50 + "\n")
        
        # Initialize dataset loader
        self.dataset_loader = STL10DatasetLoader(
            root=config.data_root,
            download=True
        )
        
        # Get contrastive transform for custom collate function
        contrastive_transform = self.dataset_loader.get_contrastive_transform(
            image_size=config.image_size,
            use_strong_augmentation=config.use_strong_augmentation
        )
        
        # Create collate function for two-view augmentation
        use_lightly_collate = False
        if LIGHTLY_AVAILABLE:
            # SimCLRCollateFunction expects PIL images, so load dataset WITHOUT transforms
            unlabeled_dataset_raw = self.dataset_loader.get_unlabeled_dataset(
                transform=None,  # No transform - SimCLRCollateFunction will handle augmentation
                download=True
            )
            
            # Wrap dataset to provide (image, label, filename) format expected by SimCLRCollateFunction
            class DatasetWithFilename:
                """Wrapper to add filename to dataset items."""
                def __init__(self, dataset):
                    self.dataset = dataset
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    # STL-10 returns (image, label) or (image,)
                    if isinstance(item, tuple):
                        if len(item) == 2:
                            image, label = item
                            # Add dummy filename
                            return (image, label, f"image_{idx}.png")
                        elif len(item) == 1:
                            image = item[0]
                            return (image, -1, f"image_{idx}.png")
                        else:
                            # Already has filename?
                            return item
                    else:
                        # Just image
                        return (item, -1, f"image_{idx}.png")
            
            unlabeled_dataset = DatasetWithFilename(unlabeled_dataset_raw)
            
            # Try to use SimCLRCollateFunction with correct API
            # Different versions of lightly may have different APIs
            try:
                # Try with basic parameters first (most compatible)
                collate_fn = SimCLRCollateFunction(
                    input_size=config.image_size,
                    cj_prob=0.8,
                    min_scale=0.2,
                    random_gray_scale=0.2,
                    gaussian_blur=0.5,
                    normalize={
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    }
                )
                use_lightly_collate = True
            except TypeError as e:
                # If that fails, try with even fewer parameters
                try:
                    collate_fn = SimCLRCollateFunction(
                        input_size=config.image_size,
                        normalize={
                            'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]
                        }
                    )
                    use_lightly_collate = True
                except TypeError:
                    # Use SimCLRTransform if available, otherwise fall back to custom
                    if SIMCLR_TRANSFORM_AVAILABLE:
                        transform = SimCLRTransform(
                            input_size=config.image_size,
                            cj_prob=0.8,
                            cj_strength=0.5,
                            min_scale=0.2,
                            random_gray_scale=0.2,
                            gaussian_blur=0.5,
                            normalize={
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]
                            }
                        )
                        # Create a simple collate that applies transform twice
                        def collate_fn(batch):
                            images = [item[0] for item in batch]
                            view0 = torch.stack([transform(img) for img in images])
                            view1 = torch.stack([transform(img) for img in images])
                            return view0, view1
                        use_lightly_collate = False  # Not using lightly collate, but using lightly transform
                    else:
                        # Fall back to custom implementation
                        print(f"Warning: SimCLRCollateFunction API not compatible: {e}")
                        print("Falling back to custom collate function.")
                        use_lightly_collate = False
        
        if not use_lightly_collate:
            # For custom collate, load dataset WITHOUT transforms (PIL images)
            # We'll apply transforms in the collate function to get two different views
            if not LIGHTLY_AVAILABLE:
                unlabeled_dataset = self.dataset_loader.get_unlabeled_dataset(
                    transform=None,  # No transform - we'll apply in collate function
                    download=True
                )
            
            # Custom collate function that creates two different augmented views
            # We need to apply the transform twice with different random states
            class TwoViewCollate:
                def __init__(self, transform):
                    self.transform = transform
                
                def __call__(self, batch):
                    # Apply transform twice to get two different views
                    view0_list = []
                    view1_list = []
                    
                    for item in batch:
                        image = item[0]  # PIL image (since transform=None)
                        # Apply transform twice (will have different random augmentations)
                        view0_list.append(self.transform(image))
                        view1_list.append(self.transform(image))
                    
                    view0 = torch.stack(view0_list)
                    view1 = torch.stack(view1_list)
                    return view0, view1
            
            # Create collate function
            collate_fn = TwoViewCollate(contrastive_transform)
        
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
            # SimCLRCollateFunction returns (views, labels, filenames) where views is a list
            try:
                view0 = None
                view1 = None
                
                # Check for lightly batch format: [views, labels, filenames] where views is a list
                if LIGHTLY_AVAILABLE and isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # Check if first element is a list/tuple (views list from lightly)
                    if isinstance(batch[0], (list, tuple)) and len(batch[0]) >= 2:
                        # Lightly format: [views, labels, filenames] or (views, labels, filenames)
                        # views is a list/tuple of tensors, one for each view
                        views = batch[0]
                        view0 = views[0].to(self.device)
                        view1 = views[1].to(self.device)
                        # Validate shapes
                        if view0.dim() != 4:
                            raise ValueError(f"view0 from lightly has wrong dimensions: {view0.dim()}, expected 4. Shape: {view0.shape}")
                        if view1.dim() != 4:
                            raise ValueError(f"view1 from lightly has wrong dimensions: {view1.dim()}, expected 4. Shape: {view1.shape}")
                
                # If not handled as lightly batch, try other formats
                if view0 is None or view1 is None:
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        # Custom collate format: (view0, view1) or [view0, view1]
                        view0, view1 = batch[0], batch[1]
                    
                    # Ensure they are tensors with correct shape
                    if view0 is not None and view1 is not None:
                        if isinstance(view0, torch.Tensor):
                            if view0.dim() == 4:
                                view0 = view0.to(self.device)
                            else:
                                # If it's a single image (3D), we need to add batch dimension
                                if view0.dim() == 3:
                                    view0 = view0.unsqueeze(0).to(self.device)
                                else:
                                    raise ValueError(f"view0 has unexpected dimensions: {view0.dim()}, shape: {view0.shape}")
                    elif isinstance(view0, (list, tuple)):
                        # Stack list of tensors
                        if len(view0) > 0 and isinstance(view0[0], torch.Tensor):
                            view0 = torch.stack(view0).to(self.device)
                        else:
                            view0 = torch.stack([torch.tensor(img).to(self.device) if not isinstance(img, torch.Tensor) else img.to(self.device) for img in view0])
                    else:
                        view0 = torch.tensor(view0).to(self.device)
                        if view0.dim() == 3:
                            view0 = view0.unsqueeze(0)
                    
                    if isinstance(view1, torch.Tensor):
                        if view1.dim() == 4:
                            view1 = view1.to(self.device)
                        else:
                            # If it's a single image (3D), we need to add batch dimension
                            if view1.dim() == 3:
                                view1 = view1.unsqueeze(0).to(self.device)
                            else:
                                raise ValueError(f"view1 has unexpected dimensions: {view1.dim()}, shape: {view1.shape}")
                    elif isinstance(view1, (list, tuple)):
                        # Stack list of tensors
                        if len(view1) > 0 and isinstance(view1[0], torch.Tensor):
                            view1 = torch.stack(view1).to(self.device)
                        else:
                            view1 = torch.stack([torch.tensor(img).to(self.device) if not isinstance(img, torch.Tensor) else img.to(self.device) for img in view1])
                    else:
                        view1 = torch.tensor(view1).to(self.device)
                        if view1.dim() == 3:
                            view1 = view1.unsqueeze(0)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # Batch might be a list/tuple of two views directly (custom collate)
                    view0_raw = batch[0]
                    view1_raw = batch[1]
                    # Convert to tensors if needed
                    if isinstance(view0_raw, torch.Tensor):
                        if view0_raw.dim() == 4:
                            view0 = view0_raw.to(self.device)
                        elif view0_raw.dim() == 3:
                            view0 = view0_raw.unsqueeze(0).to(self.device)
                        else:
                            raise ValueError(f"view0_raw has unexpected dimensions: {view0_raw.dim()}, shape: {view0_raw.shape}")
                    elif isinstance(view0_raw, (list, tuple)):
                        # Stack list of tensors
                        if len(view0_raw) > 0:
                            if isinstance(view0_raw[0], torch.Tensor):
                                view0 = torch.stack(view0_raw).to(self.device)
                            else:
                                view0 = torch.stack([torch.tensor(img).to(self.device) if not isinstance(img, torch.Tensor) else img.to(self.device) for img in view0_raw])
                        else:
                            raise ValueError("view0_raw is empty list")
                    else:
                        view0 = torch.tensor(view0_raw).to(self.device)
                        if view0.dim() == 3:
                            view0 = view0.unsqueeze(0)
                    
                    if isinstance(view1_raw, torch.Tensor):
                        if view1_raw.dim() == 4:
                            view1 = view1_raw.to(self.device)
                        elif view1_raw.dim() == 3:
                            view1 = view1_raw.unsqueeze(0).to(self.device)
                        else:
                            raise ValueError(f"view1_raw has unexpected dimensions: {view1_raw.dim()}, shape: {view1_raw.shape}")
                    elif isinstance(view1_raw, (list, tuple)):
                        # Stack list of tensors
                        if len(view1_raw) > 0:
                            if isinstance(view1_raw[0], torch.Tensor):
                                view1 = torch.stack(view1_raw).to(self.device)
                            else:
                                view1 = torch.stack([torch.tensor(img).to(self.device) if not isinstance(img, torch.Tensor) else img.to(self.device) for img in view1_raw])
                        else:
                            raise ValueError("view1_raw is empty list")
                    else:
                        view1 = torch.tensor(view1_raw).to(self.device)
                        if view1.dim() == 3:
                            view1 = view1.unsqueeze(0)
                    # Ensure they are tensors with correct shape
                    if view0 is not None and view1 is not None:
                        if isinstance(view0, torch.Tensor):
                            if view0.dim() == 4:
                                view0 = view0.to(self.device)
                            else:
                                # If it's a single image (3D), we need to add batch dimension
                                if view0.dim() == 3:
                                    view0 = view0.unsqueeze(0).to(self.device)
                                else:
                                    raise ValueError(f"view0 has unexpected dimensions: {view0.dim()}, shape: {view0.shape}")
                        elif isinstance(view0, (list, tuple)):
                            # Stack list of tensors
                            if len(view0) > 0 and isinstance(view0[0], torch.Tensor):
                                view0 = torch.stack(view0).to(self.device)
                            else:
                                view0 = torch.stack([torch.tensor(img).to(self.device) if not isinstance(img, torch.Tensor) else img.to(self.device) for img in view0])
                        else:
                            view0 = torch.tensor(view0).to(self.device)
                            if view0.dim() == 3:
                                view0 = view0.unsqueeze(0)
                        
                        if isinstance(view1, torch.Tensor):
                            if view1.dim() == 4:
                                view1 = view1.to(self.device)
                            else:
                                # If it's a single image (3D), we need to add batch dimension
                                if view1.dim() == 3:
                                    view1 = view1.unsqueeze(0).to(self.device)
                                else:
                                    raise ValueError(f"view1 has unexpected dimensions: {view1.dim()}, shape: {view1.shape}")
                        elif isinstance(view1, (list, tuple)):
                            # Stack list of tensors
                            if len(view1) > 0 and isinstance(view1[0], torch.Tensor):
                                view1 = torch.stack(view1).to(self.device)
                            else:
                                view1 = torch.stack([torch.tensor(img).to(self.device) if not isinstance(img, torch.Tensor) else img.to(self.device) for img in view1])
                        else:
                            view1 = torch.tensor(view1).to(self.device)
                            if view1.dim() == 3:
                                view1 = view1.unsqueeze(0)
                    else:
                        raise ValueError(f"Could not extract views from batch. Batch type: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
            except Exception as e:
                print(f"\nError processing batch at index {batch_idx}: {e}")
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, (list, tuple)):
                    print(f"Batch length: {len(batch)}")
                    if len(batch) > 0:
                        print(f"First element type: {type(batch[0])}")
                        if isinstance(batch[0], (list, tuple)):
                            print(f"First element length: {len(batch[0])}")
                            if len(batch[0]) > 0:
                                print(f"First element[0] type: {type(batch[0][0])}")
                                if isinstance(batch[0][0], torch.Tensor):
                                    print(f"First element[0] shape: {batch[0][0].shape}")
                raise
            
            # Forward pass with mixed precision
            # Validate tensor shapes before passing to model
            if view0.dim() != 4:
                raise ValueError(f"view0 has wrong number of dimensions: {view0.dim()}, expected 4. Shape: {view0.shape}")
            if view1.dim() != 4:
                raise ValueError(f"view1 has wrong number of dimensions: {view1.dim()}, expected 4. Shape: {view1.shape}")
            
            if self.config.mixed_precision:
                with autocast():
                    z0 = self.model(view0)
                    z1 = self.model(view1)
                    
                    # Compute contrastive loss
                    if LIGHTLY_AVAILABLE and self.config.method == "simclr":
                        # Lightly NTXentLoss expects two separate arguments: out0, out1
                        # Each view should be (batch_size, projection_dim)
                        loss = self.criterion(z0, z1)
                    else:
                        # Custom loss expects two separate tensors
                        loss = self.criterion(z0, z1)
                    
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                z0 = self.model(view0)
                z1 = self.model(view1)
                
                if LIGHTLY_AVAILABLE and self.config.method == "simclr":
                    # Lightly NTXentLoss expects two separate arguments: out0, out1
                    loss = self.criterion(z0, z1)
                else:
                    # Custom loss expects two separate tensors
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

