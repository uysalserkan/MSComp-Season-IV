"""
Self-supervised contrastive learning training script for ViT on STL-10.

This script implements SimCLR-style contrastive learning using the lightly library
for best practices and proven implementations.
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
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
        self.temperature = max(temperature, 1e-8)  # Ensure temperature is positive
    
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
        
        # Validate inputs for NaN/Inf
        if torch.isnan(z0).any() or torch.isinf(z0).any():
            raise ValueError("z0 contains NaN or Inf values")
        if torch.isnan(z1).any() or torch.isinf(z1).any():
            raise ValueError("z1 contains NaN or Inf values")
        
        # Concatenate embeddings
        z = torch.cat([z0, z1], dim=0)  # (2*batch_size, dim)
        
        # Compute similarity matrix with numerical stability
        # Add epsilon to temperature to prevent division issues
        similarity_matrix = torch.matmul(z, z.T) / (self.temperature + 1e-8)  # (2*batch_size, 2*batch_size)
        
        # Clamp logits to prevent extreme values that can cause numerical instability
        similarity_matrix = torch.clamp(similarity_matrix, min=-50.0, max=50.0)
        
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
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Loss is NaN or Inf. Similarity matrix stats: min={similarity_matrix.min():.4f}, max={similarity_matrix.max():.4f}, mean={similarity_matrix.mean():.4f}")
        
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
        
        # Progressive unfreezing strategy for numerical stability
        # Freeze the most problematic layers for the first few epochs
        self.model.freeze_patch_embed()
        self.model.freeze_pos_embed()
        self.model.freeze_early_blocks(num_blocks=2)
        self.unfreeze_epoch = 5  # Unfreeze after this many epochs
        
        # Print model summary
        print("\n" + "="*50)
        print("SSL Model Summary")
        print("="*50)
        param_counts = count_parameters(self.model)
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Method: {config.method.upper()}")
        print(f"\nProgressive Unfreezing Strategy:")
        print(f"  - Patch embed, pos embed, and first 2 blocks frozen until epoch {self.unfreeze_epoch}")
        print(f"  - This prevents NaN gradients during early training")
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
        
        # Optimizer with layer-wise discriminative learning rates
        # Early layers (patch_embed, pos_embed) get 10x lower LR to prevent NaN gradients
        param_groups = self.model.get_param_groups(
            base_lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.optimizer = optim.AdamW(param_groups)
        
        # Store initial LR for each param group (needed for warmup)
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
        # Print layer-wise learning rates for transparency
        print("\nLayer-wise Learning Rates:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            print(f"  {group['name']}: LR={group['lr']:.6f}, Params={num_params:,}")
        print()
        
        # Learning rate scheduler with warmup
        self.scheduler = self._get_scheduler()
        
        # Mixed precision scaler
        # Use new API to avoid deprecation warning
        if config.mixed_precision:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
    
    def _get_scheduler(self):
        """
        Get learning rate scheduler with warmup.
        
        For discriminative learning rates, we need to maintain the LR ratios
        between parameter groups. Standard PyTorch schedulers don't do this,
        so we implement custom scheduling logic.
        """
        # With discriminative learning rates, we'll apply scheduling manually
        # to maintain the LR ratios between groups
        # Store the LR ratios for later use
        if len(self.optimizer.param_groups) > 1:
            # Calculate LR ratios relative to the first (highest) base LR
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            max_base_lr = max(base_lrs)
            self.lr_ratios = [lr / max_base_lr for lr in base_lrs]
            self.max_base_lr = max_base_lr
            # Don't use standard scheduler with discriminative LRs
            return None
        else:
            # Standard scheduler for single LR
            if self.config.lr_scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs - self.config.warmup_epochs,
                    eta_min=self.config.lr_min,
                    last_epoch=-1
                )
                return scheduler
            elif self.config.lr_scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.config.epochs // 3,
                    gamma=0.1,
                    last_epoch=-1
                )
                return scheduler
            else:
                return None
    
    def _get_warmup_lr(self, epoch: int) -> dict:
        """
        Get learning rates for each param group with warmup.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Dictionary mapping param group index to learning rate.
        """
        if epoch < self.config.warmup_epochs:
            # Linear warmup - scale each group's base LR
            warmup_factor = (epoch + 1) / self.config.warmup_epochs
            # Each param group has its own base LR stored in 'lr' key
            # We need to get the base LR for each group (stored when optimizer was created)
            warmup_lrs = {}
            for i, param_group in enumerate(self.optimizer.param_groups):
                # The base LR for this group is stored in the param_group
                # We need to scale it by warmup factor
                base_lr = param_group['lr'] if epoch == 0 else param_group.get('initial_lr', param_group['lr'])
                warmup_lrs[i] = base_lr * warmup_factor
            return warmup_lrs
        else:
            # Return base LRs for each group
            return {i: param_group.get('initial_lr', param_group['lr']) 
                    for i, param_group in enumerate(self.optimizer.param_groups)}
    
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
        
        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Loaded scheduler state from checkpoint")
        
        # Load scaler state if available and using mixed precision
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"Loaded scaler state from checkpoint")
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def _process_view_tensor(self, view, name: str = "view") -> torch.Tensor:
        """
        Process a view tensor to ensure it's a 4D tensor on the correct device.
        
        Args:
            view: Input view (can be tensor, list of tensors, or other)
            name: Name for error messages
        
        Returns:
            4D tensor (batch_size, channels, height, width) on device
        """
        if isinstance(view, torch.Tensor):
            if view.dim() == 4:
                return view.to(self.device)
            elif view.dim() == 3:
                # Single image, add batch dimension
                return view.unsqueeze(0).to(self.device)
            else:
                raise ValueError(f"{name} has unexpected dimensions: {view.dim()}, shape: {view.shape}. Expected 3D or 4D tensor.")
        elif isinstance(view, (list, tuple)):
            if len(view) == 0:
                raise ValueError(f"{name} is empty list/tuple")
            # Stack list of tensors
            if isinstance(view[0], torch.Tensor):
                return torch.stack(view).to(self.device)
            else:
                # Convert to tensors first
                return torch.stack([torch.tensor(img).to(self.device) if not isinstance(img, torch.Tensor) else img.to(self.device) for img in view])
        else:
            # Try to convert to tensor
            view_tensor = torch.tensor(view).to(self.device)
            if view_tensor.dim() == 3:
                view_tensor = view_tensor.unsqueeze(0)
            elif view_tensor.dim() != 4:
                raise ValueError(f"{name} cannot be converted to 4D tensor. Shape: {view_tensor.shape}")
            return view_tensor
    
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
        
        # Progressive unfreezing: unfreeze problematic layers after initial epochs
        if epoch == self.unfreeze_epoch:
            print(f"\nEpoch {epoch+1}: Unfreezing patch_embed, pos_embed, and early blocks...")
            self.model.unfreeze_patch_embed()
            self.model.unfreeze_pos_embed()
            self.model.unfreeze_early_blocks()
            # Rebuild param groups with newly unfrozen parameters
            param_groups = self.model.get_param_groups(
                base_lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            # Update optimizer param groups
            self.optimizer.param_groups = param_groups
            # Store initial LR for each param group
            for param_group in self.optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']
            # Recalculate LR ratios
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            max_base_lr = max(base_lrs)
            self.lr_ratios = [lr / max_base_lr for lr in base_lrs]
            self.max_base_lr = max_base_lr
            print("Optimizer updated with new parameter groups\n")
        
        # Set learning rate with warmup (handles multiple param groups with different base LRs)
        if epoch < self.config.warmup_epochs:
            warmup_lrs = self._get_warmup_lr(epoch)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = warmup_lrs[i]
        elif epoch == self.config.warmup_epochs:
            # Ensure smooth transition: set to base LR when warmup ends
            # This ensures scheduler starts from correct base LR for each group
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr']
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [SSL]")
        
        for batch_idx, batch in enumerate(pbar):
            # Get two augmented views from collate function
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
                        view0 = views[0]
                        view1 = views[1]
                
                # If not handled as lightly batch, try custom collate format
                if view0 is None or view1 is None:
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        # Custom collate format: (view0, view1) or [view0, view1]
                        view0 = batch[0]
                        view1 = batch[1]
                    else:
                        raise ValueError(
                            f"Unrecognized batch format. Batch type: {type(batch)}, "
                            f"length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}. "
                            f"Expected lightly format [views, labels, filenames] or custom format (view0, view1)."
                        )
                
                # Process both views to ensure they're 4D tensors
                view0 = self._process_view_tensor(view0, "view0")
                view1 = self._process_view_tensor(view1, "view1")
                
                # Final validation
                if view0.dim() != 4:
                    raise ValueError(f"view0 has wrong number of dimensions: {view0.dim()}, expected 4. Shape: {view0.shape}")
                if view1.dim() != 4:
                    raise ValueError(f"view1 has wrong number of dimensions: {view1.dim()}, expected 4. Shape: {view1.shape}")
                
                # Check for NaN/Inf in input views
                if torch.isnan(view0).any() or torch.isinf(view0).any():
                    print(f"Warning: view0 contains NaN/Inf at batch {batch_idx}, epoch {epoch}. Skipping batch.")
                    continue
                if torch.isnan(view1).any() or torch.isinf(view1).any():
                    print(f"Warning: view1 contains NaN/Inf at batch {batch_idx}, epoch {epoch}. Skipping batch.")
                    continue
                
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
            if self.config.mixed_precision:
                with autocast(device_type='cuda', enabled=True):
                    try:
                        z0 = self.model(view0)
                        z1 = self.model(view1)
                    except ValueError as e:
                        print(f"Error in model forward pass at batch {batch_idx}, epoch {epoch}: {e}")
                        print(f"  View0 stats: min={view0.min():.4f}, max={view0.max():.4f}, mean={view0.mean():.4f}")
                        print(f"  View1 stats: min={view1.min():.4f}, max={view1.max():.4f}, mean={view1.mean():.4f}")
                        continue
                    except RuntimeError as e:
                        print(f"Runtime error in model forward pass at batch {batch_idx}, epoch {epoch}: {e}")
                        continue
                    
                    # Validate embeddings for NaN/Inf before loss computation
                    if torch.isnan(z0).any() or torch.isinf(z0).any():
                        print(f"Warning: z0 contains NaN/Inf at batch {batch_idx}, epoch {epoch}. Embedding stats: min={z0.min():.4f}, max={z0.max():.4f}, mean={z0.mean():.4f}")
                        # Try to get more diagnostic info
                        try:
                            with torch.no_grad():
                                features0 = self.model.backbone(view0)
                                proj0 = self.model.projection_head(features0)
                                print(f"  Backbone features0: min={features0.min():.4f}, max={features0.max():.4f}, mean={features0.mean():.4f}")
                                print(f"  Projection0 (before norm): min={proj0.min():.4f}, max={proj0.max():.4f}, mean={proj0.mean():.4f}")
                        except:
                            pass
                        continue
                    if torch.isnan(z1).any() or torch.isinf(z1).any():
                        print(f"Warning: z1 contains NaN/Inf at batch {batch_idx}, epoch {epoch}. Embedding stats: min={z1.min():.4f}, max={z1.max():.4f}, mean={z1.mean():.4f}")
                        continue
                    
                    # Check for zero vectors (can cause normalization issues)
                    z0_norm = torch.norm(z0, dim=1)
                    z1_norm = torch.norm(z1, dim=1)
                    if (z0_norm < 1e-8).any() or (z1_norm < 1e-8).any():
                        print(f"Warning: Zero or near-zero embeddings detected at batch {batch_idx}, epoch {epoch}. Skipping batch.")
                        continue
                    
                    # Compute contrastive loss
                    try:
                        if LIGHTLY_AVAILABLE and self.config.method == "simclr":
                            # Lightly NTXentLoss expects two separate arguments: out0, out1
                            # Each view should be (batch_size, projection_dim)
                            loss = self.criterion(z0, z1)
                        else:
                            # Custom loss expects two separate tensors
                            loss = self.criterion(z0, z1)
                    except (ValueError, RuntimeError) as e:
                        print(f"Error computing loss at batch {batch_idx}, epoch {epoch}: {e}")
                        continue
                    
                    # Validate loss before proceeding
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Loss is NaN/Inf at batch {batch_idx}, epoch {epoch}. Skipping batch.")
                        continue
                    
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                try:
                    z0 = self.model(view0)
                    z1 = self.model(view1)
                except ValueError as e:
                    print(f"Error in model forward pass at batch {batch_idx}, epoch {epoch}: {e}")
                    print(f"  View0 stats: min={view0.min():.4f}, max={view0.max():.4f}, mean={view0.mean():.4f}")
                    print(f"  View1 stats: min={view1.min():.4f}, max={view1.max():.4f}, mean={view1.mean():.4f}")
                    continue
                except RuntimeError as e:
                    print(f"Runtime error in model forward pass at batch {batch_idx}, epoch {epoch}: {e}")
                    continue
                
                # Validate embeddings for NaN/Inf before loss computation
                if torch.isnan(z0).any() or torch.isinf(z0).any():
                    print(f"Warning: z0 contains NaN/Inf at batch {batch_idx}, epoch {epoch}. Embedding stats: min={z0.min():.4f}, max={z0.max():.4f}, mean={z0.mean():.4f}")
                    continue
                if torch.isnan(z1).any() or torch.isinf(z1).any():
                    print(f"Warning: z1 contains NaN/Inf at batch {batch_idx}, epoch {epoch}. Embedding stats: min={z1.min():.4f}, max={z1.max():.4f}, mean={z1.mean():.4f}")
                    continue
                
                # Check for zero vectors (can cause normalization issues)
                z0_norm = torch.norm(z0, dim=1)
                z1_norm = torch.norm(z1, dim=1)
                if (z0_norm < 1e-8).any() or (z1_norm < 1e-8).any():
                    print(f"Warning: Zero or near-zero embeddings detected at batch {batch_idx}, epoch {epoch}. Skipping batch.")
                    continue
                
                try:
                    if LIGHTLY_AVAILABLE and self.config.method == "simclr":
                        # Lightly NTXentLoss expects two separate arguments: out0, out1
                        loss = self.criterion(z0, z1)
                    else:
                        # Custom loss expects two separate tensors
                        loss = self.criterion(z0, z1)
                except (ValueError, RuntimeError) as e:
                    print(f"Error computing loss at batch {batch_idx}, epoch {epoch}: {e}")
                    continue
                
                # Validate loss before proceeding
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Loss is NaN/Inf at batch {batch_idx}, epoch {epoch}. Skipping batch.")
                    continue
                
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights (with gradient accumulation)
            # Check if we should update weights (either at accumulation step or at end of epoch)
            should_update = (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)
            
            if should_update or is_last_batch:
                # Check gradients for NaN/Inf before updating
                # For mixed precision, we need to unscale first to check properly
                # But we must ensure scaler state is handled correctly
                has_nan_grad = False
                nan_grad_params = []  # Track which parameters have NaN gradients
                
                if self.config.mixed_precision:
                    # Unscale gradients (required before clipping and checking)
                    self.scaler.unscale_(self.optimizer)
                    
                    # Check for NaN/Inf gradients
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"Warning: NaN/Inf gradient detected in {name} at batch {batch_idx}, epoch {epoch}")
                                nan_grad_params.append(name)
                                has_nan_grad = True
                    
                    # Handle NaN gradients: zero out gradients for problematic parameters
                    # Positional embeddings often get NaN gradients - zero them out instead of skipping batch
                    if has_nan_grad:
                        for name, param in self.model.named_parameters():
                            if name in nan_grad_params:
                                if param.grad is not None:
                                    # Zero out NaN/Inf gradients for this parameter
                                    param.grad = torch.where(
                                        torch.isnan(param.grad) | torch.isinf(param.grad),
                                        torch.zeros_like(param.grad),
                                        param.grad
                                    )
                                    print(f"  Zeroed out NaN/Inf gradients for {name}")
                        
                        # Check if we still have NaN gradients after zeroing
                        still_has_nan = False
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    still_has_nan = True
                                    break
                        
                        if still_has_nan:
                            # If still has NaN after zeroing, skip this update
                            print(f"  Still has NaN gradients after zeroing. Skipping batch.")
                            self.optimizer.zero_grad()
                            self.scaler.update()
                            continue
                    
                    # Apply gradient clipping if needed (do this after handling NaN)
                    if self.config.gradient_clip_norm:
                        # Clip gradients more aggressively for positional embeddings and projection head
                        # Separate clipping for different parameter groups
                        pos_embed_params = []
                        projection_params = []
                        other_params = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if 'pos_embed' in name:
                                    pos_embed_params.append(param)
                                elif 'projection_head' in name:
                                    projection_params.append(param)
                                else:
                                    other_params.append(param)
                        
                        # Clip positional embeddings very aggressively
                        if pos_embed_params:
                            torch.nn.utils.clip_grad_norm_(pos_embed_params, max_norm=0.1)
                        
                        # Clip projection head moderately aggressively
                        if projection_params:
                            torch.nn.utils.clip_grad_norm_(projection_params, max_norm=0.5)
                        
                        # Clip other parameters normally
                        if other_params:
                            torch.nn.utils.clip_grad_norm_(other_params, self.config.gradient_clip_norm)
                    else:
                        # Even without general clipping, clip problematic parameters
                        pos_embed_params = []
                        projection_params = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if 'pos_embed' in name:
                                    pos_embed_params.append(param)
                                elif 'projection_head' in name:
                                    projection_params.append(param)
                        
                        if pos_embed_params:
                            torch.nn.utils.clip_grad_norm_(pos_embed_params, max_norm=0.1)
                        if projection_params:
                            torch.nn.utils.clip_grad_norm_(projection_params, max_norm=0.5)
                    
                    # Step optimizer and update scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Non-mixed precision: check gradients directly
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"Warning: NaN/Inf gradient detected in {name} at batch {batch_idx}, epoch {epoch}")
                                nan_grad_params.append(name)
                                has_nan_grad = True
                    
                    # Handle NaN gradients: zero out gradients for problematic parameters
                    if has_nan_grad:
                        for name, param in self.model.named_parameters():
                            if name in nan_grad_params:
                                if param.grad is not None:
                                    # Zero out NaN/Inf gradients for this parameter
                                    param.grad = torch.where(
                                        torch.isnan(param.grad) | torch.isinf(param.grad),
                                        torch.zeros_like(param.grad),
                                        param.grad
                                    )
                                    print(f"  Zeroed out NaN/Inf gradients for {name}")
                        
                        # Check if we still have NaN gradients after zeroing
                        still_has_nan = False
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    still_has_nan = True
                                    break
                        
                        if still_has_nan:
                            # If still has NaN after zeroing, skip this update
                            print(f"  Still has NaN gradients after zeroing. Skipping batch.")
                            self.optimizer.zero_grad()
                            continue
                    
                    # Apply gradient clipping if needed
                    if self.config.gradient_clip_norm:
                        # Clip gradients more aggressively for positional embeddings and projection head
                        pos_embed_params = []
                        projection_params = []
                        other_params = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if 'pos_embed' in name:
                                    pos_embed_params.append(param)
                                elif 'projection_head' in name:
                                    projection_params.append(param)
                                else:
                                    other_params.append(param)
                        
                        # Clip positional embeddings very aggressively
                        if pos_embed_params:
                            torch.nn.utils.clip_grad_norm_(pos_embed_params, max_norm=0.1)
                        
                        # Clip projection head moderately aggressively
                        if projection_params:
                            torch.nn.utils.clip_grad_norm_(projection_params, max_norm=0.5)
                        
                        # Clip other parameters normally
                        if other_params:
                            torch.nn.utils.clip_grad_norm_(other_params, self.config.gradient_clip_norm)
                    else:
                        # Even without general clipping, clip problematic parameters
                        pos_embed_params = []
                        projection_params = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if 'pos_embed' in name:
                                    pos_embed_params.append(param)
                                elif 'projection_head' in name:
                                    projection_params.append(param)
                        
                        if pos_embed_params:
                            torch.nn.utils.clip_grad_norm_(pos_embed_params, max_norm=0.1)
                        if projection_params:
                            torch.nn.utils.clip_grad_norm_(projection_params, max_norm=0.5)
                    
                    # Step optimizer
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate loss (only if we successfully processed the batch)
            loss_value = loss.item() * self.config.gradient_accumulation_steps
            if not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value))):
                running_loss += loss_value
            num_batches += 1
            
            # Update progress bar with additional debug info
            if (batch_idx + 1) % self.config.print_freq == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                avg_loss_display = running_loss / num_batches if num_batches > 0 else 0.0
                
                # Compute gradient norm for debugging (only if gradients exist)
                total_norm = 0.0
                grad_count = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        grad_count += 1
                if grad_count > 0:
                    total_norm = total_norm ** (1. / 2)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss_display:.4f}',
                    'lr': f'{current_lr:.6f}',
                    'grad_norm': f'{total_norm:.2f}' if grad_count > 0 else 'N/A'
                })
        
        # Defensive check for empty batches
        if num_batches == 0:
            raise RuntimeError("No batches were processed in this epoch. Check your data loader configuration.")
        
        avg_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        
        # Validate average loss
        if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
            print(f"Warning: Average loss is NaN/Inf at epoch {epoch}. This may indicate training instability.")
        
        # Update learning rate scheduler (after warmup)
        if epoch >= self.config.warmup_epochs:
            if self.scheduler is not None:
                # Standard scheduler (single param group)
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                if (epoch + 1) % 5 == 0:  # Log every 5 epochs
                    print(f"  Scheduler LR: {current_lr:.8f}")
            elif hasattr(self, 'lr_ratios'):
                # Custom scheduling for discriminative LRs
                # Apply cosine annealing to max LR, then scale other groups by their ratios
                if self.config.lr_scheduler == "cosine":
                    # Cosine annealing
                    epochs_since_warmup = epoch - self.config.warmup_epochs
                    T_max = self.config.epochs - self.config.warmup_epochs
                    current_max_lr = self.config.lr_min + (self.max_base_lr - self.config.lr_min) * \
                                     0.5 * (1 + math.cos(epochs_since_warmup / T_max * math.pi))
                    
                    # Apply to all param groups with their ratios
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        param_group['lr'] = current_max_lr * self.lr_ratios[i]
                    
                    if (epoch + 1) % 5 == 0:  # Log every 5 epochs
                        print(f"  Max LR: {current_max_lr:.8f}")
                        for i, group in enumerate(self.optimizer.param_groups):
                            print(f"    {group.get('name', f'group_{i}')}: {group['lr']:.8f}")
                elif self.config.lr_scheduler == "step":
                    # Step decay
                    decay_epochs = (epoch - self.config.warmup_epochs) // (self.config.epochs // 3)
                    current_max_lr = self.max_base_lr * (0.1 ** decay_epochs)
                    
                    # Apply to all param groups with their ratios
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        param_group['lr'] = current_max_lr * self.lr_ratios[i]
        
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
            
            # Save scheduler state if available
            if self.scheduler is not None:
                checkpoint_state["scheduler_state_dict"] = self.scheduler.state_dict()
            
            # Save scaler state if using mixed precision
            if self.scaler is not None:
                checkpoint_state["scaler_state_dict"] = self.scaler.state_dict()
            
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
    parser.add_argument("--projection_num_layers", type=int, default=3,
                        help="Number of layers in projection head")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                        help="Freeze backbone layers")
    
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
    parser.add_argument("--use_strong_augmentation", action="store_true", default=True,
                        help="Use strong data augmentation for contrastive learning")
    parser.add_argument("--pin_memory", action="store_true", default=True,
                        help="Pin memory for faster GPU transfer")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.03,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Learning rate warmup epochs")
    
    # Learning rate scheduling
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"],
                        help="Learning rate scheduler (cosine, step, or none)")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum learning rate")
    
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
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic algorithms")
    
    # Logging
    parser.add_argument("--print_freq", type=int, default=10,
                        help="Print metrics every N batches")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create config from arguments
    # Convert "none" to None for lr_scheduler
    lr_scheduler = args.lr_scheduler if args.lr_scheduler != "none" else None
    
    config = SSLConfig(
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_num_layers=args.projection_num_layers,
        freeze_backbone=args.freeze_backbone,
        method=args.method,
        temperature=args.temperature,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        image_size=args.image_size,
        use_strong_augmentation=args.use_strong_augmentation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min=args.lr_min,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        seed=args.seed,
        deterministic=args.deterministic,
        print_freq=args.print_freq,
        save_freq=args.save_freq
    )
    
    # Create trainer and start training
    trainer = SSLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

