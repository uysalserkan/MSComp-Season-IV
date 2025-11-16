"""
Self-supervised learning training script for ViT on STL-10 dataset.

This script implements SimCLR contrastive learning with periodic linear evaluation
to assess representation quality. It follows production-grade best practices including
type hints, error handling, mixed precision training, and comprehensive logging.
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Lightly imports for SimCLR
try:
    from lightly.loss import NTXentLoss
    from lightly.transforms import SimCLRTransform
    LIGHTLY_AVAILABLE = True
except ImportError:
    LIGHTLY_AVAILABLE = False
    warnings.warn(
        "lightly library not available. Please install it: pip install lightly",
        UserWarning
    )

from models.ssl_model import ContrastiveViT
from models.vit_model import ViTFinetuner
from stl10_dataset import STL10DatasetLoader
from ssl_config import SSLConfig
from utils import (
    set_seed,
    get_device,
    calculate_accuracy_simple,
    save_checkpoint,
    load_checkpoint,
    format_time,
    count_parameters
)
from torch.utils.data import Dataset
class ContrastiveSTL10Dataset(Dataset):
    """
    Dataset wrapper that generates SimCLR views from STL-10 images.

    This ensures augmented pairs are created within __getitem__, allowing the
    default PyTorch collate function to stack tensors without custom logic.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transform: SimCLRTransform
    ) -> None:
        """
        Args:
            base_dataset: STL-10 dataset returning PIL images.
            transform: SimCLRTransform that produces two augmented views.
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.base_dataset[index]
        view1, view2 = self.transform(image)
        return view1, view2, label


class LinearEvaluator:
    """
    Linear evaluation protocol for assessing SSL representation quality.
    
    Freezes the pretrained backbone and trains a linear classifier on top
    to evaluate the quality of learned representations on labeled data.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        num_classes: int = 10,
        image_size: int = 224,
        data_root: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        learning_rate: float = 0.01,
        epochs: int = 50,
        mixed_precision: bool = True
    ):
        """
        Initialize linear evaluator.
        
        Args:
            backbone: Pretrained backbone model (will be frozen).
            device: Device to run evaluation on.
            num_classes: Number of output classes.
            image_size: Input image size.
            data_root: Root directory for dataset.
            batch_size: Batch size for evaluation.
            num_workers: Number of data loading workers.
            learning_rate: Learning rate for linear classifier training.
            epochs: Number of training epochs for linear classifier.
            mixed_precision: Whether to use mixed precision training.
        """
        self.device = device
        self.num_classes = num_classes
        self.epochs = epochs
        self.mixed_precision = mixed_precision
        
        # Extract backbone and freeze it
        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size).to(device)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Create linear classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes).to(device)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if mixed_precision else None
        
        # Load data
        dataset_loader = STL10DatasetLoader(root=data_root, download=True)
        self.train_loader, self.test_loader = dataset_loader.get_vit_data_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            use_augmentation=True,
            pin_memory=True
        )
    
    def train_classifier(self) -> float:
        """
        Train linear classifier on frozen features.
        
        Returns:
            Best test accuracy achieved.
        """
        best_acc = 0.0
        
        for epoch in range(self.epochs):
            # Training phase
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features (no gradients for backbone)
                with torch.no_grad():
                    features = self.backbone(images)
                
                # Forward through classifier
                if self.mixed_precision:
                    with autocast():
                        logits = self.classifier(features)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.classifier(features)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            
            # Evaluation phase
            test_acc = self.evaluate()
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Linear Eval Epoch [{epoch+1}/{self.epochs}] "
                      f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        return best_acc
    
    def evaluate(self) -> float:
        """
        Evaluate linear classifier on test set.
        
        Returns:
            Test accuracy.
        """
        self.classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                features = self.backbone(images)
                
                # Classify
                logits = self.classifier(features)
                _, predicted = logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy


class SSLTrainer:
    """
    Self-supervised learning trainer for ViT using SimCLR.
    
    This trainer implements contrastive learning with SimCLR on unlabeled data
    and periodically evaluates learned representations using linear evaluation.
    """
    
    def __init__(self, config: SSLConfig):
        """
        Initialize SSL trainer with configuration.
        
        Args:
            config: SSL training configuration object.
        """
        self.config = config
        
        # Check lightly availability
        if not LIGHTLY_AVAILABLE:
            raise ImportError(
                "lightly library is required for SSL training. "
                "Install it with: pip install lightly"
            )
        
        # Set seed for reproducibility
        set_seed(config.seed, config.deterministic)
        
        # Get device
        self.device = get_device()
        
        # Initialize SSL model
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
        print("\n" + "="*70)
        print("SSL Model Summary")
        print("="*70)
        param_counts = self.model.count_parameters()
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Frozen parameters: {param_counts['frozen']:,}")
        print(f"Backbone parameters: {param_counts['backbone']:,}")
        print(f"Projection head parameters: {param_counts['projection']:,}")
        print("="*70 + "\n")
        
        # Initialize dataset loader
        self.dataset_loader = STL10DatasetLoader(
            root=config.data_root,
            download=True
        )
        
        # Setup SimCLR transform for two views
        self.simclr_transform = SimCLRTransform(
            input_size=config.image_size,
            cj_prob=0.8,
            cj_strength=0.5,
            min_scale=0.2,
            random_gray_scale=0.2,
            gaussian_blur=0.5,
            normalize={
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        )

        # Load unlabeled dataset without tensor conversion (keep PIL for SimCLR)
        from torchvision import transforms as T

        pil_preserving_transform = T.Resize((config.image_size, config.image_size))

        base_dataset = self.dataset_loader.get_unlabeled_dataset(
            transform=pil_preserving_transform,
            download=True
        )

        # Wrap dataset to generate contrastive pairs directly
        self.train_dataset = ContrastiveSTL10Dataset(
            base_dataset=base_dataset,
            transform=self.simclr_transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
        
        # Contrastive loss (NT-Xent loss)
        self.criterion = NTXentLoss(temperature=config.temperature, memory_bank_size=0)
        
        # Optimizer with layer-wise learning rates
        param_groups = self.model.get_param_groups(
            base_lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.optimizer = optim.AdamW(param_groups, lr=config.learning_rate)
        
        # Learning rate scheduler with warmup
        self.scheduler = self._get_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.start_epoch = 0
        self.best_linear_acc = 0.0
        self.ssl_losses = []
        self.linear_accs = []
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
        
        print(f"Training on {len(self.train_dataset)} unlabeled images")
        print(f"Batch size: {config.batch_size}\n")
    
    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Get learning rate scheduler based on config.
        
        Returns:
            Learning rate scheduler or None.
        """
        if self.config.lr_scheduler == "cosine":
            # Cosine annealing with warmup
            def lr_lambda(epoch: int) -> float:
                if epoch < self.config.warmup_epochs:
                    # Linear warmup
                    return float(epoch) / float(max(1, self.config.warmup_epochs))
                else:
                    # Cosine decay
                    progress = float(epoch - self.config.warmup_epochs) / float(
                        max(1, self.config.epochs - self.config.warmup_epochs)
                    )
                    return max(
                        self.config.lr_min / self.config.learning_rate,
                        0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)).item())
                    )
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        else:
            return None
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device
        )
        
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        if "best_linear_acc" in checkpoint:
            self.best_linear_acc = checkpoint["best_linear_acc"]
        if "ssl_losses" in checkpoint:
            self.ssl_losses = checkpoint["ssl_losses"]
        if "linear_accs" in checkpoint:
            self.linear_accs = checkpoint["linear_accs"]
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Resumed training from epoch {self.start_epoch}")
        print(f"Best linear evaluation accuracy so far: {self.best_linear_acc:.2f}%\n")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch using contrastive learning.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Average contrastive loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        forward_errors = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs} [SSL Train]"
        )
        
        for batch_idx, (view1, view2, _) in enumerate(pbar):
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            
            # Forward pass with mixed precision
            try:
                if self.config.mixed_precision:
                    with autocast():
                        proj1 = self.model(view1)
                        proj2 = self.model(view2)
                        loss = self.criterion(proj1, proj2)
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    proj1 = self.model(view1)
                    proj2 = self.model(view2)
                    loss = self.criterion(proj1, proj2)
                    loss = loss / self.config.gradient_accumulation_steps
            except ValueError as err:
                forward_errors += 1
                print(
                    f"\nWarning: Model forward failed at batch {batch_idx + 1} "
                    f"due to numerical instability: {err}. Skipping this batch."
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                print(
                    f"\nWarning: Invalid loss detected at batch {batch_idx + 1}. "
                    "Skipping this batch to maintain training stability."
                )
                continue
            
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
            
            # Track metrics
            running_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if (batch_idx + 1) % self.config.print_freq == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/num_batches:.4f}',
                    'lr': f'{self.optimizer.param_groups[-1]["lr"]:.6f}'
                })
        
        # Check if any batches were processed
        if num_batches == 0:
            raise RuntimeError(
                "No batches were processed in this epoch. "
                "This might indicate a data loading issue. "
                f"Dataset size: {len(self.train_dataset)}, Batch size: {self.config.batch_size}, "
                f"Train loader length: {len(self.train_loader)}"
            )
        
        if skipped_batches > 0 or forward_errors > 0:
            print(
                f"Epoch {epoch+1}: Skipped {skipped_batches} batches due to invalid losses "
                f"and {forward_errors} batches due to NaN/Inf features."
            )
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def linear_evaluation(self, epoch: int) -> float:
        """
        Perform linear evaluation to assess representation quality.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Best test accuracy from linear evaluation.
        """
        print(f"\n{'='*70}")
        print(f"Linear Evaluation at Epoch {epoch+1}")
        print(f"{'='*70}")
        
        # Create linear evaluator
        evaluator = LinearEvaluator(
            backbone=self.model.get_backbone(),
            device=self.device,
            num_classes=10,
            image_size=self.config.image_size,
            data_root=self.config.data_root,
            batch_size=128,
            num_workers=self.config.num_workers,
            learning_rate=0.01,
            epochs=50,  # Train linear classifier for 50 epochs
            mixed_precision=self.config.mixed_precision
        )
        
        # Train and evaluate
        best_acc = evaluator.train_classifier()
        
        print(f"{'='*70}")
        print(f"Linear Evaluation Best Accuracy: {best_acc:.2f}%")
        print(f"{'='*70}\n")
        
        return best_acc
    
    def train(self) -> None:
        """Main training loop for self-supervised learning."""
        print("\n" + "="*70)
        print("Starting Self-Supervised Learning (SimCLR)")
        print("="*70)
        print(f"Total epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"Linear evaluation frequency: every {self.config.eval_freq} epochs")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            ssl_loss = self.train_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Store metrics
            self.ssl_losses.append(ssl_loss)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.config.epochs} Summary:")
            print(f"  SSL Loss: {ssl_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[-1]['lr']:.6f}")
            print(f"  Time: {format_time(epoch_time)}")
            
            # Perform linear evaluation
            linear_acc = None
            is_best = False
            
            if (epoch + 1) % self.config.eval_freq == 0 or epoch == self.config.epochs - 1:
                linear_acc = self.linear_evaluation(epoch)
                self.linear_accs.append((epoch + 1, linear_acc))
                
                # Check for best model
                if linear_acc > self.best_linear_acc:
                    self.best_linear_acc = linear_acc
                    is_best = True
                    print(f"New best linear evaluation accuracy: {self.best_linear_acc:.2f}%")
            
            # Save checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_linear_acc": self.best_linear_acc,
                "ssl_losses": self.ssl_losses,
                "linear_accs": self.linear_accs,
                "config": self.config.to_dict()
            }
            
            checkpoint_path = Path(self.config.checkpoint_dir) / f"ssl_checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                checkpoint_state,
                str(checkpoint_path),
                is_best=is_best,
                keep_last_n=self.config.save_freq
            )
            
            print("-" * 70)
        
        total_time = time.time() - start_time
        print(f"\nSSL Training completed in {format_time(total_time)}")
        print(f"Best linear evaluation accuracy: {self.best_linear_acc:.2f}%")
        print(f"Final SSL model saved to: {self.config.checkpoint_dir}/best_model.pth\n")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Self-supervised learning with SimCLR for ViT on STL-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="vit_base_patch16_224",
        help="ViT backbone name from timm"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use pretrained ImageNet weights (usually False for SSL)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=128,
        help="Projection head output dimension"
    )
    parser.add_argument(
        "--projection_hidden_dim",
        type=int,
        default=2048,
        help="Projection head hidden dimension"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (larger is better for contrastive learning)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.03,
        help="Base learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs"
    )
    
    # Learning rate scheduling
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", None],
        help="Learning rate scheduler"
    )
    
    # Training settings
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm"
    )
    
    # Paths
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints_ssl",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs_ssl",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Evaluation
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Linear evaluation frequency (every N epochs)"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Logging
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="Print frequency (every N batches)"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="Number of checkpoints to keep"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run SSL training."""
    # Parse arguments
    args = parse_args()
    
    # Create config from arguments
    config = SSLConfig(
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        image_size=args.image_size,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        mixed_precision=args.mixed_precision,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        eval_freq=args.eval_freq,
        seed=args.seed,
        print_freq=args.print_freq,
        save_freq=args.save_freq
    )
    
    # Create trainer and start training
    try:
        trainer = SSLTrainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving checkpoint...")
        # Could add emergency checkpoint save here
        print("Exiting gracefully.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()

