"""
Linear evaluation script for SSL pretrained models.

This script implements the linear evaluation protocol to measure the quality
of self-supervised learned representations by training a linear classifier
on frozen features.
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
from models.vit_model import ViTFinetuner
from stl10_dataset import STL10DatasetLoader
from ssl_config import SSLConfig
from utils import (
    set_seed,
    get_device,
    calculate_accuracy_simple,
    format_time,
    count_parameters,
    MetricsCalculator
)


class LinearEvaluator:
    """
    Linear evaluator for SSL pretrained models.
    
    Freezes the backbone and trains only a linear classifier.
    """
    
    def __init__(
        self,
        ssl_checkpoint_path: str,
        num_classes: int = 10,
        backbone_name: str = "vit_base_patch16_224",
        image_size: int = 224,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        epochs: int = 100,
        data_root: str = "./data",
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Initialize linear evaluator.
        
        Args:
            ssl_checkpoint_path: Path to SSL pretrained checkpoint.
            num_classes: Number of classes (10 for STL-10).
            backbone_name: ViT backbone name.
            image_size: Input image size.
            batch_size: Batch size for evaluation.
            learning_rate: Learning rate for linear classifier.
            epochs: Number of training epochs.
            data_root: Dataset root directory.
            num_workers: Number of data loading workers.
            seed: Random seed.
        """
        # Set seed
        set_seed(seed, deterministic=True)
        
        # Get device
        self.device = get_device()
        
        # Load SSL model
        print("Loading SSL pretrained model...")
        checkpoint = torch.load(ssl_checkpoint_path, map_location='cpu')
        
        # Create SSL model
        ssl_config = SSLConfig()
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            ssl_config = SSLConfig.from_dict(checkpoint["config"])
        
        self.backbone_model = ContrastiveViT(
            backbone_name=backbone_name,
            pretrained=False,
            image_size=image_size,
            projection_dim=ssl_config.projection_dim,
            projection_hidden_dim=ssl_config.projection_hidden_dim,
        )
        
        # Load SSL weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.backbone_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.backbone_model.load_state_dict(checkpoint)
        
        # Freeze backbone
        self.backbone_model.freeze_backbone()
        self.backbone_model = self.backbone_model.to(self.device)
        self.backbone_model.eval()
        
        # Get backbone feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size).to(self.device)
            features = self.backbone_model.extract_features(dummy_input)
            feature_dim = features.shape[1]
        
        # Create linear classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(self.device)
        
        # Print summary
        print("\n" + "="*50)
        print("Linear Evaluation Setup")
        print("="*50)
        backbone_params = count_parameters(self.backbone_model)
        classifier_params = count_parameters(self.classifier)
        print(f"Backbone parameters: {backbone_params['total']:,} (frozen)")
        print(f"Classifier parameters: {classifier_params['total']:,} (trainable)")
        print("="*50 + "\n")
        
        # Initialize dataset loader
        self.dataset_loader = STL10DatasetLoader(root=data_root, download=True)
        
        # Get data loaders
        self.train_loader, self.val_loader = self.dataset_loader.get_vit_data_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            use_augmentation=False,  # No augmentation for linear evaluation
            pin_memory=True
        )
        
        # Get class names from dataset
        train_dataset = self.dataset_loader.get_train_dataset()
        self.class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else None
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=num_classes,
            class_names=self.class_names
        )
        
        # Loss and optimizer (only for classifier)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=0.0
        )
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_results = True  # Default to saving results
    
    def train_epoch(self, epoch: int) -> tuple:
        """
        Train linear classifier for one epoch.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.classifier.train()
        self.backbone_model.eval()  # Ensure backbone is frozen
        
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Extract features with backbone (frozen)
            with torch.no_grad():
                features = self.backbone_model.extract_features(images)
            
            # Forward pass through classifier
            logits = self.classifier(features)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            acc = calculate_accuracy_simple(logits, labels)
            running_loss += loss.item()
            running_acc += acc
            num_batches += 1
            
            # Update progress bar
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
        Validate linear classifier.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.classifier.eval()
        self.backbone_model.eval()
        
        # Reset metrics calculator for this epoch
        self.metrics_calculator.reset()
        
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                features = self.backbone_model.extract_features(images)
                
                # Forward pass
                logits = self.classifier(features)
                loss = self.criterion(logits, labels)
                
                # Calculate metrics
                acc = calculate_accuracy_simple(logits, labels)
                running_loss += loss.item()
                running_acc += acc
                num_batches += 1
                
                # Update metrics calculator
                self.metrics_calculator.update(logits, labels)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/num_batches:.4f}',
                    'acc': f'{running_acc/num_batches:.4f}'
                })
        
        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches
        
        return avg_loss, avg_acc
    
    def evaluate(self):
        """Run linear evaluation."""
        print("\n" + "="*50)
        print("Starting Linear Evaluation")
        print("="*50)
        print(f"Total epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print("="*50 + "\n")
        
        best_val_acc = 0.0
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Time: {format_time(epoch_time)}")
            
            # Track best accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            print("-" * 50)
        
        print(f"\nLinear Evaluation completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Compute and print comprehensive metrics
        print("\n" + "="*70)
        print("FINAL EVALUATION METRICS (on validation set)")
        print("="*70)
        final_metrics = self.metrics_calculator.compute()
        self.metrics_calculator.print_summary(final_metrics)
        
        # Save metrics and confusion matrix if requested
        if hasattr(self, 'save_results') and self.save_results:
            import os
            save_dir = Path("./evaluation_results")
            save_dir.mkdir(exist_ok=True)
            
            self.metrics_calculator.save_metrics_to_file(
                final_metrics,
                save_path=str(save_dir / "ssl_linear_eval_metrics.txt")
            )
            
            self.metrics_calculator.save_confusion_matrix(
                final_metrics,
                save_path=str(save_dir / "ssl_linear_eval_confusion_matrix.png")
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Linear evaluation for SSL pretrained models")
    
    parser.add_argument("--ssl_checkpoint_path", type=str, required=True,
                        help="Path to SSL pretrained checkpoint")
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_224",
                        help="ViT backbone name")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Dataset root directory")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save evaluation results and confusion matrix")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    evaluator = LinearEvaluator(
        ssl_checkpoint_path=args.ssl_checkpoint_path,
        num_classes=args.num_classes,
        backbone_name=args.backbone_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        data_root=args.data_root,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    evaluator.save_results = args.save_results
    evaluator.evaluate()


if __name__ == "__main__":
    main()

