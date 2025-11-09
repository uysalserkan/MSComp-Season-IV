"""
Evaluation script for supervised ViT models on STL-10.

This script evaluates a trained ViT model and provides comprehensive metrics
including accuracy, precision, recall, F1-score, confusion matrix, and per-class metrics.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from models.vit_model import ViTFinetuner
from stl10_dataset import STL10DatasetLoader
from utils import (
    set_seed,
    get_device,
    calculate_accuracy_simple,
    load_checkpoint,
    MetricsCalculator
)


class ModelEvaluator:
    """
    Evaluator for supervised ViT models.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 10,
        model_name: str = "vit_base_patch16_224",
        image_size: int = 224,
        batch_size: int = 128,
        data_root: str = "./data",
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Initialize model evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            num_classes: Number of classes (10 for STL-10).
            model_name: ViT model name.
            image_size: Input image size.
            batch_size: Batch size for evaluation.
            data_root: Dataset root directory.
            num_workers: Number of data loading workers.
            seed: Random seed.
        """
        # Set seed
        set_seed(seed, deterministic=True)
        
        # Get device
        self.device = get_device()
        
        # Load model
        print("Loading model checkpoint...")
        self.model = ViTFinetuner(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,  # We'll load from checkpoint
            image_size=image_size
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, self.model, device=self.device)
        
        self.model.eval()
        
        # Initialize dataset loader
        self.dataset_loader = STL10DatasetLoader(root=data_root, download=True)
        
        # Get data loaders
        _, self.test_loader = self.dataset_loader.get_vit_data_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            use_augmentation=False,  # No augmentation for evaluation
            pin_memory=True
        )
        
        # Get class names from dataset
        test_dataset = self.dataset_loader.get_test_dataset()
        self.class_names = test_dataset.classes if hasattr(test_dataset, 'classes') else None
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=num_classes,
            class_names=self.class_names
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self) -> dict:
        """
        Evaluate the model on test set.
        
        Returns:
            Dictionary containing all computed metrics.
        """
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        
        print("\n" + "="*70)
        print("Evaluating Model on Test Set")
        print("="*70 + "\n")
        
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                acc = calculate_accuracy_simple(outputs, labels)
                running_loss += loss.item()
                running_acc += acc
                num_batches += 1
                
                # Update metrics calculator
                self.metrics_calculator.update(outputs, labels)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/num_batches:.4f}',
                    'acc': f'{running_acc/num_batches:.4f}'
                })
        
        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches
        
        print(f"\nTest Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {avg_acc:.4f}")
        
        # Compute comprehensive metrics
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION METRICS")
        print("="*70)
        metrics = self.metrics_calculator.compute()
        self.metrics_calculator.print_summary(metrics)
        
        return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate supervised ViT model on STL-10")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="ViT model name")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Dataset root directory")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save evaluation results and confusion matrix")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint_path,
        num_classes=args.num_classes,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    # Save results if requested
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to file
        evaluator.metrics_calculator.save_metrics_to_file(
            metrics,
            save_path=str(output_dir / "supervised_eval_metrics.txt")
        )
        
        # Save confusion matrix
        evaluator.metrics_calculator.save_confusion_matrix(
            metrics,
            save_path=str(output_dir / "supervised_eval_confusion_matrix.png")
        )


if __name__ == "__main__":
    main()

