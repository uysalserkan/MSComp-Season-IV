"""
Utility functions for ViT finetuning training.

This module provides helper functions for reproducibility, device management,
checkpointing, and metrics calculation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from collections import defaultdict

try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support,
        accuracy_score,
        top_k_accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Confusion matrix visualization disabled.")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (may be slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    Get the available device (CUDA if available, else CPU).
    
    Returns:
        torch.device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    return device


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Dict[int, float]:
    """
    Calculate top-k classification accuracy.
    
    Args:
        outputs: Model predictions (logits) of shape (batch_size, num_classes).
        targets: Ground truth labels of shape (batch_size,).
        topk: Tuple of k values for top-k accuracy (e.g., (1, 3, 5)).
    
    Returns:
        Dictionary mapping k to accuracy (float between 0 and 1).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = (correct_k.mul_(100.0 / batch_size)).item() / 100.0
        
        return res


def calculate_accuracy_simple(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate simple top-1 classification accuracy (backward compatibility).
    
    Args:
        outputs: Model predictions (logits) of shape (batch_size, num_classes).
        targets: Ground truth labels of shape (batch_size,).
    
    Returns:
        Accuracy as a float between 0 and 1.
    """
    acc_dict = calculate_accuracy(outputs, targets, topk=(1,))
    return acc_dict[1]


def save_checkpoint(
    state: Dict,
    filepath: str,
    is_best: bool = False,
    keep_last_n: int = 5
) -> None:
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, epoch, etc.
        filepath: Path to save the checkpoint.
        is_best: If True, also save as 'best_model.pth'.
        keep_last_n: Keep only the last N checkpoints (delete older ones).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(state, filepath)
    
    # Save as best model if applicable
    if is_best:
        best_path = filepath.parent / "best_model.pth"
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")
    
    # Clean up old checkpoints
    if keep_last_n > 0:
        checkpoint_dir = filepath.parent
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0
        )
        
        if len(checkpoints) > keep_last_n:
            for old_checkpoint in checkpoints[:-keep_last_n]:
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to load the checkpoint on.
    
    Returns:
        Dictionary containing checkpoint information (epoch, best_acc, etc.).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Loaded checkpoint from {filepath}")
    
    return checkpoint


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict:
    """
    Get summary of model parameters and architecture.
    
    Args:
        model: PyTorch model.
        input_size: Input tensor size (batch, channels, height, width).
    
    Returns:
        Dictionary with model summary information.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    # Try to get forward pass time (if possible)
    try:
        model.eval()
        dummy_input = torch.randn(input_size)
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        forward_time = None  # Could add timing here if needed
    except Exception:
        forward_time = None
    
    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb,
        "forward_time": forward_time,
    }
    
    return summary


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted time string (e.g., "1h 23m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary with parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification tasks.
    
    Calculates various metrics including accuracy, precision, recall, F1-score,
    confusion matrix, and per-class metrics.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes.
            class_names: Optional list of class names. If None, uses "Class 0", "Class 1", etc.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
        # Storage for predictions and targets
        self.all_predictions = []
        self.all_targets = []
        self.all_probs = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions.
        
        Args:
            outputs: Model logits of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).
        """
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            self.all_predictions.extend(preds.cpu().numpy())
            self.all_targets.extend(targets.cpu().numpy())
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict:
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics.
        """
        if len(self.all_predictions) == 0:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)
        
        metrics = {}
        
        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # Top-k accuracy
        if SKLEARN_AVAILABLE:
            for k in [1, 3, 5]:
                if k <= self.num_classes:
                    try:
                        topk_acc = top_k_accuracy_score(targets, probs, k=k, labels=range(self.num_classes))
                        metrics[f'top_{k}_accuracy'] = topk_acc
                    except:
                        pass
        
        # Per-class metrics
        if SKLEARN_AVAILABLE:
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, labels=range(self.num_classes), zero_division=0
            )
            
            metrics['per_class'] = {}
            for i in range(self.num_classes):
                metrics['per_class'][self.class_names[i]] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
            
            # Macro averages
            metrics['macro_precision'] = float(precision.mean())
            metrics['macro_recall'] = float(recall.mean())
            metrics['macro_f1'] = float(f1.mean())
            
            # Weighted averages
            metrics['weighted_precision'] = float(np.average(precision, weights=support))
            metrics['weighted_recall'] = float(np.average(recall, weights=support))
            metrics['weighted_f1'] = float(np.average(f1, weights=support))
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(
                targets, predictions, labels=range(self.num_classes)
            ).tolist()
            
            # Classification report
            metrics['classification_report'] = classification_report(
                targets, predictions,
                target_names=self.class_names,
                labels=range(self.num_classes),
                output_dict=True,
                zero_division=0
            )
        
        return metrics
    
    def reset(self):
        """Reset all stored predictions and targets."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probs = []
    
    def print_summary(self, metrics: Optional[Dict] = None):
        """
        Print a formatted summary of metrics.
        
        Args:
            metrics: Optional metrics dictionary. If None, computes metrics first.
        """
        if metrics is None:
            metrics = self.compute()
        
        print("\n" + "="*70)
        print("EVALUATION METRICS SUMMARY")
        print("="*70)
        
        # Overall accuracy
        print(f"\nOverall Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        # Top-k accuracy
        for k in [1, 3, 5]:
            key = f'top_{k}_accuracy'
            if key in metrics:
                print(f"Top-{k} Accuracy: {metrics[key]:.4f}")
        
        # Macro averages
        if 'macro_precision' in metrics:
            print(f"\nMacro Averages:")
            print(f"  Precision: {metrics['macro_precision']:.4f}")
            print(f"  Recall:    {metrics['macro_recall']:.4f}")
            print(f"  F1-Score:  {metrics['macro_f1']:.4f}")
        
        # Weighted averages
        if 'weighted_precision' in metrics:
            print(f"\nWeighted Averages:")
            print(f"  Precision: {metrics['weighted_precision']:.4f}")
            print(f"  Recall:    {metrics['weighted_recall']:.4f}")
            print(f"  F1-Score:  {metrics['weighted_f1']:.4f}")
        
        # Per-class metrics
        if 'per_class' in metrics:
            print(f"\nPer-Class Metrics:")
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print("-" * 70)
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"{class_name:<20} "
                      f"{class_metrics['precision']:<12.4f} "
                      f"{class_metrics['recall']:<12.4f} "
                      f"{class_metrics['f1_score']:<12.4f} "
                      f"{class_metrics['support']:<10}")
        
        print("="*70 + "\n")
    
    def save_confusion_matrix(
        self,
        metrics: Optional[Dict] = None,
        save_path: str = "confusion_matrix.png",
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Save confusion matrix visualization.
        
        Args:
            metrics: Optional metrics dictionary. If None, computes metrics first.
            save_path: Path to save the confusion matrix image.
            figsize: Figure size (width, height).
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Cannot save confusion matrix.")
            return
        
        if metrics is None:
            metrics = self.compute()
        
        if 'confusion_matrix' not in metrics:
            print("Warning: Confusion matrix not available.")
            return
        
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def save_metrics_to_file(self, metrics: Optional[Dict] = None, save_path: str = "metrics.txt"):
        """
        Save metrics to a text file.
        
        Args:
            metrics: Optional metrics dictionary. If None, computes metrics first.
            save_path: Path to save the metrics file.
        """
        if metrics is None:
            metrics = self.compute()
        
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*70 + "\n\n")
            
            # Overall metrics
            f.write(f"Overall Accuracy: {metrics.get('accuracy', 0):.4f}\n\n")
            
            for k in [1, 3, 5]:
                key = f'top_{k}_accuracy'
                if key in metrics:
                    f.write(f"Top-{k} Accuracy: {metrics[key]:.4f}\n")
            
            # Macro and weighted averages
            if 'macro_precision' in metrics:
                f.write(f"\nMacro Averages:\n")
                f.write(f"  Precision: {metrics['macro_precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['macro_recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['macro_f1']:.4f}\n")
            
            if 'weighted_precision' in metrics:
                f.write(f"\nWeighted Averages:\n")
                f.write(f"  Precision: {metrics['weighted_precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['weighted_recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['weighted_f1']:.4f}\n")
            
            # Per-class metrics
            if 'per_class' in metrics:
                f.write(f"\nPer-Class Metrics:\n")
                f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
                f.write("-" * 70 + "\n")
                for class_name, class_metrics in metrics['per_class'].items():
                    f.write(f"{class_name:<20} "
                           f"{class_metrics['precision']:<12.4f} "
                           f"{class_metrics['recall']:<12.4f} "
                           f"{class_metrics['f1_score']:<12.4f} "
                           f"{class_metrics['support']:<10}\n")
            
            # Classification report
            if 'classification_report' in metrics:
                f.write(f"\n\nDetailed Classification Report:\n")
                f.write("="*70 + "\n")
                from sklearn.metrics import classification_report
                if len(self.all_predictions) > 0:
                    report = classification_report(
                        np.array(self.all_targets),
                        np.array(self.all_predictions),
                        target_names=self.class_names,
                        labels=range(self.num_classes),
                        zero_division=0
                    )
                    f.write(report)
        
        print(f"Metrics saved to {save_path}")

