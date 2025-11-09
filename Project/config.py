"""
Configuration settings for ViT finetuning on STL-10 dataset.

This module provides a centralized configuration class for all training parameters.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration class for ViT finetuning training.
    
    This class holds all hyperparameters and settings needed for training.
    """
    
    # Model parameters
    model_name: str = "vit_base_patch16_224"
    num_classes: int = 10  # STL-10 has 10 classes
    pretrained: bool = True
    image_size: int = 224
    freeze_backbone: bool = False
    
    # Data parameters
    data_root: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training hyperparameters
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    momentum: float = 0.9
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # Options: "cosine", "step", "onecycle", None
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6
    
    # Training settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: Optional[float] = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    resume_from: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    print_freq: int = 10  # Print metrics every N batches
    save_freq: int = 5  # Save checkpoint every N epochs
    use_tensorboard: bool = False
    
    # Validation
    val_split: float = 0.1  # Fraction of training data to use for validation
    
    # SSL pretraining
    ssl_pretrained_path: Optional[str] = None  # Path to SSL checkpoint
    use_ssl_pretrained: bool = False  # Flag to use SSL pretrained weights
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert self.lr_scheduler in ["cosine", "step", "onecycle", None], \
            "lr_scheduler must be one of: cosine, step, onecycle, None"
        
        # Create directories if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

