"""
Configuration settings for self-supervised contrastive learning on STL-10 dataset.

This module provides a centralized configuration class for SSL training parameters.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class SSLConfig:
    """
    Configuration class for self-supervised contrastive learning training.
    
    This class holds all hyperparameters and settings needed for SSL training.
    """
    
    # Model parameters
    backbone_name: str = "vit_base_patch16_224"
    pretrained: bool = False  # Start from scratch for SSL
    image_size: int = 224
    projection_dim: int = 128
    projection_hidden_dim: int = 2048
    projection_num_layers: int = 3
    freeze_backbone: bool = False
    
    # Contrastive learning method
    method: str = "simclr"  # Options: "simclr", "moco", "byol"
    
    # Contrastive learning hyperparameters
    temperature: float = 0.07  # Temperature for contrastive loss
    memory_bank_size: int = 4096  # For MoCo
    momentum: float = 0.999  # For MoCo momentum encoder
    
    # Data parameters
    data_root: str = "./data"
    batch_size: int = 256  # Large batch size for contrastive learning
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training hyperparameters
    epochs: int = 100
    learning_rate: float = 5e-4  # Base learning rate (lower for AdamW)
    weight_decay: float = 1e-4
    warmup_epochs: int = 15  # Learning rate warmup
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # Options: "cosine", "step", None
    lr_min: float = 1e-6
    
    # Training settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: Optional[float] = 1.0
    
    # Data augmentation
    use_strong_augmentation: bool = True
    
    # Paths
    checkpoint_dir: str = "./checkpoints_ssl"
    log_dir: str = "./logs_ssl"
    resume_from: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    print_freq: int = 10  # Print metrics every N batches
    save_freq: int = 10  # Save checkpoint every N epochs
    use_tensorboard: bool = False
    
    # Evaluation
    eval_freq: int = 10  # Evaluate every N epochs (linear evaluation)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.temperature > 0, "temperature must be positive"
        assert self.method in ["simclr", "moco", "byol"], \
            "method must be one of: simclr, moco, byol"
        assert self.lr_scheduler in ["cosine", "step", None], \
            "lr_scheduler must be one of: cosine, step, None"
        assert 0 < self.momentum < 1, "momentum must be between 0 and 1"
        
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
    def from_dict(cls, config_dict: dict) -> 'SSLConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

