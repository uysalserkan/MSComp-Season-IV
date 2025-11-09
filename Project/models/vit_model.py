"""
Vision Transformer (ViT) model for finetuning on STL-10 dataset.

This module provides a ViTFinetuner class that loads pretrained ViT models
from timm and adapts them for the STL-10 classification task.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import timm


class ViTFinetuner(nn.Module):
    """
    Vision Transformer model for finetuning on STL-10 dataset.
    
    This class loads a pretrained ViT model from timm and replaces the
    classification head to match the number of classes in STL-10 (10 classes).
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 10,
        pretrained: bool = True,
        image_size: int = 224,
        freeze_backbone: bool = False,
        ssl_pretrained_path: Optional[str] = None,
    ):
        """
        Initialize ViT model for finetuning.
        
        Args:
            model_name: Name of the ViT model from timm (e.g., 'vit_base_patch16_224').
            num_classes: Number of output classes (10 for STL-10).
            pretrained: If True, loads pretrained weights from ImageNet.
            image_size: Input image size (default 224x224).
            freeze_backbone: If True, freezes all layers except the classification head.
            ssl_pretrained_path: Path to SSL pretrained checkpoint. If provided, loads SSL weights instead of ImageNet.
        """
        super(ViTFinetuner, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Determine if we should use pretrained weights
        use_pretrained = pretrained and ssl_pretrained_path is None
        
        # Load pretrained ViT model from timm
        self.model = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,  # Remove the original classification head
            img_size=image_size,
        )
        
        # Load SSL pretrained weights if provided
        if ssl_pretrained_path:
            self._load_ssl_weights(ssl_pretrained_path)
        
        # Get the feature dimension from the model
        # For ViT models, this is typically the embedding dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            features = self.model(dummy_input)
            feature_dim = features.shape[1]
        
        # Create new classification head for STL-10
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Initialize classification head weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def _load_ssl_weights(self, checkpoint_path: str) -> None:
        """
        Load SSL pretrained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to SSL checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Extract backbone weights (remove projection head keys)
        backbone_state_dict = {}
        for key, value in state_dict.items():
            # Skip projection head parameters
            if not key.startswith("projection_head"):
                # Remove "backbone." prefix if present
                new_key = key.replace("backbone.", "")
                backbone_state_dict[new_key] = value
        
        # Load backbone weights
        missing_keys, unexpected_keys = self.model.load_state_dict(backbone_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading SSL weights: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading SSL weights: {unexpected_keys[:5]}...")
        
        print(f"Loaded SSL pretrained weights from {checkpoint_path}")
    
    def freeze_backbone(self) -> None:
        """Freeze all backbone layers, keeping only classifier trainable."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first N transformer blocks.
        
        Args:
            num_layers: Number of transformer blocks to freeze from the beginning.
        """
        if not hasattr(self.model, 'blocks'):
            raise ValueError("Model does not have 'blocks' attribute")
        
        total_blocks = len(self.model.blocks)
        if num_layers > total_blocks:
            raise ValueError(f"Cannot freeze {num_layers} layers, model has only {total_blocks} blocks")
        
        # Freeze specified number of blocks
        for i in range(num_layers):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, image_size, image_size).
        
        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # Extract features from backbone
        features = self.model(x)
        
        # Pass through classification head
        logits = self.classifier(features)
        
        return logits
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters.
        
        Returns:
            List of trainable parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> dict:
        """
        Count total and trainable parameters.
        
        Returns:
            Dictionary with 'total' and 'trainable' parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }

