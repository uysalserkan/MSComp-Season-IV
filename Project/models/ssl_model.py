"""
Self-Supervised Learning (SSL) model for contrastive learning with ViT.

This module provides a ContrastiveViT class that wraps a ViT backbone
and adds a projection head for contrastive learning (SimCLR, MoCo, etc.).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import timm


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Typically a 2-3 layer MLP that projects backbone features to
    a lower-dimensional space for contrastive learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
        use_batch_norm: bool = True,
        use_bias: bool = False,
    ):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input feature dimension (backbone output dimension).
            hidden_dim: Hidden layer dimension.
            output_dim: Output projection dimension.
            num_layers: Number of layers in projection head (2 or 3).
            use_batch_norm: If True, use batch normalization.
            use_bias: If True, use bias in final layer (typically False for normalized embeddings).
        """
        super(ProjectionHead, self).__init__()
        
        assert num_layers in [2, 3], "num_layers must be 2 or 3"
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layer (only for 3-layer head)
        if num_layers == 3:
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # Final projection layer (no bias for normalized embeddings)
        layers.append(nn.Linear(hidden_dim, output_dim, bias=use_bias))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.
        
        Args:
            x: Input features of shape (batch_size, input_dim).
        
        Returns:
            Projected features of shape (batch_size, output_dim).
        """
        return self.projection(x)


class ContrastiveViT(nn.Module):
    """
    Vision Transformer with projection head for contrastive learning.
    
    This class wraps a ViT backbone and adds a projection head for
    self-supervised contrastive learning (SimCLR, MoCo, etc.).
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        image_size: int = 224,
        projection_dim: int = 128,
        projection_hidden_dim: int = 2048,
        projection_num_layers: int = 3,
        freeze_backbone: bool = False,
    ):
        """
        Initialize contrastive ViT model.
        
        Args:
            backbone_name: Name of the ViT backbone from timm.
            pretrained: If True, loads pretrained ImageNet weights.
            image_size: Input image size.
            projection_dim: Output dimension of projection head.
            projection_hidden_dim: Hidden dimension of projection head.
            projection_num_layers: Number of layers in projection head (2 or 3).
            freeze_backbone: If True, freezes backbone weights.
        """
        super(ContrastiveViT, self).__init__()
        
        self.backbone_name = backbone_name
        self.image_size = image_size
        
        # Load ViT backbone from timm (without classification head)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            img_size=image_size,
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            backbone_output = self.backbone(dummy_input)
            backbone_dim = backbone_output.shape[1]
        
        # Create projection head
        self.projection_head = ProjectionHead(
            input_dim=backbone_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            num_layers=projection_num_layers,
            use_batch_norm=True,
            use_bias=False,  # No bias for normalized embeddings
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        normalize: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input images of shape (batch_size, 3, image_size, image_size).
            return_features: If True, also return backbone features.
            normalize: If True, normalize projection output (L2 normalization).
        
        Returns:
            If return_features=False: Projected features of shape (batch_size, projection_dim).
            If return_features=True: Tuple of (projected_features, backbone_features).
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to contrastive space
        projection = self.projection_head(features)
        
        # Normalize projection (important for contrastive learning)
        if normalize:
            # Add small epsilon to prevent division by zero for zero vectors
            # This ensures numerical stability during normalization
            projection = nn.functional.normalize(projection + 1e-8, dim=1)
        
        if return_features:
            return projection, features
        return projection
    
    def get_backbone(self) -> nn.Module:
        """
        Get the backbone model (without projection head).
        
        Returns:
            Backbone model.
        """
        return self.backbone
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone (without projection head).
        
        Args:
            x: Input images of shape (batch_size, 3, image_size, image_size).
        
        Returns:
            Backbone features of shape (batch_size, backbone_dim).
        """
        return self.backbone(x)
    
    def count_parameters(self) -> dict:
        """
        Count total and trainable parameters.
        
        Returns:
            Dictionary with 'total', 'trainable', and 'frozen' parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        projection_params = sum(p.numel() for p in self.projection_head.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'backbone': backbone_params,
            'projection': projection_params
        }

