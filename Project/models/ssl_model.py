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
    
    def _initialize_weights(self):
        """Initialize weights to prevent extreme values that could cause NaN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier/Kaiming initialization with smaller gain
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm with stable values
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
                # Initialize running stats to prevent division by zero
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.
        
        Args:
            x: Input features of shape (batch_size, input_dim).
        
        Returns:
            Projected features of shape (batch_size, output_dim).
        """
        # Check input for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(
                f"Projection head input contains NaN/Inf. Input stats: "
                f"min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}"
            )
        
        output = self.projection(x)
        
        # Check output for NaN/Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError(
                f"Projection head output contains NaN/Inf. Output stats: "
                f"min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}"
            )
        
        return output


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
        
        # Initialize projection head weights for stability
        self.projection_head._initialize_weights()
        
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
        # Validate input
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Check backbone features for NaN/Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError(
                f"Backbone output contains NaN/Inf. Features stats: "
                f"min={features.min():.4f}, max={features.max():.4f}, "
                f"mean={features.mean():.4f}, std={features.std():.4f}"
            )
        
        # Project to contrastive space
        projection = self.projection_head(features)
        
        # Check projection output before normalization
        if torch.isnan(projection).any() or torch.isinf(projection).any():
            raise ValueError(
                f"Projection head output contains NaN/Inf before normalization. "
                f"Projection stats: min={projection.min():.4f}, max={projection.max():.4f}, "
                f"mean={projection.mean():.4f}, std={projection.std():.4f}"
            )
        
        # Normalize projection (important for contrastive learning)
        if normalize:
            # Compute norm for each sample
            norms = torch.norm(projection, dim=1, keepdim=True)
            # Replace zero or very small norms with epsilon to prevent division by zero
            norms = torch.clamp(norms, min=1e-8)
            # Normalize
            projection = projection / norms
        
        # Final check after normalization
        if torch.isnan(projection).any() or torch.isinf(projection).any():
            raise ValueError(
                f"Projection contains NaN/Inf after normalization. "
                f"Final projection stats: min={projection.min():.4f}, max={projection.max():.4f}, "
                f"mean={projection.mean():.4f}"
            )
        
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

