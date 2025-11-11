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
        
        # Initialize backbone weights for numerical stability (especially patch_embed)
        self._initialize_backbone()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # Optionally freeze positional embeddings (often causes NaN gradients)
        # This can be enabled via a method call after initialization
        self._freeze_pos_embed = False
    
    def _initialize_backbone(self) -> None:
        """
        Initialize backbone weights for numerical stability.
        
        Applies very conservative initialization to patch embedding convolution
        to prevent NaN gradients in early training.
        """
        for name, module in self.backbone.named_modules():
            if 'patch_embed' in name and isinstance(module, nn.Conv2d):
                # Very conservative initialization for patch embedding
                # Using Xavier with very small gain to prevent extreme activations
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                print(f"Initialized {name} with conservative gain (0.02) for stability")
    
    def freeze_pos_embed(self) -> None:
        """Freeze positional embeddings to prevent NaN gradient issues."""
        for name, param in self.backbone.named_parameters():
            if 'pos_embed' in name:
                param.requires_grad = False
                self._freeze_pos_embed = True
        if self._freeze_pos_embed:
            print("Positional embeddings frozen to prevent NaN gradients")
    
    def unfreeze_pos_embed(self) -> None:
        """Unfreeze positional embeddings."""
        for name, param in self.backbone.named_parameters():
            if 'pos_embed' in name:
                param.requires_grad = True
                self._freeze_pos_embed = False
    
    def freeze_patch_embed(self) -> None:
        """Freeze patch embedding layer to prevent NaN issues during early training."""
        for name, param in self.backbone.named_parameters():
            if 'patch_embed' in name:
                param.requires_grad = False
        print("Patch embedding frozen for stable early training")
    
    def unfreeze_patch_embed(self) -> None:
        """Unfreeze patch embedding layer."""
        for name, param in self.backbone.named_parameters():
            if 'patch_embed' in name:
                param.requires_grad = True
        print("Patch embedding unfrozen")
    
    def freeze_early_blocks(self, num_blocks: int = 2) -> None:
        """
        Freeze first N transformer blocks to prevent NaN issues.
        
        Args:
            num_blocks: Number of early blocks to freeze (default: 2)
        """
        for name, param in self.backbone.named_parameters():
            if 'blocks' in name:
                try:
                    block_num = int(name.split('blocks.')[1].split('.')[0])
                    if block_num < num_blocks:
                        param.requires_grad = False
                except (IndexError, ValueError):
                    pass
        print(f"First {num_blocks} transformer blocks frozen for stable early training")
    
    def unfreeze_early_blocks(self) -> None:
        """Unfreeze all transformer blocks."""
        for name, param in self.backbone.named_parameters():
            if 'blocks' in name:
                param.requires_grad = True
        print("All transformer blocks unfrozen")
    
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
    
    def get_param_groups(self, base_lr: float, weight_decay: float = 1e-4) -> list:
        """
        Get parameter groups with layer-wise discriminative learning rates.
        
        Early layers (patch_embed, pos_embed) get lower LRs to prevent NaN gradients.
        Later layers get progressively higher LRs.
        
        Args:
            base_lr: Base learning rate for late layers and projection head.
            weight_decay: Weight decay for all parameter groups.
        
        Returns:
            List of parameter group dictionaries for optimizer.
        """
        patch_embed_params = []
        pos_embed_params = []
        early_block_params = []
        middle_block_params = []
        late_block_params = []
        other_params = []
        projection_params = list(self.projection_head.parameters())
        
        for name, param in self.backbone.named_parameters():
            if 'patch_embed' in name:
                patch_embed_params.append(param)
            elif 'pos_embed' in name or 'cls_token' in name:
                pos_embed_params.append(param)
            elif 'blocks' in name:
                # Extract block number from parameter name
                # Name format: blocks.N.xxx where N is the block number
                try:
                    block_num = int(name.split('blocks.')[1].split('.')[0])
                    if block_num < 4:
                        early_block_params.append(param)
                    elif block_num < 8:
                        middle_block_params.append(param)
                    else:
                        late_block_params.append(param)
                except (IndexError, ValueError):
                    # If we can't parse block number, put in other_params
                    other_params.append(param)
            else:
                # Any other backbone parameters (norm layers, etc.)
                other_params.append(param)
        
        param_groups = [
            {
                'params': patch_embed_params,
                'lr': base_lr * 0.1,
                'weight_decay': weight_decay,
                'name': 'patch_embed'
            },
            {
                'params': pos_embed_params,
                'lr': base_lr * 0.1,
                'weight_decay': weight_decay,
                'name': 'pos_embed'
            },
            {
                'params': early_block_params,
                'lr': base_lr * 0.3,
                'weight_decay': weight_decay,
                'name': 'early_blocks'
            },
            {
                'params': middle_block_params,
                'lr': base_lr * 0.6,
                'weight_decay': weight_decay,
                'name': 'middle_blocks'
            },
            {
                'params': late_block_params,
                'lr': base_lr * 1.0,
                'weight_decay': weight_decay,
                'name': 'late_blocks'
            },
            {
                'params': other_params,
                'lr': base_lr * 1.0,
                'weight_decay': weight_decay,
                'name': 'other_backbone'
            },
            {
                'params': projection_params,
                'lr': base_lr * 1.0,
                'weight_decay': weight_decay,
                'name': 'projection_head'
            }
        ]
        
        # Filter out empty parameter groups
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        return param_groups
    
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

