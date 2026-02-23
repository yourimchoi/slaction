"""
Neural Network Models for Sleep Action Recognition

This module contains custom MoviNet model implementations optimized for sleep
action recognition from video sequences. It includes modifications for handling
grayscale input and frame compression techniques.
"""

from torch import nn
from movinets import MoViNet
from movinets.config import _C


class MoviNet_A2_50(nn.Module):
    """
    MoviNet-A2 model adapted for sleep video analysis with frame compression.
    
    This model converts grayscale 150-frame sequences to RGB 50-frame sequences
    by grouping consecutive frames. It's designed for transfer learning with a
    custom classification head.
    
    Args:
        n_classes (int): Number of output classes for classification
        pretrained (bool): Whether to use pretrained MoviNet weights (default: True)
    """
    
    def __init__(self, n_classes, pretrained=True):
        super(MoviNet_A2_50, self).__init__()
        
        # Initialize MoViNet-A2 backbone
        self.model = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=pretrained)
        
        # Replace the final classifier layer with identity for feature extraction
        self.model.classifier[-1] = nn.Identity()
        
        # Add custom classification head for sleep action recognition
        self.fc = nn.Linear(_C.MODEL.MoViNetA2.dense9.hidden_dim, n_classes)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 150, 1, H, W]
            
        Returns:
            torch.Tensor: Classification logits of shape [B, n_classes]
        """
        P = 3  # Number of frames to group together for RGB conversion
        B, N, C, H, W = x.shape  # [B, 150, 1, 224, 224]
        N = N // P  # Reduce temporal dimension: N = 50
        
        # Reshape to group frames: [B, 50, 3, 224, 224]
        x = x.view(B, N, P, H, W)
        
        # Rearrange dimensions for MoviNet: [B, 3, 50, 224, 224]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Pass through MoviNet backbone
        x = self.model(x)
        
        # Remove unnecessary dimensions and apply classification head
        x = x.squeeze()
        x = self.fc(x)
        return x
    

class MoviNet_A2_150(nn.Module):
    """
    MoviNet-A2 model for processing full 150-frame grayscale sequences.
    
    This model processes the full temporal resolution without frame compression,
    suitable for applications requiring fine temporal detail.
    
    Args:
        n_classes (int): Number of output classes for classification
        pretrained (bool): Whether to use pretrained MoviNet weights (default: True)
    """
    
    def __init__(self, n_classes, pretrained=True):
        super(MoviNet_A2_150, self).__init__()
        
        # Initialize MoViNet-A2 backbone
        self.model = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=pretrained)
        
        # Replace the final classifier layer with identity for feature extraction
        self.model.classifier[-1] = nn.Identity()
        
        # Add custom classification head
        self.fc = nn.Linear(_C.MODEL.MoViNetA2.dense9.hidden_dim, n_classes)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 150, 1, H, W]
            
        Returns:
            torch.Tensor: Classification logits of shape [B, n_classes]
        """
        # Rearrange dimensions for MoviNet: [B, 1, 150, 224, 224]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Pass through MoviNet backbone
        x = self.model(x)
        
        # Remove unnecessary dimensions and apply classification head
        x = x.squeeze()
        x = self.fc(x)
        return x    