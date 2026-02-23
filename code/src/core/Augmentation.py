"""
Video Data Augmentation Module

This module provides comprehensive video transformation classes for data augmentation
in sleep action recognition tasks. It includes spatial, temporal, and appearance
transformations optimized for video sequence processing.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class VideoTransform:
    """
    Comprehensive video transformation class for data augmentation.
    
    This class applies various transformations to video frames including spatial
    transformations (crop, resize, flip, rotation), perspective transformations,
    and color jittering for robust model training.
    
    Args:
        mean (float): Mean value for normalization (default: 0.485)
        std (float): Standard deviation for normalization (default: 0.229)
        rotation (bool): Whether to apply random rotation (default: False)
        perspective (bool): Whether to apply perspective transformation (default: False)
        random_padding (bool): Whether to apply random padding (default: False)
        size (tuple): Target size for frame resizing (default: (224, 224))
        rotation_angle_range (tuple): Range of rotation angles in degrees (default: (-90, 90))
        color_jitter_brightness (tuple): Brightness adjustment range (default: (0.7, 1.3))
        color_jitter_contrast (tuple): Contrast adjustment range (default: (0.7, 1.3))
        random_resized_crop_scale (tuple): Scale range for random resized crop (default: (0.7, 1.0))
        random_resized_crop_ratio (tuple): Aspect ratio range for random resized crop (default: (0.75, 1.33))
    """
    def __init__(self, mean=0.485, std=0.229, rotation=False, perspective=False, 
                 random_padding=False, size=(224, 224), rotation_angle_range=(-90, 90),
                 color_jitter_brightness=(0.7, 1.3), color_jitter_contrast=(0.7, 1.3),
                 random_resized_crop_scale=(0.7, 1.0), random_resized_crop_ratio=(0.75, 1.33)):
        """Initialize VideoTransform with specified parameters."""
        self.mean = mean
        self.std = std
        self.rotation = rotation
        self.perspective = perspective
        self.random_padding = random_padding
        self.size = size
        self.rotation_angle_range = rotation_angle_range
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.random_resized_crop_scale = random_resized_crop_scale
        self.random_resized_crop_ratio = random_resized_crop_ratio

    def transform_train(self, x):
        """
        Apply training transformations to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Transformed video tensor
        """
        # Apply spatial augmentations
        x = self.random_padding(x, prob=0.5, max_pad_factor=6) if self.random_padding else x
        x = self.random_resized_crop(x, scale=self.random_resized_crop_scale, 
                                   ratio=self.random_resized_crop_ratio, size=self.size)
        self.scale_(x)
        x = self.hflip(x)
        
        # Apply appearance augmentations
        self.color_jitter_(x, brightness=self.color_jitter_brightness, 
                          contrast=self.color_jitter_contrast)
        
        # Apply geometric augmentations
        x = self.perspective_transform(x) if self.perspective else x
        x = self.rotate_frames(x) if self.rotation else x
        
        # Normalize
        self.normalize_(x)
        return x

    def transform_test(self, x):
        """
        Apply test/validation transformations to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Transformed video tensor
        """
        x = self.center_crop(x, size=self.size)
        self.scale_(x)
        self.normalize_(x)
        return x

    # Spatial transformation methods
    
    def random_padding(self, x, prob=0.5, max_pad_factor=4):
        """
        Apply random padding to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor
            prob (float): Probability of applying padding
            max_pad_factor (float): Maximum ratio of area increase due to padding
            
        Returns:
            torch.Tensor: Padded video tensor
        """
        if torch.rand(1).item() < prob:
            _, _, h, w = x.shape
            scale_factor = torch.sqrt(torch.FloatTensor(1).uniform_(1, max_pad_factor)).item()
            pad_h = int((scale_factor - 1) * h / 2)
            pad_w = int((scale_factor - 1) * w / 2)
            padding = (pad_w, pad_w, pad_h, pad_h)
            x = F.pad(x, padding, fill=0)
        return x

    def random_resized_crop(self, x, scale, ratio, size):
        """
        Apply random resized crop to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor
            scale (tuple): Range of crop scale
            ratio (tuple): Range of aspect ratio
            size (tuple): Target output size
            
        Returns:
            torch.Tensor: Cropped and resized video tensor
        """
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            x[0], scale=scale, ratio=ratio)
        return F.resized_crop(x, i, j, h, w, size=size)

    def center_crop(self, x, size):
        """
        Apply center crop to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor
            size (tuple): Target crop size
            
        Returns:
            torch.Tensor: Center-cropped video tensor
        """
        return transforms.CenterCrop(size)(x)

    def hflip(self, x, prob=0.5):
        """
        Apply random horizontal flip to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor
            prob (float): Probability of applying horizontal flip
            
        Returns:
            torch.Tensor: Horizontally flipped video tensor
        """
        if torch.rand(1).item() < prob:
            return x.flip(dims=[3])
        return x

    def rotate_frames(self, x, angle_range=None, prob=1):
        """
        Apply random rotation to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor
            angle_range (tuple): Range of rotation angles in degrees
            prob (float): Probability of applying rotation
            
        Returns:
            torch.Tensor: Rotated video tensor
        """
        if angle_range is None:
            angle_range = self.rotation_angle_range
        if torch.rand(1).item() < prob:
            angle = torch.FloatTensor(1).uniform_(*angle_range).item()
            x = F.rotate(x, angle)
        return x

    def perspective_transform(self, x, distortion_scale=0.5, prob=0.5):
        """
        Apply perspective transformation to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor
            distortion_scale (float): Scale of perspective distortion
            prob (float): Probability of applying perspective transformation
            
        Returns:
            torch.Tensor: Perspective-transformed video tensor
        """
        if torch.rand(1).item() < prob:
            startpoints, endpoints = self.get_perspective_params(
                x.shape[2], x.shape[3], distortion_scale)
            return F.perspective(x, startpoints, endpoints)
        return x

    def get_perspective_params(self, height, width, distortion_scale):
        """
        Generate parameters for perspective transformation.
        
        Args:
            height (int): Frame height
            width (int): Frame width
            distortion_scale (float): Scale of distortion
            
        Returns:
            tuple: Start and end points for perspective transformation
        """
        half_height = height // 2
        half_width = width // 2
        
        # Generate random corner points for perspective transformation
        topleft = [torch.randint(0, int(distortion_scale * half_width), (1,)).item(), 
                  torch.randint(0, int(distortion_scale * half_height), (1,)).item()]
        topright = [torch.randint(width - int(distortion_scale * half_width), width, (1,)).item(), 
                   torch.randint(0, int(distortion_scale * half_height), (1,)).item()]
        botright = [torch.randint(width - int(distortion_scale * half_width), width, (1,)).item(), 
                   torch.randint(height - int(distortion_scale * half_height), height, (1,)).item()]
        botleft = [torch.randint(0, int(distortion_scale * half_width), (1,)).item(), 
                  torch.randint(height - int(distortion_scale * half_height), height, (1,)).item()]
        
        startpoints = [topleft, topright, botright, botleft]
        endpoints = [[0, 0], [width, 0], [width, height], [0, height]]
        return startpoints, endpoints

    # Appearance transformation methods
    
    def color_jitter_(self, x, brightness=(0.7, 1.3), contrast=(0.7, 1.3), prob=1):
        """
        Apply color jittering to video frames.
        
        Args:
            x (torch.Tensor): Input video tensor (modified in-place)
            brightness (tuple): Range of brightness adjustment factors
            contrast (tuple): Range of contrast adjustment factors
            prob (float): Probability of applying color jittering
        """
        if torch.rand(1).item() < prob:
            brightness_factor = torch.FloatTensor(1).uniform_(*brightness).item()
            contrast_factor = torch.FloatTensor(1).uniform_(*contrast).item()
            self.adjust_brightness_(x, brightness_factor)
            self.adjust_contrast_(x, contrast_factor)

    def adjust_brightness_(self, x, brightness_factor):
        """
        Adjust brightness of video frames in-place.
        
        Args:
            x (torch.Tensor): Input video tensor (modified in-place)
            brightness_factor (float): Brightness adjustment factor
        """
        x.mul_(brightness_factor).clamp_(0, 1)

    def adjust_contrast_(self, x, contrast_factor):
        """
        Adjust contrast of video frames in-place.
        
        Args:
            x (torch.Tensor): Input video tensor (modified in-place)
            contrast_factor (float): Contrast adjustment factor
        """
        mean = x.mean(dim=[1, 2], keepdim=True)
        x.sub_(mean).mul_(contrast_factor).add_(mean).clamp_(0, 1)

    # Normalization methods
    
    def scale_(self, x):
        """
        Scale pixel values to [0, 1] range in-place.
        
        Args:
            x (torch.Tensor): Input video tensor (modified in-place)
        """
        x.div_(255.0)

    def normalize_(self, x):
        """
        Normalize video frames using mean and standard deviation in-place.
        
        Args:
            x (torch.Tensor): Input video tensor (modified in-place)
        """
        mean = torch.tensor(self.mean, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, -1, 1, 1)
        x.sub_(mean).div_(std)

    # Additional helper methods
    
    def adjust_contrast_tensor_(self, x, contrast_factor):
        """
        Adjust contrast using tensor operations (non-in-place version).
        
        Args:
            x (torch.Tensor): Input video tensor
            contrast_factor (float): Contrast adjustment factor
            
        Returns:
            torch.Tensor: Contrast-adjusted tensor
        """
        mean = x.mean(dim=[1, 2], keepdim=True)
        return (x - mean).mul_(contrast_factor).add_(mean).clamp_(0, 1)

    def adjust_brightness_tensor_(self, x, brightness_factor):
        """
        Adjust brightness using tensor operations (non-in-place version).
        
        Args:
            x (torch.Tensor): Input video tensor
            brightness_factor (float): Brightness adjustment factor
            
        Returns:
            torch.Tensor: Brightness-adjusted tensor
        """
        return x.mul_(brightness_factor).clamp_(0, 1)

if __name__ == '__main__':
    """
    Demonstration script for visualizing video transformation effects.
    
    This script shows how different transformations affect video frames,
    useful for debugging and understanding augmentation effects.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms.functional as TF

    def visualize_transform_steps(image_path, transform_module):
        """
        Visualize the step-by-step transformation process.
        
        Args:
            image_path (str): Path to sample image
            transform_module (VideoTransform): Configured transformation module
        """
        # Load and prepare the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Convert image to tensor and add batch dimension (simulate video with one frame)
        image_tensor = TF.to_tensor(image).unsqueeze(0)  # Shape: (1, 1, H, W)

        steps = ['Original']
        images = [image_tensor.clone()]

        # Apply transformations step by step
        # 1. Random resized crop
        cropped = transform_module.random_resized_crop(
            image_tensor, scale=transform_module.random_resized_crop_scale,
            ratio=transform_module.random_resized_crop_ratio, size=transform_module.size)
        steps.append('Random Resized Crop')
        images.append(cropped.clone())

        # 2. Horizontal flip
        flipped = transform_module.hflip(cropped, prob=1.0)  # Ensure horizontal flip
        steps.append('Horizontal Flip')
        images.append(flipped.clone())

        # 3. Color jittering
        transform_module.color_jitter_(flipped, prob=1.0)  # Ensure color jitter
        steps.append('Color Jitter')
        images.append(flipped.clone())

        # 4. Rotation
        rotated = transform_module.rotate_frames(flipped, angle_range=(-45, 45), prob=1.0)
        steps.append('Rotation')
        images.append(rotated.clone())

        # Visualize the transformation pipeline
        num_steps = len(steps)
        fig, axes = plt.subplots(1, num_steps, figsize=(5 * num_steps, 5))
        for idx, ax in enumerate(axes):
            img = images[idx].squeeze(0).permute(1, 2, 0).numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(steps[idx])
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Example usage
    image_path = 'sample_frame.jpg'  # Replace with actual image path
    transform_module = VideoTransform(rotation=True, perspective=False)
    # visualize_transform_steps(image_path, transform_module)  # Uncomment to run
