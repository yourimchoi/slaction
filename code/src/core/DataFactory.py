"""
Data Factory Module for Sleep Action Recognition

This module provides data loading, preprocessing, and augmentation utilities
for video-based sleep action recognition. It includes custom datasets and
PyTorch Lightning data modules for efficient training pipelines.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from tfrecord.torch.dataset import TFRecordDataset
import pytorch_lightning as pl
from Augmentation import VideoTransform


class VideoDataset(Dataset):
    """
    Custom dataset for loading and processing video data from TFRecord files.
    
    This dataset handles video sequences stored in TFRecord format, applies
    transformations, and supports variable video lengths for sleep action recognition.
    
    Args:
        file_paths (list): List of TFRecord file paths
        labels (list): Corresponding labels for each video
        is_train (bool): Whether this is training data (affects augmentation)
        rotation (bool): Enable rotation augmentation
        perspective (bool): Enable perspective transformation
        random_padding (bool): Enable random padding augmentation
        video_length_sec (int): Duration of video sequences in seconds
        rotation_angle_range (tuple): Range of rotation angles for augmentation
        color_jitter_brightness (tuple): Brightness adjustment range
        color_jitter_contrast (tuple): Contrast adjustment range
        random_resized_crop_scale (tuple): Scale range for random crop
        random_resized_crop_ratio (tuple): Aspect ratio range for random crop
    """
    
    def __init__(
        self, file_paths, labels, is_train=True,
        rotation=False, perspective=False, random_padding=False,
        video_length_sec=30, rotation_angle_range=(-90, 90),
        color_jitter_brightness=(0.7, 1.3),
        color_jitter_contrast=(0.7, 1.3),
        random_resized_crop_scale=(0.7, 1.0),
        random_resized_crop_ratio=(0.75, 1.33)
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train
        self.video_length_sec = video_length_sec
        
        # Initialize video transformation pipeline
        self.transform_module = VideoTransform(
            rotation=rotation,
            perspective=perspective,
            random_padding=random_padding,
            rotation_angle_range=rotation_angle_range,
            color_jitter_brightness=color_jitter_brightness,
            color_jitter_contrast=color_jitter_contrast,
            random_resized_crop_scale=random_resized_crop_scale,
            random_resized_crop_ratio=random_resized_crop_ratio
        )
        
        # Select appropriate transformation based on training mode
        if is_train:
            self.transform = self.transform_module.transform_train
        else:
            self.transform = self.transform_module.transform_test
            
        # TFRecord description for parsing
        self.description = {'video': 'byte', 'length': 'int'}

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and process a single video sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (video_tensor, label) pair
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Handle 60-second videos by concatenating two 30-second segments
        if self.video_length_sec == 60:
            next_file_path = self.get_next_file_path(file_path)
            video_tensor = self.load_and_concat_videos(file_path, next_file_path)
        else:
            video_tensor = self.load_video(file_path)
        
        # Apply transformations
        video_tensor = self.transform(video_tensor)
        return video_tensor, label

    def get_next_file_path(self, file_path):
        """
        Generate the path for the next consecutive video file.
        
        Args:
            file_path (str): Current file path
            
        Returns:
            str: Path to the next video file
        """
        base, ext = os.path.splitext(file_path)
        next_file_path = f"{base[:-4]}{int(base[-4:]) + 1:04d}{ext}"
        return next_file_path

    def load_and_concat_videos(self, file_path, next_file_path):
        """
        Load and concatenate two consecutive video segments.
        
        Args:
            file_path (str): First video file path
            next_file_path (str): Second video file path
            
        Returns:
            torch.Tensor: Concatenated video tensor
        """
        video_tensor1 = self.load_video(file_path)
        video_tensor2 = self.load_video(next_file_path)
        
        # Concatenate videos and downsample by taking every second frame
        video_tensor = torch.cat((video_tensor1, video_tensor2), dim=0)
        video_tensor = video_tensor[::2]  # Reduce temporal resolution
        return video_tensor

    def load_video(self, file_path):
        """
        Load video data from TFRecord file.
        
        Args:
            file_path (str): Path to TFRecord file
            
        Returns:
            torch.Tensor: Video tensor of shape (T, C, H, W)
        """
        dataset = TFRecordDataset(file_path, index_path=None, description=self.description)
        sample = next(iter(dataset))
        video_bytes = sample['video']
        return self.bytes_to_tensor(video_bytes)

    def bytes_to_tensor(self, video_bytes):
        """
        Convert byte data to video tensor.
        
        Args:
            video_bytes (bytes): Raw video bytes from TFRecord
            
        Returns:
            torch.Tensor: Video tensor of shape (150, 1, 256, 256)
        """
        frame_size = 256 * 256  # Each frame size (256x256 pixels)
        length = 150  # Number of frames
        expected_size = length * frame_size
        actual_size = len(video_bytes)
        metadata_padding = 32  # Bytes reserved for metadata
        
        # Validate video data size
        if actual_size != expected_size + metadata_padding:
            raise ValueError(f"Unexpected video size: expected {expected_size + metadata_padding}, got {actual_size}")

        # Convert bytes to tensor and reshape
        video_bytes_copy = torch.tensor(bytearray(video_bytes[metadata_padding:]), dtype=torch.uint8)
        video_tensor = video_bytes_copy.reshape(length, 1, 256, 256).float()
        
        return video_tensor


class VideoDataModuleLightning(pl.LightningDataModule):
    """
    PyTorch Lightning data module for video datasets.
    
    This class handles data loading, preprocessing, and batch creation for
    training, validation, and testing phases with support for weighted sampling
    and various augmentation strategies.
    
    Args:
        base_csv_path (str): Base path for CSV label files
        base_tfrecord_path (str): Base path for TFRecord video files  
        label_column (str): Column name for labels in CSV files
        batch_size (int): Batch size for data loaders
        shuffle (bool): Whether to shuffle training data
        num_workers (int): Number of workers for data loading
        weighted_sampling (bool): Enable weighted sampling for class balance
        train_set (str): Training set identifier
        valid_set (str): Validation set identifier
        test_set (str): Test set identifier
        rotation (bool): Enable rotation augmentation
        perspective (bool): Enable perspective transformation
        random_padding (bool): Enable random padding
        video_length_sec (int): Video duration in seconds
        rotation_angle_range (tuple): Rotation angle range
        color_jitter_brightness (tuple): Brightness adjustment range
        color_jitter_contrast (tuple): Contrast adjustment range
        random_resized_crop_scale (tuple): Random crop scale range
        random_resized_crop_ratio (tuple): Random crop aspect ratio range
        sample_ratio (float): Fraction of data to use
    """
    
    def __init__(
        self, base_csv_path, base_tfrecord_path, label_column, batch_size=32,
        shuffle=True, num_workers=4, weighted_sampling=False, train_set='train',
        valid_set='valid', test_set='test', rotation=False, perspective=False,
        random_padding=False, video_length_sec=30,
        rotation_angle_range=(-90, 90), color_jitter_brightness=(0.7, 1.3),
        color_jitter_contrast=(0.7, 1.3), random_resized_crop_scale=(0.7, 1.0),
        random_resized_crop_ratio=(0.75, 1.33), sample_ratio=1.0
    ):
        super().__init__()
        self.base_csv_path = base_csv_path
        self.base_tfrecord_path = base_tfrecord_path
        self.label_column = label_column
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.weighted_sampling = weighted_sampling
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.rotation = rotation
        self.perspective = perspective
        self.random_padding = random_padding
        self.video_length_sec = video_length_sec
        self.mean = 0.485
        self.std = 0.229
        self.rotation_angle_range = rotation_angle_range
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.random_resized_crop_scale = random_resized_crop_scale
        self.random_resized_crop_ratio = random_resized_crop_ratio
        self.sample_ratio = sample_ratio

    def setup(self, stage=None):
        # Helper function to extract split folder from dataset name
        def get_split_folder(dataset_name):
            if '_train' in dataset_name:
                return 'train'
            elif '_valid' in dataset_name:
                return 'valid'
            elif '_test' in dataset_name:
                return 'test'
            else:
                # Fallback: assume train if not specified
                return 'train'
        
        # Determine the CSV file names based on the provided sets
        if self.train_set:
            train_csv_path = os.path.join(self.base_csv_path, f'{self.train_set}.csv')
            self.train_data = pd.read_csv(train_csv_path)
            # Add appropriate train/valid/test folder to path
            split_folder = get_split_folder(self.train_set)
            self.train_data['File'] = self.train_data['File'].apply(lambda x: os.path.join(self.base_tfrecord_path, split_folder, x))
            self.train_data = self.train_data.groupby("Case_num", group_keys=False).apply(
                lambda g: pd.concat([
                    g[g[self.label_column].isin([0,1])].sample(frac=self.sample_ratio, random_state=42),
                    g[~g[self.label_column].isin([0,1])]
                ])
            )
            self.train_dataset = VideoDataset(self.train_data['File'].tolist(), self.train_data[self.label_column].tolist(), is_train=True, rotation=self.rotation, perspective=self.perspective, random_padding=self.random_padding, video_length_sec=self.video_length_sec, rotation_angle_range=self.rotation_angle_range, color_jitter_brightness=self.color_jitter_brightness, color_jitter_contrast=self.color_jitter_contrast, random_resized_crop_scale=self.random_resized_crop_scale, random_resized_crop_ratio=self.random_resized_crop_ratio)  # Use selected label column

        if self.valid_set:
            valid_csv_path = os.path.join(self.base_csv_path, f'{self.valid_set}.csv')
            self.valid_data = pd.read_csv(valid_csv_path)
            # Add appropriate train/valid/test folder to path
            split_folder = get_split_folder(self.valid_set)
            self.valid_data['File'] = self.valid_data['File'].apply(lambda x: os.path.join(self.base_tfrecord_path, split_folder, x))
            self.valid_dataset = VideoDataset(self.valid_data['File'].tolist(), self.valid_data[self.label_column].tolist(), is_train=False, video_length_sec=self.video_length_sec)  # Use selected label column

        if self.test_set:
            test_csv_path = os.path.join(self.base_csv_path, f'{self.test_set}.csv')
            self.test_data = pd.read_csv(test_csv_path)
            # Add appropriate train/valid/test folder to path
            split_folder = get_split_folder(self.test_set)
            self.test_data['File'] = self.test_data['File'].apply(lambda x: os.path.join(self.base_tfrecord_path, split_folder, x))
            self.test_dataset = VideoDataset(self.test_data['File'].tolist(), self.test_data[self.label_column].tolist(), is_train=False, video_length_sec=self.video_length_sec)  # Use selected label column

    def train_dataloader(self):
        """
        Create training data loader with optional weighted sampling.
        
        If weighted_sampling is enabled, uses class-based weighted sampling for balanced training.
        Otherwise, uses standard random sampling.
        
        Returns:
            DataLoader: Training data loader with or without weighted sampling
        """
        if self.train_set:
            if self.weighted_sampling:
                # Use basic class-based weighted sampling
                class_counts = Counter(self.train_data[self.label_column])
                class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
                sample_weights = [class_weights[label] for label in self.train_data[self.label_column]]
                sample_weights = torch.FloatTensor(sample_weights)
                sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
                
                train_loader = DataLoader(
                    self.train_dataset, 
                    batch_size=self.batch_size, 
                    sampler=sampler, 
                    num_workers=self.num_workers, 
                    pin_memory=False
                )
            else:
                # Use standard random sampling
                train_loader = DataLoader(
                    self.train_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=self.shuffle, 
                    num_workers=self.num_workers, 
                    pin_memory=False
                )
            return train_loader
        else:
            return None

    def val_dataloader(self):
        if self.valid_set:
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)
            return valid_loader
        else:
            return None

    def test_dataloader(self):
        if self.test_set:
            test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)
            return test_loader
        else:
            return None

    def visualize_first_video(self, dataloader, num_frames_to_display=5, images_per_row=5):
        for video_tensor, label in dataloader:
            first_video = video_tensor[0].numpy()
            num_frames_to_display = min(num_frames_to_display, first_video.shape[0])
            
            fig, axes = plt.subplots(
                nrows=(num_frames_to_display + images_per_row - 1) // images_per_row,
                ncols=images_per_row,
                figsize=(3 * images_per_row, 3 * ((num_frames_to_display + images_per_row - 1) // images_per_row))
            )
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i in range(num_frames_to_display):
                frame = (first_video[i, 0, :, :] * self.std + self.mean) * 255  # Denormalize the frame
                axes[i].imshow(frame, cmap='gray')
                axes[i].set_title(f"Frame {i+1}")
                axes[i].axis('off')
            
            for i in range(num_frames_to_display, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            break  # Stop after visualizing the first video

# Test Code
if __name__ == '__main__':
    config = {
        'base_csv_path': 'base_csv_path',
        'base_tfrecord_path': 'base_tfrecord_path',
        'label_column': 'RA_Pose_New',
        'batch_size':2,
        'weighted_sampling': True,
        'train_set': 'A_train',
        'valid_set': 'A_valid',
        'test_set': 'A_test',
        'rotation': True,
        'perspective': False,
        'random_padding': False,
        'video_length_sec': 60
    }

    video_data_module = VideoDataModuleLightning(
        config['base_csv_path'],
        config['base_tfrecord_path'],
        label_column=config['label_column'],
        batch_size=config['batch_size'],
        weighted_sampling=config['weighted_sampling'],
        train_set=config['train_set'],
        valid_set=config['valid_set'],
        test_set=config['test_set'],
        rotation=config['rotation'],
        perspective=config['perspective'],
        random_padding=config['random_padding'],
        video_length_sec=config['video_length_sec'],
        num_workers=1
    )

    video_data_module.setup()
    train_loader = video_data_module.train_dataloader()
    valid_loader = video_data_module.val_dataloader()
    
    if train_loader is not None:
        video_data_module.visualize_first_video(train_loader, num_frames_to_display=20, images_per_row=5)

#%%