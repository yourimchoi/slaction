"""
Training Script for Sleep Action Recognition

This module provides the main training pipeline for video-based sleep action recognition
using PyTorch Lightning. It supports distributed training, checkpoint management,
transfer learning, and comprehensive logging with Weights & Biases.

Key Features:
- Multi-GPU distributed training with DDP strategy
- Three-phase training (warmup, progressive unfreezing, fine-tuning)
- Transfer learning from pretrained models
- Automatic checkpoint management and resuming
- Model complexity analysis and profiling
- Integration with Weights & Biases for experiment tracking
"""

import wandb
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.optim.lr_scheduler import CosineAnnealingLR
from packaging import version

from Utils import set_seed, get_results_path, ensure_results_dir
import os
from VideoTrainer import SleepVideoModel, AutoFreezeCallback
from DataFactory import VideoDataModuleLightning
from torchsummary import summary
from ptflops import get_model_complexity_info
import glob

@rank_zero_only
def init_wandb(name, config):
    """
    Initialize Weights & Biases tracking for experiment logging.
    
    Args:
        name (str): Experiment name for tracking
        config (dict): Configuration dictionary with training parameters
        
    Returns:
        wandb.config: WandB configuration object
    """
    wandb.init(project=config['wandb_project'], name=name, config=config)
    return wandb.config

@rank_zero_only
def append_wandb_id_to_config(config_path, wandb_id):
    """
    Append WandB run ID to configuration file for resuming capabilities.
    
    Args:
        config_path (str): Path to YAML configuration file
        wandb_id (str): WandB run identifier
    """
    with open(config_path, 'a') as file:
        file.write(f'\nwandb_id: {wandb_id}\n')

@rank_zero_only
def log_model_summary(model, input_size):
    """
    Log model complexity metrics to WandB including parameters and FLOPs.
    
    Args:
        model (torch.nn.Module): Neural network model to analyze
        input_size (tuple): Input tensor shape for FLOP calculation
    """
    # Calculate model parameters and computational complexity
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops, params = get_model_complexity_info(model, input_res=input_size, as_strings=False, print_per_layer_stat=False, verbose=False)
    gflops = flops / 1e9
    
    # Log metrics to WandB
    wandb.log({
        "Total Parameters": total_params,
        "Trainable Parameters": total_trainable_params,
        "GFLOPs": gflops,
    })

@rank_zero_only
def get_latest_checkpoint(config, name):
    """
    Find the most recent checkpoint file for resuming training.
    
    Searches for checkpoint files matching patterns 'last.ckpt' and 'last-v*.ckpt'
    and returns the one with the highest version number and most recent creation time.
    
    Args:
        config (dict): Configuration dictionary containing path settings
        name (str): Experiment name used to locate checkpoint directory
        
    Returns:
        str: Path to the latest checkpoint file
        
    Raises:
        FileNotFoundError: If no checkpoint files are found
    """
    import re
    
    def get_version_number(filename):
        """Extract version number from checkpoint filename."""
        match = re.search(r'last-v(\d+)\.ckpt$', filename)
        if match:
            return int(match.group(1))
        elif filename.endswith('last.ckpt'):
            return 0  # Base version
        return -1

    # Search for checkpoint files with different patterns
    checkpoint_files = []
    weights_dir = get_results_path(config, 'weights', name)
    for pattern in [f'{weights_dir}/last.ckpt', f'{weights_dir}/last-v*.ckpt']:
        checkpoint_files.extend(glob.glob(pattern))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {weights_dir}/")
    
    # Sort checkpoints by version number and creation time
    latest_checkpoint = max(checkpoint_files, 
                          key=lambda x: (get_version_number(x), os.path.getctime(x)))
    
    print(f"Found checkpoints: {checkpoint_files}")
    print(f"Selected latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint

@rank_zero_only
def get_wandb_id_from_config(config_path):
    """
    Extract WandB run ID from configuration file for resuming experiments.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        str or None: WandB run ID if found, None otherwise
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('wandb_id')

@rank_zero_only
def save_best_model_path(config, exp_number, path):
    """
    Save the path of the best model checkpoint for later use.
    
    Args:
        config (dict): Configuration dictionary containing path settings
        exp_number (str): Experiment number identifier
        path (str): Path to the best model checkpoint
    """
    temp_dir = ensure_results_dir(config, 'temp')
    temp_file_path = os.path.join(temp_dir, f'best_model_path_{exp_number}.txt')
    with open(temp_file_path, 'w') as f:
        f.write(path)

def load_transfer_model(config, exp_number):
    """
    Load model from checkpoint for transfer learning.
    
    Supports loading from either a specified resume_path in config or from
    the best checkpoint of a previous experiment.
    
    Args:
        config (dict): Configuration dictionary containing model parameters
        exp_number (str): Experiment number for locating previous checkpoints
        
    Returns:
        SleepVideoModel: Loaded model with pretrained weights
    """
    # Check if resume_path is specified in config for pretrained model
    if 'resume_path' in config and config['resume_path']:
        resume_path = config['resume_path']
        print(f"Loading pretrained model from: {resume_path}")
        return SleepVideoModel.load_from_checkpoint(resume_path, config=config)
    else:
        # Original logic: load from previous experiment's best checkpoint
        temp_file_path = get_results_path(config, 'temp', f'best_model_path_{exp_number}.txt')
        with open(temp_file_path, 'r') as f:
            best_ckpt_path = f.read().strip()
        print(f"Loading model from the best checkpoint of experiment {exp_number}: {best_ckpt_path}")
        return SleepVideoModel.load_from_checkpoint(best_ckpt_path, config=config)

def main():
    """
    Main training function that orchestrates the entire training pipeline.
    
    This function handles:
    - Argument parsing and configuration loading
    - Multi-GPU setup and distributed training initialization
    - Data module and model creation
    - Training with checkpointing and logging
    - Best model path saving for transfer learning
    """
    # Setup distributed training environment
    rank = int(os.environ.get('LOCAL_RANK', 0))
    NUM_DEVICES = torch.cuda.device_count()
    
    # Configure compute environment
    set_seed(42)  # Ensure reproducible results
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')  # Optimize performance
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sleep Action Recognition Training Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Generate experiment name from config filename
    name = 'exp_' + args.config.split('/')[-1].split('.')[0]
    config['name'] = name
    NUM_WORKERS = args.num_workers

    # Extract training parameters from configuration
    SAMPLE_RATIO = config['sample_ratio']
    BATCHSIZE = config['batchsize']
    LEARNING_RATE = config['learning_rate']
    WARMUP_EPOCHS = config['epochs_warmup']
    PROGRESSIVE_UNFREEZE_EPOCHS = config['epochs_progressive_unfreeze']
    FINETUNE_EPOCHS = config['epochs_finetune']
    OPTIMIZER = config['optimizer']
    CSV_FILE_PATH = config['csv_file_path']
    BASE_TFRECORD_PATH = config['base_tfrecord_path']
    LABEL_COLUMN = config['label_column']
    ACCUMULATE_GRAD_BATCHES = config['accumulate_grad_batches']
    ROTATION = config['rotation']
    PERSPECTIVE = config['perspective']
    RANDOM_PADDING = config.get('random_padding', False)
    ROTATION_ANGLE_RANGE = config.get('rotation_angle_range', [-90, 90])
    color_jitter_brightness = config.get('color_jitter_brightness', [0.7, 1.3])
    color_jitter_contrast = config.get('color_jitter_contrast', [0.7, 1.3])
    random_resized_crop_scale = config.get('random_resized_crop_scale', [0.7, 1.0])
    random_resized_crop_ratio = config.get('random_resized_crop_ratio', [0.75, 1.33])
    TRAIN_SET = config.get('train_set', 'A_train')
    VALID_SET = config.get('valid_set', 'A_valid')
    WEIGHTED_SAMPLING = config.get('weighted_sampling', False)  # Read from config

    ###################### Wandb Initialization ######################
    resume_checkpoint = None
    transfer_from_ckpt = config.get('transfer_from_ckpt', False)
    
    # Initialize WandB tracking (only on rank 0 for distributed training)
    if rank == 0:
        wandb_id = get_wandb_id_from_config(args.config)
        if wandb_id is not None and config.get('resume', False):
            # Resume existing experiment
            wandb.init(project=config['wandb_project'], name=name, config=config, resume="allow", id=wandb_id)
            # Only look for resume checkpoint if not using transfer learning
            if not transfer_from_ckpt:
                resume_checkpoint = get_latest_checkpoint(config, name)
        else:
            # Start new experiment
            wandb.init(project=config['wandb_project'], name=name, config=config)
        
        # Save WandB ID for future resuming
        if not wandb.run.resumed:
            append_wandb_id_to_config(args.config, wandb.run.id)
        
        print(f"{'Resuming' if config['resume'] else 'Starting'} wandb run")

    # Create WandbLogger for PyTorch Lightning integration
    wandb_logger = WandbLogger(
        project=config['wandb_project'],
        name=name,
        config=config,
    )

    ###################### DataModule and Model Initialization ######################
    # Initialize data module with configuration parameters
    video_data_module = VideoDataModuleLightning(
        base_csv_path=CSV_FILE_PATH,
        base_tfrecord_path=BASE_TFRECORD_PATH,
        label_column=LABEL_COLUMN,
        batch_size=int(BATCHSIZE/NUM_DEVICES/ACCUMULATE_GRAD_BATCHES),
        shuffle=True,
        num_workers=NUM_WORKERS//NUM_DEVICES,
        weighted_sampling=WEIGHTED_SAMPLING,
        train_set=TRAIN_SET,
        valid_set=VALID_SET,
        test_set=None,
        rotation=ROTATION,
        perspective=PERSPECTIVE,
        random_padding=RANDOM_PADDING,
        rotation_angle_range=ROTATION_ANGLE_RANGE,
        color_jitter_brightness=color_jitter_brightness,
        color_jitter_contrast=color_jitter_contrast,
        random_resized_crop_scale=random_resized_crop_scale,
        random_resized_crop_ratio=random_resized_crop_ratio
    )

    ###################### Transfer Learning ######################
    # Initialize model either from checkpoint (transfer learning) or from scratch
    if transfer_from_ckpt:
        exp_number = args.config.split('/')[-1].split('.')[0]
        model = load_transfer_model(config, exp_number)
        print("Loaded model from checkpoint for transfer learning")
    else:
        model = SleepVideoModel(config)
        print("Initialized new model from scratch")
    
    ###################### Logging ######################
    # Log model complexity metrics and setup profiler
    log_model_summary(model, input_size=(150, 1, 224, 224))
    profiler_dir = ensure_results_dir(config, 'profiler')
    profiler = SimpleProfiler(dirpath=profiler_dir, filename=f"{name}_profile.log")
    
    ###################### Callbacks ######################
    # Setup checkpoint saving and model summary callbacks
    weights_dir = ensure_results_dir(config, 'weights', name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_dir,
        filename='{epoch:02d}-{val_f1:.2f}',
        save_top_k=1,  # Save only the best model
        save_last=True,  # Also save the last checkpoint
        monitor='val_f1',  # Monitor F1 score for best model selection
        mode='max'  # Higher F1 is better
    )
    modelsummary_callback = ModelSummary(max_depth=2)
    
    # Add progressive unfreezing callback if specified
    autofreeze_callback = AutoFreezeCallback(
        warmup_epochs=WARMUP_EPOCHS, 
        progressive_unfreeze_epochs=PROGRESSIVE_UNFREEZE_EPOCHS, 
        finetune_epochs=FINETUNE_EPOCHS, 
        datamodule=video_data_module
    )
    
    # Combine all callbacks
    callbacks = [checkpoint_callback, modelsummary_callback]
    if WARMUP_EPOCHS > 0 or PROGRESSIVE_UNFREEZE_EPOCHS > 0:
        callbacks.append(autofreeze_callback)

    ###################### Trainer ######################
    # Configure PyTorch Lightning trainer with all settings
    trainer_kwargs = {
        'max_epochs': WARMUP_EPOCHS + PROGRESSIVE_UNFREEZE_EPOCHS + FINETUNE_EPOCHS,
        'accumulate_grad_batches': ACCUMULATE_GRAD_BATCHES,
        'accelerator': 'gpu',
        'devices': NUM_DEVICES,
        'strategy': 'ddp' if NUM_DEVICES > 1 else 'auto',  # Use DDP for multi-GPU
        'logger': wandb_logger,
        'callbacks': callbacks,
        'precision': '16',  # Mixed precision training
        'profiler': profiler,
        'benchmark': True,  # Optimize CUDA operations
        'num_sanity_val_steps': 2,  # Quick validation check
        'sync_batchnorm': True if NUM_DEVICES > 1 else False,  # Sync batch norm for DDP
        'limit_train_batches': SAMPLE_RATIO,  # Use subset of data if specified
        'fast_dev_run': False,  # Full training run
    }

    trainer = Trainer(**trainer_kwargs)

    ###################### Training ######################
    # Start training process with optional checkpoint resuming
    ckpt_path = None if transfer_from_ckpt else resume_checkpoint
    print(f"Starting training with {NUM_DEVICES} GPU(s)")
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    
    trainer.fit(model, datamodule=video_data_module, ckpt_path=ckpt_path)
    
    ###################### Save Best Model Path ######################
    # Save best model path for potential transfer learning use
    if rank == 0:
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model checkpoint path: {best_model_path}")
        
        if not best_model_path:
            raise ValueError("No checkpoint was created during training.")
        
        # Extract experiment number and save best model path
        exp_number = args.config.split('/')[-1].split('.')[0]
        save_best_model_path(config, exp_number, best_model_path)
        print(f"Saved best model path for experiment {exp_number}")

        # Clean up WandB resources
        wandb.finish()

if __name__ == '__main__':
    main()