"""
Evaluation Script for Sleep Action Recognition

This module provides evaluation capabilities for trained sleep action recognition models.
It loads the best checkpoint from training and evaluates on specified test sets,
logging results to Weights & Biases for comprehensive analysis.

Key Features:
- Automatic best model checkpoint loading
- Multi-dataset evaluation support
- Integration with existing WandB experiment runs
- Comprehensive result logging and visualization
"""

import wandb
import yaml
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from VideoTrainer import SleepVideoModel
from DataFactory import VideoDataModuleLightning
from Utils import get_results_path

@rank_zero_only
def init_wandb(name, config):
    """
    Initialize Weights & Biases for evaluation tracking.
    
    Args:
        name (str): Experiment name for tracking
        config (dict): Configuration dictionary with evaluation parameters
        
    Returns:
        wandb.config: WandB configuration object
    """
    wandb.init(project=config['wandb_project'], name=name, config=config)
    return wandb.config

def get_wandb_id_from_config(config_path):
    """
    Extract WandB run ID from configuration file to resume existing experiment.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        str or None: WandB run ID if found, None otherwise
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('wandb_id')

def main():
    """
    Main evaluation function that orchestrates model testing on specified datasets.
    
    This function handles:
    - Configuration loading and argument parsing
    - WandB experiment resuming for result tracking
    - Best model checkpoint loading
    - Multi-dataset evaluation with result logging
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sleep Action Recognition Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Generate experiment name from config filename
    name = 'exp_' + args.config.split('/')[-1].split('.')[0]
    config['name'] = name
    
    # Resume existing WandB experiment using saved ID
    wandb_id = get_wandb_id_from_config(args.config)
    if not wandb_id:
        raise ValueError("No wandb_id found in config file. Please run training first.")

    # Initialize WandB with existing experiment
    wandb.init(
        id=wandb_id,
        project=config['wandb_project'],
        resume="allow",
        name=name
    )
    print(f"Resuming wandb run: {wandb_id}")

    # Create WandB logger linked to existing experiment
    wandb_logger = WandbLogger(
        project=config['wandb_project'],
        name=name,
        id=wandb_id,
        resume="allow",
        experiment=wandb.run  # Use current run
    )

    # Extract evaluation parameters from configuration
    BATCHSIZE = config['batchsize']
    CSV_FILE_PATH = config['csv_file_path']
    BASE_TFRECORD_PATH = config['base_tfrecord_path']
    NUM_WORKERS = args.num_workers
    LABEL_COLUMN = config['label_column']
    VIDEO_LENGTH_SEC = config.get('video_length_sec', 30)
    BATCHSIZE = 16  # Use smaller batch size for evaluation

    # Load best model checkpoint from training
    exp_number = args.config.split('/')[-1].split('.')[0]
    
    try:
        temp_file_path = get_results_path(config, 'temp', f'best_model_path_{exp_number}.txt')
        with open(temp_file_path, 'r') as f:
            best_model_path = f.read().strip()
        print(f"Loading best model from: {best_model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find best model path file for experiment {exp_number}. Please ensure training has completed successfully.")
    
    # Load trained model from checkpoint
    model = SleepVideoModel.load_from_checkpoint(best_model_path, config=config)
    
    # Get test datasets from configuration
    test_sets = config.get('test_sets', [])
    if not test_sets:
        raise ValueError("No test_sets specified in config file. Please add test_sets list to configuration.")

    # Initialize trainer for evaluation
    test_trainer = Trainer(
        accelerator='gpu',
        devices=1,  # Single GPU for evaluation
        logger=wandb_logger,
        benchmark=True,  # Optimize for consistent input sizes
    )

    # Evaluate model on each specified test set
    print(f"Starting evaluation on {len(test_sets)} test set(s)")
    for test_set in test_sets:
        print(f"Evaluating on test set: {test_set}")
        
        # Update configuration with current test set
        config['test_set'] = test_set
        
        # Create data module for current test set
        test_data_module = VideoDataModuleLightning(
            base_csv_path=CSV_FILE_PATH,
            base_tfrecord_path=BASE_TFRECORD_PATH,
            label_column=LABEL_COLUMN,
            batch_size=BATCHSIZE,
            shuffle=False,  # No shuffling for evaluation
            num_workers=NUM_WORKERS,
            weighted_sampling=False,  # No weighted sampling for evaluation
            train_set=None,
            valid_set=None,
            test_set=test_set,
            video_length_sec=VIDEO_LENGTH_SEC
        )
        
        # Setup data module and update model configuration
        test_data_module.setup(stage='test')
        model.config = config  # Update model config with current test set
        
        # Run evaluation
        test_trainer.test(model, datamodule=test_data_module)
        print(f"Completed evaluation on {test_set}")

    print("All evaluations completed successfully")
    wandb.finish()  # Clean up WandB resources

if __name__ == '__main__':
    main()
