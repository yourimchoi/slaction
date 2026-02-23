"""
Video Trainer Module for Sleep Action Recognition

This module contains PyTorch Lightning models and callbacks for training and evaluating
video-based sleep action recognition models using MoviNet architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

from Model import MoviNet_A2_50, MoviNet_A2_150
from Utils import CosineAnnealingWarmUpRestarts


class SleepVideoModel(pl.LightningModule):
    """
    PyTorch Lightning module for sleep video classification using MoviNet architecture.
    
    This class implements a complete training pipeline for video-based sleep action recognition,
    including data augmentation (mixup), metric tracking, and model checkpointing.
    
    Args:
        config (dict): Configuration dictionary containing model hyperparameters and settings.
    """
    
    def __init__(self, config):
        super(SleepVideoModel, self).__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Initialize model based on input configuration
        if config.get('squeeze_rgb', True):
            self.model = MoviNet_A2_50(config['num_class'], config.get('pretrained', True))
        else:
            self.model = MoviNet_A2_150(config['num_class'], config.get('pretrained', True))
        
        # Loss function with label smoothing for regularization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        self.num_classes = config['num_class']
        self.init_metrics(self.num_classes)
        self.test_results = [] 
        self.video_length_sec = config.get('video_length_sec', 30)

    def init_metrics(self, num_classes):
        """
        Initialize torchmetrics for training, validation, and testing phases.
        
        Args:
            num_classes (int): Number of classes for classification task.
        """
        task_type = 'binary' if num_classes == 2 else 'multiclass'
        metrics_kwargs = {'task': task_type, 'num_classes': num_classes} if num_classes != 2 else {'task': task_type}
        
        # Training metrics
        self.train_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.train_f1 = torchmetrics.F1Score(average='macro', **metrics_kwargs)
        
        # Validation metrics
        self.val_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.val_f1 = torchmetrics.F1Score(average='macro', **metrics_kwargs)
        self.val_auroc = torchmetrics.AUROC(**metrics_kwargs)

        # Test metrics
        self.test_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.test_f1 = torchmetrics.F1Score(average='macro', **metrics_kwargs)
        self.test_precision = torchmetrics.Precision(average='macro', **metrics_kwargs)
        self.test_recall = torchmetrics.Recall(average='macro', **metrics_kwargs)
        self.test_auroc = torchmetrics.AUROC(**metrics_kwargs)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(**metrics_kwargs)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer and scheduler configuration for PyTorch Lightning.
        """
        learning_rate = self.hparams['learning_rate']
        optimizer_name = self.hparams['optimizer'].lower()
        scheduler_name = self.hparams.get('scheduler', 'cosine_warmup').lower()

        # Available optimizers
        optimizers = {
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adam': optim.Adam
        }
        
        if optimizer_name not in optimizers:
            raise ValueError(f"Optimizer {self.hparams['optimizer']} is not supported.")
        
        optimizer_class = optimizers[optimizer_name]
        optimizer_args = {'params': self.model.parameters(), 'lr': learning_rate}
        
        # Add momentum for SGD optimizer
        if optimizer_name == 'sgd':
            optimizer_args['momentum'] = 0.9
        
        optimizer = optimizer_class(**optimizer_args)

        # Scheduler configuration
        scheduler_params = self.hparams.get('scheduler_params', {})
        
        if scheduler_name == 'constant':
            return optimizer
        elif scheduler_name == 'cosine_warmup':
            cosine_params = scheduler_params.get('cosine_warmup', {})
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer=optimizer,
                T_0=cosine_params.get('T_0', 1),
                T_mult=cosine_params.get('T_mult', 2),
                eta_max=learning_rate,
                eta_min=cosine_params.get('eta_min', 1e-5),
                T_up=cosine_params.get('T_up', 1),
                gamma=cosine_params.get('gamma', 1)
            )
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Scheduler {scheduler_name} is not supported.")

    def mixup_data(self, x, y, alpha=1.0):
        """
        Apply mixup data augmentation.
        
        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor): Labels
            alpha (float): Mixup interpolation parameter
            
        Returns:
            Tuple of mixed data, original labels, shuffled labels, and lambda value
        """
        lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1
        index = torch.randperm(x.size()[0]).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Calculate mixup loss by interpolating between two losses.
        
        Args:
            criterion: Loss function
            pred (torch.Tensor): Model predictions
            y_a (torch.Tensor): Original labels
            y_b (torch.Tensor): Shuffled labels
            lam (float): Interpolation parameter
            
        Returns:
            torch.Tensor: Mixup loss
        """
        pred = pred.float()  # Ensure pred is float32
        y_a = y_a.long()    # Ensure targets are long
        y_b = y_b.long()    # Ensure targets are long
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.
        
        Args:
            batch: Input batch containing (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            dict: Training metrics and loss
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        
        # Ensure batch dimension is maintained
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        if self.hparams['use_mixup']:
            inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets, self.hparams['mixup_alpha'])
            outputs = self(inputs)
            loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            preds = torch.argmax(outputs, dim=1)
            self.update_metrics(preds, targets_a, targets_b)
        else:
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            preds = torch.argmax(outputs, dim=1)
            self.update_metrics(preds, targets)
        
        # Log training metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            'loss': loss,
            'train_acc': self.train_acc,
            'train_f1': self.train_f1
        }

    def update_metrics(self, preds, targets_a, targets_b=None):
        """
        Update training metrics with predictions and targets.
        
        Args:
            preds (torch.Tensor): Model predictions
            targets_a (torch.Tensor): Primary targets
            targets_b (torch.Tensor, optional): Secondary targets for mixup
        """
        if targets_b is not None:
            # Update metrics for both original and mixed targets in mixup
            self.train_acc.update(preds, targets_a)
            self.train_acc.update(preds, targets_b)
            self.train_f1.update(preds, targets_a)
            self.train_f1.update(preds, targets_b)
        else:
            self.train_acc.update(preds, targets_a)
            self.train_f1.update(preds, targets_a)

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.
        
        Args:
            batch: Input batch containing (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            dict: Validation metrics and loss
        """
        x, targets = batch
        outputs = self.forward(x)
        
        # Ensure batch dimension is maintained
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        outputs = outputs.float()
        targets = targets.long()
        
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        probas = torch.softmax(outputs, dim=1)
        
        # Update validation metrics
        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_auroc.update(probas, targets)
        
        # Log validation metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {
            'val_loss': loss,
            'val_acc': self.val_acc,
            'val_f1': self.val_f1,
            'val_auroc': self.val_auroc
        }

    def on_validation_epoch_start(self):
        """Reset validation metrics at the start of each validation epoch."""
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

    def on_test_epoch_start(self):
        """Reset test metrics and initialize results storage at the start of test epoch."""
        self.test_results = []
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_auroc.reset()
        self.test_confusion_matrix.reset()

    def test_step(self, batch, batch_idx):
        """
        Test step for one batch.
        
        Args:
            batch: Input batch containing (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            dict: Test metrics and loss
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        outputs = self(inputs)
        
        # Ensure batch dimension is maintained
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        outputs = outputs.float()
        targets = targets.long()
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        preds = torch.argmax(outputs, dim=1)
        probas = torch.softmax(outputs, dim=1) if self.num_classes > 2 else torch.softmax(outputs, dim=1)[:, 1]

        # Update test metrics
        self.test_acc.update(preds, targets)
        self.test_f1.update(preds, targets)
        self.test_auroc.update(probas, targets)
        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_confusion_matrix.update(preds, targets)

        # Log test metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Store results for post-processing
        self.test_results.append({
            'probas': probas.cpu().numpy(),
            'preds': preds.cpu().numpy(),
            'targets': targets.cpu().numpy()
        })

        return {
            'test_loss': loss,
            'test_acc': self.test_acc,
            'test_f1': self.test_f1,
            'test_auroc': self.test_auroc
        }

    def on_test_epoch_end(self):
        """
        Process and save test results at the end of testing epoch.
        
        Generates confusion matrix visualization and saves detailed results to CSV files.
        """
        # Generate and log confusion matrix
        confusion_matrix = self.test_confusion_matrix.compute().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        # Prepare result directories
        test_set_name = self.config['test_set']
        exp_name = self.config['name']
        exp_number = exp_name.split('_')[1]  # Extract experiment number from the name
        result_dir = f'../../results/result/{exp_number}'
        os.makedirs(result_dir, exist_ok=True)
        
        # Log confusion matrix to wandb
        self.logger.experiment.log({"confusion_matrix": wandb.Image(fig, caption=test_set_name)})
        plt.close(fig)

        # Prepare detailed results for CSV export
        results_list = []
        
        for result in self.test_results:
            for i, target in enumerate(result['targets']):
                results_list.append({
                    'target': target,
                    'prediction': result['preds'][i],
                    'probabilities': result['probas'][i]
                })

        # Save results to CSV files
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'{result_dir}/{test_set_name}_results.csv', index=False)
        
        # Save confusion matrix as CSV
        confusion_matrix_df = pd.DataFrame(confusion_matrix)
        confusion_matrix_df.to_csv(f'{result_dir}/{test_set_name}_confusion_matrix.csv', index=False)


class AutoFreezeCallback(Callback):
    """
    PyTorch Lightning callback for automatic layer freezing and progressive unfreezing.
    
    This callback implements a three-phase training strategy:
    1. Warmup phase: Only the final classification layer is trainable
    2. Progressive unfreeze phase: Gradually unfreeze layers from early to late
    3. Fine-tuning phase: All layers are trainable with full learning rate
    
    Args:
        warmup_epochs (int): Number of epochs for warmup phase with frozen backbone
        progressive_unfreeze_epochs (int): Number of epochs for progressive unfreezing
        finetune_epochs (int): Number of epochs for full fine-tuning
        datamodule (optional): Lightning data module (for compatibility)
    """
    
    def __init__(self, warmup_epochs, progressive_unfreeze_epochs, finetune_epochs, datamodule=None):
        self.warmup_epochs = warmup_epochs
        self.progressive_unfreeze_epochs = progressive_unfreeze_epochs
        self.finetune_epochs = finetune_epochs
        self.total_epochs = warmup_epochs + progressive_unfreeze_epochs + finetune_epochs

    def on_train_start(self, trainer, pl_module):
        """
        Initialize layer freezing at the start of training.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        current_epoch = trainer.current_epoch
        model_type = pl_module.config['model']
        
        # Freeze all backbone parameters, keep only final layer trainable
        for param in pl_module.model.model.parameters():
            param.requires_grad = False
        print("MoviNet model layers frozen, except for the fc layer.")
            
        # If resuming from checkpoint after progressive unfreeze phase
        if current_epoch > self.warmup_epochs + self.progressive_unfreeze_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True
            print("Resuming training with all layers unfrozen.")

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Handle phase transitions at the start of each epoch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        current_epoch = trainer.current_epoch
        
        if current_epoch == self.warmup_epochs:
            print("Starting progressive unfreeze phase.")
        elif self.warmup_epochs < current_epoch <= self.warmup_epochs + self.progressive_unfreeze_epochs:
            self.progressive_unfreeze(pl_module, current_epoch - self.warmup_epochs)
        elif current_epoch == self.warmup_epochs + self.progressive_unfreeze_epochs:
            # Unfreeze all layers for fine-tuning phase
            for param in pl_module.model.model.parameters():
                param.requires_grad = True
            print("Starting fine-tuning phase with all layers unfrozen.")
            
            # Reset the scheduler to start cosine annealing fresh
            for scheduler in trainer.lr_schedulers:
                scheduler['scheduler'].last_epoch = -1

    def progressive_unfreeze(self, pl_module, epoch):
        """
        Progressively unfreeze layers based on current epoch.
        
        Args:
            pl_module: Lightning module being trained
            epoch (int): Current epoch within the progressive unfreeze phase
        """
        total_layers = len(list(pl_module.model.model.parameters()))
        layers_to_unfreeze = (epoch * total_layers) // self.progressive_unfreeze_epochs
        
        for i, param in enumerate(pl_module.model.model.parameters()):
            if i < layers_to_unfreeze:
                param.requires_grad = True
                
        print(f"Progressive unfreezing: {layers_to_unfreeze}/{total_layers} layers unfrozen.")