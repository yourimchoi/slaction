#!/usr/bin/env bash
set -e

echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

echo "Creating slaction environment..."
conda create -n slaction python=3.11 ipykernel pytorch=2.3.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Activating environment with source..."
source activate slaction

echo "Installing requirements..."
# Deep Learning Frameworks
pip install pytorch-lightning==2.4.0
pip install torchmetrics==1.4.1
pip install lightning-utilities==0.11.7
pip install tensorflow-cpu

# Experiment Tracking
pip install wandb==0.18.0

# Model Analysis & Summary
pip install torchsummary==1.5.1
pip install ptflops==0.7.3

# Video Processing
pip install pytorchvideo==0.1.5
pip install "git+https://github.com/Atze00/MoViNet-pytorch.git"
pip install av==13.0.0
pip install einops==0.8.0

# Data Processing
pip install tfrecord==1.14.5
pip install pandas==2.2.2
pip install numpy==2.0.1
pip install PyYAML==6.0.1

# Machine Learning
pip install scikit-learn==1.5.2
pip install scipy==1.14.1

# Visualization
pip install matplotlib==3.9.2
pip install seaborn==0.13.2

# Progress & Utilities
pip install tqdm==4.66.5
pip install packaging==24.1
pip install requests==2.32.3

# Image Processing
pip install Pillow==10.4.0

# Configuration
pip install yacs==0.1.8

echo "Setup completed!"