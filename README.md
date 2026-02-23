# SlAction: Infrared Video-Based Sleep Apnea Detection

## Overview

**SlAction** is a deep learning framework for automated sleep apnea detection from infrared sleep videos.
The system analyzes respiratory motion and body dynamics during sleep using a MoviNet-based video model, enabling non-contact and scalable sleep monitoring.

This repository contains code for:

* Data preprocessing from video frames
* Model training and evaluation
* Experiment configuration management
* Performance analysis and regression

---

## Repository Structure

```
SlAction/
├── ckpt/
│     ├── epoch=00-val_f1=0.70.ckpt
│     └── last.ckpt
├── code/
│   ├── play.sh
│   ├── exp_config/
│   │   └── 000.yaml
│   ├── scripts/
│   │   ├── train.sh
│   │   ├── evaluate.sh
│   │   └── regression.sh
│   └── src/
│       ├── core/
│       │   ├── Train.py
│       │   ├── Evaluate.py
│       │   ├── VideoTrainer.py
│       │   ├── Model.py
│       │   ├── DataFactory.py
│       │   └── Utils.py
│       ├── preprocess/
│       │   └── convert_jpg_to_tfrecord.py
│       └── postprocess/
│           ├── Analysis.py
│           └── Regression.py
├── data/
│   ├── frames/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── labels/
```

---

## Environment Setup

Create the environment and install dependencies:

```bash
chmod +x environment/postinstall.sh
./environment/postinstall.sh
```

Then activate:

```bash
conda activate slaction
```

Main dependencies include:

* Python 3.11
* PyTorch 2.3 (CUDA 11.8)
* PyTorch Lightning
* MoviNet (PyTorch implementation)
* TensorFlow, scikit-learn, wandb

---

## Quick Start

Run the full pipeline:

```bash
cd code
chmod +x play.sh
./play.sh
```

This executes:

1. **Preprocessing** – converts frame sequences to TFRecord format
2. **Training** – trains MoviNet model with transfer learning
3. **Evaluation** – computes metrics and generates reports

---

## Manual Execution

### Preprocessing

```bash
python code/src/preprocess/convert_jpg_to_tfrecord.py \
    --demo_frames_dir data/demo_frames \
    --output_dir results/tfrecord \
    --workers 8
```

### Training

```bash
bash code/scripts/train.sh --gpus 0 --exp 000 --num_workers 8
```

### Evaluation

```bash
bash code/scripts/evaluate.sh --gpus 0 --exp 000 --num_workers 8
```

---

## Experiment Management

Create a new experiment:

```bash
cp code/exp_config/000.yaml code/exp_config/001.yaml
vim code/exp_config/001.yaml
bash code/scripts/train.sh --gpus 0 --exp 001
```

---

## Outputs

Training produces:

* Model checkpoints in `ckpt/`
* Logs and metrics (wandb/local logs)
* Confusion matrices and classification reports
* Per-class performance analysis


---

## Acknowledgments

This project builds on:

* MoviNet video architecture
* PyTorch Lightning training framework
* Infrared video-based sleep monitoring research
