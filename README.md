# Dog or Dough 🐕🍞

A multi-class image classification project that in the current implementation distinguishes between dogs and bread (dough) using a custom neural network implementation built from scratch with NumPy. This project demonstrates deep learning fundamentals including forward/backward propagation, optimization algorithms, and regularization techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Hyperparameters](#️-hyperparameters)
- [Results](#results)

## 🎯 Overview

This project implements a complete deep learning pipeline for multi-class image classification:
- **Task**: Classify images as either "dog" or "dough"
- **Approach**: Custom neural network built from scratch using only NumPy
- **Input**: RGB images resized to 64×64 pixels
- **Output**: Multi-class classification with confidence scores

## ✨ Features

- **Custom Neural Network**: Fully implemented from scratch (no deep learning frameworks)
- **Adam Optimizer**: Adaptive learning rate optimization with momentum
- **Regularization**: Dropout and L2 weight decay to prevent overfitting
- **Data Augmentation**: On-the-fly augmentation using Albumentations (horizontal flip, rotation, brightness/contrast)
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Decay**: Adaptive learning rate reduction on plateau
- **Comprehensive Metrics**: Accuracy, precision, recall, and F1-score (per-class and macro-averaged)
- **Model Checkpointing**: Automatic saving of best models during training
- **Inference & Evaluation**: Separate modes for inferencing against or without ground truths.

## 📁 Project Structure

```
Dog-Or-Dough/
├── src/
│   ├── model.py              # NeuralNetwork class implementation
│   ├── train.py              # Training script with Trainer class
│   ├── eval.py               # Evaluation and inference script
│   └── helpers/
│       ├── activations.py    # ReLU and Softmax functions
│       ├── utils.py          # Data loading, preprocessing, augmentation
│       └── metrics.py        # Evaluation metrics and plotting
├── data/
│   ├── train/                # Training images (dog/, dough/)
│   ├── val/                  # Validation images (dog/, dough/)
│   └── test/                 # Test images (dog/, dough/)
├── checkpoints/
│   └── Exp6/                 # Experiment 6 checkpoints and results
│       ├── model_parameters_*.npz
│       └── loss_curve.png
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.7+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Dog-Or-Dough
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training

Train a new model by modifying the configuration in `src/train.py`:

```python
from model import NeuralNetwork
from train import Trainer
from helpers.utils import get_augmentation_pipeline

# Define model architecture
NN = NeuralNetwork(
    layers=[128, 64, 32, 16, 2],      # Hidden layers + output layer
    input_size=(64, 64),               # Input image dimensions
    learning_rate=1e-4,                # Initial learning rate
    weight_decay=1e-5,                 # L2 regularization strength
    dropout=[0.3, 0.2, 0.1, 0.0],      # Dropout rates per layer
)

# Create trainer
trainer = Trainer(
    NN=NN,
    epochs=200,                        # Maximum epochs
    batch_size=32,                     # Batch size
    patience=25,                       # Early stopping patience
    use_lr_decay=True,                 # Enable learning rate decay
    lr_patience=10,                    # LR decay patience
    min_lr=1e-6,                       # Minimum learning rate
    experiment_name="Exp6",            # Experiment name
    aug_pipeline=get_augmentation_pipeline()  # Data augmentation
)

# Start training
trainer.train()
```

Run training:
```bash
cd src
python train.py
```

### Evaluation

Evaluate the model against the ground truths of the test data:

1. Edit `src/eval.py` to set the model path:
```python
model_path = os.path.join(script_dir, "../checkpoints/Exp6/model_parameters_153.npz")
```

2. Set mode to `"evaluate"`:
```python
mode = "evaluate"
```

3. Run evaluation:
```bash
python eval.py
```

### Inference

Perform inference on new images:

1. Edit `src/eval.py`:
```python
mode = "inference"
```

2. Place images in `data/test/` or modify the image paths in the script

3. Run inference:
```bash
python eval.py
```

## 🏗️ Architecture

### Neural Network Structure

**Best Experiment:**
- **Input Layer**: 64×64×3 = 12,288 features
- **Hidden Layer 1**: 128 neurons (ReLU, Dropout 0.3)
- **Hidden Layer 2**: 64 neurons (ReLU, Dropout 0.2)
- **Hidden Layer 3**: 32 neurons (ReLU, Dropout 0.1)
- **Hidden Layer 4**: 16 neurons (ReLU, No Dropout)
- **Output Layer**: 2 neurons (Softmax)

**Total Trainable Parameters**: ~1.6M

### Key Components

1. **Forward Propagation**
   - Linear transformation: `Z = W·A + b`
   - ReLU activation for hidden layers
   - Softmax activation for output layer
   - Dropout during training (disabled during inference)

2. **Backward Propagation**
   - Cross-entropy loss with L2 regularization
   - Gradient computation via chain rule
   - Dropout mask applied to gradients

3. **Optimization**
   - **Adam Optimizer**: Combines momentum (β₁=0.9) and RMSprop (β₂=0.999)
   - Bias correction for moving averages
   - Adaptive learning rates per parameter

4. **Regularization**
   - **Dropout**: Randomly zeroes activations during training (0.3 → 0.2 → 0.1 → 0.0)
   - **L2 Weight Decay**: Penalizes large weights (λ=1e-5)

## ⚙️ Hyperparameters

All key tunable hyperparameters are exposed either in `model.py` (via the `NeuralNetwork` constructor) or in `train.py` (via the `Trainer` constructor):

- **Model architecture**
  - **`layers`**: List of layer sizes, including the output layer (e.g. `[128, 64, 32, 16, 2]`).
  - **`input_size`**: Input spatial resolution as `(height, width)` (default `(64, 64)`).
  - **`color`**: Whether to treat images as RGB (`True`) or grayscale (`False`).

- **Optimization (Adam)**
  - **`learning_rate`**: Initial learning rate for Adam (e.g. `1e-4`).
  - **`beta1`**: Exponential decay rate for the first moment estimates (default `0.9`).
  - **`beta2`**: Exponential decay rate for the second moment estimates (default `0.999`).
  - **`epsilon`**: Small constant for numerical stability in Adam (default `1e-8`).

- **Regularization**
  - **`weight_decay`**: L2 regularization strength (e.g. `1e-5`).
  - **`dropout`**: List of dropout rates for each hidden layer (e.g. `[0.3, 0.2, 0.1, 0.0]`).

- **Training loop (Trainer)**
  - **`epochs`**: Maximum number of training epochs.
  - **`batch_size`**: Number of samples per batch.
  - **`patience`**: Early stopping patience (number of epochs without validation loss improvement).
  - **`use_lr_decay`**: Whether to enable learning rate decay on plateau.
  - **`lr_decay`**: Multiplicative factor to decay the learning rate (e.g. `0.5`).
  - **`lr_patience`**: Number of epochs without improvement before decaying the learning rate.
  - **`min_lr`**: Lower bound for the learning rate when decay is enabled.

- **Data and experiment setup**
  - **`data_folder_name`**: Name of the data folder (default `"data"`).
  - **`ckpt_folder_name`**: Folder for saving checkpoints (default `"checkpoints"`).
  - **`experiment_name`**: Sub-folder name under checkpoints to separate runs (e.g. `"Exp6"`).
  - **`aug_pipeline`**: Albumentations pipeline used for data augmentation (can be `None` to disable).

## 🎯 Results

### Experiment 6 Performance

The best model from Experiment 6 achieved the following performance:

- **Model Checkpoint**: `checkpoints/Exp6/model_parameters_153.npz`
- **Training Epochs**: 153 (early stopped based on validation loss)

**Current Hyperparameter Choice (Exp6)**

- **Architecture**: `[128, 64, 32, 16, 2]`
- **Input Size**: `(64, 64)` with 3 color channels (RGB)
- **Optimizer**: Adam (`learning_rate=1e-4`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`)
- **Regularization**:
  - Weight decay (`L2`): `1e-5`
  - Dropout: `[0.3, 0.2, 0.1, 0.0]` on the four hidden layers
- **Training Loop**:
  - Batch size: `32`
  - Max epochs: `200`
  - Early stopping patience: `25`
  - LR decay: enabled with factor `0.5`, patience `10`, minimum LR `1e-6`
  - Data augmentation: enabled via `get_augmentation_pipeline()` (flip, rotate, brightness/contrast)

**Loss Curve**

The training and validation loss curves for Experiment 6 are shown below (saved by the training script as part of the run):

![Training and Validation Loss Curve](checkpoints/Exp6/loss_curve.png)

*Note: Specific test metrics (accuracy, precision, recall, F1) should be obtained by running the evaluation script on your own test set.*
