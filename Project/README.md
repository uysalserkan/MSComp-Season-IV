# CMP722 - Advanced Computer Vision Project

## Project Overview

This project is a computer vision application that uses advanced computer vision techniques to solve a real-world problem.

Selected Dataset: [STL-10](https://cs.stanford.edu/~acoates/stl10/)
Selected Architecture: ViT (Vision Transformer)

## Dataset

STL-10 is a dataset for developing unsupervised feature learning, deep learning, and self-taught learning algorithms. It contains:
- Training set: 5,000 labeled images (10 classes, 500 per class)
- Test set: 8,000 labeled images (10 classes, 800 per class)
- Unlabeled set: 100,000 unlabeled images
- Image size: 96x96 pixels

## Project Structure

```
Project/
├── models/
│   ├── __init__.py
│   ├── vit_model.py          # ViT model class for finetuning
│   └── ssl_model.py          # SSL contrastive learning model
├── stl10_dataset.py          # STL-10 dataset loader with ViT transforms
├── config.py                # Supervised training configuration
├── ssl_config.py            # SSL training configuration
├── utils.py                 # Utility functions
├── train.py                 # Supervised training script
├── train_ssl.py             # Self-supervised training script
├── eval_ssl.py              # Linear evaluation script
└── README.md                # This file
```

## Installation

1. Install dependencies using Poetry:
```bash
poetry install
```

Or install manually:
```bash
pip install torch torchvision timm scikit-learn lightly
```

## Usage

### Basic Training

Train a ViT model on STL-10 with default settings:

```bash
python train.py
```

### Custom Training Configuration

Train with custom hyperparameters:

```bash
python train.py \
    --model_name vit_base_patch16_224 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --mixed_precision \
    --use_augmentation
```

### Training Options

Key command-line arguments:

**Model Options:**
- `--model_name`: ViT model name from timm (default: `vit_base_patch16_224`)
- `--pretrained`: Use pretrained ImageNet weights (default: True)
- `--freeze_backbone`: Freeze backbone layers (default: False)
- `--ssl_pretrained_path`: Path to SSL pretrained checkpoint (default: None)
- `--use_ssl_pretrained`: Use SSL pretrained weights instead of ImageNet (default: False)

**Data Options:**
- `--data_root`: Dataset root directory (default: `./data`)
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: Data loading workers (default: 4)
- `--image_size`: Input image size (default: 224)
- `--use_augmentation`: Enable data augmentation (default: True)

**Training Options:**
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--mixed_precision`: Use mixed precision training (default: True)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--gradient_clip_norm`: Gradient clipping norm (default: 1.0)

**Learning Rate Scheduling:**
- `--lr_scheduler`: Scheduler type - `cosine`, `step`, `onecycle`, or `None` (default: `cosine`)
- `--lr_min`: Minimum learning rate (default: 1e-6)

**Checkpointing:**
- `--checkpoint_dir`: Checkpoint directory (default: `./checkpoints`)
- `--resume_from`: Path to checkpoint to resume from (default: None)
- `--save_freq`: Save checkpoint every N epochs (default: 5)

**Other Options:**
- `--seed`: Random seed (default: 42)
- `--early_stopping_patience`: Early stopping patience (default: 10)
- `--print_freq`: Print metrics every N batches (default: 10)

### Resume Training

Resume training from a checkpoint:

```bash
python train.py --resume_from ./checkpoints/best_model.pth
```

### Using the Dataset Loader

You can also use the dataset loader programmatically:

```python
from stl10_dataset import STL10DatasetLoader

# Initialize loader
loader = STL10DatasetLoader(root="./data", download=True)

# Get ViT-specific data loaders
train_loader, test_loader = loader.get_vit_data_loaders(
    batch_size=32,
    image_size=224,
    use_augmentation=True
)

# Or get custom transforms
train_transform = loader.get_vit_train_transform(
    image_size=224,
    use_augmentation=True
)
val_transform = loader.get_vit_val_transform(image_size=224)
```

### Using the ViT Model

Load and use the ViT model:

```python
from models.vit_model import ViTFinetuner

# Create model
model = ViTFinetuner(
    model_name="vit_base_patch16_224",
    num_classes=10,
    pretrained=True,
    image_size=224
)

# Freeze/unfreeze layers
model.freeze_backbone()  # Freeze all backbone layers
model.unfreeze_backbone()  # Unfreeze all layers
model.freeze_layers(6)  # Freeze first 6 transformer blocks

# Get parameter counts
params = model.count_parameters()
print(f"Total parameters: {params['total']:,}")
print(f"Trainable parameters: {params['trainable']:,}")
```

## Features

### Best Practices Implemented

1. **Data Handling:**
   - Proper data augmentation (RandomResizedCrop, RandomHorizontalFlip, ColorJitter, etc.)
   - ImageNet normalization for pretrained models
   - Efficient DataLoader with multiple workers

2. **Model Setup:**
   - Pretrained ViT models from timm
   - Customizable classification head
   - Layer freezing support for gradual unfreezing

3. **Training:**
   - Mixed precision training (AMP) for efficiency
   - Learning rate scheduling (CosineAnnealingLR, StepLR, OneCycleLR)
   - Gradient clipping for stability
   - Weight decay for regularization
   - Early stopping to prevent overfitting
   - Gradient accumulation for larger effective batch sizes

4. **Monitoring:**
   - Real-time training/validation metrics
   - Best model checkpointing
   - Comprehensive logging
   - Progress bars with tqdm

5. **Reproducibility:**
   - Random seed setting
   - Deterministic operations
   - Configuration management

## Output

The training script will:
- Save checkpoints in `./checkpoints/` directory
- Save the best model as `best_model.pth`
- Print training progress to console
- Display metrics (loss, accuracy) for each epoch

## Example Training Output

```
==================================================
Model Summary
==================================================
Total parameters: 86,567,690
Trainable parameters: 86,567,690
Frozen parameters: 0
==================================================

==================================================
Starting Training
==================================================
Total epochs: 50
Batch size: 32
Learning rate: 0.0001
Mixed precision: True
Gradient accumulation: 1
==================================================

Epoch 1/50 [Train]: 100%|████████| 157/157 [00:45<00:00, loss=1.2345, acc=0.4567, lr=0.000100]
Epoch 1/50 [Val]: 100%|████████| 250/250 [00:12<00:00, loss=0.9876, acc=0.6789]

Epoch 1/50 Summary:
  Train Loss: 1.2345 | Train Acc: 0.4567
  Val Loss: 0.9876 | Val Acc: 0.6789
  Learning Rate: 0.000100
  Time: 57s
--------------------------------------------------
```

## Self-Supervised Learning (SSL)

This project includes self-supervised contrastive learning capabilities using SimCLR-style training on the unlabeled STL-10 dataset (100k images).

### Two-Stage Training Pipeline

The recommended approach is a two-stage training pipeline:

1. **Stage 1: Self-Supervised Pretraining** - Learn visual representations from unlabeled data
2. **Stage 2: Supervised Finetuning** - Finetune on labeled data using SSL-pretrained weights

### Stage 1: SSL Pretraining

Train a contrastive learning model on unlabeled STL-10 data:

```bash
python train_ssl.py \
    --backbone_name vit_base_patch16_224 \
    --batch_size 256 \
    --epochs 100 \
    --learning_rate 0.03 \
    --method simclr \
    --temperature 0.07
```

**Key SSL Training Options:**
- `--method`: Contrastive method - `simclr`, `moco`, or `byol` (default: `simclr`)
- `--temperature`: Temperature for contrastive loss (default: 0.07)
- `--projection_dim`: Projection head output dimension (default: 128)
- `--projection_hidden_dim`: Projection head hidden dimension (default: 2048)
- `--warmup_epochs`: Learning rate warmup epochs (default: 10)
- `--batch_size`: Batch size (default: 256, large for contrastive learning)

The SSL training will:
- Use the 100k unlabeled STL-10 images
- Apply strong data augmentation (RandomResizedCrop, ColorJitter, GaussianBlur, etc.)
- Learn visual representations via contrastive learning
- Save checkpoints in `./checkpoints_ssl/` directory

### Stage 2: Supervised Finetuning with SSL Weights

After SSL pretraining, finetune on labeled data:

```bash
python train.py \
    --ssl_pretrained_path ./checkpoints_ssl/best_model.pth \
    --use_ssl_pretrained \
    --batch_size 64 \
    --epochs 50 \
    --learning_rate 1e-4
```

This will:
- Load SSL-pretrained backbone weights
- Initialize classification head randomly
- Finetune on 5k labeled training images
- Typically achieve better performance than ImageNet pretraining

### Linear Evaluation

Evaluate the quality of SSL-learned representations using linear evaluation:

```bash
python eval_ssl.py \
    --ssl_checkpoint_path ./checkpoints_ssl/best_model.pth \
    --epochs 100 \
    --learning_rate 0.1
```

Linear evaluation:
- Freezes the SSL-pretrained backbone
- Trains only a linear classifier on frozen features
- Measures representation quality without finetuning
- Useful for comparing different SSL methods/configurations

### SSL Features

1. **Contrastive Learning:**
   - SimCLR-style contrastive learning
   - Strong data augmentation pipeline
   - Normalized embeddings
   - Temperature-scaled contrastive loss

2. **Training:**
   - Learning rate warmup (important for contrastive learning)
   - Cosine annealing learning rate schedule
   - Mixed precision training
   - Gradient accumulation support

3. **Architecture:**
   - ViT backbone with projection head
   - 3-layer MLP projection head
   - Batch normalization in projection head
   - Normalized output embeddings

### Complete Workflow Example

```bash
# Step 1: SSL Pretraining (on unlabeled data)
python train_ssl.py \
    --backbone_name vit_base_patch16_224 \
    --batch_size 256 \
    --epochs 100 \
    --learning_rate 0.03 \
    --method simclr

# Step 2: Linear Evaluation (optional, to measure SSL quality)
python eval_ssl.py \
    --ssl_checkpoint_path ./checkpoints_ssl/best_model.pth

# Step 3: Supervised Finetuning (on labeled data)
python train.py \
    --ssl_pretrained_path ./checkpoints_ssl/best_model.pth \
    --use_ssl_pretrained \
    --batch_size 64 \
    --epochs 50 \
    --learning_rate 1e-4
```

## License

This project is part of CMP722 - Advanced Computer Vision course.
