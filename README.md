# ICCR-Net: Improved Contrastive Learning for Semi-Supervised Semantic Segmentation

ICCR-Net is a semi-supervised semantic segmentation model that leverages contrastive learning with a memory bank to achieve high performance with limited labeled data. The model combines teacher-student frameworks with advanced loss functions and data augmentation strategies.

## Features

- **Semi-Supervised Learning**: Efficiently learns from limited labeled data
- **Contrastive Learning**: Uses a memory bank with weighted sampling strategies
- **Dual Architecture Support**: 
  - Mean Teacher (MT) network
  - U-Net segmentation network
  - Optional VGG16-BN backbone for improved feature extraction
- **Advanced Loss Functions**:
  - Focal Loss
  - SSIM Loss
  - Jaccard (IoU) Loss
  - Dice Loss
  - MSE Loss with imbalance weighting
  - Combined BCE-SSIM-IoU Loss
  - Contrastive Loss with memory bank
- **Data Augmentation**: Geometry and color transformations (D4, rotation, color jitter, blur, grayscale)
- **Exponential Moving Average (EMA)**: Teacher model updates via EMA for stable training
- **Imbalanced Data Support**: Weighted loss functions for handling class imbalance

## Project Structure

```
Proposed_Contrastive3/
├── main.py                  # Main entry point and training configuration
├── model.py                 # Model architectures (MT, UNet, backbone definitions)
├── loss.py                  # Loss functions and metrics
├── data.py                  # Data loading and augmentation pipelines
├── train_proposed.py        # Main training logic with contrastive learning
├── train_mt.py             # Mean Teacher training implementation
├── train_unet.py           # Standard UNet training
├── train_contr.py          # Contrastive learning training
├── ramp.py                 # Ramp-up scheduling utilities
├── results.json            # Training results and metrics
├── pytorch_iou/            # IoU loss implementation
├── pytorch_ssim/           # SSIM loss implementation
└── weight/                 # Saved model checkpoints
```

## Installation

1. **Create and activate the Python virtual environment** (if not already done):
```bash
source khoa-env/bin/activate
```

2. **Install dependencies**:
```bash
pip install torch torchvision
pip install albumentations
pip install scipy numpy opencv-python
```

## Configuration

Edit `main.py` to configure the training parameters:

```python
# Dataset and model settings
IMAGE_SIZE = 256              # Input image resolution
DATASET = 'OTU'              # Dataset name
LABELED_RATIO = 0.1          # Ratio of labeled data (0.1 = 10%)
NUM_CLASS = 7                # Number of segmentation classes
NUM_WORKERS = 4              # DataLoader workers
DEVICE = 'cuda:1'            # GPU device

# Training parameters
PRE_EPOCHS = 10              # Pre-training epochs
EPOCHS = 100                 # Main training epochs
BATCH_SIZE = 8               # Batch size

# Contrastive learning
CAPACITY = 400               # Memory bank capacity
BANK_WEIGHT_TYPE = 'Exp'     # Weight type: 'Exp' or other
WITH_BANK_WEIGHT = True      # Use weighted memory bank
k = 0.01                     # Weighting parameter

# Loss function components
WITH_IMB_LOSS = True         # Use imbalance-aware loss
WITH_DIS_HARD_LOSS = False   # Use hard negative mining

# Model architecture
MODEL = 'ALL'                # Training mode
WITH_VGG16BN_BACKBONE = True # Use pretrained VGG16-BN backbone
```

## Usage

### Basic Training

Run the main training script:
```bash
python main.py
```

The script will automatically:
1. Load and preprocess the OTU dataset
2. Split data into labeled and unlabeled sets
3. Initialize models with the specified architecture
4. Train using the proposed contrastive learning approach
5. Save the best model checkpoint to `weight/{DATASET}/proposed-*`

### Training Different Models

Modify the `MODEL` variable in `main.py`:

- `MODEL = 'Proposed'` - Main ICCR-Net with contrastive learning
- `MODEL = 'MT'` - Mean Teacher baseline
- `MODEL = 'UNet'` - Standard UNet segmentation
- `MODEL = 'ALL'` - Train all models sequentially

### Custom Dataset

To train on your own dataset, modify the data loading in `data.py`:

1. Update the dataset path and loading logic
2. Adjust `NUM_CLASS` to match your dataset's number of classes
3. Modify augmentation parameters if needed
4. Set appropriate `LABELED_RATIO` for your semi-supervised setup

## Model Architecture

### Mean Teacher (MT) Network
A U-Net style architecture with:
- 5 encoding blocks with max pooling
- 5 decoding blocks with transpose convolution
- Skip connections between encoder and decoder
- Optional output for t-SNE embeddings

### Key Components

- **ConvBlock**: Double convolution with batch normalization and ReLU
- **Encoder**: Progressive downsampling to extract features
- **Decoder**: Progressive upsampling with skip connections for reconstruction
- **Output**: Sigmoid activation for binary or multi-class segmentation

## Loss Functions

### Main Loss Functions

| Loss | Description |
|------|-------------|
| **Focal Loss** | Addresses class imbalance, emphasizes hard examples (α=0.26, γ=2.3) |
| **SSIM Loss** | Structural similarity for perceptual quality (1 - SSIM) |
| **Jaccard Loss** | IoU-based loss for set similarity |
| **Dice Loss** | F1-score based loss for class overlap |
| **Contrastive Loss** | Memory bank-based contrastive learning |
| **MSE Loss** | Mean squared error with imbalance weighting |

### Combined Losses

- `joint_loss1`: Average of Focal + SSIM + Jaccard losses
- `bce_ssim_iou_loss`: Combined BCE + SSIM + IoU losses

## Training Strategies

### Data Augmentation

**Geometric Transforms:**
- D4 transformations (rotations/flips)
- Random resized crops (50-100% of original size)
- Rotation (±15 degrees)

**Color Transforms (Student):**
- Gaussian blur
- Color jitter
- Grayscale conversion

**Color Transforms (Teacher):**
- Minimal augmentation for teacher consistency

### EMA (Exponential Moving Average)

Teacher model is updated via EMA with a schedule:
- Start coefficient: 0.99
- End coefficient: 0.999
- Follows sigmoid rampup schedule during training

## Results

The trained models are saved to:
```
weight/{DATASET}/proposed-b{BATCH_SIZE}-c{CAPACITY}-k{k}[-BankWeight{TYPE}][-IMBLoss].pth
```

Example results are saved in `results.json` with metrics including:
- Dice coefficient
- Jaccard similarity
- Precision and recall
- Per-class performance

## Performance Metrics

The model evaluates performance using:

- **Dice Coefficient**: `2 * (TP) / (2 * TP + FP + FN)`
- **Jaccard Similarity**: `(TP) / (TP + FP + FN)`
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`

## Reproducibility

Deterministic training is ensured through:
- Fixed random seeds for NumPy, PyTorch, CUDA
- Deterministic algorithm settings
- Fixed worker initialization for DataLoader

Seed value: **42**

## Requirements

- Python 3.8+
- PyTorch 1.9+
- TorchVision
- Albumentations
- NumPy
- OpenCV
- SciPy


---

**For questions or issues**, please refer to the project's main documentation or contact the maintainers.
