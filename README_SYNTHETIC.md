# Synthetic Data Generation Pipeline for PATE Teachers

This pipeline generates labeled synthetic datasets for pre-trained teacher classifiers **without retraining the teacher models**. It uses a VAE decoder pretrained on public EMNIST data to generate candidate images, which are then labeled and filtered using each teacher classifier.

## Overview

The pipeline consists of four main steps:

1. **Pretrain Decoder**: Train a VAE decoder on public EMNIST Balanced dataset
2. **Generate Candidates**: Create a large pool of synthetic candidate images
3. **Label and Filter**: For each teacher, label candidates and filter to select high-quality samples
4. **Evaluate**: Run safety checks and quality metrics on synthetic datasets

## Requirements

- Python 3.11
- PyTorch
- torchvision
- Windows PowerShell (for running commands)

## Directory Structure

```
.
├── models/
│   ├── __init__.py
│   └── decoder.py              # VAE decoder architecture
├── utils/
│   ├── __init__.py
│   ├── io.py                   # I/O utilities
│   ├── preprocess.py           # Image normalization
│   └── metrics.py              # Evaluation metrics
├── scripts/
│   ├── pretrain_decoder.py     # Step 1: Pretrain decoder
│   ├── generate_candidates.py  # Step 2: Generate candidate pool
│   ├── label_and_filter.py     # Step 3: Label and filter per teacher
│   └── evaluate_synthetic.py  # Step 4: Evaluate synthetic datasets
├── pretrained_decoder/         # Output: Pretrained decoder weights
├── candidates/                  # Output: Candidate pools
├── teachers/                    # Input/Output: Teacher models and synthetic datasets
│   └── teacher_{id}/
│       ├── model.pth           # Teacher classifier model
│       ├── shard.pt            # Original private shard images (optional)
│       ├── metadata.json       # Shard metadata (class distribution)
│       ├── synthetic_samples.pt # Output: Selected synthetic samples
│       ├── labels.csv          # Output: Labels for synthetic samples
│       ├── selection_report.json # Output: Selection statistics
│       └── evaluation_report.json # Output: Evaluation metrics
└── README_SYNTHETIC.md         # This file
```

## Usage

### Step 0: Download EMNIST Dataset (if not already downloaded)

First, download the EMNIST dataset to the project directory:

```powershell
python download-emnist.py
```

This will download EMNIST Balanced to `wp3_d3.2_saferlearn/src/input-data/EMNIST` by default.

### Step 1: Pretrain Decoder

Train a VAE decoder on public EMNIST data:

```powershell
python scripts\pretrain_decoder.py `
    --data-root wp3_d3.2_saferlearn\src\input-data\EMNIST `
    --latent-dim 32 `
    --epochs 50 `
    --out-dir pretrained_decoder `
    --seed 42
```

**Outputs:**
- `pretrained_decoder/decoder.pth` - Decoder weights
- `pretrained_decoder/pretrain_report.json` - Training report

**Options:**
- `--data-root`: Directory for EMNIST dataset (will download if needed)
- `--latent-dim`: Latent dimension (default: 32)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-3)
- `--use-kmnist`: Use KMNIST if EMNIST unavailable

### Step 2: Generate Candidate Pool

Generate a large pool of synthetic candidate images:

```powershell
python scripts\generate_candidates.py `
    --decoder-path pretrained_decoder\decoder.pth `
    --pool-size 5000 `
    --out candidates\candidates_run1.pt `
    --seed 123 `
    --save-grid
```

**Outputs:**
- `candidates/candidates_run1.pt` - Candidate pool tensor
- `candidates/candidates_run1.png` - Visualization grid (if `--save-grid`)

**Options:**
- `--decoder-path`: Path to pretrained decoder
- `--pool-size`: Number of candidates to generate (default: 5000)
- `--latent-dim`: Must match decoder (default: 32)
- `--latent-mixing`: Fraction for latent mixing diversity (default: 0.0)

### Step 3: Label and Filter (Per Teacher)

For each teacher, label candidates and filter to select high-quality samples:

```powershell
python scripts\label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --teacher-model teachers\teacher_0\model.pth `
    --teacher-shard-path teachers\teacher_0\shard.pt `
    --shard-metadata teachers\teacher_0\metadata.json `
    --confidence 0.9 `
    --quota True `
    --min-nn-distance 1.0 `
    --out-dir teachers\teacher_0
```

**Outputs:**
- `teachers/teacher_0/synthetic_samples.pt` - Selected synthetic samples
- `teachers/teacher_0/labels.csv` - Labels for samples
- `teachers/teacher_0/selection_report.json` - Selection statistics

**Options:**
- `--candidates`: Path to candidate pool
- `--teacher-model`: Path to teacher model.pth
- `--teacher-shard-path`: Path to original shard images (for memorization check)
- `--shard-metadata`: Path to metadata JSON with class distribution
- `--confidence`: Confidence threshold (default: 0.9)
- `--quota`: Match class distribution from metadata (default: True)
- `--max-per-class`: Override max samples per class
- `--min-diversity-distance`: Minimum pairwise distance for diversity (default: 0.0, disabled)
- `--min-nn-distance`: Minimum NN distance for memorization check (default: 1.0)

### Step 4: Evaluate Synthetic Dataset

Run diagnostics on the synthetic dataset:

```powershell
python scripts\evaluate_synthetic.py `
    --samples teachers\teacher_0\synthetic_samples.pt `
    --labels teachers\teacher_0\labels.csv `
    --shard teachers\teacher_0\shard.pt `
    --out teachers\teacher_0\evaluation_report.json
```

**Outputs:**
- `teachers/teacher_0/evaluation_report.json` - Evaluation metrics

**Options:**
- `--samples`: Path to synthetic samples
- `--labels`: Path to labels CSV (auto-detected if in same directory)
- `--shard`: Path to original shard for comparison
- `--shard-labels`: Path to original shard labels (if separate)

## Example Workflow (All Teachers)

```powershell
# Step 1: Pretrain decoder (once)
python scripts\pretrain_decoder.py --data-root wp3_d3.2_saferlearn\src\input-data\EMNIST --epochs 50 --out-dir pretrained_decoder --seed 42

# Step 2: Generate candidate pool (once, shared across teachers)
python scripts\generate_candidates.py --decoder-path pretrained_decoder\decoder.pth --pool-size 10000 --out candidates\candidates_run1.pt --seed 123

# Step 3: Process each teacher (can be parallelized)
for ($i = 0; $i -lt 250; $i++) {
    python scripts\label_and_filter.py `
        --candidates candidates\candidates_run1.pt `
        --teacher-model teachers\teacher_$i\model.pth `
        --teacher-shard-path teachers\teacher_$i\shard.pt `
        --shard-metadata teachers\teacher_$i\metadata.json `
        --confidence 0.9 `
        --quota True `
        --out-dir teachers\teacher_$i
}

# Step 4: Evaluate each teacher (optional)
for ($i = 0; $i -lt 250; $i++) {
    python scripts\evaluate_synthetic.py `
        --samples teachers\teacher_$i\synthetic_samples.pt `
        --shard teachers\teacher_$i\shard.pt `
        --out teachers\teacher_$i\evaluation_report.json
}
```

## Configuration Files

### Teacher Shard Metadata (`metadata.json`)

Example structure:

```json
{
  "teacher_id": 0,
  "shard_size": 240,
  "class_distribution": {
    "0": 24,
    "1": 24,
    "2": 24,
    "3": 24,
    "4": 24,
    "5": 24,
    "6": 24,
    "7": 24,
    "8": 24,
    "9": 24
  },
  "shard_path": "teachers/teacher_0/shard.pt"
}
```

### Hyperparameter Config (`config.json`)

Example configuration:

```json
{
  "decoder": {
    "latent_dim": 32,
    "pretrain_epochs": 50,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "kl_annealing_epochs": 10
  },
  "generation": {
    "pool_size": 5000,
    "latent_mixing_ratio": 0.0
  },
  "filtering": {
    "confidence_threshold": 0.9,
    "min_nn_distance": 1.0,
    "min_diversity_distance": 0.0,
    "max_per_class": null
  }
}
```

## Key Features

### Privacy Preservation
- Decoder pretrained **only** on public EMNIST data
- Teacher models are **never retrained** or modified
- Memorization checks prevent copying private images
- Nearest-neighbor distance filtering ensures synthetic samples are sufficiently different from private shards

### Quality Assurance
- Confidence-based filtering (default: 0.9)
- Class distribution matching
- Diversity enforcement (pairwise distance)
- Comprehensive evaluation metrics

### Reproducibility
- All scripts support `--seed` parameter
- File hashes saved in reports for audit
- Versioned candidate pools and decoder weights

## Success Criteria

A successful synthetic dataset should have:
- **High average teacher confidence** (≥ 0.9)
- **Diversity ratio ≥ 0.75** (compared to original shard)
- **Min NN distance ≥ 1.0** (low memorization risk)
- **Class distribution matching** original shard (histogram L1 distance < 0.2)

## Troubleshooting

### EMNIST Not Available
If EMNIST download fails, use KMNIST:
```powershell
python scripts\pretrain_decoder.py --data-root wp3_d3.2_saferlearn\src\input-data\KMNIST --use-kmnist ...
```

### Low Candidate Quality
- Increase `--pool-size` in candidate generation
- Lower `--confidence` threshold in filtering
- Try different `--latent-dim` (16 or 64)

### Memorization Warnings
- Increase `--min-nn-distance` threshold
- Generate larger candidate pools
- Check that decoder was trained only on public data

### Class Imbalance
- Ensure `metadata.json` has correct class distribution
- Use `--max-per-class` to override quotas
- Generate multiple candidate pools and merge

## Notes

- All images are normalized using MNIST normalization: mean=0.1307, std=0.3081
- Teacher models use `UCStubModel` architecture (see `train_mnist_models.py`)
- Synthetic samples are saved in [0, 1] range (not normalized)
- Labels are saved as CSV for easy inspection

## License

See main project license.

