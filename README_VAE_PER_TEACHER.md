# Per-Teacher VAE Approach for Synthetic Data Generation

This guide describes the **per-teacher VAE approach** where each teacher trains its own VAE encoder+decoder on its private shard subset, then uses that VAE to generate synthetic samples.

## ⚠️ Important Notes

This approach was replaced by the **pretrained decoder approach** (see `README_SYNTHETIC.md`) due to several limitations:

- **Small shard size**: Only ~240 samples per teacher → insufficient for VAE training
- **Underfitting**: VAEs trained on tiny shards produce blurry, low-quality reconstructions
- **Mode collapse**: Limited diversity in generated samples
- **High memorization risk**: Small training sets increase risk of overfitting to private data
- **Computational cost**: Training 250 separate VAEs is computationally expensive

However, this approach is still available for experimental purposes and comparison studies.

## Overview

The per-teacher VAE approach consists of three main steps:

1. **Train VAE per Teacher**: Train a VAE encoder+decoder on each teacher's private shard
2. **Generate Candidates**: Use each teacher's VAE decoder to generate candidate images
3. **Label and Filter**: Label candidates using the teacher classifier and filter for quality

## Requirements

Same as the main synthetic pipeline:
- Python 3.11
- PyTorch (>=2.0.0)
- torchvision (>=0.15.0)
- numpy (>=1.20.0)
- Windows PowerShell (for running commands)

## Directory Structure

```
.
├── teacher_vaes/              # Output: Per-teacher VAE models
│   └── teacher_{id}/
│       ├── decoder.pth        # Decoder weights (for generation)
│       ├── vae_full.pth       # Full VAE (encoder + decoder)
│       └── training_report.json # Training statistics
├── candidates/                 # Output: Candidate pools (per-teacher)
│   └── candidates_teacher_{id}.pt
├── teachers/                   # Input/Output: Teacher models and synthetic datasets
│   └── teacher_{id}/
│       ├── model.pth          # Teacher classifier model
│       ├── shard.pt            # Original private shard images (optional)
│       ├── synthetic_samples.pt # Output: Selected synthetic samples
│       └── labels.csv          # Output: Labels for synthetic samples
└── README_VAE_PER_TEACHER.md  # This file
```

## Usage

**Important**: All commands should be run from the project root directory (`D:\Documents\TUWien\MA\Thesis\PATE`)

### Step 1: Train VAE for Each Teacher

#### Option A: Train Single Teacher (for testing)

```powershell
python scripts\train_teacher_vae.py `
    --teacher-id 0 `
    --shard-indices wp3_d3.2_saferlearn\shard_indices.json `
    --latent-dim 32 `
    --epochs 100 `
    --batch-size 32 `
    --out-dir teacher_vaes `
    --seed 42
```

**Options:**
- `--teacher-id`: Teacher ID (required)
- `--shard-indices`: Path to shard_indices.json (default: `wp3_d3.2_saferlearn/shard_indices.json`)
- `--shard-path`: Alternative: path to pre-saved shard.pt file
- `--mnist-data-dir`: Directory containing MNIST dataset (default: `wp3_d3.2_saferlearn/src/input-data/MNIST`)
- `--latent-dim`: Latent dimension (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32, auto-adjusted for small shards)
- `--lr`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay (default: 1e-5)
- `--kl-annealing-epochs`: Epochs for KL annealing (default: 20)
- `--out-dir`: Output directory (default: `teacher_vaes`)
- `--seed`: Random seed (default: 42)
- `--no-augmentation`: Disable data augmentation

**Outputs:**
- `teacher_vaes/teacher_{id}/decoder.pth` - Decoder weights
- `teacher_vaes/teacher_{id}/vae_full.pth` - Full VAE (encoder + decoder)
- `teacher_vaes/teacher_{id}/training_report.json` - Training statistics

#### Option B: Batch Train All Teachers

```powershell
python scripts\batch_train_teacher_vaes.py `
    --num-teachers 250 `
    --start-id 0 `
    --shard-indices wp3_d3.2_saferlearn\shard_indices.json `
    --latent-dim 32 `
    --epochs 100 `
    --batch-size 32 `
    --out-dir teacher_vaes `
    --seed 42
```

**Options:**
- `--num-teachers`: Number of teachers to process (default: 250)
- `--start-id`: Starting teacher ID (default: 0)
- `--shard-base`: Base directory for pre-saved shard.pt files (optional)
- All other options same as single teacher training

**Note**: This will train 250 VAEs sequentially, which may take a long time. Consider running in parallel or on a cluster if available.

### Step 2: Generate Candidate Pool for Each Teacher

Generate candidates using each teacher's trained VAE decoder:

```powershell
# For a single teacher
python scripts\generate_candidates_teacher_vae.py `
    --teacher-id 0 `
    --decoder-path teacher_vaes\teacher_0\decoder.pth `
    --pool-size 20000 `
    --out candidates\candidates_teacher_0.pt `
    --seed 123 `
    --config config\synthetic_generation.json
```

**Options:**
- `--teacher-id`: Teacher ID (required)
- `--decoder-path`: Path to teacher-specific decoder.pth (required)
- `--pool-size`: Number of candidates to generate (default: 20000)
- `--latent-dim`: Must match decoder (default: 32)
- `--latent-mixing`: Fraction for latent mixing diversity (default: 0.3)
- `--latent-noise`: Scale of latent noise (default: 0.1)
- `--out`: Output path for candidate pool (.pt file)
- `--seed`: Random seed (will be offset by teacher_id for uniqueness)
- `--save-grid`: Save visualization grid
- `--config`: Path to config file

**Batch Generation Script** (to be created or use a loop):

```powershell
# Generate candidates for all teachers
for ($i = 0; $i -lt 250; $i++) {
    python scripts\generate_candidates_teacher_vae.py `
        --teacher-id $i `
        --decoder-path teacher_vaes\teacher_$i\decoder.pth `
        --pool-size 20000 `
        --out candidates\candidates_teacher_$i.pt `
        --seed 123 `
        --config config\synthetic_generation.json
}
```

### Step 3: Label and Filter (Per Teacher)

Use the existing `label_and_filter.py` script with teacher-specific candidate pools:

```powershell
python scripts\label_and_filter.py `
    --candidates candidates\candidates_teacher_0.pt `
    --teacher-model wp3_d3.2_saferlearn\trained_nets_gpu\0\model.pth `
    --teacher-shard-path teachers\teacher_0\shard.pt `
    --shard-metadata teachers\teacher_0\metadata.json `
    --config config\synthetic_generation.json `
    --out-dir teachers\teacher_0
```

Or use `batch_label_and_filter.py` with per-teacher candidate pools:

```powershell
# Note: This requires modifying batch_label_and_filter.py to support
# per-teacher candidate pools, or use a loop similar to generation
```

## Example Workflow

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

# Activate virtual environment (if using one)
.\.venv\Scripts\Activate.ps1

# Step 1: Train VAEs for all teachers (this will take a while!)
python scripts\batch_train_teacher_vaes.py `
    --num-teachers 250 `
    --shard-indices wp3_d3.2_saferlearn\shard_indices.json `
    --epochs 100 `
    --batch-size 32 `
    --out-dir teacher_vaes `
    --seed 42

# Step 2: Generate candidates for each teacher
for ($i = 0; $i -lt 250; $i++) {
    python scripts\generate_candidates_teacher_vae.py `
        --teacher-id $i `
        --decoder-path teacher_vaes\teacher_$i\decoder.pth `
        --pool-size 20000 `
        --out candidates\candidates_teacher_$i.pt `
        --seed 123 `
        --config config\synthetic_generation.json
}

# Step 3: Label and filter for each teacher
for ($i = 0; $i -lt 250; $i++) {
    python scripts\label_and_filter.py `
        --candidates candidates\candidates_teacher_$i.pt `
        --teacher-model wp3_d3.2_saferlearn\trained_nets_gpu\$i\model.pth `
        --teacher-shard-path teachers\teacher_$i\shard.pt `
        --shard-metadata teachers\teacher_$i\metadata.json `
        --config config\synthetic_generation.json `
        --out-dir teachers\teacher_$i
}
```

## Training Parameters

### Recommended Settings for Small Shards (~240 samples)

- **Epochs**: 100-200 (more epochs needed due to small dataset)
- **Batch size**: 16-32 (auto-adjusted if shard is smaller)
- **Learning rate**: 1e-3 (standard)
- **KL annealing**: 20-30 epochs (gradual introduction of KL term)
- **Augmentation**: Enabled by default (helps with small datasets)

### Expected Issues

1. **High reconstruction loss**: VAEs trained on tiny shards will have high reconstruction error
2. **Blurry samples**: Limited training data leads to poor image quality
3. **Low diversity**: Mode collapse is common with small datasets
4. **Long training time**: 250 VAEs × 100 epochs = significant compute time

## Comparison with Pretrained Decoder Approach

| Aspect | Per-Teacher VAE | Pretrained Decoder |
|--------|----------------|-------------------|
| Training data | ~240 samples per teacher | 112k EMNIST samples |
| Number of models | 250 VAEs | 1 decoder |
| Training time | Very long (250×) | Short (1×) |
| Sample quality | Low (blurry, low diversity) | High (sharp, diverse) |
| Memorization risk | High (small dataset) | Low (public data only) |
| Privacy | Each VAE sees private shard | Decoder sees only public data |

## Troubleshooting

### Low Sample Quality

- Increase training epochs (try 200-300)
- Enable data augmentation (default)
- Reduce batch size further (try 16 or 8)
- Increase latent dimension (try 64)

### Training Instability

- Reduce learning rate (try 5e-4)
- Increase KL annealing epochs (try 30-40)
- Check that shard images are in [0, 1] range

### Out of Memory

- Reduce batch size
- Use CPU instead of GPU (slower but less memory)

### Missing Shard Data

- Ensure `shard_indices.json` exists and contains teacher IDs
- Or provide `--shard-path` with pre-saved shard.pt files

## Notes

- All images are normalized using MNIST normalization: mean=0.1307, std=0.3081
- VAE training uses images in [0, 1] range (not normalized)
- Decoder outputs are in [0, 1] range
- Each teacher's VAE is trained independently with a unique random seed (offset by teacher_id)

## License

See main project license.

