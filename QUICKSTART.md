# Quick Start Guide

## Prerequisites

1. Ensure you have Python 3.11 installed
2. Install dependencies:
   ```powershell
   # Option 1: Install from existing project requirements (includes torch/torchvision)
   pip install -r wp3_d3.2_saferlearn/requirements.txt
   
   # Option 2: Install minimal requirements for synthetic pipeline only
   pip install -r requirements_synthetic.txt
   
   # Option 3: Install manually
   pip install torch torchvision numpy pillow
   ```
3. Teacher models should be saved as `teachers/teacher_{id}/model.pth`
4. (Optional) Teacher shard images and metadata for better filtering

## Step-by-Step Execution

### 0. Download EMNIST Dataset (One-time, ~5-10 minutes)

**Important**: Run from the project root directory (`D:\Documents\TUWien\MA\Thesis\PATE`)

**Note**: Make sure you have activated your virtual environment and installed dependencies first.

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

# Activate virtual environment (if using one)
.\.venv\Scripts\Activate.ps1

# Download EMNIST
python download-emnist.py
```

This downloads EMNIST Balanced to `wp3_d3.2_saferlearn/src/input-data/EMNIST`.

### 1. Pretrain Decoder (One-time, ~30-60 minutes)

**Important**: Run from the project root directory (`D:\Documents\TUWien\MA\Thesis\PATE`)

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

# Activate virtual environment (if using one)
.\.venv\Scripts\Activate.ps1

# Run pretraining
python scripts\pretrain_decoder.py `
    --data-root wp3_d3.2_saferlearn\src\input-data\EMNIST `
    --latent-dim 32 `
    --epochs 50 `
    --out-dir pretrained_decoder `
    --seed 42
```

**Expected output:**
- `pretrained_decoder/decoder.pth`
- `pretrained_decoder/pretrain_report.json`

### 2. Generate Candidate Pool (One-time, ~1-2 minutes)

```powershell
python scripts\generate_candidates.py `
    --decoder-path pretrained_decoder\decoder.pth `
    --pool-size 10000 `
    --out candidates\candidates_run1.pt `
    --seed 123 `
    --save-grid
```

**Expected output:**
- `candidates/candidates_run1.pt` (tensor with 10000 images)
- `candidates/candidates_run1.png` (visualization grid)

### 3. Process All Teachers (Batch Processing)

**Important**: Run from the project root directory

**Process all 250 teachers:**

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

python scripts\batch_label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --teachers-dir wp3_d3.2_saferlearn\trained_nets_gpu `
    --output-dir teachers `
    --num-teachers 250 `
    --confidence 0.9 `
    --quota True `
    --min-nn-distance 1.0
```

**Process first 10 teachers for testing:**

```powershell
python scripts\batch_label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --num-teachers 10 `
    --confidence 0.9
```

**Expected output per teacher:**
- `teachers/teacher_{id}/synthetic_samples.pt`
- `teachers/teacher_{id}/labels.csv`
- `teachers/teacher_{id}/selection_report.json`

**Process a single teacher (if needed):**

```powershell
python scripts\label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --teacher-model wp3_d3.2_saferlearn\trained_nets_gpu\0\model.pth `
    --teacher-shard-path teachers\teacher_0\shard.pt `
    --shard-metadata teachers\teacher_0\metadata.json `
    --confidence 0.9 `
    --quota True `
    --min-nn-distance 1.0 `
    --out-dir teachers\teacher_0
```

### 4. Evaluate (Optional, for quality assurance)

**Important**: Run from the project root directory

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

python scripts\evaluate_synthetic.py `
    --samples teachers\teacher_0\synthetic_samples.pt `
    --shard teachers\teacher_0\shard.pt `
    --out teachers\teacher_0\evaluation_report.json
```

## Tips

- **Large candidate pools**: Use `--pool-size 10000` or higher for better diversity
- **Confidence threshold**: Lower to `0.85` if too few samples are selected
- **Memorization check**: Increase `--min-nn-distance` to `1.5` if you see warnings
- **Class imbalance**: Check `selection_report.json` to see if any classes are underrepresented

## Troubleshooting

**Problem**: EMNIST download fails
**Solution**: Use `--use-kmnist` flag

**Problem**: Too few samples selected
**Solution**: Lower `--confidence` threshold or increase `--pool-size`

**Problem**: Memorization warnings
**Solution**: Increase `--min-nn-distance` or generate larger candidate pools

**Problem**: Import errors
**Solution**: Ensure you're running from the project root directory

## Expected Results

After processing, each teacher should have:
- 500-2000 synthetic samples (depending on filtering settings)
- Average confidence ≥ 0.9
- Diversity ratio ≥ 0.75 (if original shard provided)
- Min NN distance ≥ 1.0 (low memorization risk)

