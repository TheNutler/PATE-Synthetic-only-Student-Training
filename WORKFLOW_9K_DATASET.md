# Workflow: Generate 9k Synthetic Dataset

This guide shows how to generate a 9,000-sample synthetic dataset by:
- Generating **1,000 candidates** per teacher (instead of 20,000)
- Selecting **36 samples** per teacher (instead of quota-based selection)
- Combining all samples: 250 teachers × 36 samples = **9,000 total samples**

## Quick Start

Run the automated workflow script:

```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE
.\scripts\workflow_9k_dataset.ps1
```

Or follow the manual steps below.

## Manual Workflow

### Step 1: Generate Candidates and Select Samples

Generate 1,000 candidates per teacher and select exactly 36 samples per teacher:

```powershell
# Generate 1000 candidates per teacher and select 36 samples per teacher
python scripts\batch_label_and_filter.py `
    --generate-candidates-per-teacher `
    --decoder-path pretrained_decoder\decoder.pth `
    --teachers-dir wp3_d3.2_saferlearn\trained_nets_gpu `
    --output-dir teachers `
    --num-teachers 250 `
    --pool-size 1000 `
    --candidates-dir candidates `
    --config config\synthetic_generation.json `
    --confidence 0.9 `
    --target-samples 36 `
    --quota False `
    --min-nn-distance 2.0 `
    --min-diversity-distance 3.0
```

**Key Parameters:**
- `--pool-size 1000`: Generate 1,000 candidates per teacher (instead of 20,000)
- `--target-samples 36`: Select exactly 36 samples per teacher (top 36 by confidence after filtering)
- `--quota False`: Disable class distribution matching (just select top 36)
- `--confidence 0.9`: Confidence threshold for filtering (lower to 0.8 for CPU/faster processing)
- `--min-nn-distance 2.0`: Minimum distance from private shard (memorization check)
- `--min-diversity-distance 3.0`: Minimum pairwise distance (diversity enforcement, set to 0.0 or use `--disable-diversity` for CPU)

**Note:** PowerShell doesn't support inline comments (`# comment`) in multi-line commands. Put comments on separate lines before the command.

**For CPU (no GPU):** Add `--disable-diversity` and use `--batch-size 64` for better performance.

**Output:**
- `candidates/candidates_teacher_{id}.pt` - 1,000 candidates per teacher
- `teachers/teacher_{id}/synthetic_samples.pt` - 36 selected samples per teacher
- `teachers/teacher_{id}/labels.csv` - Labels for 36 samples

### Step 2: Combine All Synthetic Datasets

Combine all teacher datasets into one large dataset:

```powershell
python scripts\combine_synthetic_datasets.py `
    --teachers-dir teachers `
    --output combined_dataset_9k `
    --percentage 1.0 `
    --seed 42
```

**Output:**
- `combined_dataset_9k/combined_synthetic_samples.pt` - All 9,000 samples
- `combined_dataset_9k/combined_synthetic_labels.csv` - All labels
- `combined_dataset_9k/combined_synthetic_metadata.json` - Dataset statistics

### Step 3: Verify Dataset

Check that you have exactly 9,000 samples:

```powershell
# Check label count (subtract 1 for header)
$labelCount = (Get-Content combined_dataset_9k\combined_synthetic_labels.csv | Measure-Object -Line).Lines - 1
Write-Host "Total samples: $labelCount"
```

Expected output: `Total samples: 9000`

### Step 4: Train Student Model (Optional)

Train a student model on the combined dataset:

```powershell
python scripts\train_student_on_synthetic.py `
    --samples combined_dataset_9k\combined_synthetic_samples.pt `
    --labels combined_dataset_9k\combined_synthetic_labels.csv `
    --output-dir student_models_9k `
    --epochs 20 `
    --batch-size 64 `
    --learning-rate 0.01

python scripts\train_student_on_synthetic.py `
    --samples combined_teacher_vae_dataset\combined_synthetic_samples.pt `
    --labels combined_teacher_vae_dataset\combined_synthetic_labels.csv `
    --output-dir student_models_9k `
    --epochs 20 `
    --batch-size 64 `
    --learning-rate 0.01
```

## Alternative: Single Teacher Processing

For testing or processing individual teachers:

```powershell
# Generate candidates for teacher 0
python scripts\generate_candidates.py `
    --decoder-path pretrained_decoder\decoder.pth `
    --pool-size 1000 `
    --out candidates\candidates_teacher_0.pt `
    --seed 123 `
    --config config\synthetic_generation.json

# Label and filter, selecting 36 samples
python scripts\label_and_filter.py `
    --candidates candidates\candidates_teacher_0.pt `
    --teacher-model wp3_d3.2_saferlearn\trained_nets_gpu\0\model.pth `
    --teacher-shard-path teachers\teacher_0\shard.pt `
    --shard-metadata teachers\teacher_0\metadata.json `
    --config config\synthetic_generation.json `
    --target-samples 36 `
    --quota False `
    --confidence 0.9 `
    --min-nn-distance 2.0 `
    --min-diversity-distance 3.0 `
    --out-dir teachers\teacher_0
```

## Parameters Explained

### `--target-samples 36`
- Selects exactly 36 samples per teacher
- Samples are sorted by confidence (descending) after all filtering
- If fewer than 36 samples pass all filters, uses all available samples
- If more than 36 samples pass, selects top 36 by confidence

### `--pool-size 1000`
- Generates 1,000 candidate images per teacher
- Smaller pool size = faster generation
- Still sufficient for selecting 36 high-quality samples

### `--quota False`
- Disables class distribution matching
- Simply selects top 36 samples by confidence
- Faster processing, no class balancing

### Filtering Pipeline
1. **Confidence filter**: Keep samples with confidence ≥ 0.9
2. **Memorization check**: Remove samples too close to private shard (distance < 2.0)
3. **Diversity filter**: Remove samples too similar to each other (distance < 3.0)
4. **Target selection**: Select top 36 samples by confidence

## Expected Results

- **Total samples**: 9,000 (250 teachers × 36 samples)
- **Per teacher**: 36 samples selected from 1,000 candidates
- **Selection rate**: ~3.6% (36/1000)
- **Processing time**: Faster than 20k pool size (5× fewer candidates)

## CPU Performance Optimization

If running on CPU (no GPU), use these optimized settings for maximum speed:

```powershell
# Fastest settings (lower quality, maximum speed)
python scripts\batch_label_and_filter.py `
    --generate-candidates-per-teacher `
    --decoder-path pretrained_decoder\decoder.pth `
    --teachers-dir wp3_d3.2_saferlearn\trained_nets_gpu `
    --output-dir teachers `
    --num-teachers 250 `
    --pool-size 1000 `
    --target-samples 36 `
    --quota False `
    --confidence 0.8 `
    --min-nn-distance 2.0 `
    --min-diversity-distance 1.0 `
    --batch-size 128 `
    --disable-diversity
```

**CPU Optimization Tips:**
- Use `--disable-diversity` to skip the slow diversity filter (fastest)
- Lower `--min-diversity-distance` to 0.0 or 2.0 (instead of 3.0)
- Use smaller `--batch-size` (64-128) for CPU
- Lower `--confidence` threshold (0.8 instead of 0.9) to get more samples faster
- Lower `--min-nn-distance` (1.5 instead of 2.0) for faster memorization check

**Note:** PowerShell doesn't support inline comments (`# comment`) in multi-line commands. Put comments on separate lines before the command.

## Troubleshooting

### Not Enough Samples Per Teacher

If some teachers have fewer than 36 samples:
- Lower `--confidence` threshold (e.g., 0.8 or 0.7)
- Reduce `--min-nn-distance` (e.g., 1.5)
- Reduce `--min-diversity-distance` (e.g., 2.0)
- Increase `--pool-size` (e.g., 2000)

### Too Many Samples Per Teacher

If you want exactly 36 samples:
- The `--target-samples 36` parameter ensures exactly 36 (or fewer if not enough pass filters)
- No action needed

### Low Quality Samples

- Increase `--confidence` threshold
- Increase `--pool-size` to have more candidates to choose from
- Check that decoder is properly trained

## Notes

- All images are normalized using MNIST normalization: mean=0.1307, std=0.3081
- Synthetic samples are saved in [0, 1] range (not normalized)
- Labels are saved as CSV for easy inspection
- The `--target-samples` parameter was added specifically for this workflow

