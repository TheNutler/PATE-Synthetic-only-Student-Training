# Workflow to generate 9k synthetic dataset (250 teachers × 36 samples)
# This script generates 1000 candidates per teacher and selects 36 samples per teacher

# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

# Activate virtual environment (if using one)
# .\.venv\Scripts\Activate.ps1

Write-Host "=" -NoNewline; Write-Host ("=" * 60) -NoNewline; Write-Host "="
Write-Host "Generating 9k Synthetic Dataset Workflow"
Write-Host "  - 250 teachers"
Write-Host "  - 1000 candidates per teacher"
Write-Host "  - 36 samples per teacher (total: 9,000 samples)"
Write-Host "=" -NoNewline; Write-Host ("=" * 60) -NoNewline; Write-Host "="
Write-Host ""

# Step 1: Generate 1000 candidates per teacher using pretrained decoder
Write-Host "Step 1: Generating 1000 candidates per teacher..."
Write-Host ""

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

Write-Host ""
Write-Host "Step 1 complete!"
Write-Host ""

# Step 2: Combine all synthetic datasets
Write-Host "Step 2: Combining all synthetic datasets..."
Write-Host ""

python scripts\combine_synthetic_datasets.py `
    --teachers-dir teachers `
    --output combined_dataset_9k `
    --percentage 1.0 `
    --seed 42

Write-Host ""
Write-Host "Step 2 complete!"
Write-Host ""

# Step 3: Verify the combined dataset
Write-Host "Step 3: Verifying combined dataset..."
Write-Host ""

$combinedSamples = "combined_dataset_9k\combined_synthetic_samples.pt"
$combinedLabels = "combined_dataset_9k\combined_synthetic_labels.csv"

if (Test-Path $combinedSamples) {
    Write-Host "✓ Combined samples found: $combinedSamples"
} else {
    Write-Host "✗ Combined samples not found: $combinedSamples"
}

if (Test-Path $combinedLabels) {
    $labelCount = (Get-Content $combinedLabels | Measure-Object -Line).Lines - 1
    Write-Host "✓ Combined labels found: $combinedLabels"
    Write-Host "  Total samples: $labelCount"
} else {
    Write-Host "✗ Combined labels not found: $combinedLabels"
}

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 60) -NoNewline; Write-Host "="
Write-Host "Workflow complete!"
Write-Host "Combined dataset saved to: combined_dataset_9k/"
Write-Host "=" -NoNewline; Write-Host ("=" * 60) -NoNewline; Write-Host "="

