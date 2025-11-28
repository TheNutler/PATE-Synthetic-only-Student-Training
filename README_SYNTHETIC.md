# Synthetic Data Generation Pipeline for PATE Teachers

This pipeline generates labeled synthetic datasets for pre-trained teacher classifiers **without retraining the teacher models**. It uses a VAE decoder pretrained on public EMNIST data to generate candidate images, which are then labeled and filtered using each teacher classifier.

## Overview

The pipeline consists of four main steps:

1. **Pretrain Decoder**: Train a VAE decoder on public EMNIST Balanced dataset (runs once)
2. **Generate Candidates**: Create candidate pools - either one shared pool or separate pools per teacher (20k images each)
3. **Label and Filter**: For each teacher, label candidates from their pool and filter to select teacher-specific high-quality samples (runs per teacher)
4. **Evaluate**: Run safety checks and quality metrics on synthetic datasets (runs per teacher)

### Key Features

- **Class-Balanced Selection**: Automatic matching to teacher's original shard distribution with min/max quotas
- **Rare Class Handling**: Lower confidence thresholds for underrepresented classes (e.g., classes 1 and 8)
- **Latent Steering**: Optimize latent vectors to generate samples for hard classes
- **Enhanced Filtering**: Improved diversity and memorization checks
- **Configurable Pipeline**: All hyperparameters configurable via JSON config file
- **Comprehensive Diagnostics**: Detailed JSON reports with class distributions, rejection reasons, and quality metrics

## Requirements

- Python 3.11
- PyTorch (>=2.0.0)
- torchvision (>=0.15.0)
- numpy (>=1.20.0)
- Windows PowerShell (for running commands)

### Installation

Install dependencies using one of the following methods:

```powershell
# Option 1: Use existing project requirements (recommended if you have the full project setup)
pip install -r wp3_d3.2_saferlearn/requirements.txt

# Option 2: Install minimal requirements for synthetic pipeline only
pip install -r requirements_synthetic.txt

# Option 3: Install manually
pip install torch torchvision numpy pillow
```

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
│   ├── metrics.py              # Evaluation metrics
│   ├── config_loader.py        # Config file loader
│   └── latent_steering.py      # Latent space optimization
├── config/
│   └── synthetic_generation.json # Pipeline configuration
├── scripts/
│   ├── pretrain_decoder.py     # Step 1: Pretrain decoder
│   ├── generate_candidates.py  # Step 2: Generate candidate pool
│   ├── label_and_filter.py     # Step 3: Label and filter per teacher
│   ├── batch_label_and_filter.py # Step 3: Batch process all teachers
│   ├── combine_synthetic_datasets.py # Step 4: Combine all teacher datasets
│   ├── train_student_on_synthetic.py # Step 5: Train student on synthetic data
│   ├── evaluate_student_model.py # Step 6: Evaluate student on MNIST test set
│   └── evaluate_synthetic.py  # Step 7: Evaluate synthetic datasets
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

**Important**: All commands should be run from the project root directory (`D:\Documents\TUWien\MA\Thesis\PATE`)

### Step 0: Download EMNIST Dataset (if not already downloaded)

First, download the EMNIST dataset to the project directory:

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

# Activate virtual environment (if using one)
.\.venv\Scripts\Activate.ps1

# Download EMNIST
python download-emnist.py
```

This will download EMNIST Balanced to `wp3_d3.2_saferlearn/src/input-data/EMNIST` by default.

### Step 1: Pretrain Decoder

Train a VAE decoder on public EMNIST data:

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

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

### Step 2: Generate Candidate Pool(s)

You have two options for candidate pool generation:

#### Option A: Generate Per-Teacher Pools (Recommended)

Generate a **separate candidate pool for each teacher** (20k images per teacher):

```powershell
# This is done automatically in Step 3 when using --generate-candidates-per-teacher
# No separate Step 2 needed!
```

**Benefits:**
- Each teacher gets their own unique pool of 20k images
- Better diversity across teachers
- Total: 250 teachers × 20k = 5M candidate images

#### Option B: Generate Single Shared Pool

Generate a **single shared pool** that all teachers will use:

```powershell
python scripts\generate_candidates.py `
    --decoder-path pretrained_decoder\decoder.pth `
    --pool-size 20000 `
    --out candidates\candidates_run1.pt `
    --seed 123 `
    --save-grid `
    --config config\synthetic_generation.json
```

**Outputs:**
- `candidates/candidates_run1.pt` - Shared candidate pool tensor (used by all teachers)
- `candidates/candidates_run1.png` - Visualization grid (if `--save-grid`)

**Options:**
- `--decoder-path`: Path to pretrained decoder
- `--pool-size`: Number of candidates to generate (default: 20000 from config)
- `--latent-dim`: Must match decoder (default: 32 from config)
- `--latent-mixing`: Fraction for latent mixing diversity (default: 0.3 from config)
- `--latent-noise`: Scale of latent noise (default: 0.1 from config)
- `--config`: Path to config file (default: `config/synthetic_generation.json`)

**Note:** The default pool size is 20,000. Latent mixing and noise are enabled by default to increase diversity.

### Step 3: Label and Filter (Per Teacher)

#### Option A: Batch Process All Teachers (Recommended)

Process all teachers automatically. You can choose to generate candidate pools per teacher or use a shared pool:

**With Per-Teacher Candidate Pools (Recommended):**

```powershell
python scripts\batch_label_and_filter.py `
    --generate-candidates-per-teacher `
    --decoder-path pretrained_decoder\decoder.pth `
    --teachers-dir wp3_d3.2_saferlearn\trained_nets_gpu `
    --output-dir teachers `
    --num-teachers 250 `
    --pool-size 50000 `
    --candidates-dir candidates `
    --config config\synthetic_generation.json
```

This will:
1. Generate a 20k candidate pool for teacher 0 → `candidates/candidates_teacher_0.pt`
2. Label and filter for teacher 0
3. Generate a 20k candidate pool for teacher 1 → `candidates/candidates_teacher_1.pt`
4. Label and filter for teacher 1
5. ... and so on for all 250 teachers

**With Shared Candidate Pool:**

```powershell
python scripts\batch_label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --teachers-dir wp3_d3.2_saferlearn\trained_nets_gpu `
    --output-dir teachers `
    --num-teachers 250 `
    --decoder-path pretrained_decoder\decoder.pth `
    --config config\synthetic_generation.json
```

This uses the shared pool generated in Step 2 (Option B).

**Options:**
- `--generate-candidates-per-teacher`: Generate a separate candidate pool for each teacher (requires `--decoder-path`)
- `--candidates`: Path to shared candidate pool (only if not using `--generate-candidates-per-teacher`)
- `--decoder-path`: Path to pretrained decoder (required for `--generate-candidates-per-teacher`, optional for latent steering)
- `--pool-size`: Size of candidate pool per teacher (default: 20000 from config, only used with `--generate-candidates-per-teacher`)
- `--candidates-dir`: Directory to save teacher-specific candidate pools (default: `candidates`)
- `--teachers-dir`: Directory containing teacher models (default: `wp3_d3.2_saferlearn/trained_nets_gpu`)
- `--output-dir`: Base output directory (default: `teachers`)
- `--num-teachers`: Number of teachers to process (default: all available)
- `--start-id`: Starting teacher ID (default: 0)
- `--confidence`: Confidence threshold (overrides config, default: 0.9 from config)
- `--quota`: Match class distribution (default: True)
- `--min-nn-distance`: Minimum NN distance for memorization check (overrides config, default: 2.0 from config)
- `--min-diversity-distance`: Minimum diversity distance (overrides config, default: 3.0 from config)
- `--min-per-class`: Minimum samples per class (overrides config, default: 150 from config)
- `--max-per-class`: Maximum samples per class (overrides config, default: 500 from config)
- `--teacher-shard-base`: Base directory for teacher shard images (optional)
- `--shard-metadata-base`: Base directory for shard metadata JSON files (optional)
- `--config`: Path to config file (default: `config/synthetic_generation.json`)
- `--seed`: Random seed for candidate generation (default: 123, will be offset by teacher_id for uniqueness)

**Example: Process first 10 teachers for testing:**
```powershell
python scripts\batch_label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --num-teachers 10 `
    --confidence 0.9


python scripts\batch_label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --num-teachers 10 `
    --confidence 0.9 `
    --quota False
```

#### Option B: Process Single Teacher

For a single teacher:

```powershell
python scripts\label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --teacher-model wp3_d3.2_saferlearn\trained_nets_gpu\0\model.pth `
    --teacher-shard-path teachers\teacher_0\shard.pt `
    --shard-metadata teachers\teacher_0\metadata.json `
    --decoder-path pretrained_decoder\decoder.pth `
    --config config\synthetic_generation.json `
    --out-dir teachers\teacher_0
```

**New Features:**
- **Class-Balanced Selection**: Automatically matches synthetic distribution to teacher's original shard with min/max quotas
- **Rare Class Thresholds**: Uses lower confidence threshold (0.7) for rare classes (1, 8) by default
- **Latent Steering**: Generates additional samples for underrepresented classes using latent optimization
- **Enhanced Filtering**: Improved diversity (min L2 distance: 3.0) and memorization (min NN distance: 2.0) checks
- **Comprehensive Reports**: Enhanced `selection_report.json` with detailed diagnostics

**Outputs:**
- `teachers/teacher_{id}/synthetic_samples.pt` - Selected synthetic samples
- `teachers/teacher_{id}/labels.csv` - Labels for samples
- `teachers/teacher_{id}/selection_report.json` - Selection statistics

**Options (for single teacher processing):**
- `--candidates`: Path to candidate pool
- `--teacher-model`: Path to teacher model.pth
- `--teacher-shard-path`: Path to original shard images (for memorization check)
- `--shard-metadata`: Path to metadata JSON with class distribution
- `--confidence`: Confidence threshold (default: 0.9)
- `--quota`: Match class distribution from metadata (default: True)
- `--max-per-class`: Override max samples per class
- `--min-diversity-distance`: Minimum pairwise distance for diversity (default: 0.0, disabled)
- `--min-nn-distance`: Minimum NN distance for memorization check (default: 1.0)

### Step 4: Combine Synthetic Datasets (Optional)

Combine synthetic datasets from all teachers into one large labeled dataset:

```powershell
python scripts\combine_synthetic_datasets.py `
    --teachers-dir teachers `
    --output combined_dataset `
    --percentage 0.1 `
    --seed 42
```

**Options:**
- `--teachers-dir`: Base directory containing teacher_* subdirectories (default: `teachers`)
- `--output`: Output directory for combined dataset (required)
- `--percentage`: Percentage of total dataset to combine (0.0 to 1.0, default: 1.0 = 100%)
  - Example: `0.1` = 10% of all samples, `0.5` = 50%, `1.0` = 100%
- `--start-id`: Starting teacher ID (inclusive, default: 0)
- `--end-id`: Ending teacher ID (exclusive, default: None = all)
- `--seed`: Random seed for sampling (default: 42)
- `--no-stratify`: Disable stratified sampling (use random sampling instead)
  - By default, stratified sampling maintains class distribution
- `--name`: Base name for output files (default: `combined_synthetic`)

**Outputs:**
- `{name}_samples.pt` - Combined synthetic samples tensor
- `{name}_labels.csv` - Combined labels CSV
- `{name}_teacher_ids.csv` - Teacher ID for each sample (for tracking)
- `{name}_metadata.json` - Metadata with statistics

**Example: Combine 10% of all samples:**
```powershell
python scripts\combine_synthetic_datasets.py `
    --teachers-dir teachers `
    --output combined_dataset `
    --percentage 0.1 `
    --seed 42
```

**Example: Combine 50% from teachers 0-99:**
```powershell
python scripts\combine_synthetic_datasets.py `
    --teachers-dir teachers `
    --output combined_dataset_subset `
    --percentage 0.5 `
    --start-id 0 `
    --end-id 100 `
    --seed 42
```

### Step 5: Train Student Model on Combined Dataset

Train a student model on the combined synthetic dataset:

```powershell
python scripts\train_student_on_synthetic.py `
    --samples combined_dataset\combined_synthetic_samples.pt `
    --labels combined_dataset\combined_synthetic_labels.csv `
    --output-dir student_models `
    --epochs 20 `
    --batch-size 64 `
    --learning-rate 0.01
```

**Options:**
- `--samples`: Path to combined synthetic samples .pt file (required)
- `--labels`: Path to combined synthetic labels .csv file (required)
- `--output-dir`: Output directory for trained model (default: `student_models`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate (default: 0.01)
- `--momentum`: Momentum for SGD (default: 0.5)
- `--weight-decay`: Weight decay / L2 regularization (default: 1e-4)
- `--train-split`: Fraction of data for training (default: 0.8, rest for validation)
- `--seed`: Random seed for train/val split (default: 42)
- `--no-cuda`: Disable CUDA even if available
- `--no-normalize`: Disable normalization (if images are already normalized)

**Outputs:**
- `student_models/student_model_best.pth` - Best model (highest validation accuracy)
- `student_models/student_model_final.pth` - Final model after all epochs

**Example: Train with custom parameters:**
```powershell
python scripts\train_student_on_synthetic.py `
    --samples combined_dataset\combined_synthetic_samples.pt `
    --labels combined_dataset\combined_synthetic_labels.csv `
    --output-dir student_models_synthetic `
    --epochs 30 `
    --batch-size 128 `
    --learning-rate 0.001 `
    --train-split 0.9
```

### Step 6: Evaluate Student Model on MNIST Test Set

Evaluate the trained student model on the standard MNIST test set (10,000 samples):

```powershell
python scripts\evaluate_student_model.py `
    --model student_models\student_model_best.pth `
    --mnist-dir wp3_d3.2_saferlearn\src\input-data\MNIST `
    --output student_models\evaluation_report.json
```

**Options:**
- `--model`: Path to trained student model .pth file (required)
- `--mnist-dir`: Path to MNIST dataset directory (default: `wp3_d3.2_saferlearn/src/input-data/MNIST`)
- `--batch-size`: Batch size for evaluation (default: 128)
- `--output`: Path to save evaluation report JSON (optional)
- `--no-cuda`: Disable CUDA even if available

**Output:**
- Prints overall accuracy, per-class accuracy, and loss
- If `--output` is provided, saves a JSON report with detailed metrics

**Example:**
```powershell
python scripts\evaluate_student_model.py `
    --model student_models\student_model_best.pth `
    --mnist-dir wp3_d3.2_saferlearn\src\input-data\MNIST `
    --batch-size 256 `
    --output student_models\mnist_test_evaluation.json
```

### Step 7: Evaluate Synthetic Dataset

Run diagnostics on the synthetic dataset:

```powershell
# Option 1: Load shard from indices (recommended if shard.pt doesn't exist)
python scripts\evaluate_synthetic.py `
    --samples teachers\teacher_0\synthetic_samples.pt `
    --labels teachers\teacher_0\labels.csv `
    --teacher-id 0 `
    --shard-indices wp3_d3.2_saferlearn\shard_indices.json `
    --out teachers\teacher_0\evaluation_report.json

# Option 2: Use pre-saved shard.pt file (if available)
python scripts\evaluate_synthetic.py `
    --samples teachers\teacher_0\synthetic_samples.pt `
    --labels teachers\teacher_0\labels.csv `
    --shard teachers\teacher_0\shard.pt `
    --out teachers\teacher_0\evaluation_report.json
```

**Outputs:**
- `teachers/teacher_0/evaluation_report.json` - Evaluation metrics including resemblance_to_shard

**Options:**
- `--samples`: Path to synthetic samples
- `--labels`: Path to labels CSV (auto-detected if in same directory)
- `--shard`: Path to original shard .pt file (if pre-saved)
- `--shard-labels`: Path to original shard labels (if separate file)
- `--teacher-id`: Teacher ID (used to load shard from indices if --shard not provided)
- `--shard-indices`: Path to shard_indices.json (default: `wp3_d3.2_saferlearn/shard_indices.json`)
- `--mnist-data-dir`: Directory containing MNIST dataset (default: `wp3_d3.2_saferlearn/src/input-data/MNIST`)

## Example Workflow (All Teachers)

**Important**: Run all commands from the project root directory (`D:\Documents\TUWien\MA\Thesis\PATE`)

```powershell
# Make sure you're in the project root
cd D:\Documents\TUWien\MA\Thesis\PATE

# Activate virtual environment (if using one)
.\.venv\Scripts\Activate.ps1

# Step 1: Pretrain decoder (once)
python scripts\pretrain_decoder.py --data-root wp3_d3.2_saferlearn\src\input-data\EMNIST --epochs 50 --out-dir pretrained_decoder --seed 42

# Step 2: Generate candidate pools (choose one option)

# Option A: Generate per-teacher pools (recommended, done automatically in Step 3)
# No separate step needed - pools are generated on-demand per teacher

# Option B: Generate single shared pool
python scripts\generate_candidates.py --decoder-path pretrained_decoder\decoder.pth --pool-size 20000 --out candidates\candidates_run1.pt --seed 123 --config config\synthetic_generation.json

# Step 3: Process all teachers (batch processing - recommended)

# Option A: With per-teacher candidate pools (recommended)
python scripts\batch_label_and_filter.py `
    --generate-candidates-per-teacher `
    --decoder-path pretrained_decoder\decoder.pth `
    --teachers-dir wp3_d3.2_saferlearn\trained_nets_gpu `
    --output-dir teachers `
    --num-teachers 250 `
    --pool-size 20000 `
    --candidates-dir candidates `
    --config config\synthetic_generation.json

# Option B: With shared candidate pool (if you ran Step 2 Option B)
python scripts\batch_label_and_filter.py `
    --candidates candidates\candidates_run1.pt `
    --teachers-dir wp3_d3.2_saferlearn\trained_nets_gpu `
    --output-dir teachers `
    --num-teachers 250 `
    --decoder-path pretrained_decoder\decoder.pth `
    --config config\synthetic_generation.json

# Alternative: Process first 10 teachers for testing
# python scripts\batch_label_and_filter.py `
#     --candidates candidates\candidates_run1.pt `
#     --num-teachers 10 `
#     --confidence 0.9

# Step 4: Combine all synthetic datasets (optional)
# Combine 10% of all samples into one large dataset
python scripts\combine_synthetic_datasets.py `
    --teachers-dir teachers `
    --output combined_dataset `
    --percentage 0.1 `
    --seed 42

# Step 5: Evaluate each teacher (optional)
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

