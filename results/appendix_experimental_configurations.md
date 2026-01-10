# Appendix: Experimental Configurations

This document records all experimental configurations used throughout the project to enable full reproducibility.

## Table of Contents

1. [Global Experimental Settings](#global-experimental-settings)
2. [Teacher Model Configurations](#teacher-model-configurations)
3. [Synthetic Data Generation Configurations](#synthetic-data-generation-configurations)
4. [Decoder-Based Exploratory Approach](#decoder-based-exploratory-approach)
5. [Student Model Training](#student-model-training)
6. [Logging and Output Artifacts](#logging-and-output-artifacts)

---

## Global Experimental Settings

### Hardware Environment

- **Platform**: Windows 10 (build 10.0.22631)
- **Shell**: PowerShell (C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe)
- **CUDA**: Available (device auto-detected via `torch.cuda.is_available()`)
- **Note**: Hardware specifications not explicitly documented in codebase

### Software Environment

- **Python Version**: 3.11 (required, specified in README files)
- **Core Dependencies** (from `requirements_synthetic.txt`):
  - `torch>=2.0.0`
  - `torchvision>=0.15.0`
  - `numpy>=1.20.0`
  - `pillow>=9.0.0` (optional but recommended)

### Dataset Versions and Preprocessing

#### MNIST Dataset
- **Source**: Standard MNIST dataset from torchvision
- **Location**: `wp3_d3.2_saferlearn/src/input-data/MNIST`
- **Normalization Parameters**:
  - Mean: `[0.1307]`
  - Std: `[0.3081]`
- **Preprocessing**: Images converted to tensor and normalized using above parameters
- **Image Range**: Normalized images stored in normalized space; synthetic images saved in `[0, 1]` range

#### EMNIST Dataset (for Decoder Pretraining)
- **Version**: EMNIST Balanced
- **Location**: `wp3_d3.2_saferlearn/src/input-data/EMNIST`
- **Size**: 112,800 training samples (if available, fallback to KMNIST)
- **Alternative**: KMNIST (if EMNIST unavailable, enabled via `--use-kmnist` flag)
- **Normalization**: Same as MNIST (mean=0.1307, std=0.3081)

#### Teacher Shard Configuration
- **Shard Index File**: `wp3_d3.2_saferlearn/shard_indices.json`
- **Shard Size**: 40 samples per teacher (60,000 total / 250 teachers)
- **Shard Creation**: Random permutation of MNIST training indices, partitioned into disjoint shards
- **Shard Storage**: Optional pre-saved shards at `teachers/teacher_{id}/shard.pt`

### Randomness Control

#### Random Seeds

**Base Seeds:**
- **Teacher Training**: `42` (default in `train_mnist_models.py`)
- **VAE Pretraining**: `42` (default in `pretrain_decoder.py`)
- **Per-Teacher VAE Training**: `42 + teacher_id` (offset per teacher in `batch_train_teacher_vaes.py`)
- **Candidate Generation (Shared Pool)**: `123` (default in `generate_candidates.py`)
- **Candidate Generation (Per-Teacher)**: `123 + teacher_id` (offset per teacher in `batch_label_and_filter.py`)
- **Dataset Combination**: `42` (default in `combine_synthetic_datasets.py`)
- **Student Training Split**: `42` (default in `train_student_on_synthetic.py`)

**Seed Offset Strategy:**
- Per-teacher VAE training: Base seed (`42`) + `teacher_id` for unique initialization per teacher
- Per-teacher candidate generation: Base seed (`123`) + `teacher_id` for unique candidate pools

**Seed Setting Locations:**
- **NumPy**: `np.random.seed(seed)` in generation scripts
- **PyTorch**: 
  - `torch.manual_seed(seed)` in all training scripts
  - `torch.cuda.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)` when CUDA available
- **Python random**: `random.seed(seed)` in data combination scripts

**Uncontrolled Randomness:**
- Teacher shard partitioning uses `random.shuffle()` with seed, but exact shard boundaries depend on total dataset size and number of teachers
- DataLoader shuffle uses PyTorch's internal RNG (controlled via `torch.manual_seed`)
- Some augmentation transforms may introduce uncontrolled randomness if not seeded

---

## Teacher Model Configurations

### Model Architecture

**Architecture Name**: `UCStubModel`

**Architecture Specification**:
```
- Input: (batch_size, 1, 28, 28) grayscale images
- Conv2d(1, 32, kernel_size=3, stride=1)
- ReLU activation
- Conv2d(32, 64, kernel_size=3, stride=1)
- ReLU activation
- MaxPool2d(kernel_size=2, stride=2)
- Dropout(0.25)
- Flatten
- Linear(9216, 128)
- ReLU activation
- Dropout(0.5)
- Linear(128, 10)
- LogSoftmax output
```

**Model Definition Location**: 
- `wp3_d3.2_saferlearn/src/usecases/UC_stub.py`
- `scripts/train_student_on_synthetic.py` (student model)
- `scripts/label_and_filter.py` (for loading teacher models)

### Training Hyperparameters

**Standard Configuration** (from `train_mnist_models.py`):
- **Number of Teachers**: 250 (default), configurable via `--num-models`
- **Epochs**: 10 (default)
- **Batch Size**: 64 (default)
- **Optimizer**: SGD (default)
  - Learning Rate: `0.01`
  - Momentum: `0.5` (for SGD)
  - Weight Decay: `1e-4`
- **Learning Rate Scheduler**: StepLR
  - Step Size: `20` epochs
  - Gamma: `0.1`
  - Note: LR decay occurs at epoch 20, but training typically runs for 10 epochs (no decay in practice)
- **Alternative Optimizer**: Adam (available via `--optimizer adam`)
  - Learning Rate: `0.01` (when using Adam)
  - Weight Decay: `1e-4`

**Data Augmentation**:
- **Default**: Disabled (`--augment false`)
- **If Enabled**: 
  - RandomRotation: ±10°
  - RandomAffine: translate=(0.1, 0.1), degrees=0

### Dataset Shard Sizes and Partitioning

**Partitioning Method**:
- Random permutation of all MNIST training indices
- Disjoint partitioning: each teacher receives non-overlapping subset
- Partition size: `total_samples // num_models`
- Last teacher receives any remaining samples if not evenly divisible

**Shard Size**:
- **Approximate**: ~240 samples per teacher (60,000 / 250 teachers)
- **Exact size**: Stored in `shard_indices.json` per teacher
- **Shard Indices**: Saved to `wp3_d3.2_saferlearn/shard_indices.json` for reproducibility

**Training Data**:
- Each teacher trains only on its assigned shard (disjoint subset)
- No overlap between teacher training sets
- MNIST test set: 10,000 samples (not used for teacher training)

### Training Epochs and Stopping Criteria

- **Fixed Epochs**: Training runs for exactly `--epochs` epochs (default: 10)
- **No Early Stopping**: Training completes all epochs regardless of convergence
- **Model Saving**: Final model saved after all epochs complete

---

## Synthetic Data Generation Configurations

### Approach 1: Per-Teacher VAE-Based Synthetic Data

This approach trains a separate VAE encoder+decoder for each teacher on their private shard.

#### VAE Architecture Variants

**Architecture**: Full VAE (encoder + decoder)

**Encoder** (`VAEEncoder`):
- Conv2d(1, 32, kernel_size=4, stride=2, padding=1) → 28×28 → 14×14
- ReLU
- Conv2d(32, 64, kernel_size=4, stride=2, padding=1) → 14×14 → 7×7
- ReLU
- Flatten → 7×7×64 = 3136
- Linear(3136, latent_dim) → mu
- Linear(3136, latent_dim) → logvar

**Decoder** (`VAEDecoder`):
- Linear(latent_dim, 7×7×128) → 6272
- Reshape → (128, 7, 7)
- ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) → 7×7 → 14×14
- ReLU
- ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) → 14×14 → 28×28
- ReLU
- Conv2d(32, 1, kernel_size=3, stride=1, padding=1) → 28×28
- Sigmoid → [0, 1] range

**Architecture Definition**: `models/decoder.py`

#### Latent Dimensions

- **Default**: `32` (configurable via `--latent-dim`)
- **Used in**: All per-teacher VAE training experiments

#### Loss Functions and Weighting

**Loss Function**: VAE Loss (Reconstruction + KL Divergence)

- **Reconstruction Loss**: Binary Cross-Entropy (BCE)
  - Formula: `F.binary_cross_entropy(recon_x, x, reduction='sum')`
  - Applied between decoder output (in [0,1]) and input (in [0,1])
- **KL Divergence Loss**: Standard VAE KL term
  - Formula: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`
- **Total Loss**: `recon_loss + kl_weight * kl_loss`

**KL Annealing**:
- **Enabled**: Yes
- **Annealing Epochs**: `20` (default, configurable via `--kl-annealing-epochs`)
- **Schedule**: Linear from 0 to 1 over first `kl_annealing_epochs` epochs
  - Epoch < kl_annealing_epochs: `kl_weight = epoch / kl_annealing_epochs`
  - Epoch >= kl_annealing_epochs: `kl_weight = 1.0`

#### Training Schedules

**Per-Teacher VAE Training** (`train_teacher_vae.py`):
- **Epochs**: `100` (default)
- **Batch Size**: `32` (default, auto-adjusted to shard size if shard < batch_size)
- **Learning Rate**: `1e-3` (default)
- **Optimizer**: Adam
- **Weight Decay**: `1e-5` (default)
- **KL Annealing Epochs**: `20` (default)
- **Data Augmentation**: Enabled by default (disabled via `--no-augmentation` flag)

**Augmentation for Per-Teacher VAEs**:
- RandomRotation: ±10°
- RandomAffine: translate=(2/28, 2/28) pixels
- Gaussian noise: σ=0.03
- Clamp to [0, 1]

**Training Input**:
- Shard images in [0, 1] range (not normalized)
- Shard size: ~240 samples per teacher
- Images clamped to [0, 1] before training

**Batch Training**: 
- Script: `batch_train_teacher_vaes.py`
- Processes all teachers sequentially
- Default: 250 teachers, starting from ID 0
- Seed offset: `base_seed + teacher_id` for each teacher

#### Sample Generation Sizes

**Per-Teacher Candidate Generation** (`generate_candidates_teacher_vae.py`):
- **Pool Size**: `20000` (default from config, configurable via `--pool-size`)
- **Generation Batch Size**: 256 (hardcoded in `generate_candidates()` function)
- **Latent Sampling**: Standard normal N(0, 1)
- **Latent Mixing**: 
  - Ratio: `0.3` (default from config)
  - Method: Interpolation between consecutive batches (α=0.6-1.0 range)
- **Latent Noise**: 
  - Scale: `0.1` (default from config)
  - Applied after mixing

**Approach Status**: Replaced by pretrained decoder approach due to limitations (documented in `README_VAE_PER_TEACHER.md`)

### Approach 2: Pretrained Decoder-Based Synthetic Data (Revised VAE Approach)

This approach uses a single decoder pretrained on public EMNIST data, then generates candidates for all teachers.

#### VAE Architecture (Pretrained Decoder)

**Architecture**: Same as per-teacher VAE (see above)

**Pretraining Dataset**: EMNIST Balanced (or KMNIST fallback)

**Pretraining Configuration** (`pretrain_decoder.py`):
- **Epochs**: `50` (default)
- **Batch Size**: `128` (default)
- **Learning Rate**: `1e-3` (default)
- **Optimizer**: Adam
- **Weight Decay**: `1e-5` (default)
- **KL Annealing Epochs**: `10` (default)
- **Dataset Size**: ~112,800 samples (EMNIST Balanced)

**Pretraining Augmentation**:
- Same as per-teacher VAE augmentation
- Applied via `AugmentedDataset` wrapper
- Images denormalized → augmented → renormalized

**Decoder Output**: Only decoder weights saved (`decoder.pth`), encoder discarded after pretraining

#### Sample Generation Sizes (Pretrained Decoder)

**Standard Configuration**:
- **Pool Size (Per-Teacher)**: `20000` (default from `config/synthetic_generation.json`)
- **Pool Size (Shared)**: `20000` (default, configurable)
- **Generation Batch Size**: 256 (hardcoded)

**Alternative Configuration (9k Dataset Workflow)**:
- **Pool Size**: `1000` per teacher (specified in `WORKFLOW_9K_DATASET.md`)
- **Target Samples**: 36 per teacher
- **Total Dataset Size**: 9,000 samples (250 teachers × 36 samples)

**Latent Parameters** (`config/synthetic_generation.json`):
- **Latent Dimension**: `32`
- **Latent Mixing Ratio**: `0.3`
- **Latent Noise Scale**: `0.1`

**Generation Strategy**:
- **Per-Teacher Pools**: Separate 20k pool per teacher (recommended)
- **Shared Pool**: Single shared pool for all teachers (alternative)
- **Seed Strategy**: Base seed `123` + `teacher_id` for per-teacher pools

### Approach 3: Revised VAE Approach with Larger Synthetic Sample Size

**Configuration**: Uses pretrained decoder approach (Approach 2) with larger sample sizes
- **Pool Size**: `50000` (documented in `batch_label_and_filter.py` examples)
- **Other parameters**: Same as Approach 2

### Approach 4: Decoder-Based Exploratory Approach

**Configuration**: Identical to Approach 2 (pretrained decoder)
- Uses same pretrained decoder
- Same generation parameters
- Explored different filtering and selection strategies

#### Decoder Pretraining Setup

**Same as Approach 2** (see above)

#### Public Dataset Usage

- **Primary**: EMNIST Balanced
- **Fallback**: KMNIST (if EMNIST unavailable)
- **Dataset**: Public data only (not private teacher shards)

#### Selection and Filtering Parameters

**Configuration File**: `config/synthetic_generation.json`

**Filtering Parameters**:
- **Default Confidence Threshold**: `0.90`
- **Rare Class Threshold**: `0.70`
- **Rare Classes**: `[1, 8]`
- **Min Per Class**: `150`
- **Max Per Class**: `500`
- **Diversity Threshold**: `3.0` (minimum pairwise L2 distance)
- **Min Memorization Distance**: `2.0` (minimum nearest-neighbor distance to private shard)
- **Quota Flexibility Percent**: `10.0`

**Class Balancing**:
- **Enabled**: By default (via `--quota True` flag)
- **Method**: Matches teacher's original shard class distribution
- **Constraints**: Min/max per class enforced with flexibility

#### Latent Steering Configuration

**Enabled**: `true` (from config)

**Parameters** (`config/synthetic_generation.json`):
- **Steps**: `20` (optimization iterations)
- **Learning Rate**: `0.05`
- **Noise Scale**: `0.1`
- **Use For Classes Below Quota**: `true`

**Method**: Gradient-based optimization in latent space to maximize teacher confidence for target class

---

## Student Model Training

### Architectures

**Architecture**: `UCStubModel` (identical to teacher architecture)
- Same specification as teacher models (see Teacher Model Configurations)

**Architecture Reuse**: Student uses identical architecture to teachers for fair comparison

### Training Hyperparameters

#### Standard Configuration (`train_student_on_synthetic.py`)

**Hyperparameters**:
- **Epochs**: `20` (default)
- **Batch Size**: `64` (default)
- **Optimizer**: SGD
- **Learning Rate**: `0.01` (default)
- **Momentum**: `0.5` (default)
- **Weight Decay**: `1e-4` (default)
- **Train/Val Split**: `0.8` (80% train, 20% validation, default)

**Training Procedure**:
- **Loss Function**: Negative Log Likelihood (NLL) loss
- **Model Selection**: Best model based on validation accuracy
- **Checkpointing**: 
  - Best model: `student_model_best.pth` (highest validation accuracy)
  - Final model: `student_model_final.pth` (after all epochs)

**Data Preprocessing**:
- **Normalization**: Enabled by default (MNIST normalization: mean=0.1307, std=0.3081)
- **Image Range**: Synthetic samples in [0, 1] → normalized for training
- **Disable Normalization**: Available via `--no-normalize` flag

#### Baseline PATE Student Training (`wp3_d3.2_saferlearn/train_student_model.py`)

**Hyperparameters**:
- **Epochs**: `20` (default)
- **Batch Size**: `64` (default)
- **Optimizer**: SGD
- **Learning Rate**: `0.01` (default)
- **Momentum**: `0.5` (default)

**Dataset**: 
- Public MNIST test set (10,000 samples)
- Labels: PATE aggregated labels from teacher ensemble

**Normalization**: 
- Transform: `transforms.Normalize((0.5,), (0.5,))` (different from synthetic approach)

### Input Dataset Sizes

**Synthetic Dataset Training**:
- **Variable**: Depends on combined synthetic dataset size
- **9k Dataset**: 9,000 samples (250 teachers × 36 samples)
- **Standard Dataset**: Variable (depends on filtering and combination settings)
- **Train/Val Split**: 80/20 by default (configurable via `--train-split`)

**Baseline PATE Training**:
- **Dataset Size**: Variable (depends on number of samples with PATE labels)
- **Source**: Public MNIST test set with PATE aggregated labels

---

## Logging and Output Artifacts

### Configuration Files (JSON/YAML)

**Primary Configuration**:
- **File**: `config/synthetic_generation.json`
- **Structure**:
  ```json
  {
    "filtering": { ... },
    "generation": { ... },
    "latent_steering": { ... },
    "preprocessing": { ... }
  }
  ```

**Example Configuration**:
- **File**: `config_example.json`
- **Purpose**: Example/template configuration

**Shard Indices**:
- **File**: `wp3_d3.2_saferlearn/shard_indices.json`
- **Format**: JSON mapping teacher_id → list of MNIST training indices
- **Purpose**: Reproducible teacher shard assignment

### Output Report Formats

**VAE Training Reports**:
- **Location**: `teacher_vaes/teacher_{id}/training_report.json`
- **Content**: 
  - Teacher ID, latent dim, epochs, batch size, learning rate
  - Weight decay, KL annealing epochs, seed
  - Shard size, shard source, augmentation status
  - Final losses (total, reconstruction, KL)
  - Training history (per-epoch metrics)
  - Model paths and file hashes

**Decoder Pretraining Report**:
- **Location**: `pretrained_decoder/pretrain_report.json`
- **Content**:
  - Latent dim, epochs, batch size, learning rate, weight decay
  - KL annealing epochs, seed
  - Dataset name and size
  - Final losses and training history
  - Decoder path and hash

**Selection Reports**:
- **Location**: `teachers/teacher_{id}/selection_report.json`
- **Content**:
  - Teacher ID, candidate pool size
  - Selection statistics (selected, rejected, reasons)
  - Class distribution (target and achieved)
  - Filtering parameters used
  - Quality metrics

**Evaluation Reports**:
- **Location**: `teachers/teacher_{id}/evaluation_report.json` or `results/*.json`
- **Content**:
  - Evaluation metrics (diversity, memorization, similarity)
  - Class distribution
  - Pixel statistics
  - Pass/fail criteria results

**Student Training**:
- **No JSON report**: Training progress printed to console
- **Model Checkpoints**: Best and final models saved as `.pth` files

**Combined Dataset Metadata**:
- **Location**: `combined_dataset/{name}_metadata.json`
- **Content**:
  - Total samples, number of teachers
  - Teacher IDs, percentage used
  - Seed, stratified sampling flag
  - Class distribution, teacher distribution
  - Average pairwise distance
  - File paths and hashes

### Naming Conventions

**Model Files**:
- **Teacher Models**: `wp3_d3.2_saferlearn/trained_nets_gpu/{teacher_id}/model.pth`
- **Per-Teacher VAE Decoders**: `teacher_vaes/teacher_{id}/decoder.pth`
- **Per-Teacher VAE Full**: `teacher_vaes/teacher_{id}/vae_full.pth`
- **Pretrained Decoder**: `pretrained_decoder/decoder.pth`
- **Student Models**: `student_models/student_model_best.pth`, `student_models/student_model_final.pth`

**Candidate Pools**:
- **Shared Pool**: `candidates/candidates_{run_id}.pt`
- **Per-Teacher Pools**: `candidates/candidates_teacher_{id}.pt`

**Synthetic Datasets**:
- **Samples**: `teachers/teacher_{id}/synthetic_samples.pt`
- **Labels**: `teachers/teacher_{id}/labels.csv`
- **Combined Samples**: `combined_dataset/{name}_samples.pt`
- **Combined Labels**: `combined_dataset/{name}_labels.csv`
- **Teacher IDs**: `combined_dataset/{name}_teacher_ids.csv`

**Reports**:
- **Selection**: `teachers/teacher_{id}/selection_report.json`
- **Evaluation**: `teachers/teacher_{id}/evaluation_report.json`
- **Training**: `teacher_vaes/teacher_{id}/training_report.json`
- **Pretraining**: `pretrained_decoder/pretrain_report.json`

**Metadata Files**:
- **Shard Metadata**: `teachers/teacher_{id}/metadata.json` (optional)
- **Shard Images**: `teachers/teacher_{id}/shard.pt` (optional)
- **Combined Metadata**: `combined_dataset/{name}_metadata.json`

### Output Format Requirements

**Tensor Files (`.pt`)**:
- PyTorch tensor format
- Images: Shape `(N, 1, 28, 28)` in [0, 1] range
- Saved via `torch.save()` or custom `save_tensor()` utility

**CSV Files**:
- **Labels**: Header row `['index', 'label']` or `['label']`
- **Teacher IDs**: Header row `['index', 'teacher_id']`
- UTF-8 encoding

**JSON Files**:
- UTF-8 encoding
- Pretty-printed with indentation
- Standard JSON format

**File Hashes**:
- Computed for model checkpoints and datasets
- Algorithm: SHA-256 (via `compute_file_hash()` utility)
- Stored in metadata/report files for audit trail

**Visualization Grids**:
- **Format**: PNG
- **Naming**: `{candidate_pool_name}.png`
- **Content**: 8×8 grid of sample images (64 samples)
- **Saved via**: `torchvision.utils.save_image()`

---

## Configuration Reuse Across Approaches

**Reused Configurations**:
- **VAE Architecture**: Same encoder/decoder architecture for all VAE-based approaches
- **Latent Dimension**: 32 for all approaches
- **Teacher Architecture**: UCStubModel used consistently
- **Normalization**: MNIST normalization (mean=0.1307, std=0.3081) used consistently
- **Random Seeds**: Base seeds (42 for training, 123 for generation) used consistently with offsets

**Unique Configurations**:
- **Per-Teacher VAE**: Separate training per teacher on private shards
- **Pretrained Decoder**: Single decoder trained on public data
- **Sample Sizes**: Vary by approach (20k vs 1k vs 50k pool sizes)
- **Filtering Strategies**: Different parameters for different experimental runs

---

## Missing or Implicit Information

**Not Explicitly Documented**:
- Exact hardware specifications (GPU model, CPU, RAM)
- Exact PyTorch version beyond minimum (>=2.0.0)
- Exact NumPy version beyond minimum (>=1.20.0)
- CUDA version
- Operating system build details (beyond Windows 10)

**Implicit Assumptions**:
- All random number generators seeded before use (via seed parameters)
- Data loaders use deterministic behavior when seeds are set
- File system paths are case-insensitive (Windows)
- Images are stored as float32 tensors
- Teacher models expect normalized input (mean=0.1307, std=0.3081)

**Configuration Inference**:
- Batch sizes may be auto-adjusted if dataset smaller than batch size (documented in code comments)
- Device selection: CUDA if available, else CPU (auto-detected)
- Some parameters may be inferred from model architecture (e.g., latent_dim from decoder weights)

---

*End of Experimental Configurations Document*

