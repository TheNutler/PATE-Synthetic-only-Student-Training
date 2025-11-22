# PATE Framework Setup and Usage Guide

This document summarizes the complete setup, execution, and troubleshooting of the PATE (Private Aggregation of Teacher Ensembles) framework with MNIST dataset on Windows using PowerShell.

## Table of Contents

1. [System Overview](#system-overview)
2. [Initial Setup](#initial-setup)
3. [Running the PATE Framework](#running-the-pate-framework)
4. [Results and Analysis](#results-and-analysis)
5. [Student Model Training](#student-model-training)
6. [Privacy Guarantee Calculations](#privacy-guarantee-calculations)
7. [Known Issues and Fixes](#known-issues-and-fixes)
8. [Recommended Improvements](#recommended-improvements)

---

## System Overview

**Framework**: Saferlearn PATE Framework
**Dataset**: MNIST (10,000 test samples used as public dataset)
**Environment**: Windows 10/11, PowerShell
**Dependencies**: Python 3.11, PyTorch, Kafka (Docker), Flask

### Architecture Components

1. **Kafka** (Message broker): Docker containers for Zookeeper and Kafka
2. **API/Orchestrator**: Flask API server that coordinates the PATE job
3. **Teachers/Data Owners**: Multiple teacher models that make predictions
4. **Student Model**: Trained on aggregated labels from teachers

---

## Initial Setup

### Prerequisites

1. **Virtual Environment**: `.venv` in `wp3_d3.2_saferlearn/`
2. **Trained Models**: Models must exist in `trained_nets_gpu/0/`, `1/`, `2/`, etc.
3. **MNIST Dataset**: Located in `src/input-data/MNIST/`
4. **Kafka Docker Setup**: `kafka_docker.yml` for Docker Compose

### Dependencies Installation

The framework requires several Python packages. Key dependencies:

```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
pip install flask flask-cors torch torchvision kafka-python huggingface-hub rpyc typer typeguard requests python-dotenv alive-progress
```

**Note**: The full `requirements.txt` includes NVIDIA CUDA packages that may not install on Windows. Install essential packages separately as shown above.

### Training Teacher Models

Before running PATE, train teacher models:

```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
python train_mnist_models.py --num-models 3 --epochs 10
```

This creates models in `trained_nets_gpu/0/`, `1/`, `2/`, etc.

---

## Running the PATE Framework

### Step-by-Step Process (PowerShell Commands)

#### Step 1: Start Kafka

```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
docker compose -f kafka_docker.yml up -d
```

**Verify Kafka is running:**
```powershell
docker ps | Select-String -Pattern "kafka|zookeeper"
```

Kafka runs on port `29092` (mapped from container port).

#### Step 2: Start API/Orchestrator

```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
$env:KAFKA_HOST="localhost"
$env:KAFKA_PORT="29092"
python src/api.py --datasets-path src/input-data/
```

API starts on `http://localhost:5000`

**Verify API is running:**
```powershell
Invoke-WebRequest -Uri http://localhost:5000/datasets -Method GET | Select-Object -ExpandProperty Content
```

#### Step 3: Start Teachers (Data Owners)

**Each teacher needs a separate terminal with unique RPyC port.**

**Terminal 2 - Teacher 1:**
```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
$env:RPYC_PORT="1244"
$env:KAFKA_HOST="localhost"
$env:KAFKA_PORT="29092"
python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1244 --nteachers 3
```

**Terminal 3 - Teacher 2:**
```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
$env:RPYC_PORT="1245"
$env:KAFKA_HOST="localhost"
$env:KAFKA_PORT="29092"
python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1245 --nteachers 3
```

**Terminal 4 - Teacher 3:**
```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
$env:RPYC_PORT="1246"
$env:KAFKA_HOST="localhost"
$env:KAFKA_PORT="29092"
python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1246 --nteachers 3
```

**Verify teachers are registered:**
```powershell
Invoke-WebRequest -Uri http://localhost:5000/clients -Method GET | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 5
```

#### Step 4: Launch PATE Job

**Get worker UUIDs first:**
```powershell
$workers = (Invoke-WebRequest -Uri http://localhost:5000/clients -Method GET).Content | ConvertFrom-Json
$workerUuids = $workers | ForEach-Object { $_.id }
$workerUuids  # Copy these UUIDs
```

**Launch job (PowerShell):**
```powershell
$body = @{
    algorithm = "pate"
    datatype = "mnist"
    dataset = "MNIST"
    workers = @('uuid1', 'uuid2', 'uuid3')  # Replace with actual UUIDs
    nbClasses = 10
    dpValue = 0.5
    useDP = $false
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:5000/job -Method PUT -ContentType "application/json" -Body $body
```

**Job parameters:**
- `algorithm`: "pate" for clear PATE (no encryption)
- `datatype`: "mnist" (must match teacher's --data-format)
- `dataset`: "MNIST"
- `workers`: Array of worker UUIDs from `/clients` endpoint
- `nbClasses`: 10 (for MNIST digits 0-9)
- `useDP`: `$true` or `$false` (enable differential privacy)
- `dpValue`: Sigma value for noise (e.g., 0.5, 1.0, 2.0)

**Note**: The code hardcodes `nb_samples = 1000`, but the framework may process all available samples (10,000 for MNIST test set).

---

## Results and Analysis

### Monitoring Job Status

**Check job status:**
```powershell
Invoke-WebRequest -Uri http://localhost:5000/jobs -Method GET | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json
```

**Get specific job details:**
```powershell
$jobUuid = "your-job-uuid"
Invoke-WebRequest -Uri "http://localhost:5000/job/$jobUuid" -Method GET | Select-Object -ExpandProperty Content | ConvertFrom-Json
```

**Check registered workers:**
```powershell
Invoke-WebRequest -Uri http://localhost:5000/clients -Method GET | Select-Object -ExpandProperty Content | ConvertFrom-Json
```

### Retrieving Results

Results are published to Kafka topic `model_pate`. Use the provided script:

```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
$env:KAFKA_HOST="localhost"
$env:KAFKA_PORT="29092"
python get_pate_results.py
```

This creates `pate_results.csv` with format:
```
sample_id,label
0,5
1,2
2,5
...
```

### Expected Runtime

- **Model loading**: ~5-10 seconds per teacher
- **Predictions**: ~30-120 seconds for 10,000 samples (depends on CPU/GPU)
- **Aggregation**: ~5-10 seconds
- **Total**: Approximately 1-3 minutes

---

## Student Model Training

After getting PATE results, train a student model on the aggregated labels:

```powershell
cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn
.\.venv\Scripts\Activate.ps1
python train_student_model.py --pate-results pate_results.csv --data-dir src/input-data/MNIST --epochs 20
```

**Parameters:**
- `--pate-results`: Path to PATE results CSV (default: `pate_results.csv`)
- `--data-dir`: MNIST dataset directory (default: `src/input-data/MNIST`)
- `--output-dir`: Where to save trained model (default: `./student_models`)
- `--epochs`: Training epochs (default: 20)
- `--batch-size`: Batch size (default: 64)
- `--learning-rate`: Learning rate (default: 0.01)
- `--momentum`: Momentum for SGD (default: 0.5)

The script:
1. Loads PATE results CSV
2. Loads corresponding MNIST images
3. Creates dataset pairing images with PATE labels
4. Trains student model (same architecture as teachers: UCStubModel)
5. Evaluates on both PATE labels and original labels
6. Saves model to `student_models/student_model.pth`

---

## Privacy Guarantee Calculations

**Important**: The framework does NOT automatically calculate or report privacy guarantees (epsilon, delta). You must calculate them yourself.

### Manual Calculation

The framework uses the **Noisy Max mechanism** with Gaussian noise:

- **Without DP** (`useDP = false`): ε = ∞ (no privacy)
- **With DP** (`useDP = true`): ε depends on sigma, number of queries, and number of teachers

**Simplified formula** (for noisy max with Gaussian noise):
```
epsilon_per_query ≈ sqrt(2*ln(1.25/delta)) / sigma
epsilon_total ≈ num_queries * epsilon_per_query
```

Where:
- `sigma` = `dpValue` parameter
- `num_queries` = number of samples labeled (typically 10,000 for MNIST)
- `delta` = typically 1e-5

### Example Calculations

**With 3 teachers, 10,000 queries, sigma=1.0:**
- Epsilon per query: ~4.84
- Total epsilon: ~48,448 (very weak privacy)

**With 20 teachers, 1,000 queries, sigma=5.0:**
- Epsilon per query: ~0.48
- Total epsilon: ~484 (still weak, but better)

**To get strong privacy (ε < 1.0):**
- Need very high sigma (>50) OR
- Very few queries (<100) OR
- Use advanced composition theorems

### Privacy Analysis Script

Use the provided script to calculate guarantees:

```powershell
python calculate_privacy_guarantee.py --teachers 3 --queries 10000 --sigma 1.0 --delta 1e-5
```

---

## Known Issues and Fixes

### Issue 1: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'torchvision'` (or 'flask', 'alive-progress', etc.)

**Fix**: Install missing packages:
```powershell
.\.venv\Scripts\Activate.ps1
pip install flask flask-cors torch torchvision kafka-python huggingface-hub rpyc alive-progress
```

### Issue 2: Orchestrator Crash - Redundant Subscribe

**Error**: `TypeError: Topics must be a list (or non-str sequence)`

**Location**: `src/orchestrator.py` line 747

**Fix Applied**: Removed redundant `consumer.subscribe(topic_name)` call because `connect_kafka_consumer()` already subscribes via constructor.

**Fixed code** (line 747 removed):
```python
# OLD (line 747):
consumer.subscribe(topic_name)  # REMOVED - redundant

# NEW:
# connect_kafka_consumer already subscribes, no need to subscribe again
```

### Issue 3: PowerShell vs Bash Syntax

**Problem**: Documentation shows bash commands, but Windows requires PowerShell.

**Key differences:**
- Bash: `source .venv/bin/activate` → PowerShell: `.\.venv\Scripts\Activate.ps1`
- Bash: `export VAR=value` → PowerShell: `$env:VAR="value"`
- Bash: `curl` → PowerShell: `Invoke-WebRequest`

### Issue 4: Poor Results Quality

**Symptoms:**
- Accuracy on PATE labels: ~47%
- Accuracy on original labels: ~10% (random)
- Extreme label imbalance (47% label 2, 0.03% label 9)

**Root causes:**
1. **Too few teachers** (3 is insufficient, need 10-100)
2. **No differential privacy** (useDP = false)
3. **Possible teacher model quality issues**
4. **Aggregation may have bugs**

---

## Recommended Improvements

### Critical Fixes Needed

1. **Increase Number of Teachers** (MOST IMPORTANT)
   ```powershell
   # Train 20 teachers instead of 3
   python train_mnist_models.py --num-models 20 --epochs 15
   ```
   - Current: 3 teachers → Expected accuracy: ~45-50%
   - With 20 teachers → Expected accuracy: ~80-90%
   - With 50-100 teachers → Expected accuracy: ~90-98%

2. **Enable Differential Privacy**
   ```powershell
   $body = @{
       ...
       useDP = $true
       dpValue = 1.0  # Start with 1.0, adjust as needed
   }
   ```
   - Provides privacy protection
   - May slightly reduce accuracy (noise can flip close votes)
   - For strong privacy, need sigma=5-10+ AND fewer queries

3. **Verify Teacher Model Quality**
   - Each teacher should achieve >90% test accuracy
   - Use `test_teacher_models.py` script (if available)
   - Ensure teachers are trained on different private data

4. **Reduce Sample Count for Better Privacy**
   - Current: 10,000 samples → Very high epsilon
   - Better: 1,000 samples → Lower epsilon
   - Trade-off: Less training data for student model

### Recommended Configuration

**For good accuracy with moderate privacy:**
```json
{
  "algorithm": "pate",
  "datatype": "mnist",
  "dataset": "MNIST",
  "workers": ["uuid1", "uuid2", ..., "uuid20"],  // 20 teachers
  "nbClasses": 10,
  "useDP": true,
  "dpValue": 1.0,
  // Note: nb_samples is hardcoded to 1000 in api.py
}
```

**Expected results:**
- Accuracy: 80-90%
- Privacy: ε ≈ 4,800 (still weak, but better than ∞)
- Runtime: ~2-5 minutes

---

## Code Modifications Made

### Fixed Files

1. **`src/orchestrator.py`** (line 747):
   - **Removed**: `consumer.subscribe(topic_name)` (redundant, caused TypeError)
   - **Reason**: `connect_kafka_consumer()` already subscribes via KafkaConsumer constructor

### Created Scripts

1. **`get_pate_results.py`**:
   - Consumes results from Kafka topic `model_pate`
   - Saves to `pate_results.csv`
   - Shows results in terminal

2. **`train_student_model.py`**:
   - Trains student model on PATE aggregated labels
   - Evaluates on both PATE labels and original labels
   - Saves trained model

3. **`calculate_privacy_guarantee.py`**:
   - Calculates epsilon (ε) and delta (δ) privacy guarantees
   - Provides recommendations for better privacy
   - Shows privacy level interpretation

---

## File Structure

```
wp3_d3.2_saferlearn/
├── src/
│   ├── api.py                    # API/Orchestrator server
│   ├── orchestrator.py           # PATE job orchestration (FIXED: line 747)
│   ├── data_owner_cli.py         # Teacher/Data owner CLI
│   ├── model_owner.py            # Model owner script
│   ├── usecases/
│   │   └── data_owner_example.py # Teacher implementation
│   └── input-data/
│       └── MNIST/                # MNIST dataset
├── trained_nets_gpu/             # Teacher models (must exist)
│   ├── 0/
│   │   └── model.pth
│   ├── 1/
│   │   └── model.pth
│   └── 2/
│       └── model.pth
├── kafka_docker.yml              # Kafka Docker Compose config
├── train_mnist_models.py         # Train teacher models
├── get_pate_results.py           # NEW: Retrieve PATE results
├── train_student_model.py        # NEW: Train student model
├── calculate_privacy_guarantee.py # NEW: Calculate privacy guarantees
├── pate_results.csv              # PATE aggregated labels (output)
└── student_models/               # Trained student model (output)
```

---

## Current Results Summary

### Last Successful Run

- **Teachers**: 3
- **Samples**: 10,000
- **DP enabled**: No (useDP = false)
- **Accuracy on PATE labels**: 47.39%
- **Accuracy on original labels**: 10.32% (essentially random)
- **Label distribution**: Extremely imbalanced (47% label 2, 0.03% label 9)
- **Privacy guarantee**: None (ε = ∞)

### Issues Identified

1. **Too few teachers** (3 is insufficient)
2. **No differential privacy** applied
3. **Poor aggregation quality** (severe label imbalance)
4. **Possible teacher model quality issues**
5. **Hardcoded sample count mismatch** (code says 1000, got 10000)

---

## API Endpoints

### Available Endpoints

- `GET /datasets` - List available datasets
- `GET /clients` - List registered workers/teachers
- `PUT /clients` - Register/update worker (called by teachers)
- `PUT /job` - Launch a new PATE job
- `GET /jobs` - List all jobs
- `GET /job/<uuid>` - Get specific job details
- `PUT /job/<uuid>/<state>` - Update job state
- `DELETE /jobs` - Clear all jobs
- `DELETE /clients` - Clear all workers

### Example Usage

```powershell
# Get workers
$workers = (Invoke-WebRequest -Uri http://localhost:5000/clients -Method GET).Content | ConvertFrom-Json

# Launch job
$body = @{ algorithm = "pate"; datatype = "mnist"; dataset = "MNIST"; workers = @("uuid1", "uuid2", "uuid3"); nbClasses = 10; dpValue = 0.5; useDP = $false } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:5000/job -Method PUT -ContentType "application/json" -Body $body

# Check job status
$jobs = (Invoke-WebRequest -Uri http://localhost:5000/jobs -Method GET).Content | ConvertFrom-Json
```

---

## Stopping Services

```powershell
# Stop teachers: Ctrl+C in each teacher terminal

# Stop API: Ctrl+C in API terminal

# Stop Kafka:
docker compose -f kafka_docker.yml down

# Kill all Python processes:
Get-Process python | Stop-Process -Force

# Clear database:
Invoke-WebRequest -Uri http://localhost:5000/jobs -Method DELETE
Invoke-WebRequest -Uri http://localhost:5000/clients -Method DELETE
```

---

## Troubleshooting

### Teachers Not Registering

- Check Kafka is running: `docker ps`
- Verify KAFKA_HOST and KAFKA_PORT environment variables
- Check teacher terminal outputs for errors
- Verify models exist in `trained_nets_gpu/X/` directories

### Job Stuck or Crashed

- Check orchestrator terminal for errors
- Verify all teachers are online: `GET /clients`
- Clear old jobs: `DELETE /jobs`
- Restart services if needed

### Poor Accuracy

- **Most likely**: Too few teachers (need 10+)
- Verify teacher models are well-trained (>90% accuracy)
- Check label distribution in results CSV
- Enable DP may help if votes are tied

### No Results in Kafka

- Wait for job to complete (~1-3 minutes)
- Check orchestrator terminal for "Publishing result" message
- Verify Kafka topic exists: `docker exec -it kafka-container kafka-topics --list`
- Use `get_pate_results.py` script to consume results

---

## Key Learnings

1. **3 teachers is insufficient** - Need 10-100 for good results
2. **DP must be calculated manually** - Framework doesn't provide epsilon/delta
3. **PowerShell syntax differs from bash** - Use `$env:VAR` not `export VAR`
4. **Models must exist before starting teachers** - Check `trained_nets_gpu/` directories
5. **Results are in Kafka topic** - Need consumer script to retrieve
6. **Student model trains on aggregated labels** - Not original labels
7. **Label imbalance indicates aggregation issues** - Check teacher predictions
8. **Privacy-accuracy tradeoff** - Higher sigma = more privacy but lower accuracy

---

## Next Steps for Better Results

1. **Train 20 teacher models** with good accuracy (>90%)
2. **Start 20 teachers** with unique RPyC ports
3. **Enable DP** with sigma=1.0 initially
4. **Reduce sample count** to 1,000 for better privacy
5. **Launch new PATE job** with all improvements
6. **Calculate privacy guarantees** using the script
7. **Train student model** on new results
8. **Expect 80-90% accuracy** with proper setup

---

## Important Notes

- **Windows PowerShell**: All commands use PowerShell syntax, not bash
- **Virtual environment**: Always activate `.\.venv\Scripts\Activate.ps1`
- **Kafka port**: Use `29092` (mapped from container), not `9092`
- **Sample count**: Hardcoded to 1000 in `api.py` line 72, but may process all available
- **Privacy accounting**: Not automatic - must calculate yourself
- **Model quality**: Critical for good results - verify each teacher achieves >90% accuracy

---

## Contact / Context

This setup was configured for a thesis project on PATE (Private Aggregation of Teacher Ensembles) for differential privacy in machine learning. The framework was successfully run end-to-end, but results were poor due to too few teachers (3) and no differential privacy enabled. Improvements are needed for production-quality results.

**Key files created during setup:**
- `get_pate_results.py` - Consume results from Kafka
- `train_student_model.py` - Train student on aggregated labels
- `calculate_privacy_guarantee.py` - Calculate epsilon/delta
- `PATE_ANALYSIS.md` - Detailed analysis of issues
- `RUN_PATE_MNIST.md` - Original documentation (bash-based)



