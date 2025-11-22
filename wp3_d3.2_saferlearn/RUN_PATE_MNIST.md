# Running PATE Framework with MNIST Dataset

This guide will walk you through running the PATE (Private Aggregation of Teacher Ensembles) framework with the MNIST dataset.

## Prerequisites

1. **MNIST Dataset**: Already downloaded in `src/input-data/MNIST/`
2. **Python Environment**: Virtual environment with dependencies installed
3. **Kafka**: Required for message passing between components
4. **Trained Models**: You need at least one trained model for the teachers

## Step 1: Set Up Kafka

The PATE framework uses Kafka for communication between teachers and the orchestrator. Start Kafka using Docker Compose:

```bash
cd wp3_d3.2_saferlearn
docker compose -f ./kafka_docker.yml up -d
```

Verify Kafka is running:
```bash
docker ps | grep kafka
```

## Step 2: Prepare Trained Models

The framework expects trained models in numbered subdirectories. Create the models directory structure:

```bash
mkdir -p trained_nets_gpu/0
mkdir -p trained_nets_gpu/1
mkdir -p trained_nets_gpu/2
```

### Option A: Use the Training Script (Recommended)

Use the provided training script to automatically train multiple models:

```bash
cd wp3_d3.2_saferlearn
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Train 3 models with default settings (10 epochs each)
python train_mnist_models.py

# Or customize the training:
python train_mnist_models.py --num-models 3 --epochs 10 --batch-size 64
```

The script will:
- Train multiple models using the `UCStubModel` architecture
- Save each model to `trained_nets_gpu/0/`, `trained_nets_gpu/1/`, etc.
- Show training progress and test accuracy for each model

**Training script options:**
- `--num-models`: Number of models to train (default: 3)
- `--epochs`: Training epochs per model (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate (default: 0.01)
- `--momentum`: Momentum for SGD (default: 0.5)
- `--data-dir`: Path to MNIST dataset (default: `src/input-data/MNIST`)
- `--output-dir`: Output directory (default: `trained_nets_gpu`)

### Option B: Train Models Manually

If you want to train models manually, you'll need to train MNIST models using the `UCStubModel` architecture:

```python
import torch
from src.usecases.data_owner_example import UCStubModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('src/input-data/MNIST', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create and train model
model = UCStubModel()
# ... train your model ...
# Save the model state dict
torch.save(model.state_dict(), 'trained_nets_gpu/0/model.pth')
```

### Option C: Use Existing Models

If you have pre-trained models, place them in the numbered subdirectories:
- `trained_nets_gpu/0/model.pth` (or any .pth/.pkl file)
- `trained_nets_gpu/1/model.pth`
- `trained_nets_gpu/2/model.pth`
- etc.

**Important**: Each subdirectory should contain at least one model file (`.pth` or `.pkl`). The framework will automatically select available models.

## Step 3: Set Environment Variables (Optional)

If you want to customize paths, set environment variables:

```bash
export MODELS_PATH="./trained_nets_gpu"
export ORCHESTRATOR_HOST="127.0.0.1"
export ORCHESTRATOR_PORT="5000"
export KAFKA_HOST="localhost"
export KAFKA_PORT="29092"
```

## Step 4: Start the Orchestrator/API

In one terminal, start the API server (which also acts as the orchestrator):

```bash
cd wp3_d3.2_saferlearn
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python src/api.py --datasets-path src/input-data/


cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn; .\.venv\Scripts\Activate.ps1; $env:KAFKA_HOST="localhost"; $env:KAFKA_PORT="29092"; Write-Host "Starting API/Orchestrator with Kafka at localhost:29092..."; python src/api.py --datasets-path src/input-data/
```

The API will start on `http://localhost:5000`.

## Step 5: Start Data Owners (Teachers)

Open **separate terminals** for each teacher. Each teacher needs:
- A unique RPyC port
- The same data format tag
- Access to models

### Terminal 2 - Teacher 1:
```bash
cd wp3_d3.2_saferlearn
source .venv/bin/activate
export RPYC_PORT=1244
python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1244 --nteachers 3


cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn; .\.venv\Scripts\Activate.ps1; $env:RPYC_PORT="1244"; $env:KAFKA_HOST="localhost"; $env:KAFKA_PORT="29092"; Write-Host "Starting Teacher 1 (Data Owner) on RPyC port 1244..." -ForegroundColor Green; python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1244 --nteachers 3
```

### Terminal 3 - Teacher 2:
```bash
cd wp3_d3.2_saferlearn
source .venv/bin/activate
export RPYC_PORT=1245
python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1245 --nteachers 3


cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn; .\.venv\Scripts\Activate.ps1; $env:RPYC_PORT="1245"; $env:KAFKA_HOST="localhost"; $env:KAFKA_PORT="29092"; Write-Host "Starting Teacher 2 (Data Owner) on RPyC port 1245..." -ForegroundColor Green; python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1245 --nteachers 3
```

### Terminal 4 - Teacher 3:
```bash
cd wp3_d3.2_saferlearn
source .venv/bin/activate
export RPYC_PORT=1246
python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1246 --nteachers 3


cd D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn; .\.venv\Scripts\Activate.ps1; $env:RPYC_PORT="1246"; $env:KAFKA_HOST="localhost"; $env:KAFKA_PORT="29092"; Write-Host "Starting Teacher 3 (Data Owner) on RPyC port 1246..." -ForegroundColor Green; python src/data_owner_cli.py --type Stub --data-format mnist --rpyc-port 1246 --nteachers 3
```

**Note**: You can start as many teachers as you have models available.

## Step 6: Verify Workers Are Registered

Check if your teachers are registered with the orchestrator:

```bash
curl http://localhost:5000/clients
```

You should see JSON output with your workers' information including their UUIDs, IPs, and ports.

## Step 7: Launch a PATE Job

Use the API to create a PATE job. You'll need the worker UUIDs from Step 6:

```bash
curl -X PUT -H 'Content-Type: application/json' \
  http://localhost:5000/job \
  --data '{
    "algorithm": "pate",
    "datatype": "mnist",
    "dataset": "MNIST",
    "workers": ["<uuid1>", "<uuid2>", "<uuid3>"],
    "nbClasses": 10,
    "dpValue": 0.5,
    "useDP": false
  }'
```


```powershell
$body = @{
    algorithm = "pate"
    datatype = "mnist"
    dataset = "MNIST"
    workers = @('8cc936b9-6a49-4c1c-8c6a-b0b47165e490', '8fcb95f2-5a8b-40fc-bb8f-8a3b74aa894c', 'dde4337a-2353-4cbf-ab81-42c19ac35583')
    nbClasses = 10
    dpValue = 1.0
    useDP = $true
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:5000/job -Method PUT -ContentType "application/json" -Body $body

Check Job Status
Invoke-WebRequest -Uri http://localhost:5000/jobs -Method GET | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json

Delete Job
   Invoke-WebRequest -Uri http://127.0.0.1:5000/jobs -Method DELETE

```

Or using the sample script (after updating worker UUIDs):
```bash
curl 'http://localhost:5000/job' -X PUT -H 'Content-Type: application/json' \
  --data-raw '{
    "algorithm":"pate",
    "datatype":"mnist",
    "dataset":"MNIST",
    "workers":["<uuid1>", "<uuid2>", "<uuid3>"],
    "nbClasses":10,
    "useDP":false,
    "dpValue":0.05
  }'
```

### Job Parameters Explained:
- **algorithm**: `"pate"` for clear PATE (no encryption) or `"fhe"` for FHE-encrypted PATE
- **datatype**: Should match the `--data-format` used when starting teachers (`"mnist"`)
- **dataset**: Name of the dataset (`"MNIST"`)
- **workers**: List of worker UUIDs (from `/clients` endpoint)
- **nbClasses**: Number of classes (10 for MNIST)
- **useDP**: Enable/disable differential privacy (`true`/`false`)
- **dpValue**: Differential privacy parameter (sigma value for noise)

## Step 8: Monitor the Job

### Check Job Status:
```bash
curl http://localhost:5000/jobs
```

### Get Specific Job Details:
```bash
curl http://localhost:5000/job/<job_uuid>
```

### View Logs:
Check the terminal outputs from:
- API/Orchestrator (Terminal 1)
- Each Teacher (Terminals 2, 3, 4, ...)

The orchestrator will:
1. Send parameters to teachers via Kafka
2. Teachers load models and predict on the public MNIST dataset
3. Teachers send votes back via Kafka
4. Orchestrator aggregates votes
5. Results are published to Kafka topic `model_pate`

## Step 9: Retrieve Results

Results are published to Kafka topic `model_pate`. You can consume them using:

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'model_pate',
    bootstrap_servers=['localhost:29092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    print(f"Sample {message.key}: Label {message.value}")
```

Or check the results using the model owner script (if available):
```bash
python src/model_owner.py
```

## Troubleshooting

### Issue: "No workers available"
- **Solution**: Make sure teachers are running and registered. Check with `curl http://localhost:5000/clients`

### Issue: "No free model found"
- **Solution**: Ensure models exist in `trained_nets_gpu/0/`, `trained_nets_gpu/1/`, etc.

### Issue: Kafka connection errors
- **Solution**: Verify Kafka is running: `docker ps | grep kafka`
- Check KAFKA_HOST and KAFKA_PORT environment variables

### Issue: Dataset not found
- **Solution**: Verify MNIST is in `src/input-data/MNIST/` and the path is correct

### Issue: Port already in use
- **Solution**: Use different RPyC ports for each teacher (1244, 1245, 1246, ...)

## Stopping Services

```bash
# Stop teachers: Ctrl+C in each teacher terminal

# Stop API: Ctrl+C in API terminal

# Stop Kafka:
docker compose -f ./kafka_docker.yml down
```

## Clean Up

To clear the database and start fresh:
```bash
# Clear workers
curl -X DELETE http://localhost:5000/clients

# Clear jobs
curl -X DELETE http://localhost:5000/jobs
```

