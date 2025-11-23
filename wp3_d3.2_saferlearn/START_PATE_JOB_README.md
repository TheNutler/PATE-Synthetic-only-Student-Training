# Starting PATE Job with Differential Privacy

This directory contains PowerShell scripts to automate starting and managing PATE jobs with differential privacy enabled.

## Scripts

### 1. `start_pate_job.ps1`
Main script that starts the orchestrator/API server:
- **Kafka** (via Docker Compose)
- **Orchestrator/API Server** (running in background with logs)

**Note:** Teachers are started automatically by the orchestrator in sequential batches when you create a PATE job. No need to start teachers manually!

### 2. `create_pate_job.ps1`
Helper script that creates a PATE job via the API:
- Fetches active worker UUIDs from the orchestrator
- Creates a job request with DP enabled
- Submits the job via PUT request


### Get Results
`cd wp3_d3.2_saferlearn; .\.venv\Scripts\Activate.ps1; python get_pate_results.py`

`cd "D:\Documents\TUWien\MA\Thesis\PATE\wp3_d3.2_saferlearn"; .\.venv\Scripts\Activate.ps1; python train_student_model.py --pate-results pate_results.csv`


### 3. `stop_pate_job.ps1`
Helper script to stop Kafka and provide instructions for stopping other components.

## Usage

### Step 1: Start All Components

```powershell
cd wp3_d3.2_saferlearn

# Use smaller batches with longer delays
.\start_pate_job.ps1 -NumTeachers 30 -BatchSize 3 -BatchDelaySeconds 5

cd wp3_d3.2_saferlearn
>> .\kill_all_pate_processes.ps1
```

Or with custom parameters:
```powershell
.\start_pate_job.ps1 -NumTeachers 30 -DpValue 0.5 -BasePort 1244
```

**Parameters:**
- `-NumTeachers`: Number of teachers to start (default: 30)
- `-DpValue`: Differential privacy parameter value (default: 0.5)
- `-BasePort`: Starting RPyC port for teachers (default: 1244)
- `-KafkaPort`: Kafka port (default: 29092)
- `-OrchestratorHost`: Orchestrator host (default: 127.0.0.1)
- `-OrchestratorPort`: Orchestrator port (default: 5000)
- `-DataFormat`: Data format (default: "mnist")
- `-Dataset`: Dataset name (default: "MNIST")
- `-NumClasses`: Number of classes (default: 10)

### Step 2: Create PATE Job with Sequential Batching

Create a PATE job and specify how many teachers and batch size:

```powershell
.\create_pate_job.ps1 -NumTeachers 30 -BatchSize 5 -DpValue 0.5 -UseDP $true
```

**Parameters:**
- `-NumTeachers`: Total number of teachers to use (default: 30)
- `-BatchSize`: Number of teachers per batch (default: 5)
- `-DpValue`: Differential privacy parameter value (default: 0.5)
- `-UseDP`: Enable/disable differential privacy (default: $true)

**What happens:**
1. The orchestrator will start teachers in batches (e.g., 5 at a time)
2. Each batch processes the dataset and publishes votes
3. Orchestrator collects votes from that batch
4. Teachers are stopped
5. Process repeats for the next batch
6. All votes are aggregated at the end

**Example:**
- 30 teachers, batch size 5 = 6 batches
- Each batch: start 5 teachers → collect votes → stop teachers
- Finally: aggregate all votes from all 30 teachers

**Parameters:**
- `-DpValue`: Differential privacy parameter value (default: 0.5)
- `-UseDP`: Enable/disable differential privacy (default: $true)
- `-NumSamples`: Number of samples to process (default: 1000)
- `-Algorithm`: Algorithm to use, "pate" for clear PATE or "fhe" for FHE-encrypted (default: "pate")

### Step 4: Monitor the Job

Check job status:
```powershell
curl http://localhost:5000/jobs
```

Get specific job details:
```powershell
curl http://localhost:5000/job/<job_uuid>
```

**Monitor Logs:**
```powershell
# View orchestrator log
.\view_teacher_logs.ps1 -Orchestrator

# Follow orchestrator log live (updates continuously)
Get-Content teacher_logs\orchestrator_api.log -Tail 50 -Wait

# View recent logs from all teachers
.\view_teacher_logs.ps1

# View specific teacher log
.\view_teacher_logs.ps1 -TeacherNum 1

# View more lines
.\view_teacher_logs.ps1 -TeacherNum 1 -Lines 100
```

### Step 5: Stop Components

Stop Kafka:
```powershell
.\stop_pate_job.ps1
```

Stop teachers and orchestrator:
- Press `Ctrl+C` in each terminal window
- Or close the terminal windows manually

### 4. `view_teacher_logs.ps1`
Helper script to view teacher logs:
- View logs from all teachers
- View specific teacher log
- Configure number of lines to display

## Background Processes

The script runs all components in the background without opening any terminal windows:
- **No visible windows** are opened at all
- All processes (orchestrator + teachers) run silently in the background
- All logs are saved to `teacher_logs/` directory:
  - Orchestrator: `orchestrator_api.log`
  - Teachers: `teacher_<number>_port_<port>.log`
- Process IDs are saved for easy cleanup:
  - Orchestrator: `orchestrator_pid.txt`
  - Teachers: `teacher_pids.txt`

**Benefits:**
- No desktop clutter from 30 terminal windows
- Reduced system resource usage
- Easier to monitor via logs
- Prevents crashes from too many windows

## Prerequisites

1. **Trained Models**: Ensure you have 30 trained models in `trained_nets_gpu/0/` through `trained_nets_gpu/29/`
2. **Virtual Environment**: Activate your Python virtual environment (the script does this automatically)
3. **Docker**: Ensure Docker is running for Kafka

## Troubleshooting

### Teachers Not Starting
**Problem:** Teachers fail to start in a batch.

**Solution:**
- Check orchestrator logs: `Get-Content teacher_logs\orchestrator_api.log -Tail 50`
- Verify models exist in `trained_nets_gpu/0/` through `trained_nets_gpu/29/`
- Check Kafka is running: `docker ps | grep kafka`
- Try smaller batch size: `-BatchSize 3`

### Teachers Not Registering
- Check that all teachers have models in their respective directories
- Verify Kafka is running: `docker ps | grep kafka`
- Check teacher logs: `.\view_teacher_logs.ps1 -TeacherNum 1`

### Orchestrator Not Starting
- Verify port 5000 is not in use
- Check orchestrator log: `.\view_teacher_logs.ps1 -Orchestrator`

### Job Creation Fails
- Ensure all teachers are registered (check `/clients` endpoint)
- Verify the data format matches what teachers are using
- Check that you have enough workers for the job

## Example: Complete Workflow

```powershell
# 1. Start all components
.\start_pate_job.ps1 -NumTeachers 30 -DpValue 0.5

# 2. Wait ~30 seconds for registration, then check workers
curl http://localhost:5000/clients

# 3. Create job with DP enabled
.\create_pate_job.ps1 -DpValue 0.5 -UseDP $true

# 4. Monitor job status
curl http://localhost:5000/jobs

# 5. When done, stop Kafka
.\stop_pate_job.ps1
# Then manually stop orchestrator and teachers (Ctrl+C in each terminal)
```

## Notes

- **Sequential batching** - teachers are started/stopped automatically per batch
- Only the orchestrator runs continuously in the background
- Teachers are temporary - started when needed, stopped after voting
- All votes from all batches are aggregated together
- No memory issues - only a few teachers run at once
- Logs are saved to `teacher_logs/` directory
- Monitor orchestrator logs: `Get-Content teacher_logs\orchestrator_api.log -Tail 50 -Wait`

## Log Files

All logs are automatically created in `teacher_logs/`:
- **Orchestrator**: `orchestrator_api.log`
- **Teachers**: `teacher_<number>_port_<port>.log`
- Example: `teacher_1_port_1244.log`, `teacher_2_port_1245.log`, etc.
- Logs include all output (stdout and stderr)
- Use `.\view_teacher_logs.ps1` to view them easily

**Example log viewing:**
```powershell
# View orchestrator log
.\view_teacher_logs.ps1 -Orchestrator

# View specific teacher
.\view_teacher_logs.ps1 -TeacherNum 1

# Follow orchestrator log live
Get-Content teacher_logs\orchestrator_api.log -Tail 50 -Wait
```

