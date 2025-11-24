# Run the PATE MNIST job

Use this document as the single source of truth for preparing teachers, running the orchestrator job, retrieving the votes, and training the student model. Every command below assumes you start in the repository root (`D:\Documents\TUWien\MA\Thesis\PATE`) and have the Python virtual environment set up in `.venv/`.

## 1. Reset teacher checkpoints

Remove any previously trained teacher directories so you always start from a clean slate.

```powershell
cd wp3_d3.2_saferlearn; Get-ChildItem -Path trained_nets_gpu -Directory | Remove-Item -Recurse -Force
```

## 2. Train the MNIST teachers

Train 250 teachers (10 epochs, batch size 64, deterministic seed).

```powershell
cd wp3_d3.2_saferlearn; python train_mnist_models.py --num-models 250 --epochs 10 --batch-size 64 --seed 42
```

## 3. Start the PATE environment

Start Kafka, orchestrator, teachers, and supporting services with the provided helper script. Leave it running.

```powershell
.\start_pate_job.ps1
```

## 4. Launch the aggregation job

Create the job with 250 teachers processed in batches of 10.

```powershell
.\create_pate_job.ps1 -NumTeachers 250 -BatchSize 10
```

## 5. Retrieve aggregated votes

Activate the virtual environment and export the PATE aggregation to `pate_results.csv`.

```powershell
.\.venv\Scripts\Activate.ps1; python get_pate_results.py
```

## 6. Train the student model

Train a student on the teacher votes file you just generated (still inside the virtual environment).

```powershell
.\.venv\Scripts\Activate.ps1; python train_student_model.py --pate-results pate_results.csv
```

## 7. Shut everything down

Force-stop the orchestrator, teachers, and related services to free ports and resources.

```powershell
.\kill_all_pate_processes.ps1
```

Keep this file updated if the workflow changes—any other markdown guides for running PATE should be removed so this remains the only canonical runbook.

