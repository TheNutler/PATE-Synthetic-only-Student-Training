# PATE Framework - Complete Reading Plan

This document provides a systematic order for reading through the codebase to fully understand the PATE (Private Aggregation of Teacher Ensembles) framework project.

## Project Overview

**Saferlearn** is a secure collaborative learning framework that uses PATE, MPC (Multi-Party Computation), and Federated Learning to enable privacy-preserving model training. The framework allows multiple data owners (teachers) to collaborate without sharing their private training data.

**Key Components:**
- **Orchestrator/API**: Coordinates jobs and manages communication
- **Data Owners (Teachers)**: Hold private models and datasets
- **Model Owner (Student)**: Receives aggregated predictions
- **Kafka**: Message broker for communication
- **RPyC**: Remote procedure calls for direct communication

---

## Phase 1: Foundation & Documentation (Start Here)

**Goal:** Understand what the project does and how to use it.

### 1.1 Overview Documentation
1. **`README.md`**
   - Project purpose and high-level architecture
   - Installation and basic setup
   - Quick start commands

2. **`RUN_PATE_MNIST.md`** (Currently open file)
   - Step-by-step guide for running PATE with MNIST
   - Complete workflow from setup to execution
   - **Important:** This shows the end-to-end user workflow

3. **`ROADMAP.md`** (Optional)
   - Architecture notes
   - Known issues and future plans

### 1.2 Configuration & Environment
4. **`src/config.py`**
   - All configuration constants
   - Environment variables
   - Default paths and ports
   - **Key concepts:** Ports, Kafka settings, model paths, dataset paths

5. **`requirements.txt`**
   - Project dependencies
   - Python packages needed

---

## Phase 2: Core Infrastructure

**Goal:** Understand the basic building blocks and communication mechanisms.

### 2.1 Data Structures & Models
6. **`src/saferlearn.py`**
   - Core data structures: `Dataset`, `MPCParameters`
   - Protocol enum: `SecureCollaborativeLearning` (PATE, PATE_MPC, PATE_FHE, FL)
   - **Why first:** These are used throughout the codebase

### 2.2 Database & Storage
7. **`src/db_utilities.py`**
   - SQLite database for workers and jobs
   - Worker registration and state management
   - Job tracking
   - **Key functions:** `get_active_workers()`, `register_job()`, `update_db_worker()`

### 2.3 Communication Utilities
8. **`src/utilities.py`**
   - Kafka producer/consumer helpers
   - Message publishing/consuming
   - HTTP requests to orchestrator
   - Topic management
   - **Key functions:** `publish_message()`, `consume_topic()`, `register_worker_to_orchestrator()`

---

## Phase 3: Main Entry Points & Orchestration

**Goal:** Understand how the system starts and coordinates work.

### 3.1 API Server (Orchestrator Entry Point)
9. **`src/api.py`**
   - Flask REST API server
   - Endpoints: `/clients`, `/job`, `/jobs`
   - Accepts job requests and coordinates execution
   - **Key endpoints:**
     - `PUT /job` - Create a new job
     - `GET /clients` - List registered workers
     - `GET /jobs` - List jobs
   - **Important:** This is the main entry point for starting jobs

### 3.2 Job Orchestration
10. **`src/orchestrator.py`**
    - `Job` class - represents a collaborative learning job
    - Job creation for different algorithms (PATE, FHE, MPC, FL)
    - Communication with teachers via Kafka
    - Vote aggregation
    - **Key methods:**
      - `create_job_clear()` - PATE without encryption
      - `create_job()` - PATE with FHE (Fully Homomorphic Encryption)
      - `create_job_mpc()` - PATE with MPC
    - **Critical:** Understand the flow of how orchestrator sends parameters to teachers and receives votes

---

## Phase 4: Data Owner (Teacher) Implementation

**Goal:** Understand how teachers (data owners) work and interact with the system.

### 4.1 Abstract Base Class
11. **`src/usecases/data_owner_abstract_class.py`**
    - `DataOwner` abstract base class (inherits from `rpyc.Service`)
    - Core interface all data owners must implement
    - State management (WAITING, RUNNING, DONE)
    - Model selection and locking
    - Dataset loading
    - **Key methods:**
      - `exposed_get_state()` - RPyC exposed method
      - `exposed_run()` - Execute teacher predictions
      - `select_available_model()` - Pick an unused model
      - `get_dataset()` - Load dataset for predictions
    - **Critical:** This defines the contract for all data owners

### 4.2 Example Implementation
12. **`src/usecases/data_owner_example.py`**
    - `ThalesDataOwner` - Concrete implementation of `DataOwner`
    - `UCStubModel` - Neural network model architecture
    - Dataset loading for different data types
    - Model prediction logic
    - **Why important:** Shows how to implement a data owner

### 4.3 Use Case Configuration
13. **`src/usecases/UC_stub.py`**
    - Dataset path configurations
    - Use case specific settings

### 4.4 Data Owner CLI
14. **`src/data_owner_cli.py`**
    - Command-line interface for starting a teacher
    - RPyC server setup
    - Registration with orchestrator
    - Starts the `DataOwner` service
    - **Key:** This is how you launch a teacher instance

---

## Phase 5: Model Owner (Student)

**Goal:** Understand how the student receives and processes aggregated results.

15. **`src/model_owner.py`**
    - Consumes aggregated predictions from Kafka
    - Handles results for different algorithms:
      - `get_results()` - PATE results
      - `get_he_results()` - FHE results
      - `get_mpc_results()` - MPC results
    - **Flow:** Waits for orchestrator to send job parameters, then consumes results from Kafka topic

---

## Phase 6: Advanced Features

**Goal:** Understand MPC and Computing Parties.

### 6.1 Computing Parties (MPC)
16. **`src/computing_party.py`**
    - `WorkerService` - RPyC service for computing parties
    - Used in MPC (Multi-Party Computation) aggregation
    - Separate from data owners - these are the parties that perform secure computation

### 6.2 MPC Programs (Optional - Advanced)
17. **`pate_aggregation.mpc`**
    - MP-SPDZ secure computation program
    - Aggregates votes securely using MPC
18. **`pate_aggregation_one_hot.mpc`**
    - One-hot encoded aggregation variant
19. **`pate_aggregation_privacy_guardian.mpc`**
    - Privacy guardian variant

---

## Phase 7: Training & Setup Scripts

**Goal:** Understand how to prepare models and datasets.

20. **`train_mnist_models.py`**
    - Script to train multiple teacher models
    - Uses `UCStubModel` architecture
    - Saves models to numbered directories
    - **Important:** Shows the training process for teachers

21. **`setup_pate_mnist.py`**
    - Setup script for MNIST experiments
    - Creates necessary directories

22. **`src/download-mnist.py`** (Optional)
    - Dataset download utility

---

## Phase 8: Docker & Deployment (Optional)

**Goal:** Understand containerization and deployment.

23. **`Dockerfile`**
    - Container image definition

24. **`docker-compose.yml`** / **`docker-compose_distributed.yml`**
    - Multi-container setup
    - Kafka and Zookeeper configuration

25. **`kafka_docker.yml`**
    - Kafka-only Docker Compose setup

---

## Reading Strategy & Tips

### Recommended Reading Order:
1. **Start with Phase 1** - Get the big picture
2. **Phase 2** - Understand infrastructure
3. **Phase 3** - See how it all connects
4. **Phase 4** - Deep dive into teachers (most complex part)
5. **Phase 5** - Understand result aggregation
6. **Phase 6-8** - Advanced topics as needed

### Key Concepts to Understand:

1. **Communication Flow:**
   - Orchestrator → Kafka → Teachers (send parameters)
   - Teachers → Kafka → Orchestrator (send votes)
   - Orchestrator → Kafka → Model Owner (send results)

2. **Component Relationships:**
   - **API** (`api.py`) creates **Jobs** (`orchestrator.py`)
   - **Jobs** communicate with **DataOwners** via Kafka
   - **DataOwners** load models and make predictions
   - **Orchestrator** aggregates votes
   - **Model Owner** consumes final results

3. **Data Flow:**
   - Public dataset → Teachers → Predictions (votes)
   - Votes → Orchestrator → Aggregated labels
   - Aggregated labels → Model Owner

### Files You Can Skip Initially:
- `main.py` - Just a hello world placeholder
- `src/usecases/aminer_deep.py` - Specialized use case
- `src/usecases/data_owner.backup.py` - Backup file
- C++ files (`pate-teacher.cpp`, etc.) - Low-level implementations
- Kafka-docker subdirectory - Just Kafka setup

### Questions to Answer as You Read:

1. How does a teacher register with the orchestrator?
2. How are job parameters sent to teachers?
3. How do teachers load models and make predictions?
4. How are votes aggregated in PATE?
5. How does differential privacy noise get added?
6. What's the difference between clear PATE and FHE PATE?
7. How does MPC aggregation work?

### Debugging Tips:

- Check Kafka topics: `docker exec -it <kafka-container> kafka-topics --list`
- View worker database: SQLite browser for `worker.db`
- Check logs: Each component logs to stdout
- API endpoints: Use `curl` or browser for testing

---

## Quick Reference: Key Files by Role

| Role | Files |
|------|-------|
| **Entry Points** | `src/api.py`, `src/data_owner_cli.py`, `src/model_owner.py` |
| **Core Logic** | `src/orchestrator.py`, `src/usecases/data_owner_abstract_class.py` |
| **Infrastructure** | `src/config.py`, `src/db_utilities.py`, `src/utilities.py`, `src/saferlearn.py` |
| **Implementations** | `src/usecases/data_owner_example.py`, `src/computing_party.py` |
| **Documentation** | `README.md`, `RUN_PATE_MNIST.md` |
| **Setup** | `train_mnist_models.py`, `setup_pate_mnist.py` |

---

## Next Steps After Reading

1. **Run the system** following `RUN_PATE_MNIST.md`
2. **Add print statements** to trace execution flow
3. **Read logs** to understand runtime behavior
4. **Experiment** with different parameters
5. **Implement a new use case** by extending `DataOwner` class

---

*Last updated: Based on current codebase structure*

