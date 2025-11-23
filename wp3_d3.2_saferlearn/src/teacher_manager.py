#!/usr/bin/env python3
"""Teacher Manager - Start and stop teachers programmatically for sequential batching"""

import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import utilities
from db_utilities import get_active_workers

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_BASE_PORT = 1244
DEFAULT_DATA_FORMAT = "mnist"
DEFAULT_ORCHESTRATOR_HOST = "127.0.0.1"
DEFAULT_ORCHESTRATOR_PORT = 5000
DEFAULT_KAFKA_HOST = "localhost"
DEFAULT_KAFKA_PORT = 29092


class TeacherProcess:
    """Represents a teacher process"""

    def __init__(self, teacher_id: int, rpyc_port: int, process: subprocess.Popen):
        self.teacher_id = teacher_id
        self.rpyc_port = rpyc_port
        self.process = process
        self.uuid: Optional[str] = None


def start_teachers(
    num_teachers: int,
    base_port: int = DEFAULT_BASE_PORT,
    data_format: str = DEFAULT_DATA_FORMAT,
    orchestrator_host: str = DEFAULT_ORCHESTRATOR_HOST,
    orchestrator_port: int = DEFAULT_ORCHESTRATOR_PORT,
    kafka_host: str = DEFAULT_KAFKA_HOST,
    kafka_port: int = DEFAULT_KAFKA_PORT,
    base_dir: Optional[str] = None,
) -> List[TeacherProcess]:
    """Start teacher processes programmatically

    Args:
        num_teachers: Number of teachers to start
        base_port: Starting RPyC port
        data_format: Data format (e.g., "mnist")
        orchestrator_host: Orchestrator host
        orchestrator_port: Orchestrator port
        kafka_host: Kafka host
        kafka_port: Kafka port
        base_dir: Base directory (defaults to script directory)

    Returns:
        List of TeacherProcess objects
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)

    python_exe = base_dir / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = base_dir / ".venv" / "bin" / "python"
        if not python_exe.exists():
            python_exe = "python"

    teacher_processes = []

    for i in range(num_teachers):
        teacher_id = i
        rpyc_port = base_port + i

        # Prepare environment
        env = os.environ.copy()
        env["RPYC_PORT"] = str(rpyc_port)
        env["KAFKA_HOST"] = kafka_host
        env["KAFKA_PORT"] = str(kafka_port)

        # Prepare command - use absolute path to script
        script_path = base_dir / "src" / "data_owner_cli.py"
        cmd = [
            str(python_exe),
            str(script_path),
            "--type",
            "Stub",
            "--data-format",
            data_format,
            "--rpyc-port",
            str(rpyc_port),
            "--nteachers",
            str(num_teachers),
            "--orchestrator-host",
            orchestrator_host,
            "--orchestrator-port",
            str(orchestrator_port),
            "pate-teacher",
        ]

        logger.info(f"Starting Teacher {teacher_id + 1} on port {rpyc_port}")

        # Start process in background
        process = subprocess.Popen(
            cmd,
            cwd=str(base_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        teacher_processes.append(TeacherProcess(teacher_id, rpyc_port, process))
        time.sleep(0.5)  # Small delay between starts

    return teacher_processes


def wait_for_teachers_registration(
    num_teachers: int,
    data_format: str = DEFAULT_DATA_FORMAT,
    timeout: int = 60,
    check_interval: float = 2.0,
) -> List[Dict]:
    """Wait for teachers to register with orchestrator

    Args:
        num_teachers: Number of teachers expected
        data_format: Data format to filter by
        timeout: Maximum time to wait (seconds)
        check_interval: Interval between checks (seconds)

    Returns:
        List of registered worker dictionaries
    """
    logger.info(f"Waiting for {num_teachers} teachers to register...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        active_workers = get_active_workers(datatype=data_format, state="online")

        if len(active_workers) >= num_teachers:
            logger.info(
                f"All {num_teachers} teachers registered! Found {len(active_workers)} workers"
            )
            return active_workers[:num_teachers]

        logger.debug(
            f"Found {len(active_workers)}/{num_teachers} teachers registered, waiting..."
        )
        time.sleep(check_interval)

    # Return what we have, even if not all registered
    active_workers = get_active_workers(datatype=data_format, state="online")
    logger.warning(
        f"Timeout waiting for teachers. Only {len(active_workers)}/{num_teachers} registered"
    )
    return active_workers[:num_teachers]


def stop_teachers(teacher_processes: List[TeacherProcess], timeout: int = 10) -> None:
    """Stop teacher processes

    Args:
        teacher_processes: List of TeacherProcess objects to stop
        timeout: Timeout for graceful shutdown
    """
    logger.info(f"Stopping {len(teacher_processes)} teachers...")

    # First, try graceful termination
    for teacher in teacher_processes:
        try:
            teacher.process.terminate()
        except Exception as e:
            logger.warning(f"Error terminating teacher {teacher.teacher_id}: {e}")

    # Wait for processes to terminate
    time.sleep(timeout)

    # Force kill any remaining processes
    for teacher in teacher_processes:
        if teacher.process.poll() is None:  # Still running
            try:
                logger.warning(f"Force killing teacher {teacher.teacher_id}")
                teacher.process.kill()
            except Exception as e:
                logger.warning(f"Error killing teacher {teacher.teacher_id}: {e}")

    # Wait a bit more for cleanup
    time.sleep(1)

    logger.info("All teachers stopped")


def get_rpyc_host() -> str:
    """Get the local RPyC host address"""
    return socket.gethostbyname(socket.gethostname())

