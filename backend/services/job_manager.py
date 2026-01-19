"""
Training job lifecycle management.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from typing import Dict, Optional, Callable

from api.schemas.training import JobStatus, TrainingProgress


# Configuration
JOBS_DIR = Path(__file__).parent.parent / "data" / "jobs"


def get_job_dir(job_id: str) -> Path:
    """Get the directory for a job."""
    return JOBS_DIR / job_id


def load_job_config(job_id: str) -> Optional[dict]:
    """Load job configuration from JSON file."""
    config_path = get_job_dir(job_id) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_job_config(job_id: str, config: dict) -> None:
    """Save job configuration to JSON file."""
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def load_job_progress(job_id: str) -> Optional[dict]:
    """Load job progress from JSON file."""
    progress_path = get_job_dir(job_id) / "progress.json"
    if not progress_path.exists():
        return None
    with open(progress_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_job_progress(job_id: str, progress: dict) -> None:
    """Save job progress to JSON file."""
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.json"
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, default=str)


def list_all_jobs() -> list[dict]:
    """List all jobs from the jobs directory."""
    if not JOBS_DIR.exists():
        return []

    jobs = []
    for job_dir in JOBS_DIR.iterdir():
        if job_dir.is_dir():
            config = load_job_config(job_dir.name)
            progress = load_job_progress(job_dir.name)
            if config:
                job_info = {
                    "job_id": job_dir.name,
                    "status": progress.get("status", JobStatus.PENDING) if progress else JobStatus.PENDING,
                    "model_name": config.get("model_name", ""),
                    "dataset_name": config.get("dataset_name", ""),
                    "created_at": config.get("created_at", ""),
                    "current_epoch": progress.get("current_epoch", 0) if progress else 0,
                    "total_epochs": config.get("epochs", 0),
                }
                jobs.append(job_info)

    # Sort by created_at descending
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return jobs


class JobManager:
    """Manages training jobs in background threads."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.active_jobs: Dict[str, Thread] = {}
        self.cancel_flags: Dict[str, Event] = {}
        self._lock = Event()  # Simple lock for single-job constraint

    def is_job_running(self) -> bool:
        """Check if any job is currently running."""
        return any(t.is_alive() for t in self.active_jobs.values())

    def get_running_job_id(self) -> Optional[str]:
        """Get the ID of the currently running job, if any."""
        for job_id, thread in self.active_jobs.items():
            if thread.is_alive():
                return job_id
        return None

    def create_job(
        self,
        dataset_id: str,
        dataset_name: str,
        model_config: str,
        model_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seq_len: int,
        estimated_minutes: float,
        estimated_cost_usd: float,
    ) -> str:
        """Create a new training job (does not start it)."""
        job_id = str(uuid.uuid4())[:8]

        config = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "model_config": model_config,
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seq_len": seq_len,
            "estimated_minutes": estimated_minutes,
            "estimated_cost_usd": estimated_cost_usd,
            "created_at": datetime.utcnow().isoformat(),
        }
        save_job_config(job_id, config)

        # Initialize progress
        progress = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "current_epoch": 0,
            "total_epochs": epochs,
            "current_step": 0,
            "total_steps": 0,
            "train_loss": None,
            "val_loss": None,
            "train_ppl": None,
            "val_ppl": None,
            "elapsed_seconds": 0,
            "estimated_remaining_seconds": int(estimated_minutes * 60),
            "last_update": datetime.utcnow().isoformat(),
            "error_message": None,
        }
        save_job_progress(job_id, progress)

        return job_id

    def start_job(self, job_id: str, worker_fn: Callable[[str, Event], None]) -> bool:
        """
        Start a training job in a background thread.

        Args:
            job_id: The job ID to start
            worker_fn: Function that takes (job_id, cancel_event) and runs training

        Returns:
            True if job started, False if another job is already running
        """
        if self.is_job_running():
            return False

        # Create cancel flag
        cancel_event = Event()
        self.cancel_flags[job_id] = cancel_event

        # Start worker thread
        thread = Thread(
            target=worker_fn,
            args=(job_id, cancel_event),
            daemon=True,
            name=f"training-{job_id}"
        )
        self.active_jobs[job_id] = thread
        thread.start()

        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Signal a job to cancel.

        Returns:
            True if cancel signal sent, False if job not found
        """
        if job_id not in self.cancel_flags:
            return False

        self.cancel_flags[job_id].set()

        # Update progress to cancelled
        progress = load_job_progress(job_id)
        if progress and progress.get("status") == JobStatus.RUNNING.value:
            progress["status"] = JobStatus.CANCELLED.value
            progress["last_update"] = datetime.utcnow().isoformat()
            save_job_progress(job_id, progress)

        return True

    def get_job_status(self, job_id: str) -> Optional[TrainingProgress]:
        """Get current progress for a job."""
        progress = load_job_progress(job_id)
        if not progress:
            return None
        return TrainingProgress(**progress)

    def cleanup_job(self, job_id: str) -> None:
        """Clean up after a job completes."""
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        if job_id in self.cancel_flags:
            del self.cancel_flags[job_id]


# Global singleton
_job_manager = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
