"""
Modal app definition for cloud GPU training.

This module defines the Modal app, Docker image, GPU configurations,
and the main training function that runs on Modal's cloud infrastructure.
"""
import os
from typing import Optional

import modal

# Modal app definition
app = modal.App("ozera-training")

# Volume names (must match modal_volumes.py)
DATASETS_VOLUME_NAME = "ozera-datasets"
MODELS_VOLUME_NAME = "ozera-models"

# Create or get volumes
datasets_volume = modal.Volume.from_name(DATASETS_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

# Get backend directory path
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")

# Docker image with all required dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tiktoken>=0.5.0",
        "requests>=2.28.0",
        "safetensors>=0.4.0",
        "packaging",
    )
    # Add only the necessary backend code directories
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
    .add_local_dir(os.path.join(BACKEND_DIR, "services"), remote_path="/app/backend/services")
    .add_local_dir(os.path.join(BACKEND_DIR, "api"), remote_path="/app/backend/api")
    .add_local_dir(os.path.join(BACKEND_DIR, "middleware"), remote_path="/app/backend/middleware")
)

# GPU configurations (string format - new Modal API)
GPU_CONFIGS = {
    "t4": "T4",
    "a10g": "A10G",
    "a100": "A100-40GB",
}

def send_progress_update(
    backend_url: str,
    webhook_secret: str,
    job_id: str,
    progress_data: dict,
) -> bool:
    """
    Send progress update to the backend webhook.

    Args:
        backend_url: Backend URL (e.g., https://your-backend.com)
        webhook_secret: Shared secret for authentication
        job_id: Training job ID
        progress_data: Progress data dict

    Returns:
        True if successful
    """
    import requests

    try:
        response = requests.post(
            f"{backend_url}/webhooks/training/{job_id}/progress",
            json=progress_data,
            headers={"X-Modal-Secret": webhook_secret},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to send progress update: {e}")
        return False


def job_should_continue(backend_url: str, webhook_secret: str, job_id: str) -> bool:
    """
    Whether the backend still wants the job to run (it is QUEUED or RUNNING).

    A job cancelled while it was being submitted can still be spawned, and stopping it on
    Modal can fail, so the job checks for itself. If the backend can't be reached the job
    keeps going, so an outage doesn't kill running jobs.
    """
    import requests

    try:
        response = requests.get(
            f"{backend_url}/webhooks/training/{job_id}/status",
            headers={"X-Modal-Secret": webhook_secret},
            timeout=10,
        )
        if response.status_code == 200:
            return bool(response.json().get("should_continue", True))
        print(f"Job status check returned {response.status_code}")
    except Exception as e:
        print(f"Failed to check job status: {e}")
    return True


@app.function(
    image=training_image,
    volumes={
        "/datasets": datasets_volume,
        "/models": models_volume,
    },
    gpu="T4",  # Default, will be overridden by spawn() call
    timeout=7200,  # 2 hour timeout
    scaledown_window=2,  # One-off jobs: don't pay for an idle container afterwards
    secrets=[modal.Secret.from_name("ozera-secrets")],
)
def run_training_on_modal(
    job_id: str,
    user_id: int,
    dataset_id: str,
    model_config: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len: int,
    dataset_name: str,
    gpu_type: str = "a10g",
) -> dict:
    """
    Run a training job on Modal cloud GPU.

    Args:
        job_id: Unique job identifier
        user_id: User ID (for data isolation)
        dataset_id: Dataset ID
        model_config: Model configuration ('nano' or 'mini')
        model_name: Name for the trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        seq_len: Sequence length
        dataset_name: Human-readable dataset name
        gpu_type: GPU type ('t4', 'a10g', 'a100')

    Returns:
        Dict with status, actual_minutes, and optional error_message
    """
    return _run_training_impl(
        job_id, user_id, dataset_id, model_config, model_name,
        epochs, batch_size, learning_rate, seq_len, dataset_name, gpu_type
    )


# Separate functions for each GPU type to allow proper GPU selection
@app.function(
    image=training_image,
    volumes={
        "/datasets": datasets_volume,
        "/models": models_volume,
    },
    gpu="T4",
    timeout=7200,
    scaledown_window=2,  # One-off jobs: don't pay for an idle container afterwards
    secrets=[modal.Secret.from_name("ozera-secrets")],
)
def run_training_t4(
    job_id: str,
    user_id: int,
    dataset_id: str,
    model_config: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len: int,
    dataset_name: str,
) -> dict:
    """Run training on T4 GPU."""
    return _run_training_impl(
        job_id, user_id, dataset_id, model_config, model_name,
        epochs, batch_size, learning_rate, seq_len, dataset_name, "t4"
    )


@app.function(
    image=training_image,
    volumes={
        "/datasets": datasets_volume,
        "/models": models_volume,
    },
    gpu="A10G",
    timeout=7200,
    scaledown_window=2,  # One-off jobs: don't pay for an idle container afterwards
    secrets=[modal.Secret.from_name("ozera-secrets")],
)
def run_training_a10g(
    job_id: str,
    user_id: int,
    dataset_id: str,
    model_config: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len: int,
    dataset_name: str,
) -> dict:
    """Run training on A10G GPU."""
    return _run_training_impl(
        job_id, user_id, dataset_id, model_config, model_name,
        epochs, batch_size, learning_rate, seq_len, dataset_name, "a10g"
    )


@app.function(
    image=training_image,
    volumes={
        "/datasets": datasets_volume,
        "/models": models_volume,
    },
    gpu="A100-40GB",
    timeout=7200,
    scaledown_window=2,  # One-off jobs: don't pay for an idle container afterwards
    secrets=[modal.Secret.from_name("ozera-secrets")],
)
def run_training_a100(
    job_id: str,
    user_id: int,
    dataset_id: str,
    model_config: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len: int,
    dataset_name: str,
) -> dict:
    """Run training on A100 GPU."""
    return _run_training_impl(
        job_id, user_id, dataset_id, model_config, model_name,
        epochs, batch_size, learning_rate, seq_len, dataset_name, "a100"
    )


def _run_training_impl(
    job_id: str,
    user_id: int,
    dataset_id: str,
    model_config: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len: int,
    dataset_name: str,
    gpu_type: str,
) -> dict:
    """Internal implementation for all GPU types."""
    import sys
    import time
    function_start = time.time()
    sys.path.insert(0, "/app/backend")

    from core.model_names import model_folder
    from services.training_logic import TrainingConfig, TrainingProgress, run_training

    # Get environment variables
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    webhook_secret = os.environ.get("MODAL_WEBHOOK_SECRET", "")

    def progress_callback(progress: TrainingProgress):
        progress_data = {
            "status": progress.status,
            "current_epoch": progress.current_epoch,
            "total_epochs": progress.total_epochs,
            "train_loss": progress.train_loss,
            "val_loss": progress.val_loss,
            "train_ppl": progress.train_ppl,
            "val_ppl": progress.val_ppl,
            "elapsed_seconds": progress.elapsed_seconds,
            "error_message": progress.error_message,
        }
        send_progress_update(backend_url, webhook_secret, job_id, progress_data)

    def cancelled() -> bool:
        return not job_should_continue(backend_url, webhook_secret, job_id)

    # Construct paths
    # Handle generic datasets (prefixed with 'generic:') vs user-uploaded datasets
    if dataset_id.startswith("generic:"):
        actual_dataset_id = dataset_id[8:]  # Remove 'generic:' prefix
        dataset_path = f"/datasets/generic/{actual_dataset_id}/raw.txt"
    else:
        dataset_path = f"/datasets/{user_id}/{dataset_id}/raw.txt"

    try:
        model_output_path = f"/models/{model_folder(user_id, model_name)}"
    except ValueError as e:
        # The API only accepts valid names; never write outside the user's folder
        status, error_message = "failed", str(e)
    else:
        if cancelled():
            # Cancelled before it started (e.g. while it was being submitted)
            status, error_message = "cancelled", None
        else:
            config = TrainingConfig(
                job_id=job_id,
                user_id=user_id,
                dataset_path=dataset_path,
                model_output_path=model_output_path,
                model_config=model_config,
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seq_len=seq_len,
                dataset_name=dataset_name,
            )
            status, _, error_message = run_training(config, progress_callback, should_stop=cancelled)

            # Commit model volume changes
            models_volume.commit()

    # Bill the container time from function start through saving the model, not just the
    # training loop
    actual_minutes = (time.time() - function_start) / 60

    # Send final update (a cancelled job was already settled by the cancel)
    if status != "cancelled":
        final_progress = {
            "status": status,
            "current_epoch": epochs if status == "completed" else 0,
            "total_epochs": epochs,
            "actual_minutes": actual_minutes,
            "error_message": error_message,
        }
        send_progress_update(backend_url, webhook_secret, job_id, final_progress)

    return {
        "status": status,
        "actual_minutes": actual_minutes,
        "error_message": error_message,
    }


def get_training_function(gpu_type: str):
    """
    Get the appropriate training function for a GPU type.

    This looks up the deployed function on Modal. The app must be deployed
    first using: modal deploy backend/services/modal_worker.py

    Args:
        gpu_type: GPU type ('t4', 'a10g', 'a100')

    Returns:
        Modal function reference for the specified GPU
    """
    function_names = {
        "t4": "run_training_t4",
        "a10g": "run_training_a10g",
        "a100": "run_training_a100",
    }
    func_name = function_names.get(gpu_type, "run_training_a10g")

    # Look up the deployed function by name
    return modal.Function.from_name("ozera-training", func_name)


# For local testing
if __name__ == "__main__":
    # Deploy the app
    print("Deploying Ozera training app to Modal...")
    print("Run: modal deploy backend/services/modal_worker.py")
