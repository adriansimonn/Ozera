#!/usr/bin/env python3
"""
Upload ozera-mini model to Modal volume with progress tracking.

Run:
    python backend/scripts/upload_base_models.py
"""

import os
import sys
import time
import threading

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def format_size(bytes_size):
    """Format bytes as human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def upload_mini_model():
    """Upload ozera-mini model to Modal volume with progress."""
    import modal

    MODELS_VOLUME_NAME = "ozera-models"
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BACKEND_DIR, "models")

    model_path = os.path.join(MODELS_DIR, "ozera-mini", "model.pt")
    remote_path = "base/ozera-mini/model.pt"

    # Verify model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False

    file_size = os.path.getsize(model_path)
    print(f"Uploading ozera-mini model: {format_size(file_size)}")
    print(f"  Local:  {model_path}")
    print(f"  Remote: /models/{remote_path}")
    print()

    # Get volume
    volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

    # Progress tracking
    upload_complete = threading.Event()
    start_time = time.time()

    def show_progress():
        """Show upload progress spinner."""
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while not upload_complete.is_set():
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            sys.stdout.write(f"\r{spinner[idx]} Uploading... {mins:02d}:{secs:02d} elapsed")
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner)
            time.sleep(0.1)

    # Start progress thread
    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()

    try:
        # Upload with batch_upload
        with volume.batch_upload(force=True) as batch:
            batch.put_file(model_path, remote_path)

        upload_complete.set()
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)

        print(f"\r✓ Upload complete! Total time: {mins:02d}:{secs:02d}         ")
        print(f"\nVerify with: modal volume ls {MODELS_VOLUME_NAME} base/ozera-mini/")
        return True

    except Exception as e:
        upload_complete.set()
        print(f"\r✗ Upload failed: {e}                    ")
        return False


if __name__ == "__main__":
    success = upload_mini_model()
    sys.exit(0 if success else 1)
