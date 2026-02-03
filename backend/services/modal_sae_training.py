"""
Modal worker for SAE training on cloud GPUs.

Trains Sparse Autoencoders on ozera-nano and ozera-mini activations
using OpenWebText dataset for diverse inputs.

Deployment:
    modal deploy backend/services/modal_sae_training.py

Usage (from Python):
    from services.modal_sae_training import train_sae_remote
    result = train_sae_remote("nano", 0, "residual")
"""

import os
import modal

# Modal app definition
app = modal.App("ozera-sae-training")

# Volume for storing trained SAEs
SAES_VOLUME_NAME = "ozera-saes"
MODELS_VOLUME_NAME = "ozera-models"

saes_volume = modal.Volume.from_name(SAES_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

# Get backend directory path
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")

# Docker image with SAE training dependencies
sae_training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tiktoken>=0.5.0",
        "safetensors>=0.4.0",
        "h5py>=3.9.0",           # HDF5 for activation storage
        "datasets>=2.14.0",      # HuggingFace datasets for The Pile
        "huggingface_hub",       # For dataset access
    )
    # Add backend code
    .add_local_dir(os.path.join(BACKEND_DIR, "core"), remote_path="/app/backend/core")
    .add_local_dir(os.path.join(BACKEND_DIR, "inference"), remote_path="/app/backend/inference")
    .add_local_dir(os.path.join(BACKEND_DIR, "scripts", "sae_training"), remote_path="/app/backend/scripts/sae_training")
)

# Model configurations
MODEL_CONFIGS = {
    "nano": {
        "num_layers": 6,
        "d_model": 192,
        "sae_num_steps": 50_000,
    },
    "mini": {
        "num_layers": 8,
        "d_model": 512,
        "sae_num_steps": 50_000,  # Reduced from 100k - same as nano
    },
}

# Checkpoint interval - save every N steps to prevent progress loss
CHECKPOINT_INTERVAL = 2_500


@app.function(
    image=sae_training_image,
    volumes={
        "/saes": saes_volume,
        "/models": models_volume,
    },
    gpu="L4",
    timeout=86400,  # 24 hours
    memory=32768,   # 32GB RAM
)
def train_single_sae_l4(
    model_name: str,
    layer: int,
    activation_type: str,
    num_activations: int = 2_000_000,
) -> dict:
    """Train a single SAE on L4 GPU ($0.80/hr - most cost effective)."""
    return _train_sae_impl(model_name, layer, activation_type, num_activations)


@app.function(
    image=sae_training_image,
    volumes={
        "/saes": saes_volume,
        "/models": models_volume,
    },
    gpu="A10G",
    timeout=86400,  # 24 hours
    memory=32768,   # 32GB RAM for activation buffer
)
def train_single_sae_a10g(
    model_name: str,
    layer: int,
    activation_type: str,
    num_activations: int = 2_000_000,
) -> dict:
    """Train a single SAE on A10G GPU."""
    return _train_sae_impl(model_name, layer, activation_type, num_activations)


@app.function(
    image=sae_training_image,
    volumes={
        "/saes": saes_volume,
        "/models": models_volume,
    },
    gpu="A100-40GB",
    timeout=86400,  # 24 hours
    memory=65536,   # 64GB RAM
)
def train_single_sae_a100(
    model_name: str,
    layer: int,
    activation_type: str,
    num_activations: int = 2_000_000,
) -> dict:
    """Train a single SAE on A100 GPU."""
    return _train_sae_impl(model_name, layer, activation_type, num_activations)


def _train_sae_impl(
    model_name: str,
    layer: int,
    activation_type: str,
    num_activations: int,
) -> dict:
    """Internal SAE training implementation."""
    import sys
    import torch
    from pathlib import Path
    from datetime import datetime

    sys.path.insert(0, "/app/backend")

    from core.sae.config import SAEConfig, ActivationSource
    from core.sae.trainer import SAETrainer, SAETrainingConfig
    from core.sae.model_torch import SparseAutoencoderTorch
    from core.sae.checkpoints import save_sae_checkpoint, load_sae_checkpoint, load_training_state
    from core.sae.activation_buffer import DiskActivationBuffer
    from inference.model_loader import load_model
    from scripts.sae_training.pile_data_loader import PileActivationCollector

    config = MODEL_CONFIGS[model_name]
    device = "cuda"

    # Adjust steps based on activation count (test mode uses fewer)
    is_test_mode = num_activations < 100_000
    num_steps = 1000 if is_test_mode else config["sae_num_steps"]

    # Paths
    base_model_path = f"/models/base/ozera-{model_name}"
    sae_output_dir = f"/saes/{model_name}/layer_{layer}_{activation_type}"
    activations_path = f"{sae_output_dir}/activations.h5"
    checkpoint_path = f"{sae_output_dir}/model.safetensors"
    training_state_path = f"{sae_output_dir}/training_state.pt"

    print(f"Training SAE: {model_name} layer {layer} {activation_type}")
    print(f"  Base model: {base_model_path}")
    print(f"  Output: {sae_output_dir}")

    # Check for existing checkpoint to resume from
    resume_from_checkpoint = False
    resume_step = 0
    if Path(checkpoint_path).exists() and Path(training_state_path).exists():
        # Check if training is already complete
        metadata_path = Path(sae_output_dir) / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if metadata.get("is_final", False):
                print(f"\n  Training already complete! Skipping.")
                return {
                    "status": "completed",
                    "model_name": model_name,
                    "layer": layer,
                    "activation_type": activation_type,
                    "message": "Already completed in previous run",
                    "checkpoint_path": sae_output_dir,
                }
        # Not complete, resume from checkpoint
        training_state = load_training_state(sae_output_dir, device="cpu")
        if training_state and "step" in training_state:
            resume_step = training_state["step"]
            if resume_step < num_steps:
                resume_from_checkpoint = True
                print(f"\n  Found checkpoint at step {resume_step}/{num_steps} - will resume")

    # Create output directory
    Path(sae_output_dir).mkdir(parents=True, exist_ok=True)

    # Step 2: Check for existing activations (REUSE if available)
    activations_file = Path(activations_path)
    if activations_file.exists() and activations_file.stat().st_size > 0:
        # Reuse existing activations - saves ~1hr of collection time
        print(f"\n[1/4] Reusing existing activations from {activations_path}")
        buffer = DiskActivationBuffer(
            filepath=activations_path,
            d_input=config["d_model"],
        )
        print(f"  Loaded {len(buffer):,} activation vectors")
        model = None  # Don't need to load model
    else:
        # Need to collect new activations
        print("\n[1/4] Loading transformer model...")
        model, model_config = load_model(
            model_name,
            models_dir="/models/base",
            device=device,
        )

        print(f"\n[2/4] Collecting {num_activations:,} activations from OpenWebText...")

        buffer = DiskActivationBuffer(
            filepath=activations_path,
            d_input=config["d_model"],
            max_samples=num_activations,
        )

        collector = PileActivationCollector(
            model=model,
            config=model_config,
            target_layer=layer,
            activation_type=activation_type,
            batch_size=32,
            seq_len=256,
            device=device,
        )

        collected = 0
        samples_per_batch = 32 * 256

        while collected < num_activations:
            activations = collector.collect_batch()
            buffer.add(activations.cpu())
            collected += activations.shape[0]

            if collected % (samples_per_batch * 50) == 0:
                pct = 100.0 * collected / num_activations
                print(f"  Collected {collected:,}/{num_activations:,} ({pct:.1f}%)")

        print(f"  Total collected: {len(buffer):,}")

        # Flush and reopen buffer to ensure all data is written
        buffer.h5file.flush()
        buffer.close()

        # Commit activations to volume immediately (prevents loss)
        print("  Committing activations to storage...")
        saes_volume.commit()

        # Reopen for reading
        buffer = DiskActivationBuffer(
            filepath=activations_path,
            d_input=config["d_model"],
        )
        print(f"  Verified: {len(buffer):,} activations in buffer")

        # Clear model from memory
        del model
        torch.cuda.empty_cache()

    # Step 3: Create or load SAE
    activation_source = (
        ActivationSource.RESIDUAL if activation_type == "residual"
        else ActivationSource.MLP_OUTPUT
    )

    # Use smaller batch in test mode to avoid memory issues
    batch_size = 1024 if is_test_mode else 4096
    warmup_steps = 100 if is_test_mode else 1000
    checkpoint_interval = 200 if is_test_mode else CHECKPOINT_INTERVAL

    training_config = SAETrainingConfig(
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=1e-4,
        warmup_steps=warmup_steps,
        sparsity_coefficient=0.01,
        dead_feature_threshold=10_000,
        dead_feature_resample=True,
        log_interval=100 if is_test_mode else 500,
        eval_interval=200 if is_test_mode else 1000,
        checkpoint_interval=checkpoint_interval,
    )

    if resume_from_checkpoint:
        # Load existing model and training state
        print(f"\n[3/4] Resuming SAE from checkpoint at step {resume_step}...")
        sae_model, sae_config, _ = load_sae_checkpoint(sae_output_dir, device=device)
        print(f"  Loaded model: {sae_model.d_input} -> {sae_model.d_hidden}")
        print(f"  Parameters: {sae_model.count_parameters():,}")

        # Create trainer and load training state
        trainer = SAETrainer(sae_model, training_config, device=device)
        saved_training_state = load_training_state(sae_output_dir, device=device)
        if saved_training_state:
            trainer.load_training_state(saved_training_state)
            print(f"  Restored training state at step {trainer.step}")

        remaining_steps = num_steps - trainer.step
        print(f"\n[4/4] Resuming training for {remaining_steps:,} more steps...")
    else:
        # Create new SAE from scratch
        print("\n[3/4] Initializing new SAE...")

        sae_config = SAEConfig(
            d_input=config["d_model"],
            expansion_factor=8,
            sparsity_coefficient=0.01,
            learning_rate=1e-4,
            batch_size=batch_size,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            target_layer=layer,
            activation_source=activation_source,
            normalize_decoder=True,
            dead_feature_threshold=10_000,
            dead_feature_resample=True,
            device=device,
        )

        sae_model = SparseAutoencoderTorch(sae_config)
        print(f"  Input dim: {sae_config.d_input}")
        print(f"  Hidden dim: {sae_config.d_hidden}")
        print(f"  Parameters: {sae_model.count_parameters():,}")

        trainer = SAETrainer(sae_model, training_config, device=device)
        print(f"\n[4/4] Training SAE for {training_config.num_steps:,} steps...")

    print(f"  Checkpointing every {checkpoint_interval} steps")

    # Progress callback - print every 1000 steps
    def progress_callback(progress):
        if progress.step % 1000 == 0:
            pct = 100.0 * progress.step / progress.total_steps
            print(
                f"  [{progress.step:>6d}/{progress.total_steps}] ({pct:5.1f}%) "
                f"loss={progress.loss:.4f} L0={progress.avg_l0:.1f} dead={progress.dead_features}"
            )

    # Checkpoint callback - save and commit to volume every checkpoint_interval steps
    def checkpoint_callback(step):
        print(f"  Saving checkpoint at step {step}...")
        save_sae_checkpoint(
            model=sae_model,
            path=sae_output_dir,
            config=sae_config,
            training_state=trainer.get_training_state(),
            metadata={
                "base_model": model_name,
                "layer": layer,
                "activation_type": activation_type,
                "dataset": "Skylion007/openwebtext",
                "num_activations": len(buffer),
                "checkpoint_step": step,
                "is_final": False,
            },
        )
        # Commit to volume immediately - prevents loss on timeout/crash
        saes_volume.commit()
        print(f"  Checkpoint saved and committed at step {step}")

    results = trainer.train(
        buffer=buffer,
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback,
    )

    # Save final checkpoint
    print(f"\nSaving final checkpoint to {sae_output_dir}...")
    save_sae_checkpoint(
        model=sae_model,
        path=sae_output_dir,
        config=sae_config,
        training_state=trainer.get_training_state(),
        metadata={
            "base_model": model_name,
            "layer": layer,
            "activation_type": activation_type,
            "dataset": "Skylion007/openwebtext",
            "num_activations": len(buffer),
            "training_results": {
                "status": results["status"],
                "best_loss": results.get("best_loss"),
                "total_steps": results.get("total_steps"),
            },
            "completed_at": datetime.utcnow().isoformat(),
            "is_final": True,
        },
    )

    # Final commit
    saes_volume.commit()

    # Cleanup
    buffer.close()

    return {
        "status": results["status"],
        "model_name": model_name,
        "layer": layer,
        "activation_type": activation_type,
        "best_loss": results.get("best_loss"),
        "total_steps": results.get("total_steps"),
        "error_message": results.get("error_message"),
        "checkpoint_path": sae_output_dir,
    }


@app.function(
    image=sae_training_image,
    volumes={
        "/saes": saes_volume,
        "/models": models_volume,
    },
    timeout=86400,  # 24 hours for full training
)
def train_all_saes_orchestrator(
    model_filter: str = None,
    gpu_type: str = "a10g",
) -> dict:
    """
    Orchestrate training of all SAEs.

    Args:
        model_filter: Optional filter for model ("nano" or "mini")
        gpu_type: GPU type to use ("a10g" or "a100")

    Returns:
        Summary of all training jobs
    """
    jobs = []

    for model_name, config in MODEL_CONFIGS.items():
        if model_filter and model_name != model_filter:
            continue

        for layer in range(config["num_layers"]):
            for activation_type in ["residual", "mlp_output"]:
                jobs.append({
                    "model_name": model_name,
                    "layer": layer,
                    "activation_type": activation_type,
                })

    print(f"Launching {len(jobs)} SAE training jobs on {gpu_type.upper()}...")

    # Select training function based on GPU type
    if gpu_type == "a100":
        train_fn = train_single_sae_a100
    elif gpu_type == "l4":
        train_fn = train_single_sae_l4
    else:
        train_fn = train_single_sae_a10g

    # Launch all jobs in parallel
    futures = []
    for job in jobs:
        future = train_fn.spawn(
            model_name=job["model_name"],
            layer=job["layer"],
            activation_type=job["activation_type"],
        )
        futures.append((job, future))

    # Wait for all jobs and collect results
    results = []
    for job, future in futures:
        try:
            result = future.get()
            results.append(result)
            print(f"Completed: {job['model_name']} L{job['layer']} {job['activation_type']}: {result['status']}")
        except Exception as e:
            results.append({
                "status": "failed",
                "error": str(e),
                **job,
            })
            print(f"Failed: {job['model_name']} L{job['layer']} {job['activation_type']}: {e}")

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")

    return {
        "total_jobs": len(jobs),
        "completed": completed,
        "failed": failed,
        "results": results,
    }


@app.local_entrypoint()
def main(
    model: str = None,
    layer: int = None,
    activation: str = None,
    all_saes: bool = False,
    gpu: str = "l4",
    test: bool = False,
):
    """
    Local entrypoint for SAE training.

    Examples:
        # Train single SAE (uses L4 by default - most cost effective)
        modal run modal_sae_training.py --model nano --layer 0 --activation residual

        # Train all SAEs for a model
        modal run modal_sae_training.py --model nano --all-saes

        # Train all 28 SAEs with L4 (default, $0.80/hr)
        modal run modal_sae_training.py --all-saes

        # Train all 28 SAEs with A100 (faster but more expensive)
        modal run modal_sae_training.py --all-saes --gpu a100
    """
    if all_saes:
        print(f"Training all SAEs{f' for {model}' if model else ''} on {gpu.upper()}...")
        result = train_all_saes_orchestrator.remote(
            model_filter=model,
            gpu_type=gpu,
        )
        print(f"\nCompleted: {result['completed']}/{result['total_jobs']}")
        if result['failed'] > 0:
            print(f"Failed: {result['failed']}")

    elif model and layer is not None and activation:
        # Test mode: fewer activations and steps for quick validation
        num_activations = 50_000 if test else 2_000_000

        print(f"Training single SAE: {model} layer {layer} {activation} on {gpu.upper()}")
        if test:
            print(f"  TEST MODE: {num_activations:,} activations, 1000 steps")

        if gpu == "a100":
            result = train_single_sae_a100.remote(model, layer, activation, num_activations)
        elif gpu == "l4":
            result = train_single_sae_l4.remote(model, layer, activation, num_activations)
        else:
            result = train_single_sae_a10g.remote(model, layer, activation, num_activations)

        print(f"\nResult: {result}")
        if result.get("error_message"):
            print(f"Error: {result['error_message']}")

    else:
        print("Usage:")
        print("  # Quick test run (50k activations, 1000 steps)")
        print("  modal run modal_sae_training.py --model nano --layer 0 --activation residual --test")
        print()
        print("  # Full training (uses L4 by default @ $0.80/hr)")
        print("  modal run modal_sae_training.py --model nano --layer 0 --activation residual")
        print("  modal run modal_sae_training.py --model nano --all-saes")
        print("  modal run modal_sae_training.py --all-saes")
        print()
        print("  # Use different GPU")
        print("  modal run modal_sae_training.py --all-saes --gpu a100  # Faster but $2.10/hr")
        print("  modal run modal_sae_training.py --all-saes --gpu a10g  # Mid-range $1.10/hr")


# Helper function to call from other code
def train_sae_remote(
    model_name: str,
    layer: int,
    activation_type: str,
    gpu_type: str = "l4",
) -> dict:
    """
    Train an SAE remotely on Modal.

    Args:
        model_name: "nano" or "mini"
        layer: Layer index
        activation_type: "residual" or "mlp_output"
        gpu_type: "l4", "a10g", or "a100"

    Returns:
        Training result dictionary
    """
    if gpu_type == "a100":
        fn = modal.Function.from_name("ozera-sae-training", "train_single_sae_a100")
    elif gpu_type == "l4":
        fn = modal.Function.from_name("ozera-sae-training", "train_single_sae_l4")
    else:
        fn = modal.Function.from_name("ozera-sae-training", "train_single_sae_a10g")

    return fn.remote(model_name, layer, activation_type)
