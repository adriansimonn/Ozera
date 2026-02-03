#!/usr/bin/env python3
"""
SAE Training Script for Ozera Models.

Trains Sparse Autoencoders on ozera-nano and ozera-mini activations
using OpenWebText dataset for diverse inputs.

Usage:
    # Train all 28 SAEs (12 for nano, 16 for mini)
    python train_saes.py --all

    # Train all SAEs for a specific model
    python train_saes.py --model nano
    python train_saes.py --model mini

    # Train specific layer/activation combination
    python train_saes.py --model nano --layer 0 --activation residual

    # Resume training from checkpoint
    python train_saes.py --model nano --layer 0 --activation residual --resume
"""

import argparse
import torch
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from core.sae.config import SAEConfig, ActivationSource
from core.sae.trainer import SAETrainer, SAETrainingConfig, SAETrainingProgress
from core.sae.model_torch import SparseAutoencoderTorch
from core.sae.checkpoints import save_sae_checkpoint, load_sae_checkpoint, load_training_state
from core.sae.activation_buffer import DiskActivationBuffer
from inference.model_loader import load_model

from pile_data_loader import PileActivationCollector


# Model configurations
MODEL_CONFIGS = {
    "nano": {
        "num_layers": 6,
        "d_model": 192,
        "checkpoint": "ozera-nano",
        "sae_num_steps": 50_000,
    },
    "mini": {
        "num_layers": 8,
        "d_model": 512,
        "checkpoint": "ozera-mini",
        "sae_num_steps": 100_000,
    },
}

ACTIVATION_TYPES = ["residual", "mlp_output"]


@dataclass
class TrainingJob:
    """Configuration for a single SAE training job."""
    model_name: str
    layer: int
    activation_type: str
    d_input: int
    num_steps: int

    @property
    def job_id(self) -> str:
        return f"{self.model_name}_layer{self.layer}_{self.activation_type}"

    @property
    def checkpoint_dir(self) -> str:
        return f"layer_{self.layer}_{self.activation_type}"


def generate_all_jobs() -> List[TrainingJob]:
    """Generate all SAE training jobs."""
    jobs = []

    for model_name, config in MODEL_CONFIGS.items():
        for layer in range(config["num_layers"]):
            for activation_type in ACTIVATION_TYPES:
                jobs.append(TrainingJob(
                    model_name=model_name,
                    layer=layer,
                    activation_type=activation_type,
                    d_input=config["d_model"],
                    num_steps=config["sae_num_steps"],
                ))

    return jobs


def get_sae_config(job: TrainingJob, device: str = "cuda") -> SAEConfig:
    """Create SAE config for a training job."""
    activation_source = (
        ActivationSource.RESIDUAL if job.activation_type == "residual"
        else ActivationSource.MLP_OUTPUT
    )

    return SAEConfig(
        d_input=job.d_input,
        expansion_factor=8,
        sparsity_coefficient=0.01,
        learning_rate=1e-4,
        batch_size=4096,
        num_steps=job.num_steps,
        warmup_steps=1000,
        target_layer=job.layer,
        activation_source=activation_source,
        normalize_decoder=True,
        dead_feature_threshold=10_000,
        dead_feature_resample=True,
        device=device,
    )


def get_training_config(job: TrainingJob) -> SAETrainingConfig:
    """Create training config for a job."""
    return SAETrainingConfig(
        num_steps=job.num_steps,
        batch_size=4096,
        learning_rate=1e-4,
        warmup_steps=1000,
        use_cosine_decay=True,
        sparsity_coefficient=0.01,
        dead_feature_threshold=10_000,
        dead_feature_resample=True,
        resample_check_interval=5_000,
        log_interval=100,
        eval_interval=500,
        checkpoint_interval=5_000,
    )


def print_progress(progress: SAETrainingProgress):
    """Print training progress."""
    pct = 100.0 * progress.step / progress.total_steps
    print(
        f"\r[{progress.step:>6d}/{progress.total_steps}] "
        f"({pct:5.1f}%) "
        f"loss={progress.loss:.4f} "
        f"recon={progress.recon_loss:.4f} "
        f"sparse={progress.sparse_loss:.4f} "
        f"L0={progress.avg_l0:.1f} "
        f"dead={progress.dead_features} "
        f"lr={progress.learning_rate:.2e}",
        end="",
        flush=True,
    )


def train_single_sae(
    job: TrainingJob,
    output_dir: str,
    models_dir: str,
    num_activations: int = 2_000_000,
    device: str = "cuda",
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Train a single SAE.

    Args:
        job: Training job configuration
        output_dir: Base directory for SAE checkpoints
        models_dir: Directory containing transformer checkpoints
        num_activations: Number of activations to collect
        device: Device for training
        resume: Whether to resume from checkpoint

    Returns:
        Training results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training SAE: {job.job_id}")
    print(f"{'='*60}")

    # Paths
    sae_dir = Path(output_dir) / job.model_name / job.checkpoint_dir
    activations_path = sae_dir / "activations.h5"

    # Check for existing checkpoint if resuming
    existing_checkpoint = None
    if resume and (sae_dir / "model.safetensors").exists():
        print(f"Found existing checkpoint, will resume training")
        existing_checkpoint = sae_dir

    # Step 1: Load transformer model
    print(f"\n[1/4] Loading {job.model_name} model...")
    model, config = load_model(job.model_name, models_dir=models_dir, device=device)

    # Step 2: Collect activations (or reuse existing)
    if activations_path.exists() and activations_path.stat().st_size > 0:
        print(f"\n[2/4] Reusing existing activations from {activations_path}")
        buffer = DiskActivationBuffer(
            filepath=str(activations_path),
            d_input=job.d_input,
            max_samples=num_activations,
        )
        print(f"  Loaded {len(buffer):,} activation vectors")
    else:
        print(f"\n[2/4] Collecting {num_activations:,} activations from OpenWebText...")
        sae_dir.mkdir(parents=True, exist_ok=True)

        buffer = DiskActivationBuffer(
            filepath=str(activations_path),
            d_input=job.d_input,
            max_samples=num_activations,
        )

        collector = PileActivationCollector(
            model=model,
            config=config,
            target_layer=job.layer,
            activation_type=job.activation_type,
            batch_size=32,
            seq_len=256,
            device=device,
        )

        collected = 0
        samples_per_batch = 32 * 256  # batch_size * seq_len

        while collected < num_activations:
            activations = collector.collect_batch()
            buffer.add(activations.cpu())
            collected += activations.shape[0]

            if collected % (samples_per_batch * 10) == 0:
                pct = 100.0 * collected / num_activations
                print(f"\r  Collected {collected:,}/{num_activations:,} ({pct:.1f}%)", end="", flush=True)

        print(f"\r  Collected {len(buffer):,} activation vectors")

    # Clear model from memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 3: Create/load SAE
    print(f"\n[3/4] Initializing SAE...")
    sae_config = get_sae_config(job, device=device)
    training_config = get_training_config(job)

    if existing_checkpoint:
        print(f"  Loading from checkpoint...")
        sae_model, loaded_config, metadata = load_sae_checkpoint(existing_checkpoint, device=device)
        training_state = load_training_state(existing_checkpoint, device=device)
    else:
        sae_model = SparseAutoencoderTorch(sae_config)
        training_state = None

    print(f"  Input dim: {sae_config.d_input}")
    print(f"  Hidden dim: {sae_config.d_hidden}")
    print(f"  Parameters: {sae_model.count_parameters():,}")

    # Step 4: Train
    print(f"\n[4/4] Training SAE for {training_config.num_steps:,} steps...")

    trainer = SAETrainer(sae_model, training_config, device=device)

    if training_state:
        trainer.load_training_state(training_state)
        print(f"  Resuming from step {trainer.step}")

    def checkpoint_callback(step: int):
        """Save checkpoint during training."""
        save_sae_checkpoint(
            model=sae_model,
            path=sae_dir,
            config=sae_config,
            training_state=trainer.get_training_state(),
            metadata={
                "base_model": job.model_name,
                "layer": job.layer,
                "activation_type": job.activation_type,
                "dataset": "Skylion007/openwebtext",
                "num_activations": len(buffer),
                "checkpoint_step": step,
            },
        )

    results = trainer.train(
        buffer=buffer,
        progress_callback=print_progress,
        checkpoint_callback=checkpoint_callback,
    )
    print()  # Newline after progress

    # Save final checkpoint
    print(f"\nSaving final checkpoint to {sae_dir}...")
    save_sae_checkpoint(
        model=sae_model,
        path=sae_dir,
        config=sae_config,
        training_state=trainer.get_training_state(),
        metadata={
            "base_model": job.model_name,
            "layer": job.layer,
            "activation_type": job.activation_type,
            "dataset": "Skylion007/openwebtext",
            "num_activations": len(buffer),
            "training_results": results,
            "completed_at": datetime.utcnow().isoformat(),
        },
    )

    # Cleanup
    buffer.close()

    return {
        "job_id": job.job_id,
        "status": results["status"],
        "final_loss": results.get("final_metrics", {}).get("loss"),
        "best_loss": results.get("best_loss"),
        "total_steps": results.get("total_steps"),
        "checkpoint_path": str(sae_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train SAEs on Ozera model activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all SAEs
    python train_saes.py --all

    # Train all SAEs for nano model
    python train_saes.py --model nano

    # Train specific SAE
    python train_saes.py --model nano --layer 0 --activation residual

    # Resume interrupted training
    python train_saes.py --model nano --layer 0 --activation residual --resume
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 28 SAEs (nano + mini, all layers, both activation types)",
    )
    parser.add_argument(
        "--model",
        choices=["nano", "mini"],
        help="Model to train SAEs for",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Specific layer to train SAE for (0-indexed)",
    )
    parser.add_argument(
        "--activation",
        choices=["residual", "mlp_output"],
        help="Activation type to train SAE on",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(backend_dir / "models" / "saes"),
        help="Output directory for SAE checkpoints",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(backend_dir / "models"),
        help="Directory containing transformer checkpoints",
    )
    parser.add_argument(
        "--num-activations",
        type=int,
        default=2_000_000,
        help="Number of activations to collect per SAE",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and args.model is None:
        parser.error("Must specify --all or --model")

    if args.layer is not None and args.model is None:
        parser.error("--layer requires --model")

    if args.activation is not None and args.model is None:
        parser.error("--activation requires --model")

    # Generate jobs
    if args.all:
        jobs = generate_all_jobs()
        print(f"Training all {len(jobs)} SAEs")
    elif args.layer is not None and args.activation is not None:
        # Single specific job
        config = MODEL_CONFIGS[args.model]
        if args.layer >= config["num_layers"]:
            parser.error(f"{args.model} only has {config['num_layers']} layers (0-{config['num_layers']-1})")
        jobs = [TrainingJob(
            model_name=args.model,
            layer=args.layer,
            activation_type=args.activation,
            d_input=config["d_model"],
            num_steps=config["sae_num_steps"],
        )]
        print(f"Training single SAE: {jobs[0].job_id}")
    elif args.layer is not None:
        # All activation types for specific layer
        config = MODEL_CONFIGS[args.model]
        if args.layer >= config["num_layers"]:
            parser.error(f"{args.model} only has {config['num_layers']} layers (0-{config['num_layers']-1})")
        jobs = [
            TrainingJob(
                model_name=args.model,
                layer=args.layer,
                activation_type=act_type,
                d_input=config["d_model"],
                num_steps=config["sae_num_steps"],
            )
            for act_type in ACTIVATION_TYPES
        ]
        print(f"Training {len(jobs)} SAEs for {args.model} layer {args.layer}")
    else:
        # All layers for model
        config = MODEL_CONFIGS[args.model]
        jobs = [
            TrainingJob(
                model_name=args.model,
                layer=layer,
                activation_type=act_type,
                d_input=config["d_model"],
                num_steps=config["sae_num_steps"],
            )
            for layer in range(config["num_layers"])
            for act_type in ACTIVATION_TYPES
        ]
        print(f"Training {len(jobs)} SAEs for {args.model}")

    # Train jobs
    results = []
    for i, job in enumerate(jobs):
        print(f"\n{'#'*60}")
        print(f"# Job {i+1}/{len(jobs)}: {job.job_id}")
        print(f"{'#'*60}")

        try:
            result = train_single_sae(
                job=job,
                output_dir=args.output_dir,
                models_dir=args.models_dir,
                num_activations=args.num_activations,
                device=args.device,
                resume=args.resume,
            )
            results.append(result)
            print(f"\nCompleted: {result['status']}")
            if result.get("final_loss"):
                print(f"  Final loss: {result['final_loss']:.4f}")
            if result.get("best_loss"):
                print(f"  Best loss: {result['best_loss']:.4f}")
        except Exception as e:
            print(f"\nFailed: {e}")
            results.append({
                "job_id": job.job_id,
                "status": "failed",
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"Completed: {completed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed jobs:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['job_id']}: {r.get('error', 'Unknown error')}")

    # Save results summary
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "completed_at": datetime.utcnow().isoformat(),
            "total_jobs": len(results),
            "completed": completed,
            "failed": failed,
            "results": results,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
