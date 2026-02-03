#!/usr/bin/env python3
"""
SAE Verification Script.

Verifies the quality of trained SAEs by checking:
- Reconstruction loss
- Sparsity levels (L0)
- Dead features
- Explained variance

Usage:
    # Verify all SAEs
    python verify_saes.py --all

    # Verify specific model
    python verify_saes.py --model nano

    # Verify specific SAE
    python verify_saes.py --model nano --layer 0 --activation residual
"""

import argparse
import torch
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from core.sae.checkpoints import load_sae_checkpoint, get_checkpoint_info, list_checkpoints
from core.sae.activation_buffer import DiskActivationBuffer
from core.sae.metrics import (
    compute_sparsity_metrics,
    compute_feature_health_metrics,
    compute_reconstruction_metrics,
    SAEQualityMetrics,
)


@dataclass
class VerificationResult:
    """Result of SAE verification."""
    sae_path: str
    model_name: str
    layer: int
    activation_type: str

    # Quality metrics
    reconstruction_loss: float
    explained_variance: float
    avg_l0: float
    dead_feature_fraction: float
    cosine_similarity: float

    # Pass/fail checks
    recon_loss_passed: bool  # < 0.1
    l0_range_passed: bool    # 10-100
    dead_features_passed: bool  # < 10%
    variance_passed: bool    # > 0.9

    @property
    def all_checks_passed(self) -> bool:
        return (
            self.recon_loss_passed and
            self.l0_range_passed and
            self.dead_features_passed and
            self.variance_passed
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_sae(
    sae_path: str,
    activations_path: Optional[str] = None,
    num_samples: int = 10_000,
    device: str = "cuda",
) -> VerificationResult:
    """
    Verify a single SAE.

    Args:
        sae_path: Path to SAE checkpoint directory
        activations_path: Path to activations HDF5 (default: sae_path/activations.h5)
        num_samples: Number of samples for evaluation
        device: Device for computation

    Returns:
        VerificationResult
    """
    sae_path = Path(sae_path)

    # Load SAE
    print(f"Loading SAE from {sae_path}...")
    sae_model, config, metadata = load_sae_checkpoint(sae_path, device=device)
    sae_model.eval()

    # Get metadata
    model_name = metadata.get("base_model", "unknown")
    layer = metadata.get("layer", -1)
    activation_type = metadata.get("activation_type", "unknown")

    # Load activations
    if activations_path is None:
        activations_path = sae_path / "activations.h5"

    if not Path(activations_path).exists():
        raise FileNotFoundError(f"Activations not found at {activations_path}")

    print(f"Loading activations from {activations_path}...")
    buffer = DiskActivationBuffer(
        filepath=str(activations_path),
        d_input=config.d_input,
    )

    # Sample activations
    actual_samples = min(num_samples, len(buffer))
    print(f"Evaluating on {actual_samples:,} samples...")

    activations = buffer.sample(actual_samples).to(device)

    # Run through SAE
    with torch.no_grad():
        x_hat, hidden = sae_model(activations, return_hidden=True)

    # Compute metrics
    print("Computing metrics...")

    # Sparsity metrics
    sparsity = compute_sparsity_metrics(hidden)

    # Feature health metrics
    feature_health = compute_feature_health_metrics(hidden)

    # Reconstruction metrics
    reconstruction = compute_reconstruction_metrics(activations, x_hat)

    # Cleanup
    buffer.close()

    # Checks
    recon_loss_passed = reconstruction.normalized_mse < 0.1
    l0_range_passed = 10 <= sparsity.avg_l0 <= 100
    dead_features_passed = feature_health.dead_feature_fraction < 0.10
    variance_passed = reconstruction.explained_variance > 0.9

    return VerificationResult(
        sae_path=str(sae_path),
        model_name=model_name,
        layer=layer,
        activation_type=activation_type,
        reconstruction_loss=reconstruction.normalized_mse,
        explained_variance=reconstruction.explained_variance,
        avg_l0=sparsity.avg_l0,
        dead_feature_fraction=feature_health.dead_feature_fraction,
        cosine_similarity=reconstruction.cosine_similarity,
        recon_loss_passed=recon_loss_passed,
        l0_range_passed=l0_range_passed,
        dead_features_passed=dead_features_passed,
        variance_passed=variance_passed,
    )


def print_result(result: VerificationResult):
    """Print verification result."""
    status = "PASSED" if result.all_checks_passed else "FAILED"
    print(f"\n{result.model_name} layer {result.layer} {result.activation_type}: {status}")
    print(f"  Path: {result.sae_path}")
    print(f"  Reconstruction loss: {result.reconstruction_loss:.4f} {'PASS' if result.recon_loss_passed else 'FAIL'} (threshold: <0.1)")
    print(f"  Explained variance:  {result.explained_variance:.4f} {'PASS' if result.variance_passed else 'FAIL'} (threshold: >0.9)")
    print(f"  Average L0:          {result.avg_l0:.1f} {'PASS' if result.l0_range_passed else 'FAIL'} (range: 10-100)")
    print(f"  Dead features:       {result.dead_feature_fraction*100:.1f}% {'PASS' if result.dead_features_passed else 'FAIL'} (threshold: <10%)")
    print(f"  Cosine similarity:   {result.cosine_similarity:.4f}")


def find_saes(saes_dir: str, model_filter: str = None) -> List[Path]:
    """
    Find all SAE checkpoint directories.

    Args:
        saes_dir: Base directory for SAEs
        model_filter: Optional model name filter ("nano" or "mini")

    Returns:
        List of SAE checkpoint paths
    """
    saes_dir = Path(saes_dir)
    sae_paths = []

    if not saes_dir.exists():
        return sae_paths

    for model_dir in saes_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Filter by model if specified
        if model_filter and model_dir.name != f"ozera-{model_filter}" and model_dir.name != model_filter:
            continue

        for layer_dir in model_dir.iterdir():
            if not layer_dir.is_dir():
                continue

            # Check if it's a valid SAE checkpoint
            if (layer_dir / "model.safetensors").exists() or (layer_dir / "model.pt").exists():
                sae_paths.append(layer_dir)

    return sorted(sae_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Verify trained SAE quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all SAEs",
    )
    parser.add_argument(
        "--model",
        choices=["nano", "mini"],
        help="Verify SAEs for specific model",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Verify specific layer",
    )
    parser.add_argument(
        "--activation",
        choices=["residual", "mlp_output"],
        help="Verify specific activation type",
    )
    parser.add_argument(
        "--saes-dir",
        type=str,
        default=str(backend_dir / "models" / "saes"),
        help="Directory containing trained SAEs",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10_000,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Find SAEs to verify
    if args.layer is not None and args.activation is not None and args.model is not None:
        # Specific SAE
        sae_path = Path(args.saes_dir) / args.model / f"layer_{args.layer}_{args.activation}"
        if not sae_path.exists():
            print(f"SAE not found: {sae_path}")
            sys.exit(1)
        sae_paths = [sae_path]
    elif args.all or args.model:
        # Find matching SAEs
        sae_paths = find_saes(args.saes_dir, model_filter=args.model)
        if not sae_paths:
            print(f"No SAEs found in {args.saes_dir}")
            sys.exit(1)
    else:
        parser.error("Must specify --all, --model, or specific --model --layer --activation")

    print(f"Found {len(sae_paths)} SAEs to verify")

    # Verify each SAE
    results = []
    for sae_path in sae_paths:
        try:
            result = verify_sae(
                sae_path=str(sae_path),
                num_samples=args.num_samples,
                device=args.device,
            )
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"\nFailed to verify {sae_path}: {e}")
            results.append({
                "sae_path": str(sae_path),
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*60}")
    print("Verification Summary")
    print(f"{'='*60}")

    valid_results = [r for r in results if isinstance(r, VerificationResult)]
    passed = sum(1 for r in valid_results if r.all_checks_passed)
    failed = sum(1 for r in valid_results if not r.all_checks_passed)
    errors = len(results) - len(valid_results)

    print(f"Passed: {passed}/{len(valid_results)}")
    print(f"Failed: {failed}/{len(valid_results)}")
    if errors > 0:
        print(f"Errors: {errors}")

    if failed > 0:
        print("\nFailed SAEs:")
        for r in valid_results:
            if not r.all_checks_passed:
                checks = []
                if not r.recon_loss_passed:
                    checks.append(f"recon_loss={r.reconstruction_loss:.4f}")
                if not r.l0_range_passed:
                    checks.append(f"L0={r.avg_l0:.1f}")
                if not r.dead_features_passed:
                    checks.append(f"dead={r.dead_feature_fraction*100:.1f}%")
                if not r.variance_passed:
                    checks.append(f"variance={r.explained_variance:.4f}")
                print(f"  - {r.model_name} L{r.layer} {r.activation_type}: {', '.join(checks)}")

    # Save results
    output_path = args.output or (Path(args.saes_dir) / "verification_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "verified_at": datetime.utcnow().isoformat(),
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "results": [r.to_dict() if isinstance(r, VerificationResult) else r for r in results],
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
