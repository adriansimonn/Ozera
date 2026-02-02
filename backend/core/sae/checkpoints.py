"""
Checkpoint saving and loading for Sparse Autoencoders.

Supports:
- Safetensors format (recommended for security)
- PyTorch native format
- Metadata storage alongside weights
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from .config import SAEConfig
from .model_torch import SparseAutoencoderTorch


def save_sae_checkpoint(
    model: SparseAutoencoderTorch,
    path: Union[str, Path],
    config: Optional[SAEConfig] = None,
    training_state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_safetensors: bool = True,
) -> Path:
    """
    Save SAE checkpoint to disk.

    Args:
        model: SAE model to save
        path: Directory or file path for saving
        config: SAE configuration (saved alongside weights)
        training_state: Optional training state from trainer
        metadata: Optional additional metadata
        use_safetensors: Use safetensors format (recommended)

    Returns:
        Path to saved checkpoint directory
    """
    path = Path(path)

    # If path is a file, use its parent directory
    if path.suffix:
        save_dir = path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = path
        save_dir.mkdir(parents=True, exist_ok=True)

    # Get model parameters
    params = model.get_parameters_dict()

    # Save weights
    if use_safetensors:
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )
        weights_path = save_dir / "model.safetensors"
        # Convert to CPU and ensure contiguous
        cpu_params = {k: v.cpu().contiguous() for k, v in params.items()}
        save_file(cpu_params, weights_path)
    else:
        weights_path = save_dir / "model.pt"
        torch.save(params, weights_path)

    # Build config dict
    config_dict = {}
    if config is not None:
        config_dict = config.to_dict()
    elif hasattr(model, "config"):
        config_dict = model.config.to_dict()

    # Build metadata
    save_metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "format": "safetensors" if use_safetensors else "pytorch",
        "d_input": model.d_input,
        "d_hidden": model.d_hidden,
        "num_parameters": model.count_parameters(),
    }

    if metadata:
        save_metadata.update(metadata)

    # Save config
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save metadata
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(save_metadata, f, indent=2)

    # Save training state if provided
    if training_state is not None:
        training_path = save_dir / "training_state.pt"
        torch.save(training_state, training_path)

    return save_dir


def load_sae_checkpoint(
    path: Union[str, Path],
    device: str = "cuda",
    config_override: Optional[Dict[str, Any]] = None,
) -> tuple[SparseAutoencoderTorch, SAEConfig, Dict[str, Any]]:
    """
    Load SAE checkpoint from disk.

    Args:
        path: Path to checkpoint directory or weights file
        device: Device to load model to
        config_override: Optional config values to override

    Returns:
        Tuple of (model, config, metadata)
    """
    path = Path(path)

    # Determine checkpoint directory
    if path.is_file():
        checkpoint_dir = path.parent
        weights_path = path
    else:
        checkpoint_dir = path
        # Find weights file
        if (checkpoint_dir / "model.safetensors").exists():
            weights_path = checkpoint_dir / "model.safetensors"
        elif (checkpoint_dir / "model.pt").exists():
            weights_path = checkpoint_dir / "model.pt"
        else:
            raise FileNotFoundError(f"No weights file found in {checkpoint_dir}")

    # Load config
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        if config_override:
            config_dict.update(config_override)
        config = SAEConfig.from_dict(config_dict)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load metadata
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Load weights
    if weights_path.suffix == ".safetensors":
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )
        params = load_file(weights_path, device=device)
    else:
        params = torch.load(weights_path, map_location=device)

    # Create model
    model = SparseAutoencoderTorch(config)
    model.load_parameters_dict(params)
    model = model.to(device)

    return model, config, metadata


def load_training_state(
    path: Union[str, Path],
    device: str = "cuda",
) -> Optional[Dict[str, Any]]:
    """
    Load training state from checkpoint directory.

    Args:
        path: Path to checkpoint directory
        device: Device to load tensors to

    Returns:
        Training state dictionary or None if not found
    """
    path = Path(path)

    if path.is_file():
        checkpoint_dir = path.parent
    else:
        checkpoint_dir = path

    training_path = checkpoint_dir / "training_state.pt"
    if training_path.exists():
        return torch.load(training_path, map_location=device)
    return None


def get_checkpoint_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.

    Args:
        path: Path to checkpoint directory

    Returns:
        Dictionary with checkpoint information
    """
    path = Path(path)

    if path.is_file():
        checkpoint_dir = path.parent
    else:
        checkpoint_dir = path

    info = {"path": str(checkpoint_dir)}

    # Load config
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            info["config"] = json.load(f)

    # Load metadata
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            info["metadata"] = json.load(f)

    # Check for weights file
    if (checkpoint_dir / "model.safetensors").exists():
        info["weights_format"] = "safetensors"
        info["weights_path"] = str(checkpoint_dir / "model.safetensors")
    elif (checkpoint_dir / "model.pt").exists():
        info["weights_format"] = "pytorch"
        info["weights_path"] = str(checkpoint_dir / "model.pt")

    # Check for training state
    info["has_training_state"] = (checkpoint_dir / "training_state.pt").exists()

    return info


def convert_checkpoint_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_format: str = "safetensors",
    device: str = "cpu",
) -> Path:
    """
    Convert checkpoint between formats.

    Args:
        input_path: Path to source checkpoint
        output_path: Path for converted checkpoint
        target_format: "safetensors" or "pytorch"
        device: Device for loading

    Returns:
        Path to converted checkpoint
    """
    # Load original
    model, config, metadata = load_sae_checkpoint(input_path, device=device)

    # Load training state if exists
    training_state = load_training_state(input_path, device=device)

    # Update metadata
    metadata["converted_from"] = str(input_path)
    metadata["conversion_date"] = datetime.utcnow().isoformat()

    # Save in new format
    use_safetensors = target_format == "safetensors"
    return save_sae_checkpoint(
        model=model,
        path=output_path,
        config=config,
        training_state=training_state,
        metadata=metadata,
        use_safetensors=use_safetensors,
    )


def list_checkpoints(directory: Union[str, Path]) -> list[Dict[str, Any]]:
    """
    List all SAE checkpoints in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of checkpoint info dictionaries
    """
    directory = Path(directory)
    checkpoints = []

    for path in directory.iterdir():
        if path.is_dir():
            # Check if it's a checkpoint directory
            if (path / "config.json").exists():
                try:
                    info = get_checkpoint_info(path)
                    checkpoints.append(info)
                except Exception:
                    pass  # Skip invalid checkpoints

    return checkpoints
