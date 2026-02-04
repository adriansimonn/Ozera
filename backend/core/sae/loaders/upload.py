"""
User upload handler for SAE safetensors files.

Handles direct safetensors file uploads, auto-detecting the weight format
and building the appropriate SAE model. Supports:
  - Ozera native format (W_enc, W_dec, b_enc, b_dec)
  - EleutherAI/sparsify format (encoder.weight, encoder.bias, W_dec, b_dec)
  - Gemma Scope format (W_enc, W_dec, b_enc, b_dec, threshold)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from ..config import SAEConfig, SAEActivationType
from .base import (
    SAELoader,
    ExternalSAESource,
    ExternalSAEMetadata,
    LoadedSAE,
)


class UploadLoader(SAELoader):
    """Load SAEs from user-uploaded safetensors files."""

    def can_load(self, identifier: str) -> bool:
        """Check if identifier is a path to a safetensors file."""
        return identifier.endswith(".safetensors") and Path(identifier).exists()

    def load(
        self,
        identifier: str,
        hookpoint: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ) -> LoadedSAE:
        """
        Load an SAE from a local safetensors file.

        Args:
            identifier: Path to the safetensors file
            hookpoint: Ignored for uploads
            device: Target device
            cache_dir: Ignored for uploads
        """
        from safetensors.torch import load_file

        file_path = Path(identifier)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {identifier}")
        if not file_path.suffix == ".safetensors":
            raise ValueError("Only .safetensors files are supported for upload")

        # Load raw weights
        raw_weights = load_file(str(file_path), device="cpu")

        # Detect format and normalize weights
        weights, format_name, activation_type = self._detect_and_normalize(raw_weights)

        d_input = weights["W_enc"].shape[0]
        d_hidden = weights["W_enc"].shape[1]

        # Try loading companion config file
        config_data = self._try_load_config(file_path)

        # Determine TopK k value if applicable
        topk_k = 32
        if config_data:
            topk_k = config_data.get("k", config_data.get("topk_k", 32))

        config = self.build_config(
            d_input=d_input,
            d_hidden=d_hidden,
            activation=activation_type,
            topk_k=topk_k,
            normalize_decoder=config_data.get("normalize_decoder", True) if config_data else True,
            use_encoder_bias="b_enc" in weights,
            use_decoder_bias="b_dec" in weights,
        )

        metadata = ExternalSAEMetadata(
            source=ExternalSAESource.USER_UPLOAD,
            source_id=file_path.name,
            display_name=file_path.stem,
            base_model=config_data.get("model", "unknown") if config_data else "unknown",
            hookpoint=config_data.get("hookpoint", "") if config_data else "",
            d_input=d_input,
            d_hidden=d_hidden,
            activation_type=activation_type.value,
            extra={
                "format": format_name,
                "file_path": str(file_path),
                "file_size_bytes": file_path.stat().st_size,
                **(config_data if config_data else {}),
            },
        )

        model = self.build_model_from_weights(config, weights, device)

        return LoadedSAE(
            model=model,
            config=config,
            metadata=metadata,
            weights=weights,
        )

    def load_from_bytes(
        self,
        data: bytes,
        filename: str,
        config_data: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> LoadedSAE:
        """
        Load an SAE from raw bytes (for HTTP upload handling).

        Args:
            data: Raw safetensors file bytes
            filename: Original filename
            config_data: Optional config dict provided alongside upload
            device: Target device
        """
        from safetensors.torch import load_file
        import tempfile

        # Write bytes to temp file for safetensors loading
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(data)
            tmp_path = f.name

        try:
            raw_weights = load_file(tmp_path, device="cpu")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Detect format and normalize
        weights, format_name, activation_type = self._detect_and_normalize(raw_weights)

        d_input = weights["W_enc"].shape[0]
        d_hidden = weights["W_enc"].shape[1]

        topk_k = 32
        if config_data:
            topk_k = config_data.get("k", config_data.get("topk_k", 32))

        config = self.build_config(
            d_input=d_input,
            d_hidden=d_hidden,
            activation=activation_type,
            topk_k=topk_k,
            normalize_decoder=config_data.get("normalize_decoder", True) if config_data else True,
            use_encoder_bias="b_enc" in weights,
            use_decoder_bias="b_dec" in weights,
        )

        metadata = ExternalSAEMetadata(
            source=ExternalSAESource.USER_UPLOAD,
            source_id=filename,
            display_name=Path(filename).stem,
            base_model=config_data.get("model", "unknown") if config_data else "unknown",
            hookpoint=config_data.get("hookpoint", "") if config_data else "",
            d_input=d_input,
            d_hidden=d_hidden,
            activation_type=activation_type.value,
            extra={
                "format": format_name,
                "filename": filename,
                "file_size_bytes": len(data),
                **(config_data if config_data else {}),
            },
        )

        model = self.build_model_from_weights(config, weights, device)

        return LoadedSAE(
            model=model,
            config=config,
            metadata=metadata,
            weights=weights,
        )

    def list_available(self, identifier: str) -> list[Dict[str, Any]]:
        """For uploads, just return info about the file."""
        path = Path(identifier)
        if path.exists() and path.suffix == ".safetensors":
            return [{
                "hookpoint": "",
                "file_path": str(path),
                "file_size_bytes": path.stat().st_size,
            }]
        return []

    @staticmethod
    def _detect_and_normalize(
        raw_weights: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], str, SAEActivationType]:
        """
        Detect weight format and normalize to standard keys.

        Returns:
            (normalized_weights, format_name, activation_type)
        """
        keys = set(raw_weights.keys())

        # Detect format
        if "encoder.weight" in keys:
            # EleutherAI/sparsify format
            format_name = "sparsify"
            W_enc = raw_weights["encoder.weight"].T  # (out, in) -> (in, out)
            b_enc = raw_weights.get("encoder.bias")
            W_dec = raw_weights["W_dec"]
            b_dec = raw_weights.get("b_dec")

            activation_type = SAEActivationType.TOPK

        elif "W_enc" in keys:
            # Ozera native or Gemma Scope format
            W_enc = raw_weights["W_enc"]
            W_dec = raw_weights["W_dec"]
            b_enc = raw_weights.get("b_enc")
            b_dec = raw_weights.get("b_dec")

            if "threshold" in keys:
                format_name = "gemma_scope"
                activation_type = SAEActivationType.JUMPRELU
            else:
                format_name = "standard"
                activation_type = SAEActivationType.RELU

        else:
            # Try to find any weight-like tensors
            raise ValueError(
                f"Unrecognized weight format. Found keys: {sorted(keys)}. "
                f"Expected W_enc/W_dec or encoder.weight/W_dec."
            )

        weights: Dict[str, torch.Tensor] = {
            "W_enc": W_enc,
            "W_dec": W_dec,
        }
        if b_enc is not None:
            weights["b_enc"] = b_enc
        if b_dec is not None:
            weights["b_dec"] = b_dec
        if "threshold" in raw_weights:
            weights["threshold"] = raw_weights["threshold"]

        return weights, format_name, activation_type

    @staticmethod
    def _try_load_config(weights_path: Path) -> Optional[Dict[str, Any]]:
        """Try to load a companion config file next to the weights file."""
        for config_name in ["config.json", "cfg.json", "sae_config.json"]:
            config_path = weights_path.parent / config_name
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        return json.load(f)
                except Exception:
                    continue
        return None
