"""
HuggingFace SAE loader.

Supports loading SAEs stored in the EleutherAI/sparsify format:
  - Weight keys: encoder.weight, encoder.bias, W_dec, b_dec
  - Config: cfg.json per hookpoint
  - Activation: TopK

Also supports generic HuggingFace SAE repos with standard weight keys
(W_enc, W_dec, b_enc, b_dec).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from ..config import SAEConfig, SAEActivationType
from ..model_torch import SparseAutoencoderTorch
from .base import (
    SAELoader,
    ExternalSAESource,
    ExternalSAEMetadata,
    LoadedSAE,
)


class HuggingFaceLoader(SAELoader):
    """Load SAEs from HuggingFace Hub repositories."""

    # Known repo formats
    SPARSIFY_FORMAT = "sparsify"  # EleutherAI format
    STANDARD_FORMAT = "standard"  # W_enc/W_dec format

    def can_load(self, identifier: str) -> bool:
        """Check if identifier looks like a HuggingFace repo ID."""
        # HF repo IDs are in format "org/repo-name"
        parts = identifier.strip("/").split("/")
        return len(parts) == 2 and not identifier.startswith("google/gemma-scope")

    def load(
        self,
        identifier: str,
        hookpoint: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ) -> LoadedSAE:
        """
        Load an SAE from a HuggingFace repository.

        Args:
            identifier: HuggingFace repo ID (e.g. "EleutherAI/sae-SmolLM2-135M-64x")
            hookpoint: Specific hookpoint to load (e.g. "layers.0.mlp")
            device: Target device
            cache_dir: Cache directory for downloads
        """
        from huggingface_hub import hf_hub_download, list_repo_tree

        # Detect format by checking repo contents
        repo_format = self._detect_format(identifier)

        if repo_format == self.SPARSIFY_FORMAT:
            return self._load_sparsify(identifier, hookpoint, device, cache_dir)
        else:
            return self._load_standard(identifier, hookpoint, device, cache_dir)

    def list_available(self, identifier: str) -> list[Dict[str, Any]]:
        """List available hookpoints/SAEs in a HuggingFace repository."""
        from huggingface_hub import list_repo_tree

        available = []
        try:
            tree = list(list_repo_tree(identifier, recursive=True))
            dirs = set()

            for item in tree:
                path = item.rfilename if hasattr(item, 'rfilename') else str(item)
                # Look for sae.safetensors or cfg.json in subdirectories
                if path.endswith("/sae.safetensors") or path.endswith("/cfg.json"):
                    hookpoint = path.rsplit("/", 1)[0]
                    dirs.add(hookpoint)
                # Also check for model.safetensors in subdirectories
                elif path.endswith("/model.safetensors"):
                    hookpoint = path.rsplit("/", 1)[0]
                    dirs.add(hookpoint)

            # Try loading config for each hookpoint
            for hookpoint in sorted(dirs):
                info = {"hookpoint": hookpoint, "repo_id": identifier}
                try:
                    cfg = self._load_hookpoint_config(identifier, hookpoint)
                    if cfg:
                        info.update(cfg)
                except Exception:
                    pass
                available.append(info)

            # If no subdirectories found, check if SAE is at repo root
            if not available:
                root_files = [
                    item.rfilename if hasattr(item, 'rfilename') else str(item)
                    for item in tree
                ]
                has_weights = any(
                    f.endswith(".safetensors") and "/" not in f
                    for f in root_files
                )
                if has_weights:
                    available.append({
                        "hookpoint": "",
                        "repo_id": identifier,
                    })

        except Exception as e:
            available.append({
                "hookpoint": "",
                "repo_id": identifier,
                "error": str(e),
            })

        return available

    def _detect_format(self, identifier: str) -> str:
        """Detect the SAE format used in a repository."""
        from huggingface_hub import list_repo_tree

        try:
            tree = list(list_repo_tree(identifier, recursive=True))
            filenames = [
                item.rfilename if hasattr(item, 'rfilename') else str(item)
                for item in tree
            ]

            # EleutherAI/sparsify format has cfg.json and sae.safetensors
            has_cfg_json = any(f.endswith("cfg.json") for f in filenames)
            has_sae_safetensors = any(f.endswith("sae.safetensors") for f in filenames)

            if has_cfg_json and has_sae_safetensors:
                return self.SPARSIFY_FORMAT

        except Exception:
            pass

        return self.STANDARD_FORMAT

    def _load_hookpoint_config(
        self, repo_id: str, hookpoint: str
    ) -> Optional[Dict[str, Any]]:
        """Load config for a specific hookpoint."""
        from huggingface_hub import hf_hub_download

        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{hookpoint}/cfg.json" if hookpoint else "cfg.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _load_sparsify(
        self,
        repo_id: str,
        hookpoint: Optional[str],
        device: str,
        cache_dir: Optional[Path],
    ) -> LoadedSAE:
        """
        Load EleutherAI/sparsify format SAE.

        Weight keys: encoder.weight (num_latents, d_in), encoder.bias (num_latents,),
                     W_dec (num_latents, d_in), b_dec (d_in,)
        Config: cfg.json with d_in, num_latents, k, normalize_decoder, etc.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if not hookpoint:
            # Pick first available hookpoint
            available = self.list_available(repo_id)
            if not available:
                raise ValueError(f"No SAE hookpoints found in {repo_id}")
            hookpoint = available[0]["hookpoint"]

        # Download config
        cfg_filename = f"{hookpoint}/cfg.json" if hookpoint else "cfg.json"
        cfg_path = hf_hub_download(
            repo_id=repo_id,
            filename=cfg_filename,
            cache_dir=cache_dir,
        )
        with open(cfg_path) as f:
            cfg = json.load(f)

        d_input = cfg["d_in"]
        num_latents = cfg.get("num_latents", 0)
        k = cfg.get("k", 32)
        normalize_decoder = cfg.get("normalize_decoder", True)

        if num_latents == 0:
            expansion = cfg.get("expansion_factor", 64)
            num_latents = d_input * expansion

        # Download weights
        weights_filename = f"{hookpoint}/sae.safetensors" if hookpoint else "sae.safetensors"
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filename,
            cache_dir=cache_dir,
        )
        raw_weights = load_file(weights_path, device="cpu")

        # Normalize to our format
        # sparsify: encoder.weight is (num_latents, d_in) - needs transpose to (d_in, num_latents)
        W_enc = raw_weights["encoder.weight"].T  # (d_in, num_latents) -> our (d_input, d_hidden)
        b_enc = raw_weights["encoder.bias"]  # (num_latents,)
        W_dec = raw_weights["W_dec"]  # (num_latents, d_in) -> our (d_hidden, d_input)
        b_dec = raw_weights["b_dec"]  # (d_in,)

        weights = {
            "W_enc": W_enc,
            "W_dec": W_dec,
            "b_enc": b_enc,
            "b_dec": b_dec,
        }

        # Build config
        config = self.build_config(
            d_input=d_input,
            d_hidden=num_latents,
            activation=SAEActivationType.TOPK,
            topk_k=k,
            normalize_decoder=normalize_decoder,
            use_encoder_bias=True,
            use_decoder_bias=True,
        )

        # Load top-level config for base model name
        base_model = repo_id
        try:
            top_cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                cache_dir=cache_dir,
            )
            with open(top_cfg_path) as f:
                top_cfg = json.load(f)
            base_model = top_cfg.get("model", repo_id)
        except Exception:
            pass

        # Build metadata
        metadata = ExternalSAEMetadata(
            source=ExternalSAESource.HUGGINGFACE,
            source_id=repo_id,
            display_name=f"{repo_id.split('/')[-1]} ({hookpoint})",
            base_model=base_model,
            hookpoint=hookpoint,
            d_input=d_input,
            d_hidden=num_latents,
            activation_type="topk",
            extra={
                "k": k,
                "normalize_decoder": normalize_decoder,
                "format": self.SPARSIFY_FORMAT,
                "skip_connection": cfg.get("skip_connection", False),
            },
        )

        model = self.build_model_from_weights(config, weights, device)

        return LoadedSAE(
            model=model,
            config=config,
            metadata=metadata,
            weights=weights,
        )

    def _load_standard(
        self,
        repo_id: str,
        hookpoint: Optional[str],
        device: str,
        cache_dir: Optional[Path],
    ) -> LoadedSAE:
        """
        Load standard format SAE (W_enc, W_dec, b_enc, b_dec keys).

        Falls back to trying common weight file names.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Try common weight filenames
        weight_filenames = []
        if hookpoint:
            weight_filenames.extend([
                f"{hookpoint}/model.safetensors",
                f"{hookpoint}/sae.safetensors",
            ])
        weight_filenames.extend([
            "model.safetensors",
            "sae.safetensors",
        ])

        raw_weights = None
        for filename in weight_filenames:
            try:
                weights_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                )
                raw_weights = load_file(weights_path, device="cpu")
                break
            except Exception:
                continue

        if raw_weights is None:
            raise ValueError(
                f"Could not find weights in {repo_id}. "
                f"Tried: {weight_filenames}"
            )

        # Normalize weight keys
        weights = self._normalize_weight_keys(raw_weights)

        d_input = weights["W_enc"].shape[0]
        d_hidden = weights["W_enc"].shape[1]

        # Determine activation type from weights
        has_threshold = "threshold" in weights
        activation = SAEActivationType.JUMPRELU if has_threshold else SAEActivationType.RELU

        # Try loading config
        config_data = {}
        for cfg_name in ["config.json", "cfg.json"]:
            prefix = f"{hookpoint}/" if hookpoint else ""
            try:
                cfg_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{prefix}{cfg_name}",
                    cache_dir=cache_dir,
                )
                with open(cfg_path) as f:
                    config_data = json.load(f)
                break
            except Exception:
                continue

        config = self.build_config(
            d_input=d_input,
            d_hidden=d_hidden,
            activation=activation,
            normalize_decoder=config_data.get("normalize_decoder", True),
            use_encoder_bias="b_enc" in weights,
            use_decoder_bias="b_dec" in weights,
        )

        metadata = ExternalSAEMetadata(
            source=ExternalSAESource.HUGGINGFACE,
            source_id=repo_id,
            display_name=f"{repo_id.split('/')[-1]}" + (f" ({hookpoint})" if hookpoint else ""),
            base_model=config_data.get("model", repo_id),
            hookpoint=hookpoint or "",
            d_input=d_input,
            d_hidden=d_hidden,
            activation_type=activation.value,
            extra={"format": self.STANDARD_FORMAT, **config_data},
        )

        model = self.build_model_from_weights(config, weights, device)

        return LoadedSAE(
            model=model,
            config=config,
            metadata=metadata,
            weights=weights,
        )

    @staticmethod
    def _normalize_weight_keys(raw_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize various weight key formats to our standard."""
        weights = {}

        # Encoder weights
        if "W_enc" in raw_weights:
            weights["W_enc"] = raw_weights["W_enc"]
        elif "encoder.weight" in raw_weights:
            # nn.Linear format: (out, in) -> we need (in, out)
            weights["W_enc"] = raw_weights["encoder.weight"].T
        elif "w_enc" in raw_weights:
            weights["W_enc"] = raw_weights["w_enc"]
        else:
            raise ValueError("No encoder weights found. Expected W_enc or encoder.weight")

        # Decoder weights
        if "W_dec" in raw_weights:
            weights["W_dec"] = raw_weights["W_dec"]
        elif "decoder.weight" in raw_weights:
            weights["W_dec"] = raw_weights["decoder.weight"]
        elif "w_dec" in raw_weights:
            weights["W_dec"] = raw_weights["w_dec"]
        else:
            raise ValueError("No decoder weights found. Expected W_dec or decoder.weight")

        # Encoder bias (optional)
        if "b_enc" in raw_weights:
            weights["b_enc"] = raw_weights["b_enc"]
        elif "encoder.bias" in raw_weights:
            weights["b_enc"] = raw_weights["encoder.bias"]

        # Decoder bias (optional)
        if "b_dec" in raw_weights:
            weights["b_dec"] = raw_weights["b_dec"]
        elif "decoder.bias" in raw_weights:
            weights["b_dec"] = raw_weights["decoder.bias"]

        # JumpReLU threshold (optional)
        if "threshold" in raw_weights:
            weights["threshold"] = raw_weights["threshold"]

        return weights
