"""
Gemma Scope SAE loader.

Supports loading SAEs from Google's Gemma Scope repositories:
  - Gemma Scope 1: params.npz format (Gemma 2 models)
  - Gemma Scope 2: params.safetensors format (Gemma 3 models)
  - Weight keys: W_enc, W_dec, b_enc, b_dec, threshold
  - Activation: JumpReLU with learned per-feature thresholds

Repo naming conventions:
  - Gemma Scope 1: google/gemma-scope-{size}-{pt|it}-{res|mlp|att}
    Path: layer_{N}/width_{W}/average_l0_{L}/params.npz
  - Gemma Scope 2: google/gemma-scope-2-{size}-{pt|it}
    Path: {site}/{layer_name}/params.safetensors
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import numpy as np

from ..config import SAEConfig, SAEActivationType
from .base import (
    SAELoader,
    ExternalSAESource,
    ExternalSAEMetadata,
    LoadedSAE,
)


class GemmaScopeLoader(SAELoader):
    """Load SAEs from Google's Gemma Scope repositories."""

    # Pattern for Gemma Scope repo IDs
    GEMMA_SCOPE_PATTERN = re.compile(
        r"^google/gemma-scope(-2)?-(\w+)-(pt|it)(-res|-mlp|-att)?$"
    )

    def can_load(self, identifier: str) -> bool:
        """Check if identifier is a Gemma Scope repository."""
        return bool(self.GEMMA_SCOPE_PATTERN.match(identifier))

    def load(
        self,
        identifier: str,
        hookpoint: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ) -> LoadedSAE:
        """
        Load a Gemma Scope SAE.

        Args:
            identifier: Gemma Scope repo ID (e.g. "google/gemma-scope-2b-pt")
            hookpoint: Path within repo (e.g. "layer_20/width_16k/average_l0_71")
            device: Target device
            cache_dir: Cache directory
        """
        from huggingface_hub import hf_hub_download

        # If no hookpoint specified, get the first available one
        if not hookpoint:
            available = self.list_available(identifier)
            if not available:
                raise ValueError(f"No SAEs found in {identifier}")
            if "error" in available[0]:
                raise ValueError(available[0]["error"])
            hookpoint = available[0]["hookpoint"]

        # Auto-detect format by trying to download the file
        # Try v2 format (safetensors) first, then fall back to v1 (npz)
        try:
            # Try v2 format
            weights_path = hf_hub_download(
                repo_id=identifier,
                filename=f"{hookpoint}/params.safetensors",
                cache_dir=cache_dir,
            )
            return self._load_v2(identifier, hookpoint, device, cache_dir)
        except Exception:
            # Fall back to v1 format
            try:
                npz_path = hf_hub_download(
                    repo_id=identifier,
                    filename=f"{hookpoint}/params.npz",
                    cache_dir=cache_dir,
                )
                return self._load_v1(identifier, hookpoint, device, cache_dir)
            except Exception as e:
                raise ValueError(
                    f"Could not load SAE from {identifier}/{hookpoint}. "
                    f"Expected params.safetensors or params.npz. Error: {str(e)}"
                )

    def list_available(self, identifier: str) -> list[Dict[str, Any]]:
        """List available SAEs in a Gemma Scope repository."""
        from huggingface_hub import list_repo_tree

        available = []

        try:
            tree = list(list_repo_tree(identifier, recursive=True))
            filenames = [
                item.rfilename if hasattr(item, 'rfilename') else str(item)
                for item in tree
            ]

            # Auto-detect format by looking for both safetensors and npz files
            has_safetensors = any(f.endswith("params.safetensors") for f in filenames)
            has_npz = any(f.endswith("params.npz") for f in filenames)

            if has_safetensors:
                # v2 format: Look for params.safetensors
                for f in filenames:
                    if f.endswith("params.safetensors"):
                        hookpoint = f.rsplit("/", 1)[0]
                        info = self._parse_v2_path(hookpoint)
                        info["repo_id"] = identifier
                        info["hookpoint"] = hookpoint
                        available.append(info)
            elif has_npz:
                # v1 format: Look for params.npz
                for f in filenames:
                    if f.endswith("params.npz"):
                        hookpoint = f.rsplit("/", 1)[0]
                        info = self._parse_v1_path(hookpoint)
                        info["repo_id"] = identifier
                        info["hookpoint"] = hookpoint
                        available.append(info)

            # If no SAE files found, provide a helpful error
            if not available:
                if not has_safetensors and not has_npz:
                    raise ValueError(
                        f"No Gemma Scope SAE files found in {identifier}. "
                        f"Expected params.safetensors or params.npz files. "
                        f"Please check the repository name is correct."
                    )

        except Exception as e:
            # Return error in a way that the frontend can display it
            available.append({
                "hookpoint": "",
                "repo_id": identifier,
                "error": str(e),
            })

        return available

    def _is_gemma_scope_2(self, identifier: str) -> bool:
        """
        Check if this is a Gemma Scope 2 repository.

        Gemma Scope 2 uses safetensors format, Gemma Scope 1 uses npz.
        v2 repos have pattern: google/gemma-scope-2-{size}-{pt|it}
        v1 repos have pattern: google/gemma-scope-{size}-{pt|it}(-{res|mlp|att})?

        But for newer releases, all repos use safetensors (v2 format) even without the -2-
        """
        # Explicitly check for v2 naming pattern
        if "gemma-scope-2-" in identifier:
            return True
        # For other Gemma Scope repos, we'll detect dynamically by checking file types

    def _parse_v1_path(self, path: str) -> Dict[str, Any]:
        """Parse Gemma Scope 1 path: layer_N/width_W/average_l0_L."""
        info: Dict[str, Any] = {}
        parts = path.split("/")
        for part in parts:
            if part.startswith("layer_"):
                try:
                    info["layer"] = int(part.split("_")[1])
                except (IndexError, ValueError):
                    pass
            elif part.startswith("width_"):
                width_str = part.split("_")[1]
                if width_str.endswith("k"):
                    info["width"] = int(width_str[:-1]) * 1000
                elif width_str.endswith("m"):
                    info["width"] = int(width_str[:-1]) * 1000000
                else:
                    try:
                        info["width"] = int(width_str)
                    except ValueError:
                        pass
            elif part.startswith("average_l0_"):
                try:
                    info["l0"] = int(part.split("average_l0_")[1])
                except (IndexError, ValueError):
                    pass
        return info

    def _parse_v2_path(self, path: str) -> Dict[str, Any]:
        """Parse Gemma Scope 2 path: site/layer_N_width_W_l0_size."""
        info: Dict[str, Any] = {}
        parts = path.split("/")
        if parts:
            info["site"] = parts[0]  # resid_post, mlp_out, attn_out, etc.

        if len(parts) > 1:
            subparts = parts[1].split("_")
            for i, sp in enumerate(subparts):
                if sp == "layer" and i + 1 < len(subparts):
                    try:
                        info["layer"] = int(subparts[i + 1])
                    except ValueError:
                        pass
                elif sp == "width" and i + 1 < len(subparts):
                    width_str = subparts[i + 1]
                    if width_str.endswith("k"):
                        info["width"] = int(width_str[:-1]) * 1000
                    else:
                        try:
                            info["width"] = int(width_str)
                        except ValueError:
                            pass
        return info

    def _load_v1(
        self,
        repo_id: str,
        hookpoint: Optional[str],
        device: str,
        cache_dir: Optional[Path],
    ) -> LoadedSAE:
        """Load Gemma Scope 1 format (params.npz)."""
        from huggingface_hub import hf_hub_download

        if not hookpoint:
            available = self.list_available(repo_id)
            if not available:
                raise ValueError(f"No SAEs found in {repo_id}")
            hookpoint = available[0]["hookpoint"]

        # Download params.npz
        npz_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{hookpoint}/params.npz",
            cache_dir=cache_dir,
        )

        # Load numpy arrays and convert to torch
        params = np.load(npz_path)
        weights = {k: torch.from_numpy(v.copy()) for k, v in params.items()}

        d_input = weights["W_enc"].shape[0]
        d_hidden = weights["W_enc"].shape[1]

        # Parse path info
        path_info = self._parse_v1_path(hookpoint)

        # Determine base model from repo ID
        match = self.GEMMA_SCOPE_PATTERN.match(repo_id)
        size = match.group(2) if match else "unknown"
        variant = match.group(3) if match else "pt"
        site = ""
        if match and match.group(4):
            site = match.group(4).lstrip("-")  # res, mlp, att

        base_model = f"google/gemma-2-{size}"

        config = self.build_config(
            d_input=d_input,
            d_hidden=d_hidden,
            activation=SAEActivationType.JUMPRELU,
            normalize_decoder=True,
            use_encoder_bias=True,
            use_decoder_bias=True,
        )

        metadata = ExternalSAEMetadata(
            source=ExternalSAESource.GEMMA_SCOPE,
            source_id=repo_id,
            display_name=f"Gemma Scope {size} L{path_info.get('layer', '?')} ({site})",
            base_model=base_model,
            hookpoint=hookpoint,
            d_input=d_input,
            d_hidden=d_hidden,
            activation_type="jumprelu",
            extra={
                "version": 1,
                "size": size,
                "variant": variant,
                "site": site,
                "layer": path_info.get("layer"),
                "width": path_info.get("width", d_hidden),
                "l0": path_info.get("l0"),
            },
        )

        model = self.build_model_from_weights(config, weights, device)

        return LoadedSAE(
            model=model,
            config=config,
            metadata=metadata,
            weights=weights,
        )

    def _load_v2(
        self,
        repo_id: str,
        hookpoint: Optional[str],
        device: str,
        cache_dir: Optional[Path],
    ) -> LoadedSAE:
        """Load Gemma Scope 2 format (params.safetensors + config.json)."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if not hookpoint:
            available = self.list_available(repo_id)
            if not available:
                raise ValueError(f"No SAEs found in {repo_id}")
            hookpoint = available[0]["hookpoint"]

        # Download weights
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{hookpoint}/params.safetensors",
            cache_dir=cache_dir,
        )
        weights = load_file(weights_path, device="cpu")

        d_input = weights["W_enc"].shape[0]
        d_hidden = weights["W_enc"].shape[1]

        # Try loading config
        config_data = {}
        try:
            cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{hookpoint}/config.json",
                cache_dir=cache_dir,
            )
            with open(cfg_path) as f:
                config_data = json.load(f)
        except Exception:
            pass

        path_info = self._parse_v2_path(hookpoint)

        # Determine base model
        match = self.GEMMA_SCOPE_PATTERN.match(repo_id)
        size = match.group(2) if match else "unknown"
        variant = match.group(3) if match else "pt"
        base_model = config_data.get("model_name", f"google/gemma-3-{size}")

        config = self.build_config(
            d_input=d_input,
            d_hidden=d_hidden,
            activation=SAEActivationType.JUMPRELU,
            normalize_decoder=True,
            use_encoder_bias=True,
            use_decoder_bias=True,
        )

        site = path_info.get("site", config_data.get("hf_hook_point_in", ""))
        layer = path_info.get("layer")

        metadata = ExternalSAEMetadata(
            source=ExternalSAESource.GEMMA_SCOPE,
            source_id=repo_id,
            display_name=f"Gemma Scope 2 {size} L{layer or '?'} ({site})",
            base_model=base_model,
            hookpoint=hookpoint,
            d_input=d_input,
            d_hidden=d_hidden,
            activation_type="jumprelu",
            extra={
                "version": 2,
                "size": size,
                "variant": variant,
                "site": site,
                "layer": layer,
                "width": config_data.get("width", d_hidden),
                "l0": config_data.get("l0"),
                "architecture": config_data.get("architecture", "jump_relu"),
                "affine_connection": config_data.get("affine_connection", False),
            },
        )

        model = self.build_model_from_weights(config, weights, device)

        return LoadedSAE(
            model=model,
            config=config,
            metadata=metadata,
            weights=weights,
        )
