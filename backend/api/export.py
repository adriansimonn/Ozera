"""
Export API endpoints for paper-ready figure generation.

Provides endpoints for exporting attention heatmaps and activation histograms
in publication-quality formats (PNG, PDF, SVG).
"""

import re

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
from io import BytesIO
import zipfile

from core.export import (
    AttentionHeatmapExporter,
    ActivationHistogramExporter,
    PUBLICATION_PRESETS,
)
from core.patching import get_patching_engine
from inference.activation_store import get_activation_store
from middleware.auth_middleware import get_current_user
from models.database import User


router = APIRouter(prefix="/export", tags=["Export"])

activation_store = get_activation_store()


# ============= Request Models =============

class ExportConfig(BaseModel):
    """Common export configuration."""
    preset: str = Field(default="default", description="Publication preset")
    format: Literal["png", "pdf", "svg"] = Field(default="png", description="Output format")
    width: Optional[float] = Field(default=None, description="Custom width in inches")
    height: Optional[float] = Field(default=None, description="Custom height in inches")
    dpi: Optional[int] = Field(default=None, description="Custom DPI")
    transparent: bool = Field(default=False, description="Transparent background")


class AttentionHeatmapRequest(BaseModel):
    """Request to export attention heatmap."""
    activation_id: str = Field(..., description="ID of captured activations")
    layer: int = Field(..., description="Layer index")
    head: int = Field(..., description="Head index")
    config: ExportConfig = Field(default_factory=ExportConfig)
    colormap: str = Field(default="inferno", description="Matplotlib colormap")
    title: Optional[str] = Field(default=None, description="Custom title")
    show_colorbar: bool = Field(default=True, description="Show colorbar")
    show_values: bool = Field(default=False, description="Show values in cells")
    token_labels: Literal["text", "number"] = Field(default="text", description="Label axes with token text or position numbers")


class MultiHeadHeatmapRequest(BaseModel):
    """Request to export multi-head attention heatmap."""
    activation_id: str = Field(..., description="ID of captured activations")
    layer: int = Field(..., description="Layer index")
    heads: Optional[list[int]] = Field(default=None, description="Head indices (None for all)")
    config: ExportConfig = Field(default_factory=ExportConfig)
    colormap: str = Field(default="inferno", description="Matplotlib colormap")
    title: Optional[str] = Field(default=None, description="Custom title")
    cols: int = Field(default=4, description="Number of columns in grid")


class ActivationHistogramRequest(BaseModel):
    """Request to export activation histogram."""
    activation_id: str = Field(..., description="ID of captured activations")
    layer: Optional[int] = Field(default=None, description="Layer index (None for all)")
    activation_type: Literal["attn_output", "ff_output", "all"] = Field(
        default="all", description="Type of activation to plot"
    )
    config: ExportConfig = Field(default_factory=ExportConfig)
    color: str = Field(default="#22d3ee", description="Bar color")
    title: Optional[str] = Field(default=None, description="Custom title")
    bins: int = Field(default=50, description="Number of histogram bins")
    show_stats: bool = Field(default=True, description="Show statistics box")
    log_scale: bool = Field(default=False, description="Use log scale for y-axis")


class BatchExportRequest(BaseModel):
    """Request to export multiple figures as ZIP."""
    exports: list[dict] = Field(..., description="List of export requests")


class PresetInfo(BaseModel):
    """Information about a publication preset."""
    name: str
    width_inches: float
    height_inches: float
    dpi: int
    font_family: str
    font_size: int


# ============= Helper Functions =============

def _stored_summary(activation_id: str, user_id: int) -> dict:
    """Summary (no tensors) of the user's stored activations, or 404."""
    summary = activation_store.get_activation_summary(activation_id, user_id)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"Activations not found: {activation_id}"
        )
    return summary


def get_attention_matrix(activation_id: str, layer: int, user_id: int) -> tuple[np.ndarray, list[str]]:
    """Get attention matrix from the user's stored activations."""
    # Try patching engine first (for captured activations)
    engine = get_patching_engine()
    captured = engine.get_captured_activations(activation_id, user_id)

    if captured is not None:
        # Look for attention weights in captured activations
        attn_key = f"layer_{layer}_attn_weights"
        if attn_key not in captured.activations:
            # Try alternative key formats
            for key in captured.activations:
                if f"layer_{layer}" in key and ("attn" in key or "attention" in key):
                    if "weight" in key or "pattern" in key:
                        attn_key = key
                        break

        if attn_key not in captured.activations:
            raise HTTPException(
                status_code=404,
                detail=f"Attention weights not found for layer {layer}"
            )

        attn_weights = captured.activations[attn_key]
        tokens = captured.decoded_tokens or [f"t{i}" for i in range(attn_weights.shape[-1])]
        return attn_weights, tokens

    # Try activation store (activations from a generation)
    summary = _stored_summary(activation_id, user_id)
    num_layers = summary["num_layers"]
    if layer < 0 or layer >= num_layers:
        raise HTTPException(
            status_code=404,
            detail=f"Layer {layer} not found (have {num_layers} layers)"
        )

    attn_weights = activation_store.get_layer_array(activation_id, layer, "attn_weights", user_id)
    if attn_weights is None:
        raise HTTPException(
            status_code=404,
            detail=f"Attention weights not found for layer {layer}"
        )

    # Label with the decoded tokens (the attention covers them all; fall back to positions
    # if they don't match)
    seq_len = attn_weights.shape[-1]
    tokens = summary["metadata"].get("decoded_tokens") or []
    if len(tokens) >= seq_len:
        tokens = tokens[len(tokens) - seq_len:]
    else:
        tokens = [f"t{i}" for i in range(seq_len)]

    return attn_weights, tokens


def get_activations_for_histogram(
    activation_id: str,
    layer: Optional[int],
    activation_type: str,
    user_id: int,
) -> np.ndarray:
    """Get activations for histogram from the user's stored data."""
    # Try patching engine first
    engine = get_patching_engine()
    captured = engine.get_captured_activations(activation_id, user_id)

    all_activations = []

    if captured is not None:
        if layer is not None:
            # Get specific layer
            if activation_type in ["attn_output", "all"]:
                key = f"layer_{layer}_attn_output"
                if key in captured.activations:
                    all_activations.append(captured.activations[key])

            if activation_type in ["ff_output", "all"]:
                key = f"layer_{layer}_ff_output"
                if key in captured.activations:
                    all_activations.append(captured.activations[key])
        else:
            # Get all layers
            for key, value in captured.activations.items():
                if activation_type == "all":
                    if "output" in key:
                        all_activations.append(value)
                elif activation_type in key:
                    all_activations.append(value)
    else:
        # Try activation store (activations from a generation)
        summary = _stored_summary(activation_id, user_id)
        layer_indices = [layer] if layer is not None else range(summary["num_layers"])
        keys = ["attn_output", "ff_output"] if activation_type == "all" else [activation_type]

        for layer_idx in layer_indices:
            for key in keys:
                values = activation_store.get_layer_array(activation_id, layer_idx, key, user_id)
                if values is not None:
                    all_activations.append(values)

    if not all_activations:
        raise HTTPException(
            status_code=404,
            detail=f"No activations found for type: {activation_type}"
        )

    return np.concatenate([a.flatten() for a in all_activations])


# ============= API Endpoints =============

@router.get("/presets")
async def list_presets() -> dict[str, PresetInfo]:
    """
    List available publication presets.

    Returns:
        Dictionary of preset names to preset information
    """
    return {
        name: PresetInfo(
            name=preset.name,
            width_inches=preset.width_inches,
            height_inches=preset.height_inches,
            dpi=preset.dpi,
            font_family=preset.font_family,
            font_size=preset.font_size,
        )
        for name, preset in PUBLICATION_PRESETS.items()
    }


@router.post("/attention-heatmap")
async def export_attention_heatmap(
    request: AttentionHeatmapRequest,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Export attention heatmap as publication-ready figure.

    Args:
        request: AttentionHeatmapRequest with activation_id, layer, head, and config

    Returns:
        Image file in requested format
    """
    # Get attention matrix
    attn_weights, tokens = get_attention_matrix(request.activation_id, request.layer, current_user.id)

    # Extract specific head
    # Shape could be [batch, heads, seq, seq] or [heads, seq, seq]
    if len(attn_weights.shape) == 4:
        attn_matrix = attn_weights[0, request.head]
    elif len(attn_weights.shape) == 3:
        attn_matrix = attn_weights[request.head]
    else:
        attn_matrix = attn_weights

    # Create exporter and figure
    exporter = AttentionHeatmapExporter(
        preset=request.config.preset,
        colormap=request.colormap,
        width=request.config.width,
        height=request.config.height,
        dpi=request.config.dpi,
    )

    # Use token text or numeric indices based on token_labels setting
    display_tokens = tokens if request.token_labels == "text" else None

    fig = exporter.create_figure(
        attention_matrix=attn_matrix,
        tokens=display_tokens,
        layer=request.layer,
        head=request.head,
        title=request.title,
        show_colorbar=request.show_colorbar,
        show_values=request.show_values,
    )

    # Export
    image_bytes = exporter.export(
        fig,
        format=request.config.format,
        transparent=request.config.transparent,
    )

    media_types = {
        "png": "image/png",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
    }

    return Response(
        content=image_bytes,
        media_type=media_types[request.config.format],
        headers={
            "Content-Disposition": f"attachment; filename=attention_L{request.layer}_H{request.head}.{request.config.format}"
        }
    )


@router.post("/attention-heatmap/multi-head")
async def export_multi_head_heatmap(
    request: MultiHeadHeatmapRequest,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Export multi-head attention heatmap as publication-ready figure.

    Args:
        request: MultiHeadHeatmapRequest with activation_id, layer, and config

    Returns:
        Image file in requested format
    """
    # Get attention matrix
    attn_weights, tokens = get_attention_matrix(request.activation_id, request.layer, current_user.id)

    # Extract all heads
    if len(attn_weights.shape) == 4:
        attn_matrices = attn_weights[0]  # [heads, seq, seq]
    elif len(attn_weights.shape) == 3:
        attn_matrices = attn_weights
    else:
        raise HTTPException(
            status_code=400,
            detail="Unexpected attention weight shape"
        )

    # Create exporter and figure
    exporter = AttentionHeatmapExporter(
        preset=request.config.preset,
        colormap=request.colormap,
        width=request.config.width,
        height=request.config.height,
        dpi=request.config.dpi,
    )

    fig = exporter.create_multi_head_figure(
        attention_matrices=attn_matrices,
        tokens=tokens,
        layer=request.layer,
        heads=request.heads,
        title=request.title,
        cols=request.cols,
    )

    # Export
    image_bytes = exporter.export(
        fig,
        format=request.config.format,
        transparent=request.config.transparent,
    )

    media_types = {
        "png": "image/png",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
    }

    return Response(
        content=image_bytes,
        media_type=media_types[request.config.format],
        headers={
            "Content-Disposition": f"attachment; filename=attention_L{request.layer}_multihead.{request.config.format}"
        }
    )


@router.post("/activation-histogram")
async def export_activation_histogram(
    request: ActivationHistogramRequest,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Export activation histogram as publication-ready figure.

    Args:
        request: ActivationHistogramRequest with activation_id and config

    Returns:
        Image file in requested format
    """
    # Get activations
    activations = get_activations_for_histogram(
        request.activation_id,
        request.layer,
        request.activation_type,
        current_user.id,
    )

    # Build title
    if request.title:
        title = request.title
    else:
        layer_str = f"Layer {request.layer}" if request.layer is not None else "All Layers"
        type_str = request.activation_type.replace("_", " ").title()
        title = f"{type_str} Distribution - {layer_str}"

    # Create exporter and figure
    exporter = ActivationHistogramExporter(
        preset=request.config.preset,
        color=request.color,
        width=request.config.width,
        height=request.config.height,
        dpi=request.config.dpi,
    )

    fig = exporter.create_figure(
        activations=activations,
        title=title,
        bins=request.bins,
        show_stats=request.show_stats,
        log_scale=request.log_scale,
    )

    # Export
    image_bytes = exporter.export(
        fig,
        format=request.config.format,
        transparent=request.config.transparent,
    )

    media_types = {
        "png": "image/png",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
    }

    safe_type = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.activation_type)
    filename = f"histogram_{'L' + str(request.layer) if request.layer is not None else 'all'}_{safe_type}.{request.config.format}"

    return Response(
        content=image_bytes,
        media_type=media_types[request.config.format],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.post("/batch")
async def export_batch(
    request: BatchExportRequest,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Export multiple figures as a ZIP archive.

    Args:
        request: BatchExportRequest with list of export requests

    Returns:
        ZIP file containing all exported figures
    """
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, export_req in enumerate(request.exports):
            export_type = export_req.get("type", "attention-heatmap")

            try:
                if export_type == "attention-heatmap":
                    req = AttentionHeatmapRequest(**export_req)
                    attn_weights, tokens = get_attention_matrix(req.activation_id, req.layer, current_user.id)

                    if len(attn_weights.shape) == 4:
                        attn_matrix = attn_weights[0, req.head]
                    elif len(attn_weights.shape) == 3:
                        attn_matrix = attn_weights[req.head]
                    else:
                        attn_matrix = attn_weights

                    exporter = AttentionHeatmapExporter(
                        preset=req.config.preset,
                        colormap=req.colormap,
                        width=req.config.width,
                        height=req.config.height,
                        dpi=req.config.dpi,
                    )

                    fig = exporter.create_figure(
                        attention_matrix=attn_matrix,
                        tokens=tokens,
                        layer=req.layer,
                        head=req.head,
                        title=req.title,
                        show_colorbar=req.show_colorbar,
                        show_values=req.show_values,
                    )

                    image_bytes = exporter.export(fig, format=req.config.format)
                    filename = f"attention_L{req.layer}_H{req.head}.{req.config.format}"

                elif export_type == "activation-histogram":
                    req = ActivationHistogramRequest(**export_req)
                    activations = get_activations_for_histogram(
                        req.activation_id,
                        req.layer,
                        req.activation_type,
                        current_user.id,
                    )

                    exporter = ActivationHistogramExporter(
                        preset=req.config.preset,
                        color=req.color,
                        width=req.config.width,
                        height=req.config.height,
                        dpi=req.config.dpi,
                    )

                    title = req.title or f"Activation Distribution"
                    fig = exporter.create_figure(
                        activations=activations,
                        title=title,
                        bins=req.bins,
                        show_stats=req.show_stats,
                        log_scale=req.log_scale,
                    )

                    image_bytes = exporter.export(fig, format=req.config.format)
                    filename = f"histogram_{i}.{req.config.format}"

                else:
                    continue

                zip_file.writestr(filename, image_bytes)

            except Exception as e:
                # Skip failed exports but continue
                print(f"Failed to export item {i}: {e}")
                continue

    zip_buffer.seek(0)

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=figures_export.zip"
        }
    )
