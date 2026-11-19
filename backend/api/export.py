"""
Export API endpoints for paper-ready figure generation.

Provides endpoints for exporting attention heatmaps, the logit lens and logit attribution
in publication-quality formats (PNG, PDF, SVG).
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
from io import BytesIO
import zipfile

from core.export.figure_generator import figure_text
from core.export import (
    AttentionHeatmapExporter,
    LogitAttributionExporter,
    LogitLensExporter,
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


class LogitLensRequest(BaseModel):
    """Request to export the logit lens of a generation."""
    activation_id: str = Field(..., description="ID of captured activations")
    color_by: Literal["target", "final"] = Field(
        default="target",
        description="Color cells by the probability of the next token (target) or of the model's final prediction",
    )
    start: int = Field(default=0, ge=0, description="First position shown")
    end: Optional[int] = Field(default=None, ge=1, description="One past the last position shown (None for the last)")
    config: ExportConfig = Field(default_factory=ExportConfig)
    colormap: str = Field(default="inferno", description="Matplotlib colormap")
    title: Optional[str] = Field(default=None, description="Custom title")
    show_colorbar: bool = Field(default=True, description="Show colorbar")
    show_tokens: bool = Field(default=True, description="Write each cell's top token (small grids only)")
    token_labels: Literal["text", "number"] = Field(default="text", description="Label positions with token text or numbers")


class LogitAttributionRequest(BaseModel):
    """Request to export one position's logit attribution."""
    activation_id: str = Field(..., description="ID of captured activations")
    position: int = Field(..., ge=0, description="Position whose next-token logit is attributed")
    config: ExportConfig = Field(default_factory=ExportConfig)
    title: Optional[str] = Field(default=None, description="Custom title")


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


def _readout(activation_id: str, name: str, user_id: int) -> tuple[dict, dict]:
    """A stored generation's logit lens or attribution (decoded) and its summary, or 404."""
    summary = _stored_summary(activation_id, user_id)
    readout = activation_store.get_readout(activation_id, name, user_id)
    if readout is None:
        raise HTTPException(
            status_code=404,
            detail="This generation was captured without it. Generate again to see it.",
        )
    return readout, summary


def _position_tokens(summary: dict, seq_len: int) -> list[str]:
    """The sequence's decoded tokens (positions as text if they aren't stored)."""
    tokens = summary["metadata"].get("decoded_tokens") or []
    return list(tokens) if len(tokens) == seq_len else [str(i) for i in range(seq_len)]


MEDIA_TYPES = {
    "png": "image/png",
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
}


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

    return Response(
        content=image_bytes,
        media_type=MEDIA_TYPES[request.config.format],
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

    return Response(
        content=image_bytes,
        media_type=MEDIA_TYPES[request.config.format],
        headers={
            "Content-Disposition": f"attachment; filename=attention_L{request.layer}_multihead.{request.config.format}"
        }
    )


@router.post("/logit-lens")
async def export_logit_lens(
    request: LogitLensRequest,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Export a generation's logit lens as a publication-ready figure: the embeddings and each
    layer's output by position, colored by the probability each gives the next token (or the
    model's final prediction).

    Returns:
        Image file in requested format
    """
    lens, summary = _readout(request.activation_id, "logit_lens", current_user.id)
    seq_len = lens["seq_len"]
    end = min(request.end if request.end is not None else seq_len, seq_len)
    if request.start >= end:
        raise HTTPException(status_code=400, detail=f"Positions run from 0 to {seq_len - 1}")
    span = slice(request.start, end)

    probabilities = lens[f"{request.color_by}_prob"][:, span]
    top_ids = lens["top_ids"][:, span, 0].astype(int)
    texts = lens["tokens"]
    tokens = _position_tokens(summary, seq_len)[span]

    exporter = LogitLensExporter(
        preset=request.config.preset,
        colormap=request.colormap,
        width=request.config.width,
        height=request.config.height,
        dpi=request.config.dpi,
    )
    fig = exporter.create_figure(
        probabilities=probabilities,
        top_tokens=[[texts.get(str(i), "") for i in row] for row in top_ids] if request.show_tokens else None,
        row_labels=["Emb"] + [f"L{i}" for i in range(lens["rows"] - 1)],
        position_labels=tokens if request.token_labels == "text" else [str(i) for i in range(request.start, end)],
        title=request.title,
        colorbar_label="P(next token)" if request.color_by == "target" else "P(final prediction)",
        show_colorbar=request.show_colorbar,
    )
    image_bytes = exporter.export(fig, format=request.config.format, transparent=request.config.transparent)

    return Response(
        content=image_bytes,
        media_type=MEDIA_TYPES[request.config.format],
        headers={"Content-Disposition": f"attachment; filename=logit_lens.{request.config.format}"},
    )


@router.post("/logit-attribution")
async def export_logit_attribution(
    request: LogitAttributionRequest,
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Export one position's logit attribution as a publication-ready figure: what the
    embeddings, each layer's attention and MLP, and every head wrote toward the logit of the
    token that came next.

    Returns:
        Image file in requested format
    """
    attribution, summary = _readout(request.activation_id, "logit_attribution", current_user.id)
    seq_len = attribution["seq_len"]
    position = request.position
    if position >= seq_len:
        raise HTTPException(status_code=400, detail=f"Positions run from 0 to {seq_len - 1}")

    title = request.title
    if title is None:
        token = _position_tokens(summary, seq_len)[position]
        target = attribution["tokens"].get(str(attribution["target_ids"][position]), "")
        title = f"Logit attribution: {figure_text(token, 16)} \u2192 {figure_text(target, 16)}"

    exporter = LogitAttributionExporter(
        preset=request.config.preset,
        width=request.config.width,
        height=request.config.height,
        dpi=request.config.dpi,
    )
    fig = exporter.create_figure(
        embed=float(attribution["embed"][position]),
        attn=attribution["attn"][position],
        mlp=attribution["mlp"][position],
        heads=attribution["heads"][position],
        title=title,
    )
    image_bytes = exporter.export(fig, format=request.config.format, transparent=request.config.transparent)

    return Response(
        content=image_bytes,
        media_type=MEDIA_TYPES[request.config.format],
        headers={"Content-Disposition": f"attachment; filename=logit_attribution_{position}.{request.config.format}"},
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
