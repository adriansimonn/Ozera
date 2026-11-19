"""
Paper-ready figure generation for attention heatmaps, the logit lens, logit attribution, and patching results.

Supports multiple output formats (PNG, PDF, SVG) with publication-specific presets.
"""

from dataclasses import dataclass
from typing import Optional, Literal
from io import BytesIO
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


@dataclass
class PublicationPreset:
    """Publication-specific figure settings."""
    name: str
    width_inches: float
    height_inches: float
    dpi: int
    font_family: str
    font_size: int
    title_size: int
    label_size: int
    tick_size: int
    linewidth: float
    colorbar_width: float


# Publication presets
PUBLICATION_PRESETS = {
    "nature": PublicationPreset(
        name="Nature",
        width_inches=3.5,  # 89mm single column
        height_inches=3.5,
        dpi=300,
        font_family="Helvetica",
        font_size=7,
        title_size=8,
        label_size=7,
        tick_size=6,
        linewidth=0.5,
        colorbar_width=0.03,
    ),
    "neurips": PublicationPreset(
        name="NeurIPS",
        width_inches=5.5,
        height_inches=4.0,
        dpi=300,
        font_family="Times New Roman",
        font_size=10,
        title_size=11,
        label_size=10,
        tick_size=9,
        linewidth=0.75,
        colorbar_width=0.04,
    ),
    "iclr": PublicationPreset(
        name="ICLR",
        width_inches=6.0,
        height_inches=4.5,
        dpi=300,
        font_family="Times New Roman",
        font_size=10,
        title_size=11,
        label_size=10,
        tick_size=9,
        linewidth=0.75,
        colorbar_width=0.04,
    ),
    "icml": PublicationPreset(
        name="ICML",
        width_inches=6.75,
        height_inches=5.0,
        dpi=300,
        font_family="Times New Roman",
        font_size=10,
        title_size=11,
        label_size=10,
        tick_size=9,
        linewidth=0.75,
        colorbar_width=0.04,
    ),
    "presentation": PublicationPreset(
        name="Presentation",
        width_inches=10.0,
        height_inches=7.5,
        dpi=150,
        font_family="Arial",
        font_size=14,
        title_size=18,
        label_size=14,
        tick_size=12,
        linewidth=1.5,
        colorbar_width=0.05,
    ),
    "default": PublicationPreset(
        name="Default",
        width_inches=8.0,
        height_inches=6.0,
        dpi=150,
        font_family="DejaVu Sans",
        font_size=10,
        title_size=12,
        label_size=10,
        tick_size=9,
        linewidth=1.0,
        colorbar_width=0.04,
    ),
}


class FigureExporter:
    """Base class for figure exporters."""

    def __init__(
        self,
        preset: str = "default",
        width: Optional[float] = None,
        height: Optional[float] = None,
        dpi: Optional[int] = None,
    ):
        """
        Initialize exporter with optional custom dimensions.

        Args:
            preset: Publication preset name
            width: Override width in inches
            height: Override height in inches
            dpi: Override DPI
        """
        self.preset = PUBLICATION_PRESETS.get(preset, PUBLICATION_PRESETS["default"])
        self.width = width or self.preset.width_inches
        self.height = height or self.preset.height_inches
        self.dpi = dpi or self.preset.dpi

    def _apply_style(self, fig: Figure, ax) -> None:
        """Apply publication preset styling to figure."""
        plt.rcParams.update({
            'font.family': self.preset.font_family,
            'font.size': self.preset.font_size,
            'axes.titlesize': self.preset.title_size,
            'axes.labelsize': self.preset.label_size,
            'xtick.labelsize': self.preset.tick_size,
            'ytick.labelsize': self.preset.tick_size,
            'axes.linewidth': self.preset.linewidth,
        })

    def export(
        self,
        fig: Figure,
        format: Literal["png", "pdf", "svg"] = "png",
        transparent: bool = False,
    ) -> bytes:
        """
        Export figure to bytes in specified format.

        Args:
            fig: Matplotlib figure to export
            format: Output format
            transparent: Whether background should be transparent

        Returns:
            Figure as bytes
        """
        buf = BytesIO()
        fig.savefig(
            buf,
            format=format,
            dpi=self.dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=transparent,
            facecolor='white' if not transparent else 'none',
            edgecolor='none',
        )
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()


class AttentionHeatmapExporter(FigureExporter):
    """Export attention heatmaps as publication-ready figures."""

    def __init__(
        self,
        preset: str = "default",
        colormap: str = "inferno",
        **kwargs,
    ):
        """
        Initialize attention heatmap exporter.

        Args:
            preset: Publication preset name
            colormap: Matplotlib colormap name
            **kwargs: Additional arguments passed to FigureExporter
        """
        super().__init__(preset=preset, **kwargs)
        self.colormap = colormap

    def create_figure(
        self,
        attention_matrix: np.ndarray,
        tokens: Optional[list[str]] = None,
        layer: int = 0,
        head: int = 0,
        title: Optional[str] = None,
        show_colorbar: bool = True,
        show_values: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Figure:
        """
        Create attention heatmap figure.

        Args:
            attention_matrix: 2D attention weights [seq_len, seq_len]
            tokens: Token labels for axes
            layer: Layer index for title
            head: Head index for title
            title: Custom title (overrides default)
            show_colorbar: Whether to show colorbar
            show_values: Whether to show attention values in cells
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        self._apply_style(fig, ax)

        # Ensure attention_matrix is 2D numpy array
        if isinstance(attention_matrix, list):
            attention_matrix = np.array(attention_matrix)

        seq_len = attention_matrix.shape[0]

        # Create heatmap
        im = ax.imshow(
            attention_matrix,
            cmap=self.colormap,
            aspect='equal',
            vmin=vmin or 0,
            vmax=vmax or attention_matrix.max(),
        )

        # Add colorbar
        if show_colorbar:
            cbar = fig.colorbar(
                im, ax=ax,
                fraction=self.preset.colorbar_width,
                pad=0.02,
            )
            cbar.ax.tick_params(labelsize=self.preset.tick_size)
            cbar.set_label('Attention Weight', fontsize=self.preset.label_size)

        # Set labels
        if tokens and len(tokens) <= 20:
            # Show all tokens if there aren't too many
            def format_token(t):
                s = str(t) if not isinstance(t, str) else t
                return s[:8] if len(s) > 8 else s
            display_tokens = [format_token(t) for t in tokens]
            ax.set_xticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=45, ha='right',
                               fontsize=self.preset.tick_size)
            ax.set_yticks(range(len(display_tokens)))
            ax.set_yticklabels(display_tokens, fontsize=self.preset.tick_size)
        else:
            # For many tokens, show sparse numeric indices
            step = max(1, seq_len // 10)
            tick_positions = list(range(0, seq_len, step))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(i) for i in tick_positions],
                               fontsize=self.preset.tick_size)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([str(i) for i in tick_positions],
                               fontsize=self.preset.tick_size)

        # Set title
        if title:
            ax.set_title(title, fontsize=self.preset.title_size, pad=10)
        else:
            ax.set_title(f'Layer {layer}, Head {head}', fontsize=self.preset.title_size, pad=10)

        ax.set_xlabel('To Token', fontsize=self.preset.label_size)
        ax.set_ylabel('From Token', fontsize=self.preset.label_size)

        # Show values in cells if requested
        if show_values and seq_len <= 20:
            for i in range(seq_len):
                for j in range(seq_len):
                    val = attention_matrix[i, j]
                    color = 'white' if val > (vmax or attention_matrix.max()) / 2 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=color, fontsize=self.preset.tick_size - 2)

        fig.tight_layout()
        return fig

    def create_multi_head_figure(
        self,
        attention_matrices: np.ndarray,
        tokens: Optional[list[str]] = None,
        layer: int = 0,
        heads: Optional[list[int]] = None,
        title: Optional[str] = None,
        cols: int = 4,
    ) -> Figure:
        """
        Create figure with multiple attention heads.

        Args:
            attention_matrices: 3D array [num_heads, seq_len, seq_len]
            tokens: Token labels for axes
            layer: Layer index for title
            heads: Specific head indices to show (default: all)
            title: Custom title
            cols: Number of columns in grid

        Returns:
            Matplotlib figure
        """
        if isinstance(attention_matrices, list):
            attention_matrices = np.array(attention_matrices)

        num_heads = attention_matrices.shape[0]
        heads_to_show = heads or list(range(num_heads))
        n_heads = len(heads_to_show)

        rows = (n_heads + cols - 1) // cols

        # Adjust figure size for grid
        fig_width = self.width * min(cols, n_heads)
        fig_height = self.height * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        self._apply_style(fig, axes)

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        vmax = attention_matrices.max()

        for idx, head_idx in enumerate(heads_to_show):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]

            im = ax.imshow(
                attention_matrices[head_idx],
                cmap=self.colormap,
                aspect='equal',
                vmin=0,
                vmax=vmax,
            )

            ax.set_title(f'Head {head_idx}', fontsize=self.preset.label_size)

            # Only show ticks on edge plots
            if row == rows - 1:
                if tokens:
                    display_tokens = [t[:6] for t in tokens]
                    ax.set_xticks(range(len(display_tokens)))
                    ax.set_xticklabels(display_tokens, rotation=45, ha='right',
                                       fontsize=self.preset.tick_size - 2)
            else:
                ax.set_xticks([])

            if col == 0:
                if tokens:
                    display_tokens = [t[:6] for t in tokens]
                    ax.set_yticks(range(len(display_tokens)))
                    ax.set_yticklabels(display_tokens, fontsize=self.preset.tick_size - 2)
            else:
                ax.set_yticks([])

        # Hide unused subplots
        for idx in range(n_heads, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        # Add overall title
        if title:
            fig.suptitle(title, fontsize=self.preset.title_size, y=1.02)
        else:
            fig.suptitle(f'Layer {layer} Attention Heads', fontsize=self.preset.title_size, y=1.02)

        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=self.preset.tick_size)

        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        return fig


def figure_text(token: str, limit: int = 8) -> str:
    """A token as figure text: whitespace made visible, cut to limit characters, and no mathtext."""
    text = token.replace(" ", "\u00b7").replace("\n", "\u21b5").replace("\t", "\u2192")
    if len(text) > limit:
        text = text[:limit - 1] + "\u2026"
    return text.replace("$", r"\$")


class LogitLensExporter(FigureExporter):
    """Export the logit lens: layers by positions, each cell colored by a token's probability."""

    def __init__(self, preset: str = "default", colormap: str = "inferno", **kwargs):
        """
        Args:
            preset: Publication preset name
            colormap: Matplotlib colormap name
            **kwargs: Additional arguments passed to FigureExporter
        """
        super().__init__(preset=preset, **kwargs)
        self.colormap = colormap

    def create_figure(
        self,
        probabilities: np.ndarray,
        top_tokens: Optional[list[list[str]]] = None,
        row_labels: Optional[list[str]] = None,
        position_labels: Optional[list[str]] = None,
        title: Optional[str] = None,
        colorbar_label: str = "Probability",
        show_colorbar: bool = True,
    ) -> Figure:
        """
        Create a logit lens figure.

        Args:
            probabilities: [rows, positions] the probability each cell is colored by
            top_tokens: [rows][positions] each cell's top token, written in the cell when the
                grid is small enough to read
            row_labels: A label per row (the embeddings, then each layer)
            position_labels: A label per position
            title: Figure title
            colorbar_label: What the color is the probability of
            show_colorbar: Whether to show the colorbar

        Returns:
            Matplotlib figure
        """
        # Before the axes exist: their tick labels take the style current when they're made
        self._apply_style(None, None)
        fig, ax = plt.subplots(figsize=(self.width, self.height))

        probabilities = np.asarray(probabilities)
        rows, positions = probabilities.shape
        cmap = plt.get_cmap(self.colormap)
        im = ax.imshow(probabilities, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')

        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=self.preset.colorbar_width, pad=0.02)
            cbar.ax.tick_params(labelsize=self.preset.tick_size)
            cbar.set_label(colorbar_label, fontsize=self.preset.label_size)

        if top_tokens is not None and positions <= 24 and rows <= 40:
            size = max(self.preset.tick_size - 2, 4)
            for i in range(rows):
                for j in range(positions):
                    r, g, b, _ = cmap(float(probabilities[i, j]))
                    dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
                    ax.text(j, i, figure_text(top_tokens[i][j], 6), ha='center', va='center',
                            color='white' if dark else 'black', fontsize=size)

        if row_labels:
            step = max(1, rows // 20)
            ax.set_yticks(range(0, rows, step))
            ax.set_yticklabels([row_labels[i] for i in range(0, rows, step)], fontsize=self.preset.tick_size)
        if position_labels:
            step = max(1, positions // 24)
            ticks = list(range(0, positions, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([figure_text(position_labels[i]) for i in ticks], rotation=45, ha='right',
                               fontsize=self.preset.tick_size)

        ax.set_xlabel('Position', fontsize=self.preset.label_size)
        ax.set_ylabel('Residual stream after', fontsize=self.preset.label_size)
        ax.set_title(title or 'Logit lens', fontsize=self.preset.title_size, pad=10)

        fig.tight_layout()
        return fig


class LogitAttributionExporter(FigureExporter):
    """Export one position's logit attribution: each layer's attention and MLP, and every head."""

    # Attention, MLP and embeddings (the labs' light-surface series colors)
    COLORS = {"attn": "#2a78d6", "mlp": "#eb6834", "embed": "#898781"}

    def create_figure(
        self,
        embed: float,
        attn: np.ndarray,
        mlp: np.ndarray,
        heads: np.ndarray,
        title: Optional[str] = None,
    ) -> Figure:
        """
        Create a logit attribution figure.

        Args:
            embed: The embeddings' share of the logit
            attn, mlp: [layers] each layer's attention and MLP share
            heads: [layers, heads] each head's share
            title: Figure title

        Returns:
            Matplotlib figure
        """
        attn, mlp, heads = np.asarray(attn), np.asarray(mlp), np.asarray(heads)
        layers = len(attn)

        self._apply_style(None, None)
        fig, (bars, grid) = plt.subplots(
            2, 1, figsize=(self.width, self.height * 1.5), gridspec_kw={"height_ratios": [1, 1.3]},
        )

        x = np.arange(layers)
        bars.bar([-1], [embed], width=0.8, color=self.COLORS["embed"], label='Embeddings')
        bars.bar(x - 0.2, attn, width=0.4, color=self.COLORS["attn"], label='Attention')
        bars.bar(x + 0.2, mlp, width=0.4, color=self.COLORS["mlp"], label='MLP')
        bars.axhline(0, color='black', linewidth=self.preset.linewidth * 0.75)
        step = max(1, layers // 16)
        ticks = list(range(0, layers, step))
        bars.set_xticks([-1] + ticks)
        bars.set_xticklabels(['Emb'] + [str(i) for i in ticks], fontsize=self.preset.tick_size)
        bars.set_xlim(-1.8, layers - 0.2)
        bars.set_xlabel('Layer', fontsize=self.preset.label_size)
        bars.set_ylabel('Logit contribution', fontsize=self.preset.label_size)
        bars.legend(fontsize=self.preset.tick_size, frameon=False, ncol=3, loc='best')
        bars.spines['top'].set_visible(False)
        bars.spines['right'].set_visible(False)
        bars.set_title(title or 'Logit attribution', fontsize=self.preset.title_size, pad=10)

        limit = float(np.abs(heads).max()) or 1.0
        im = grid.imshow(
            heads, cmap='RdBu_r', aspect='auto', interpolation='nearest',
            norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        cbar = fig.colorbar(im, ax=grid, fraction=self.preset.colorbar_width, pad=0.02)
        cbar.ax.tick_params(labelsize=self.preset.tick_size)
        cbar.set_label('Logit contribution', fontsize=self.preset.label_size)
        head_count = heads.shape[1]
        head_step = max(1, head_count // 16)
        grid.set_xticks(range(0, head_count, head_step))
        grid.set_xticklabels([str(i) for i in range(0, head_count, head_step)], fontsize=self.preset.tick_size)
        grid.set_yticks(ticks)
        grid.set_yticklabels([str(i) for i in ticks], fontsize=self.preset.tick_size)
        grid.set_xlabel('Head', fontsize=self.preset.label_size)
        grid.set_ylabel('Layer', fontsize=self.preset.label_size)

        fig.tight_layout()
        return fig


class PatchingResultExporter(FigureExporter):
    """Export Attribution results: effect heatmaps, estimates against exact effects, and ablation bars."""

    def __init__(
        self,
        preset: str = "default",
        **kwargs,
    ):
        """
        Initialize patching result exporter.

        Args:
            preset: Publication preset name
            **kwargs: Additional arguments passed to FigureExporter
        """
        super().__init__(preset=preset, **kwargs)

    def create_effect_heatmap(
        self,
        effects: np.ndarray,
        layer_labels: Optional[list[str]] = None,
        position_labels: Optional[list[str]] = None,
        title: str = "Patching Effects",
        colormap: str = "RdBu_r",
        center_zero: bool = True,
        xlabel: str = "Position",
        ylabel: str = "Layer",
        colorbar_label: str = "Effect",
    ) -> Figure:
        """
        Create heatmap of effects by layer and position (or head).

        Args:
            effects: 2D array [num_layers, num_positions]
            layer_labels: Labels for layers
            position_labels: Labels for positions/tokens
            title: Figure title
            colormap: Matplotlib colormap
            center_zero: Whether to center colormap at zero
            xlabel, ylabel, colorbar_label: Axis and colorbar titles

        Returns:
            Matplotlib figure
        """
        # Before the axes exist: their tick labels take the style current when they're made
        self._apply_style(None, None)
        fig, ax = plt.subplots(figsize=(self.width, self.height))

        effects = np.asarray(effects, dtype=float)
        rows, cols = effects.shape

        # Determine color normalization
        if center_zero:
            vmax = float(np.nanmax(np.abs(effects))) or 1.0
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(effects, cmap=colormap, aspect='auto', norm=norm, interpolation='nearest')
        else:
            im = ax.imshow(effects, cmap=colormap, aspect='auto', interpolation='nearest')

        # Colorbar
        cbar = fig.colorbar(
            im, ax=ax,
            fraction=self.preset.colorbar_width,
            pad=0.02,
        )
        cbar.ax.tick_params(labelsize=self.preset.tick_size)
        cbar.set_label(colorbar_label, fontsize=self.preset.label_size)

        # Labels, thinned so they stay readable
        if layer_labels:
            step = max(1, rows // 24)
            ticks = list(range(0, rows, step))
            ax.set_yticks(ticks)
            ax.set_yticklabels([layer_labels[i] for i in ticks], fontsize=self.preset.tick_size)
        if position_labels:
            step = max(1, cols // 32)
            ticks = list(range(0, cols, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([figure_text(position_labels[i], 10) for i in ticks], rotation=45, ha='right',
                               fontsize=self.preset.tick_size)
        ax.set_xlabel(xlabel, fontsize=self.preset.label_size)
        ax.set_ylabel(ylabel, fontsize=self.preset.label_size)
        ax.set_title(title, fontsize=self.preset.title_size, pad=10)

        fig.tight_layout()
        return fig

    def create_estimate_scatter(
        self,
        estimates: list[float],
        exact: list[float],
        errors: Optional[list[float]] = None,
        title: str = "Attribution vs. activation patching",
        xlabel: str = "Attribution estimate",
        ylabel: str = "Exact patching effect",
        note: Optional[str] = None,
    ) -> Figure:
        """
        Create a scatter of estimated effects against exact ones, with the line where they agree.

        Args:
            estimates, exact: One value per cell
            errors: Each exact effect's standard error (drawn as a bar)
            note: A line under the title (the rank correlation, say)

        Returns:
            Matplotlib figure
        """
        self._apply_style(None, None)
        fig, ax = plt.subplots(figsize=(self.width, self.height))

        x, y = np.asarray(estimates, dtype=float), np.asarray(exact, dtype=float)
        low, high = float(min(x.min(), y.min(), 0.0)), float(max(x.max(), y.max(), 0.0))
        pad = 0.06 * ((high - low) or 1.0)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color='#898781', linestyle='--',
                linewidth=self.preset.linewidth, zorder=1)
        ax.axhline(0, color='#c9c8c4', linewidth=self.preset.linewidth * 0.75, zorder=0)
        ax.axvline(0, color='#c9c8c4', linewidth=self.preset.linewidth * 0.75, zorder=0)
        if errors is not None:
            ax.errorbar(x, y, yerr=np.asarray(errors, dtype=float), fmt='none', ecolor='#2a78d6', alpha=0.45,
                        linewidth=self.preset.linewidth, zorder=2)
        ax.scatter(x, y, s=18, color='#2a78d6', edgecolors='white', linewidths=0.4, zorder=3)

        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel(xlabel, fontsize=self.preset.label_size)
        ax.set_ylabel(ylabel, fontsize=self.preset.label_size)
        ax.set_title(title if not note else f"{title}\n{note}", fontsize=self.preset.title_size, pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def create_comparison_bar(
        self,
        values: list[float],
        labels: list[str],
        baseline: Optional[float] = None,
        title: str = "Patching Comparison",
        ylabel: str = "Value",
        colors: Optional[list[str]] = None,
        errors: Optional[list[Optional[float]]] = None,
        marks: Optional[list[list[float]]] = None,
        baseline_label: str = "Baseline",
    ) -> Figure:
        """
        Create bar chart comparing results.

        Args:
            values: Values to compare
            labels: Labels for each bar
            baseline: Optional baseline value to show as horizontal line
            title: Figure title
            ylabel: Y-axis label
            colors: Optional custom colors for bars
            errors: Each bar's standard error (None for none)
            marks: Per bar, values drawn as ticks over it (each random set's, say)
            baseline_label: The baseline's legend label

        Returns:
            Matplotlib figure
        """
        self._apply_style(None, None)
        fig, ax = plt.subplots(figsize=(self.width, self.height))

        x = list(range(len(values)))
        bar_colors = colors or ['#2a78d6'] * len(values)

        ax.bar(x, values, color=bar_colors, alpha=0.9, edgecolor='white')
        if errors is not None:
            shown = [(i, values[i], e) for i, e in enumerate(errors) if e]
            if shown:
                ax.errorbar([i for i, _, _ in shown], [v for _, v, _ in shown], yerr=[e for _, _, e in shown],
                            fmt='none', ecolor='#111111', capsize=3, linewidth=self.preset.linewidth)
        for i, row in enumerate(marks or []):
            if row:
                ax.scatter([i] * len(row), row, marker='_', s=260, color='#111111', linewidths=self.preset.linewidth, zorder=3)

        ax.axhline(0, color='#111111', linewidth=self.preset.linewidth * 0.75)
        if baseline is not None:
            ax.axhline(y=baseline, color='#898781', linestyle='--',
                       linewidth=self.preset.linewidth, label=baseline_label)
            ax.legend(fontsize=self.preset.tick_size, frameon=False)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0, ha='right' if len(labels) > 4 else 'center')
        ax.set_ylabel(ylabel, fontsize=self.preset.label_size)
        ax.set_title(title, fontsize=self.preset.title_size, pad=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def create_layer_effect_line(
        self,
        effects_by_layer: list[float],
        layer_indices: Optional[list[int]] = None,
        title: str = "Effect by Layer",
        ylabel: str = "Effect Size",
        show_points: bool = True,
        fill: bool = True,
    ) -> Figure:
        """
        Create line plot of effects across layers.

        Args:
            effects_by_layer: Effect values for each layer
            layer_indices: Custom layer indices
            title: Figure title
            ylabel: Y-axis label
            show_points: Whether to show data points
            fill: Whether to fill under the line

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        self._apply_style(fig, ax)

        x = layer_indices or list(range(len(effects_by_layer)))

        ax.plot(x, effects_by_layer, color='#22d3ee',
                linewidth=self.preset.linewidth * 2, marker='o' if show_points else None,
                markersize=6 if show_points else None)

        if fill:
            ax.fill_between(x, effects_by_layer, alpha=0.2, color='#22d3ee')

        ax.set_xlabel('Layer', fontsize=self.preset.label_size)
        ax.set_ylabel(ylabel, fontsize=self.preset.label_size)
        ax.set_title(title, fontsize=self.preset.title_size, pad=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks(x)

        fig.tight_layout()
        return fig
