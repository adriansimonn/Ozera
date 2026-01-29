"""
Paper-ready figure generation for attention heatmaps, activation histograms, and patching results.

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


class ActivationHistogramExporter(FigureExporter):
    """Export activation distribution histograms."""

    def __init__(
        self,
        preset: str = "default",
        color: str = "#22d3ee",
        **kwargs,
    ):
        """
        Initialize activation histogram exporter.

        Args:
            preset: Publication preset name
            color: Bar color
            **kwargs: Additional arguments passed to FigureExporter
        """
        super().__init__(preset=preset, **kwargs)
        self.color = color

    def create_figure(
        self,
        activations: np.ndarray,
        title: str = "Activation Distribution",
        xlabel: str = "Activation Value",
        ylabel: str = "Frequency",
        bins: int = 50,
        show_stats: bool = True,
        log_scale: bool = False,
    ) -> Figure:
        """
        Create activation histogram figure.

        Args:
            activations: Array of activation values (any shape, will be flattened)
            title: Figure title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of histogram bins
            show_stats: Whether to show statistics box
            log_scale: Whether to use log scale for y-axis

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        self._apply_style(fig, ax)

        # Flatten activations
        if isinstance(activations, list):
            activations = np.array(activations)
        flat_values = activations.flatten()

        # Create histogram
        n, bins_edges, patches = ax.hist(
            flat_values,
            bins=bins,
            color=self.color,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
        )

        ax.set_title(title, fontsize=self.preset.title_size, pad=10)
        ax.set_xlabel(xlabel, fontsize=self.preset.label_size)
        ax.set_ylabel(ylabel, fontsize=self.preset.label_size)

        if log_scale:
            ax.set_yscale('log')

        # Add statistics box
        if show_stats:
            stats_text = (
                f'Mean: {flat_values.mean():.4f}\n'
                f'Std: {flat_values.std():.4f}\n'
                f'Min: {flat_values.min():.4f}\n'
                f'Max: {flat_values.max():.4f}'
            )
            ax.text(
                0.98, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=self.preset.tick_size,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig

    def create_comparison_figure(
        self,
        activations_list: list[np.ndarray],
        labels: list[str],
        title: str = "Activation Distribution Comparison",
        bins: int = 50,
        alpha: float = 0.6,
    ) -> Figure:
        """
        Create overlaid histogram comparison.

        Args:
            activations_list: List of activation arrays to compare
            labels: Labels for each distribution
            title: Figure title
            bins: Number of histogram bins
            alpha: Transparency for overlaid histograms

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        self._apply_style(fig, ax)

        colors = plt.cm.tab10(np.linspace(0, 1, len(activations_list)))

        for i, (acts, label) in enumerate(zip(activations_list, labels)):
            if isinstance(acts, list):
                acts = np.array(acts)
            flat_values = acts.flatten()
            ax.hist(
                flat_values,
                bins=bins,
                color=colors[i],
                alpha=alpha,
                edgecolor='white',
                linewidth=0.5,
                label=label,
            )

        ax.set_title(title, fontsize=self.preset.title_size, pad=10)
        ax.set_xlabel("Activation Value", fontsize=self.preset.label_size)
        ax.set_ylabel("Frequency", fontsize=self.preset.label_size)
        ax.legend(fontsize=self.preset.tick_size)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        return fig


class PatchingResultExporter(FigureExporter):
    """Export patching experiment results."""

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
    ) -> Figure:
        """
        Create heatmap of patching effects by layer and position.

        Args:
            effects: 2D array [num_layers, num_positions]
            layer_labels: Labels for layers
            position_labels: Labels for positions/tokens
            title: Figure title
            colormap: Matplotlib colormap
            center_zero: Whether to center colormap at zero

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        self._apply_style(fig, ax)

        if isinstance(effects, list):
            effects = np.array(effects)

        # Determine color normalization
        if center_zero:
            vmax = max(abs(effects.min()), abs(effects.max()))
            vmin = -vmax
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            vmin, vmax = effects.min(), effects.max()
            norm = None

        im = ax.imshow(
            effects,
            cmap=colormap,
            aspect='auto',
            norm=norm,
            vmin=vmin if not center_zero else None,
            vmax=vmax if not center_zero else None,
        )

        # Colorbar
        cbar = fig.colorbar(
            im, ax=ax,
            fraction=self.preset.colorbar_width,
            pad=0.02,
        )
        cbar.ax.tick_params(labelsize=self.preset.tick_size)
        cbar.set_label('Effect Size', fontsize=self.preset.label_size)

        # Labels
        if layer_labels:
            ax.set_yticks(range(len(layer_labels)))
            ax.set_yticklabels(layer_labels)
        else:
            ax.set_ylabel('Layer', fontsize=self.preset.label_size)

        if position_labels:
            display_labels = [l[:8] for l in position_labels]
            ax.set_xticks(range(len(display_labels)))
            ax.set_xticklabels(display_labels, rotation=45, ha='right')
        else:
            ax.set_xlabel('Position', fontsize=self.preset.label_size)

        ax.set_title(title, fontsize=self.preset.title_size, pad=10)

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
    ) -> Figure:
        """
        Create bar chart comparing patching results.

        Args:
            values: Values to compare
            labels: Labels for each bar
            baseline: Optional baseline value to show as horizontal line
            title: Figure title
            ylabel: Y-axis label
            colors: Optional custom colors for bars

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        self._apply_style(fig, ax)

        x = range(len(values))
        bar_colors = colors or ['#22d3ee'] * len(values)

        bars = ax.bar(x, values, color=bar_colors, alpha=0.8, edgecolor='white')

        if baseline is not None:
            ax.axhline(y=baseline, color='#ef4444', linestyle='--',
                       linewidth=self.preset.linewidth * 2, label='Baseline')
            ax.legend(fontsize=self.preset.tick_size)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
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
