"""Base visualization utilities."""

from app.utils.visualization.config import VisualizationConfig
from app.utils.visualization.overlays import (
    add_colorbar,
    add_kernel_overlay,
    add_search_window_overlay,
    add_statistics,
    highlight_pixel,
    plot_similarity_map,
)
from app.utils.visualization.plots import plot_weight_distribution
from app.utils.visualization.utils import add_value_annotations, setup_figure

__all__ = [
    "VisualizationConfig",
    "add_colorbar",
    "add_statistics",
    "add_kernel_overlay",
    "add_search_window_overlay",
    "highlight_pixel",
    "plot_similarity_map",
    "plot_weight_distribution",
    "setup_figure",
    "add_value_annotations",
]
