"""Visualization utilities."""
from .config import (
    VisualizationConfig,
    KernelOverlayConfig,
    SearchWindowOverlayConfig,
    create_visualization_config,
    create_kernel_overlay_config
)
from .overlays import (
    add_colorbar,
    add_statistics,
    add_kernel_overlay,
    add_search_window_overlay,
    highlight_pixel,
    plot_similarity_map,
    plot_weight_distribution
)
from .utils import setup_figure, add_value_annotations

__all__ = [
    # Configs
    'VisualizationConfig',
    'KernelOverlayConfig',
    'SearchWindowOverlayConfig',
    'create_visualization_config',
    'create_kernel_overlay_config',
    
    # Overlays
    'add_colorbar',
    'add_statistics',
    'add_kernel_overlay',
    'add_search_window_overlay',
    'highlight_pixel',
    'plot_similarity_map',
    'plot_weight_distribution',
    
    # Utils
    'setup_figure',
    'add_value_annotations'
] 