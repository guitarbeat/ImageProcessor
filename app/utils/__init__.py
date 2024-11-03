"""Utility modules for the application."""

from .config import AppConfig
from .constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    DEFAULT_FILTER_TYPE,
    DEFAULT_KERNEL_SIZE,
    DISPLAY_MODES,
    PROGRESS_BAR_STYLES,
)
from .context_managers import figure_context, visualization_context
from .file_utils import get_image_files
from .latex import (
    NLM_FORMULA_CONFIG,
    SPECKLE_FORMULA_CONFIG,
    create_kernel_matrix_latex,
    get_search_window_bounds,
)
from .visualization import (
    KernelOverlayConfig,
    SearchWindowOverlayConfig,
    VisualizationConfig,
    add_colorbar,
    add_kernel_overlay,
    add_search_window_overlay,
    add_statistics,
    create_kernel_overlay_config,
    create_visualization_config,
    highlight_pixel,
    plot_similarity_map,
    plot_weight_distribution,
)

__all__ = [
    "AppConfig",
    "get_image_files",
    # Constants
    "DEFAULT_KERNEL_SIZE",
    "DEFAULT_FILTER_TYPE",
    "ALLOWED_IMAGE_EXTENSIONS",
    "DISPLAY_MODES",
    "PROGRESS_BAR_STYLES",
    # Context managers
    "figure_context",
    "visualization_context",
    # LaTeX utilities
    "SPECKLE_FORMULA_CONFIG",
    "NLM_FORMULA_CONFIG",
    "create_kernel_matrix_latex",
    "get_search_window_bounds",
    # Visualization utilities
    "VisualizationConfig",
    "KernelOverlayConfig",
    "SearchWindowOverlayConfig",
    "create_visualization_config",
    "create_kernel_overlay_config",
    "add_colorbar",
    "add_statistics",
    "add_kernel_overlay",
    "add_search_window_overlay",
    "highlight_pixel",
    "plot_similarity_map",
    "plot_weight_distribution",
]
