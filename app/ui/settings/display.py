from dataclasses import dataclass
from typing import TYPE_CHECKING
import streamlit as st

if TYPE_CHECKING:
    from app.utils.visualization import VisualizationConfig, KernelOverlayConfig

@dataclass
class DisplaySettings:
    """Display settings configuration."""
    # Basic display settings
    show_colorbar: bool = True
    show_stats: bool = True
    colormap: str = "viridis"
    decimals: int = 3
    
    # Kernel overlay settings
    kernel_color: str = "#FF0000"
    kernel_width: int = 2
    grid_color: str = "#FF0000"
    grid_width: int = 1
    center_color: str = "#FF0000"
    center_alpha: float = 0.5
    
    # Annotation settings
    annotation_color: str = "#FFFFFF"
    annotation_bg_color: str = "#000000"
    annotation_bg_alpha: float = 0.5

    def to_visualization_config(self) -> 'VisualizationConfig':
        """Convert to VisualizationConfig."""
        from app.utils.visualization import VisualizationConfig
        return VisualizationConfig(
            show_colorbar=self.show_colorbar,
            show_stats=self.show_stats,
            colormap=self.colormap,
            decimals=self.decimals
        )

    def to_kernel_overlay_config(self, kernel_size: int) -> 'KernelOverlayConfig':
        """Convert to KernelOverlayConfig."""
        from app.utils.visualization import KernelOverlayConfig
        return KernelOverlayConfig(
            kernel_size=kernel_size,
            outline_color=self.kernel_color,
            outline_width=self.kernel_width,
            grid_color=self.grid_color,
            grid_width=self.grid_width,
            center_color=self.center_color,
            center_alpha=self.center_alpha
        )

    @classmethod
    def from_session_state(cls) -> 'DisplaySettings':
        """Create settings from session state with defaults."""
        return cls(
            show_colorbar=bool(st.session_state.get('show_colorbar', True)),
            show_stats=bool(st.session_state.get('show_stats', True)),
            colormap=str(st.session_state.get('colormap', 'viridis')),
            decimals=int(st.session_state.get('annotation_decimals', 3)),
            kernel_color=str(st.session_state.get('kernel_color', '#FF0000')),
            kernel_width=int(st.session_state.get('kernel_width', 2)),
            grid_color=str(st.session_state.get('grid_color', '#FF0000')),
            grid_width=int(st.session_state.get('grid_width', 1)),
            center_color=str(st.session_state.get('center_color', '#FF0000')),
            center_alpha=float(st.session_state.get('center_alpha', 0.5)),
            annotation_color=str(st.session_state.get('annotation_color', '#FFFFFF')),
            annotation_bg_color=str(st.session_state.get('annotation_bg_color', '#000000')),
            annotation_bg_alpha=float(st.session_state.get('annotation_bg_alpha', 0.5))
        )
