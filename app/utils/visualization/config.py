"""Configuration classes for visualization."""

from dataclasses import dataclass

import streamlit as st


@dataclass
class VisualizationConfig:
    """Base configuration for visualizations."""

    show_colorbar: bool = True
    show_stats: bool = True
    colormap: str = "viridis"
    decimals: int = 3


@dataclass
class KernelOverlayConfig:
    """Configuration for kernel overlay visualization."""

    kernel_size: int
    outline_color: str = "#FF0000"
    outline_width: int = 2
    grid_color: str = "#FF0000"
    grid_width: int = 1
    center_color: str = "#FF0000"
    center_alpha: float = 0.5


@dataclass
class SearchWindowOverlayConfig:
    """Configuration for search window overlay visualization."""

    outline_color: str = "#0000FF"
    outline_width: int = 2
    outline_style: str = "dashed"

    @classmethod
    def from_session_state(cls) -> "SearchWindowOverlayConfig":
        """Create from session state."""
        return cls(
            outline_color=st.session_state.get(
                "search_window_color", "#0000FF"),
            outline_width=st.session_state.get("search_window_width", 2),
            outline_style=st.session_state.get(
                "search_window_style", "dashed"),
        )


def create_visualization_config() -> VisualizationConfig:
    """Create visualization config from session state."""
    return VisualizationConfig(
        show_colorbar=bool(st.session_state.get("show_colorbar", True)),
        show_stats=bool(st.session_state.get("show_stats", True)),
        colormap=str(st.session_state.get("colormap", "viridis")),
        decimals=int(st.session_state.get("annotation_decimals", 3)),
    )


def create_kernel_overlay_config() -> KernelOverlayConfig:
    """Create kernel overlay config from session state."""
    return KernelOverlayConfig(
        kernel_size=int(st.session_state.get("kernel_size", 7)),
        outline_color=str(st.session_state.get("kernel_color", "#FF0000")),
        outline_width=int(st.session_state.get("kernel_width", 2)),
        grid_color=str(st.session_state.get("grid_color", "#FF0000")),
        grid_width=int(st.session_state.get("grid_width", 1)),
        center_color=str(st.session_state.get("center_color", "#FF0000")),
        center_alpha=float(st.session_state.get("center_alpha", 0.5)),
    )


@dataclass
class VisualizationTheme:
    """Unified visualization theme."""

    # Colors
    primary_color: str = "#FF0000"
    secondary_color: str = "#0000FF"
    accent_color: str = "#00FF00"

    # Text
    font_family: str = "sans-serif"
    title_size: int = 14
    label_size: int = 12

    # Layout
    padding: float = 0.1
    spacing: float = 0.05

    @classmethod
    def from_session_state(cls) -> "VisualizationTheme":
        """Create theme from session state."""
        return cls(
            primary_color=st.session_state.get("primary_color", "#FF0000"),
            secondary_color=st.session_state.get("secondary_color", "#0000FF"),
            accent_color=st.session_state.get("accent_color", "#00FF00"),
        )
