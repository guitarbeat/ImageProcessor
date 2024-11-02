"""
Component for the application sidebar.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st
from PIL import Image

from app.utils.constants import DEFAULT_COLORMAP

from app.components import Component
from app.components.processing_params import (LSCIParams, ProcessingParams,
                                ProcessingParamsControl)


@dataclass
class DisplaySettings:
    """Display settings configuration."""
    show_colorbar: bool = True
    show_stats: bool = True
    kernel_color: str = "#FF0000"
    kernel_width: int = 2
    grid_color: str = "#FF0000"
    grid_width: int = 1
    center_color: str = "#FF0000"
    center_alpha: float = 0.5
    # Add annotation settings
    annotation_color: str = "#FFFFFF"
    annotation_bg_color: str = "#000000"
    annotation_bg_alpha: float = 0.5
    annotation_decimals: int = 3


@dataclass
class SidebarConfig:
    """Configuration for sidebar component."""
    title: str
    colormaps: list[str]
    on_colormap_changed: Callable[[str], None]
    on_display_settings_changed: Callable[[DisplaySettings], None]
    on_params_changed: Callable[[ProcessingParams], None]
    initial_colormap: str = DEFAULT_COLORMAP


class Sidebar(Component):
    """Sidebar component with settings and controls."""

    def __init__(self, config: SidebarConfig):
        self.config = config

    def render(self, image: Optional[Image.Image]) -> None:
        """Render the sidebar interface."""
        with st.sidebar:
            st.title(self.config.title)

            with st.expander("Display Settings", expanded=True):
                self._render_display_settings()

            # Only show processing parameters if an image is loaded
            if image is not None:
                with st.expander("Processing Parameters", expanded=True):
                    self._render_processing_params()

    def _render_display_settings(self) -> None:
        """Render display-related settings."""
        # Initialize session state if needed
        if 'colormap' not in st.session_state:
            st.session_state.colormap = self.config.initial_colormap
        if 'show_colorbar' not in st.session_state:
            st.session_state.show_colorbar = True
        if 'show_stats' not in st.session_state:
            st.session_state.show_stats = True

        # Add visualization settings to session state
        if 'kernel_color' not in st.session_state:
            st.session_state.kernel_color = "#FF0000"
        if 'kernel_width' not in st.session_state:
            st.session_state.kernel_width = 2
        if 'grid_color' not in st.session_state:
            st.session_state.grid_color = "#FF0000"
        if 'grid_width' not in st.session_state:
            st.session_state.grid_width = 1
        if 'center_color' not in st.session_state:
            st.session_state.center_color = "#FF0000"
        if 'center_alpha' not in st.session_state:
            st.session_state.center_alpha = 0.5

        # Add annotation settings to session state
        if 'annotation_color' not in st.session_state:
            st.session_state.annotation_color = "#FFFFFF"
        if 'annotation_bg_color' not in st.session_state:
            st.session_state.annotation_bg_color = "#000000"
        if 'annotation_bg_alpha' not in st.session_state:
            st.session_state.annotation_bg_alpha = 0.5
        if 'annotation_decimals' not in st.session_state:
            st.session_state.annotation_decimals = 3

        def on_display_settings_change():
            """Handle display settings change."""
            # Update the base session state values
            st.session_state.kernel_color = st.session_state.kernel_color_select
            st.session_state.kernel_width = st.session_state.kernel_width_select
            st.session_state.grid_color = st.session_state.grid_color_select
            st.session_state.grid_width = st.session_state.grid_width_select
            st.session_state.center_color = st.session_state.center_color_select
            st.session_state.center_alpha = st.session_state.center_alpha_select
            st.session_state.show_colorbar = st.session_state.show_colorbar_select
            st.session_state.show_stats = st.session_state.show_stats_select
            
            # Update annotation settings
            st.session_state.annotation_color = st.session_state.annotation_color_select
            st.session_state.annotation_bg_color = st.session_state.annotation_bg_color_select
            st.session_state.annotation_bg_alpha = st.session_state.annotation_bg_alpha_select
            st.session_state.annotation_decimals = st.session_state.annotation_decimals_select

            # Create and pass settings object
            settings = DisplaySettings(
                show_colorbar=st.session_state.show_colorbar_select,
                show_stats=st.session_state.show_stats_select,
                kernel_color=st.session_state.kernel_color_select,
                kernel_width=st.session_state.kernel_width_select,
                grid_color=st.session_state.grid_color_select,
                grid_width=st.session_state.grid_width_select,
                center_color=st.session_state.center_color_select,
                center_alpha=st.session_state.center_alpha_select,
                annotation_color=st.session_state.annotation_color_select,
                annotation_bg_color=st.session_state.annotation_bg_color_select,
                annotation_bg_alpha=st.session_state.annotation_bg_alpha_select,
                annotation_decimals=st.session_state.annotation_decimals_select
            )
            self.config.on_display_settings_changed(settings)

        def on_colormap_change():
            """Handle colormap change."""
            self.config.on_colormap_changed(st.session_state.colormap_select)

        # Create tabs for different settings groups
        basic_tab, visual_tab = st.tabs(["Basic Settings", "Visualization"])

        with basic_tab:
            st.selectbox(
                "Colormap",
                options=self.config.colormaps,
                index=self.config.colormaps.index(st.session_state.colormap),
                key='colormap_select',
                on_change=on_colormap_change,
                help="Choose a colormap for image display"
            )

            st.checkbox(
                "Show Colorbar",
                value=st.session_state.show_colorbar,
                key='show_colorbar_select',
                on_change=on_display_settings_change,
                help="Toggle colorbar visibility"
            )

            st.checkbox(
                "Show Statistics",
                value=st.session_state.show_stats,
                key='show_stats_select',
                on_change=on_display_settings_change,
                help="Toggle statistics visibility"
            )

        with visual_tab:
            # Kernel settings
            st.subheader("Kernel Overlay")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Outline Color",
                    value=st.session_state.kernel_color,
                    key='kernel_color_select',
                    on_change=on_display_settings_change,
                    help="Color of the kernel outline"
                )
            with col2:
                st.slider(
                    "Width",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.kernel_width,
                    key='kernel_width_select',
                    on_change=on_display_settings_change,
                    help="Width of the kernel outline"
                )

            # Grid settings
            st.subheader("Grid Lines")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Color",
                    value=st.session_state.grid_color,
                    key='grid_color_select',
                    on_change=on_display_settings_change,
                    help="Color of the grid lines"
                )
            with col2:
                st.slider(
                    "Width",
                    min_value=1,
                    max_value=3,
                    value=st.session_state.grid_width,
                    key='grid_width_select',
                    on_change=on_display_settings_change,
                    help="Width of the grid lines"
                )

            # Center pixel settings
            st.subheader("Center Pixel")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Color",
                    value=st.session_state.center_color,
                    key='center_color_select',
                    on_change=on_display_settings_change,
                    help="Color of the center pixel highlight"
                )
            with col2:
                st.slider(
                    "Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.center_alpha,
                    key='center_alpha_select',
                    on_change=on_display_settings_change,
                    help="Opacity of the center pixel highlight"
                )

            # Add annotation settings
            st.subheader("Value Annotations")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Text Color",
                    value=st.session_state.annotation_color,
                    key='annotation_color_select',
                    on_change=on_display_settings_change,
                    help="Color of the value annotations"
                )
                st.color_picker(
                    "Background Color",
                    value=st.session_state.annotation_bg_color,
                    key='annotation_bg_color_select',
                    on_change=on_display_settings_change,
                    help="Background color of the value annotations"
                )
            with col2:
                st.slider(
                    "Background Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.annotation_bg_alpha,
                    key='annotation_bg_alpha_select',
                    on_change=on_display_settings_change,
                    help="Opacity of the annotation background"
                )
                st.slider(
                    "Decimal Places",
                    min_value=0,
                    max_value=5,
                    value=st.session_state.annotation_decimals,
                    key='annotation_decimals_select',
                    on_change=on_display_settings_change,
                    help="Number of decimal places to show"
                )

    def _render_processing_params(self) -> None:
        """Render processing parameters section."""
        # Initialize kernel size in session state if needed
        if 'kernel_size' not in st.session_state:
            st.session_state.kernel_size = 7

        params = ProcessingParams(
            lsci=LSCIParams(
                kernel_size=st.session_state.kernel_size
            )
        )
        params_control = ProcessingParamsControl(
            params=params,
            on_change=self.config.on_params_changed
        )
        params_control.render()
