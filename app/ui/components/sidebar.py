"""Component for the application sidebar."""

from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st
from PIL import Image

from app.ui.components.base import Component
from app.ui.components.processing_params import (CommonParams, NLMParams,
                                                 ProcessingParams,
                                                 ProcessingParamsControl)
from app.ui.settings.display import DisplaySettings
from app.utils.constants import DEFAULT_COLORMAP


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
        """Initialize sidebar with config."""
        self.config = config
        self.settings = DisplaySettings.from_session_state()

    def render(self, image: Optional[Image.Image]) -> None:
        """Render the sidebar interface."""
        with st.sidebar:
            st.title(self.config.title)

            with st.expander("Display Settings", expanded=True):
                self._render_display_settings()

            if image is not None:
                with st.expander("Processing Parameters", expanded=True):
                    self._render_processing_params()

    def on_colormap_change(self) -> None:
        """Handle colormap change."""
        colormap = st.session_state.colormap_select
        st.session_state.colormap = colormap
        self.config.on_colormap_changed(colormap)

    def on_display_settings_change(self) -> None:
        """Handle display settings change."""
        # Update session state values
        st.session_state.update(
            {
                "kernel_color": st.session_state.kernel_color_select,
                "kernel_width": st.session_state.kernel_width_select,
                "grid_color": st.session_state.grid_color_select,
                "grid_width": st.session_state.grid_width_select,
                "center_color": st.session_state.center_color_select,
                "center_alpha": st.session_state.center_alpha_select,
                "show_colorbar": st.session_state.show_colorbar_select,
                "show_stats": st.session_state.show_stats_select,
                "annotation_color": st.session_state.annotation_color_select,
                "annotation_bg_color": st.session_state.annotation_bg_color_select,
                "annotation_bg_alpha": st.session_state.annotation_bg_alpha_select,
                "annotation_decimals": st.session_state.annotation_decimals_select,
            }
        )

        # Create and pass settings object using centralized settings
        settings = DisplaySettings.from_session_state()
        self.config.on_display_settings_changed(settings)

    def _render_display_settings(self) -> None:
        """Render display-related settings."""
        # Initialize session state variables using centralized defaults
        defaults = {
            "colormap": self.config.initial_colormap,
            "show_colorbar": self.settings.show_colorbar,
            "show_stats": self.settings.show_stats,
            "kernel_color": self.settings.kernel_color,
            "kernel_width": self.settings.kernel_width,
            "grid_color": self.settings.grid_color,
            "grid_width": self.settings.grid_width,
            "center_color": self.settings.center_color,
            "center_alpha": self.settings.center_alpha,
            "annotation_color": self.settings.annotation_color,
            "annotation_bg_color": self.settings.annotation_bg_color,
            "annotation_bg_alpha": self.settings.annotation_bg_alpha,
            "annotation_decimals": self.settings.decimals,
        }

        # Initialize state variables
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
            select_key = f"{key}_select"
            if select_key not in st.session_state:
                st.session_state[select_key] = st.session_state[key]

        # Create tabs for different settings groups
        basic_tab, visual_tab, annotation_tab = st.tabs(
            ["Basic", "Visualization", "Annotations"]
        )

        self._render_basic_settings(basic_tab)
        self._render_visual_settings(visual_tab)
        self._render_annotation_settings(annotation_tab)

        # Update all settings at once
        self.on_display_settings_change()

    def _render_basic_settings(self, tab) -> None:
        """Render basic display settings."""
        with tab:
            st.selectbox(
                "Colormap",
                options=self.config.colormaps,
                key="colormap_select",
                on_change=self.on_colormap_change,
                help="Choose a colormap for image display",
            )

            st.checkbox(
                "Show Colorbar",
                key="show_colorbar_select",
                on_change=self.on_display_settings_change,
                help="Toggle colorbar visibility",
            )

            st.checkbox(
                "Show Statistics",
                key="show_stats_select",
                on_change=self.on_display_settings_change,
                help="Toggle statistics visibility",
            )

    def _render_visual_settings(self, tab) -> None:
        """Render visualization settings."""
        with tab:
            # Kernel overlay settings
            st.subheader("Kernel Overlay")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Outline Color",
                    key="kernel_color_select",
                    help="Color of the kernel outline",
                )
                st.color_picker(
                    "Grid Color",
                    key="grid_color_select",
                    help="Color of the grid lines",
                )
            with col2:
                st.slider(
                    "Outline Width",
                    min_value=1,
                    max_value=5,
                    key="kernel_width_select",
                    help="Width of the kernel outline",
                )
                st.slider(
                    "Grid Width",
                    min_value=1,
                    max_value=3,
                    key="grid_width_select",
                    help="Width of the grid lines",
                )

            # Center pixel settings
            st.subheader("Center Pixel")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Color",
                    key="center_color_select",
                    help="Color of the center pixel highlight",
                )
            with col2:
                st.slider(
                    "Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    key="center_alpha_select",
                    help="Opacity of the center pixel highlight",
                )

            # Search window settings
            st.subheader("Search Window")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Search Window Color",
                    key="search_window_color_select",
                    value="#0000FF",  # Default blue
                    help="Color of the search window outline",
                )
            with col2:
                st.slider(
                    "Search Window Width",
                    min_value=1,
                    max_value=5,
                    key="search_window_width_select",
                    value=2,
                    help="Width of the search window outline",
                )

            st.selectbox(
                "Search Window Style",
                options=["solid", "dashed", "dotted", "dashdot"],
                key="search_window_style_select",
                help="Line style of the search window",
            )

    def _render_annotation_settings(self, tab) -> None:
        """Render annotation settings."""
        with tab:
            st.subheader("Value Annotations")
            col1, col2 = st.columns(2)
            with col1:
                st.color_picker(
                    "Text Color",
                    key="annotation_color_select",
                    help="Color of the value text",
                )
                st.color_picker(
                    "Background Color",
                    key="annotation_bg_color_select",
                    help="Color of the text background",
                )
            with col2:
                st.slider(
                    "Background Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    key="annotation_bg_alpha_select",
                    help="Opacity of the text background",
                )
                st.slider(
                    "Decimal Places",
                    min_value=0,
                    max_value=6,
                    key="annotation_decimals_select",
                    help="Number of decimal places in value annotations",
                )

    def _render_processing_params(self) -> None:
        """Render processing parameters section."""
        # Initialize all parameters at once
        defaults = {
            "kernel_size": 7,
            "filter_type": "lsci",
            "filter_strength": 10.0,
            "use_full_image": True,
            "search_size": 21,
            "selected_filters": ["LSCI", "Mean", "Standard Deviation"],
        }

        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

        # Create parameters object with current state
        params = ProcessingParams(
            common=CommonParams(kernel_size=st.session_state.kernel_size),
            nlm=NLMParams(
                filter_strength=st.session_state.filter_strength,
                search_window_size=(
                    None
                    if st.session_state.use_full_image
                    else st.session_state.search_size
                ),
            ),
            filter_type=st.session_state.filter_type,
        )

        # Create and render the control
        params_control = ProcessingParamsControl(
            params=params, on_change=self._on_params_change
        )
        params_control.render()

    def _on_params_change(self, params: ProcessingParams) -> None:
        """Handle parameter changes and trigger processing."""
        # Update session state
        st.session_state.update(
            {
                "kernel_size": params.common.kernel_size,
                "filter_type": params.filter_type,
                "filter_strength": params.nlm.filter_strength,
                "search_window_size": params.nlm.search_window_size,
                "use_full_image": params.nlm.search_window_size is None,
                "search_size": params.nlm.search_window_size or 21,
                "needs_processing": True,
            }
        )

        # Call the provided callback
        if self.config.on_params_changed:
            self.config.on_params_changed(params)
