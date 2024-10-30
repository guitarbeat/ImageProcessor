"""
Component for the application sidebar.
"""
from dataclasses import dataclass
from typing import Callable
import streamlit as st

from . import Component
from .processing_params import ProcessingParamsControl, ProcessingParams, LSCIParams
from utils.constants import DEFAULT_COLORMAP

@dataclass
class DisplaySettings:
    """Display settings configuration."""
    show_colorbar: bool = True
    show_stats: bool = True

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
        
    def render(self) -> None:
        """Render the sidebar interface."""
        with st.sidebar:
            st.title(self.config.title)
            
            with st.expander("Display Settings", expanded=True):
                self._render_display_settings()
            
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

        def on_colormap_change():
            """Handle colormap change."""
            self.config.on_colormap_changed(st.session_state.colormap_select)
            
        def on_display_settings_change():
            """Handle display settings change."""
            settings = DisplaySettings(
                show_colorbar=st.session_state.show_colorbar_select,
                show_stats=st.session_state.show_stats_select
            )
            self.config.on_display_settings_changed(settings)
            
        # Use session state for persistent values with callbacks
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