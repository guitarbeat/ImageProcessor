"""
Components for processing parameter controls.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import streamlit as st

from app.ui.components.base import Component

FilterType = Literal["lsci", "nlm"]


@dataclass
class CommonParams:
    """Common parameters for all filters."""

    kernel_size: int = 7

    def validate(self) -> None:
        """Validate common parameters."""
        if self.kernel_size % 2 != 1:
            raise ValueError("Kernel size must be odd")
        if self.kernel_size < 3:
            raise ValueError("Kernel size must be at least 3")


@dataclass
class NLMParams:
    """Parameters specific to Non-Local Means processing."""

    filter_strength: float = 10.0
    search_window_size: Optional[int] = None

    def validate(self, kernel_size: int) -> None:
        """Validate NLM parameters."""
        if self.filter_strength <= 0:
            raise ValueError("Filter strength must be positive")
        if self.search_window_size is not None:
            if self.search_window_size < kernel_size:
                raise ValueError("Search window must be larger than kernel")
            if self.search_window_size % 2 == 0:
                raise ValueError("Search window size must be odd")


@dataclass
class ProcessingParams:
    """Processing parameters configuration."""

    common: CommonParams = field(default_factory=CommonParams)
    nlm: NLMParams = field(default_factory=NLMParams)
    filter_type: FilterType = "lsci"

    def validate(self) -> None:
        """Validate all parameters."""
        self.common.validate()
        if self.filter_type == "nlm":
            self.nlm.validate(self.common.kernel_size)


class ProcessingParamsControl(Component):
    """Component for controlling processing parameters."""

    def __init__(
        self,
        params: ProcessingParams,
        on_change: Optional[Callable[[ProcessingParams], None]] = None,
    ):
        self.params = params
        self.on_change = on_change

    def render(self) -> None:
        """Render processing parameters controls."""
        # Common Parameters
        kernel_size = st.slider(
            "Neighborhood Size",
            min_value=3,
            max_value=15,
            value=st.session_state.get("kernel_size", 7),
            step=2,
            help="Size of the local neighborhood",
        )
        st.caption(f"Selected kernel: {kernel_size}×{kernel_size} pixels")

        # Filter Selection
        st.markdown("---")
        filter_options = {"LSCI": "🌟 Speckle Contrast", "NLM": "🔍 Non-Local Means"}
        filter_type = st.radio(
            "Filter Type",
            options=list(filter_options.keys()),
            format_func=lambda x: filter_options[x],
            horizontal=True,
            help="Select processing method",
        )

        self.params.common.kernel_size = kernel_size
        self.params.filter_type = filter_type.lower()

        # NLM Parameters
        if filter_type == "NLM":
            st.markdown("---")

            # Filter strength with better visual feedback
            filter_strength = st.slider(
                "Filter Strength",
                min_value=1.0,
                max_value=100.0,
                value=st.session_state.get("filter_strength", 10.0),
                step=0.5,
                help="Controls smoothing intensity",
            )
            effect = (
                "Low"
                if filter_strength < 25
                else "Medium" if filter_strength < 75 else "High"
            )
            st.caption(f"Smoothing: {effect}")

            st.markdown("---")

            # Search window settings in a cleaner layout
            use_full_image = st.toggle(
                "Global Search",
                value=st.session_state.get("use_full_image", True),
                help="Search entire image for similar patches",
            )

            if use_full_image:
                st.info("🔍 Searching entire image for similar patches")
            else:
                search_size = st.select_slider(
                    "Local Search Window Size",
                    options=list(range(kernel_size, 32, 2)),
                    value=st.session_state.get("search_size", 21),
                    help="Size of local neighborhood to search",
                )
                st.caption(f"Search area: {search_size}×{search_size} pixels")
                self.params.nlm.search_window_size = search_size

            self.params.nlm.filter_strength = filter_strength

        try:
            self.params.validate()
            if self.on_change:
                self.on_change(self.params)
        except ValueError as e:
            st.error(f"⚠️ {str(e)}", icon="🚨")
