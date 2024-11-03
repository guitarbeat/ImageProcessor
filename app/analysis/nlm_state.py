"""State management for NLM analysis."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import streamlit as st


@dataclass
class NLMState:
    """Encapsulate NLM-specific state."""

    input_coords: Tuple[int, int]  # (x,y) in input space
    output_coords: Tuple[int, int]  # (i,j) in output space
    kernel_size: int
    filter_strength: float
    use_full_image: bool
    search_size: Optional[int]
    filter_type: str = "nlm"
    display_mode: str = "side_by_side"
    show_analysis: bool = True
    show_formulas: bool = True
    active_tab: str = "overview"

    @classmethod
    def from_session_state(cls) -> "NLMState":
        """Create state from session state with validation."""
        # Get selected pixel with safe default
        selected_pixel = st.session_state.get("selected_pixel")
        if selected_pixel is None:
            x, y = 0, 0
        else:
            try:
                x, y = selected_pixel
            except (TypeError, ValueError):
                x, y = 0, 0

        kernel_size = st.session_state.get("kernel_size", 7)
        half_kernel = kernel_size // 2

        return cls(
            input_coords=(x, y),
            output_coords=(x - half_kernel, y - half_kernel),
            kernel_size=kernel_size,
            filter_strength=st.session_state.get("filter_strength", 10.0),
            use_full_image=st.session_state.get("use_full_image", True),
            search_size=(
                None
                if st.session_state.get("use_full_image", True)
                else st.session_state.get("search_size", 21)
            ),
            filter_type=st.session_state.get("filter_type", "nlm"),
        )

    @property
    def search_bounds(self) -> Dict[str, int]:
        """Get search window boundaries."""
        if self.use_full_image:
            return {
                "x_min": self.kernel_size // 2,
                "x_max": st.session_state.image_array.shape[1] - self.kernel_size // 2,
                "y_min": self.kernel_size // 2,
                "y_max": st.session_state.image_array.shape[0] - self.kernel_size // 2,
            }
        else:
            x, y = self.input_coords
            half_search = self.search_size // 2
            return {
                "x_min": max(self.kernel_size // 2, x - half_search),
                "x_max": min(
                    st.session_state.image_array.shape[1] - self.kernel_size // 2,
                    x + half_search,
                ),
                "y_min": max(self.kernel_size // 2, y - half_search),
                "y_max": min(
                    st.session_state.image_array.shape[0] - self.kernel_size // 2,
                    y + half_search,
                ),
            }

    @classmethod
    def initialize(cls) -> None:
        """Initialize all required state variables with consistent defaults."""
        defaults = {
            "filter_type": "nlm",
            "kernel_size": 7,
            "filter_strength": 10.0,
            "use_full_image": True,
            "search_size": 21,
            "show_analysis": True,
            "show_formulas": True,
            "active_tab": "overview",
            "display_mode": "side_by_side",
            "selected_filters": ["NLM"],
            "needs_processing": True,
            "selected_pixel": None,  # Add explicit default for selected_pixel
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
