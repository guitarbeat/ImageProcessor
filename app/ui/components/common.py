"""Shared UI components."""
import streamlit as st
from typing import List, Tuple

def create_pixel_selector(x_range: Tuple[int, int], y_range: Tuple[int, int]) -> Tuple[int, int]:
    """Create consistent pixel selection UI."""
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input(
            "X Position",
            min_value=x_range[0],
            max_value=x_range[1],
            value=x_range[0],
            step=1,
            help=f"X coordinate ({x_range[0]} to {x_range[1]})"
        )
    with col2:
        y = st.number_input(
            "Y Position",
            min_value=y_range[0],
            max_value=y_range[1],
            value=y_range[0],
            step=1,
            help=f"Y coordinate ({y_range[0]} to {y_range[1]})"
        )
    return x, y 