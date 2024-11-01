"""
Module for adding visual overlays to image processing visualizations.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


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


def add_kernel_overlay(
    ax: plt.Axes,
    center: Tuple[int, int],
    kernel_size: int,
    image_shape: Tuple[int, int],
    config: Optional[KernelOverlayConfig] = None
) -> None:
    """
    Add kernel overlay to the input image.

    Args:
        ax: Matplotlib axes to draw on
        center: (x, y) coordinates of kernel center
        kernel_size: Size of the kernel window
        image_shape: (height, width) of the image
        config: Optional configuration for overlay styling
    """
    if config is None:
        config = KernelOverlayConfig(
            kernel_size=kernel_size,
            outline_color=st.session_state.get('kernel_color', "#FF0000"),
            outline_width=st.session_state.get('kernel_width', 2),
            grid_color=st.session_state.get('grid_color', "#FF0000"),
            grid_width=st.session_state.get('grid_width', 1),
            center_color=st.session_state.get('center_color', "#FF0000"),
            center_alpha=st.session_state.get('center_alpha', 0.5)
        )

    try:
        half = kernel_size // 2
        x, y = center

        # Create kernel boundary lines
        x_min, y_min = x - half - 0.5, y - half - 0.5
        x_max, y_max = x + half + 0.5, y + half + 0.5

        # Ensure boundaries are within image
        x_min = max(-0.5, x_min)
        y_min = max(-0.5, y_min)
        x_max = min(image_shape[1] - 0.5, x_max)
        y_max = min(image_shape[0] - 0.5, y_max)

        # Create line segments for kernel boundary
        segments = [
            [(x_min, y_min), (x_max, y_min)],  # Bottom
            [(x_max, y_min), (x_max, y_max)],  # Right
            [(x_max, y_max), (x_min, y_max)],  # Top
            [(x_min, y_max), (x_min, y_min)]   # Left
        ]

        # Add kernel outline
        line_collection = LineCollection(
            segments,
            colors=config.outline_color,
            linewidths=config.outline_width
        )
        ax.add_collection(line_collection)

        # Add grid lines
        grid_segments = []
        for i in range(1, kernel_size):
            offset = i - half - 0.5
            # Vertical lines
            if x + offset >= x_min and x + offset <= x_max:
                grid_segments.append(
                    [(x + offset, y_min), (x + offset, y_max)]
                )
            # Horizontal lines
            if y + offset >= y_min and y + offset <= y_max:
                grid_segments.append(
                    [(x_min, y + offset), (x_max, y + offset)]
                )

        grid_collection = LineCollection(
            grid_segments,
            colors=config.grid_color,
            linewidths=config.grid_width,
            linestyles=':'
        )
        ax.add_collection(grid_collection)

        # Highlight center pixel with a semi-transparent square
        ax.add_patch(Rectangle(
            (x - 0.5, y - 0.5),  # Offset by 0.5 to center on pixel
            1, 1,                 # Width and height of 1 pixel
            facecolor=config.center_color,
            alpha=config.center_alpha,
            edgecolor='none'
        ))

    except ValueError as ve:
        print(f"ValueError in add_kernel_overlay: {str(ve)}")
    except TypeError as te:
        print(f"TypeError in add_kernel_overlay: {str(te)}")
    except Exception as e:
        print(f"Unexpected error in add_kernel_overlay: {str(e)}")


def highlight_pixel(
    ax: plt.Axes,
    position: Tuple[int, int],
    color: str = "#FF0000",
    alpha: float = 0.5
) -> None:
    """
    Highlight a single pixel in the processed image.

    Args:
        ax: Matplotlib axes to draw on
        position: (x, y) coordinates of pixel to highlight
        color: Color of the highlight
        alpha: Transparency of the highlight
    """
    try:
        x, y = position
        ax.add_patch(Rectangle(
            (x - 0.5, y - 0.5),  # Offset by 0.5 to center on pixel
            1, 1,                 # Width and height of 1 pixel
            facecolor=color,
            alpha=alpha,
            edgecolor='none'
        ))
    except ValueError as ve:
        print(f"ValueError in highlight_pixel: {str(ve)}")
    except TypeError as te:
        print(f"TypeError in highlight_pixel: {str(te)}")
    except Exception as e:
        print(f"Unexpected error in highlight_pixel: {str(e)}")
