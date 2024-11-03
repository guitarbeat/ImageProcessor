"""
Module for adding visual overlays to image processing visualizations.
"""

from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from .config import KernelOverlayConfig, SearchWindowOverlayConfig, VisualizationConfig

# Add orientation literal type
Orientation = Literal["vertical", "horizontal"]


def add_colorbar(
    ax: plt.Axes, im: plt.cm.ScalarMappable, label: str, config: VisualizationConfig
) -> None:
    """Add colorbar with consistent styling."""
    if config.show_colorbar and im is not None:
        plt.colorbar(mappable=im, ax=ax, label=label)


def add_statistics(data: np.ndarray, title: str, config: VisualizationConfig) -> None:
    """Add statistics with consistent formatting."""
    if config.show_stats:
        stats = {
            "Min": float(np.min(data)),
            "Max": float(np.max(data)),
            "Mean": float(np.mean(data)),
            "Std": float(np.std(data)),
        }
        st.write(
            f"{title} Statistics:",
            {k: f"{v:.{config.decimals}f}" for k, v in stats.items()},
        )


def plot_similarity_map(
    ax: plt.Axes,
    similarity_map: np.ndarray,
    center: Tuple[int, int],
    kernel_config: KernelOverlayConfig,
    search_config: Optional[SearchWindowOverlayConfig],
    vis_config: VisualizationConfig,
    title: str,
    is_full_image: bool = False,
    show_kernel: bool = False,
) -> None:
    """Plot similarity map with consistent styling."""
    # Display map
    im = ax.imshow(
        similarity_map,
        cmap=vis_config.colormap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    # Add colorbar if enabled
    add_colorbar(ax, im, "Similarity Weight (w)", vis_config)

    # Add overlays
    x, y = center
    if not is_full_image:
        x, y = 0, 0  # Center coordinates for cropped view

    # Always show center pixel
    highlight_pixel(ax, (x, y), color=kernel_config.center_color)

    # Only show kernel if requested
    if show_kernel:
        add_kernel_overlay(
            ax=ax,
            center=(x, y),
            kernel_size=kernel_config.kernel_size,
            image_shape=(
                int(similarity_map.shape[1]),
                int(similarity_map.shape[0]),
            ),  # Cast to tuple[int, int]
            config=kernel_config,
        )

    if search_config and not is_full_image:
        add_search_window_overlay(
            ax=ax,
            center=(x, y),
            search_window_size=similarity_map.shape[0],
            image_shape=(
                int(similarity_map.shape[1]),
                int(similarity_map.shape[0]),
            ),  # Cast to tuple[int, int]
            config=search_config,
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plot_weight_distribution(
    ax: plt.Axes,
    weights: np.ndarray,
    vis_config: VisualizationConfig,
    orientation: Orientation = "vertical",  # Use Literal type
    show_percentiles: bool = True,
) -> None:
    """Plot weight distribution with consistent styling."""
    non_zero_weights = weights[weights > 0]

    # Plot histogram with properly typed orientation
    ax.hist(
        non_zero_weights,
        bins=50,
        orientation=orientation,  # Now properly typed as Literal
        color=st.session_state.get("center_color", "#FF0000"),
        alpha=0.7,
        label=f"n={len(non_zero_weights)}",
    )

    # Add statistics lines
    mean_val = np.mean(non_zero_weights)
    median_val = np.median(non_zero_weights)

    if orientation == "vertical":
        ax.axvline(
            mean_val,
            color="r",
            linestyle="--",
            label=f"Mean={mean_val:.{vis_config.decimals}f}",
        )
        ax.axvline(
            median_val,
            color="g",
            linestyle=":",
            label=f"Median={median_val:.{vis_config.decimals}f}",
        )

        if show_percentiles:
            for p in [25, 75]:
                p_val = np.percentile(non_zero_weights, p)
                ax.axvline(
                    p_val,
                    color=f"C{p//25}",
                    linestyle=":",
                    label=f"{p}th={p_val:.{vis_config.decimals}f}",
                )
    else:
        ax.axhline(
            mean_val,
            color="r",
            linestyle="--",
            label=f"Mean={mean_val:.{vis_config.decimals}f}",
        )
        ax.axhline(
            median_val,
            color="g",
            linestyle=":",
            label=f"Median={median_val:.{vis_config.decimals}f}",
        )

    ax.legend()


def add_kernel_overlay(
    ax: plt.Axes,
    center: Tuple[int, int],
    kernel_size: int,
    image_shape: Tuple[int, int],
    config: KernelOverlayConfig,
) -> None:
    """Add kernel overlay to the input image."""
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
            [(x_min, y_max), (x_min, y_min)],  # Left
        ]

        # Add kernel outline
        line_collection = LineCollection(
            segments,
            colors=config.outline_color,
            linewidths=config.outline_width,
            label="Kernel",
        )
        ax.add_collection(line_collection)

        # Add grid lines
        grid_segments = []
        for i in range(1, kernel_size):
            offset = i - half - 0.5
            # Vertical lines
            if x + offset >= x_min and x + offset <= x_max:
                grid_segments.append([(x + offset, y_min), (x + offset, y_max)])
            # Horizontal lines
            if y + offset >= y_min and y + offset <= y_max:
                grid_segments.append([(x_min, y + offset), (x_max, y + offset)])

        grid_collection = LineCollection(
            grid_segments,
            colors=config.grid_color,
            linewidths=config.grid_width,
            linestyles=":",
            label="Grid",
        )
        ax.add_collection(grid_collection)

    except Exception as e:
        print(f"Error in add_kernel_overlay: {str(e)}")


def highlight_pixel(
    ax: plt.Axes,
    position: Tuple[int, int],
    color: str = "#FF0000",
    alpha: float = 0.5,
    label: str = "Center",
) -> None:
    """Highlight a single pixel in the image."""
    try:
        x, y = position
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),  # Offset by 0.5 to center on pixel
                1,
                1,  # Width and height of 1 pixel
                facecolor=color,
                alpha=alpha,
                edgecolor="none",
                label=label,
            )
        )
    except Exception as e:
        print(f"Error in highlight_pixel: {str(e)}")


def add_search_window_overlay(
    ax: plt.Axes,
    center: Tuple[int, int],
    search_window_size: Optional[int],
    image_shape: Tuple[int, int],
    config: SearchWindowOverlayConfig,
) -> None:
    """Add search window overlay to the image."""
    try:
        x, y = center

        if search_window_size is None:
            # Use full image boundaries
            x_min, y_min = -0.5, -0.5
            x_max = image_shape[1] - 0.5
            y_max = image_shape[0] - 0.5
        else:
            # Use search window boundaries
            half = search_window_size // 2
            x_min, y_min = x - half - 0.5, y - half - 0.5
            x_max, y_max = x + half + 0.5, y + half + 0.5

            # Ensure boundaries are within image
            x_min = max(-0.5, x_min)
            y_min = max(-0.5, y_min)
            x_max = min(image_shape[1] - 0.5, x_max)
            y_max = min(image_shape[0] - 0.5, y_max)

        # Create line segments for search window boundary
        segments = [
            [(x_min, y_min), (x_max, y_min)],  # Bottom
            [(x_max, y_min), (x_max, y_max)],  # Right
            [(x_max, y_max), (x_min, y_max)],  # Top
            [(x_min, y_max), (x_min, y_min)],  # Left
        ]

        # Add search window outline
        line_collection = LineCollection(
            segments,
            colors=config.outline_color,
            linewidths=config.outline_width,
            linestyles=config.outline_style,
            label="Search Window",
        )
        ax.add_collection(line_collection)

    except Exception as e:
        print(f"Error in add_search_window_overlay: {str(e)}")
