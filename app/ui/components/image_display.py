"""
Component for image display with matplotlib integration.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from app.ui.components.base import Component
from app.ui.settings import DisplaySettings
from app.utils.constants import DEFAULT_DISPLAY_SIZE
from app.utils.context_managers import figure_context
from app.utils.visualization import (
    KernelOverlayConfig,
    SearchWindowOverlayConfig,
    add_colorbar,
    add_kernel_overlay,
    add_search_window_overlay,
    add_statistics,
    highlight_pixel,
)
from app.utils.visualization.utils import add_value_annotations, setup_figure


@dataclass
class ImageDisplayConfig:
    """Configuration for image display."""

    colormap: str
    title: str = ""
    show_colorbar: bool = True
    show_stats: bool = True
    figsize: Tuple[int, int] = DEFAULT_DISPLAY_SIZE


@contextmanager
def display_context(
    settings: DisplaySettings, figsize: Optional[Tuple[int, int]] = None
):
    """Context manager for image display."""
    with figure_context(figsize=figsize) as fig:
        ax = fig.add_subplot(111)
        yield fig, ax
        plt.tight_layout()


class ImageDisplay(Component):
    """Component for displaying images with matplotlib."""

    def __init__(self, config: ImageDisplayConfig):
        self.config = config
        self.settings = DisplaySettings.from_session_state()
        self.vis_config = self.settings.to_visualization_config()
        self.kernel_config = KernelOverlayConfig(
            kernel_size=st.session_state.get("kernel_size", 7),
            outline_color=st.session_state.get("kernel_color", "#FF0000"),
            outline_width=st.session_state.get("kernel_width", 2),
            grid_color=st.session_state.get("grid_color", "#FF0000"),
            grid_width=st.session_state.get("grid_width", 1),
            center_color=st.session_state.get("center_color", "#FF0000"),
            center_alpha=st.session_state.get("center_alpha", 0.5),
        )

    def render(self, image: Image.Image) -> None:
        """Render the image with optional overlays."""
        try:
            # Convert image to grayscale and normalize
            if image.mode != "L":
                image = image.convert("L")
            img_array = np.array(image, dtype=np.float32) / 255.0

            # Store current image array in session state for NLM computation
            st.session_state.current_image_array = img_array

            # Create figure and axis
            with display_context(self.settings, figsize=self.config.figsize) as (
                fig,
                ax,
            ):
                ax.axis("off")

                # Display image
                im = ax.imshow(
                    img_array,
                    cmap=self.config.colormap,
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                    aspect="equal",
                )

                # Add overlays based on session state
                selected_pixel = st.session_state.get("selected_pixel")
                if selected_pixel:
                    x, y = selected_pixel
                    half_kernel = self.kernel_config.kernel_size // 2

                    if self.config.title == "Input Image":
                        # For input image, use original coordinates (x,y)
                        highlight_pixel(
                            ax=ax,
                            position=(x, y),
                            color=self.kernel_config.center_color,
                            alpha=self.kernel_config.center_alpha,
                        )

                        # Add kernel overlay only on input image
                        add_kernel_overlay(
                            ax=ax,
                            center=(x, y),
                            kernel_size=self.kernel_config.kernel_size,
                            image_shape=img_array.shape,
                            config=self.kernel_config,
                        )

                        # Show search window for NLM
                        if st.session_state.get("filter_type") == "nlm":
                            search_config = (
                                SearchWindowOverlayConfig.from_session_state()
                            )
                            search_size = (
                                None
                                if st.session_state.get("use_full_image", True)
                                else st.session_state.get("search_size", 21)
                            )
                            add_search_window_overlay(
                                ax=ax,
                                center=(x, y),
                                search_window_size=search_size,
                                image_shape=img_array.shape,
                                config=search_config,
                            )
                    else:
                        # For processed images, use output coordinates (i,j)
                        # Adjust coordinates by half kernel size
                        i, j = x - half_kernel, y - half_kernel
                        highlight_pixel(
                            ax=ax,
                            position=(i, j),  # Use output coordinates
                            color=self.kernel_config.center_color,
                            alpha=self.kernel_config.center_alpha,
                        )

                # Restore title
                if self.config.title:
                    ax.set_title(self.config.title)

                # Add colorbar if enabled
                if self.vis_config.show_colorbar:
                    add_colorbar(ax, im, "Intensity", self.vis_config)

                # Display plot
                st.pyplot(fig)

                # Display statistics if enabled
                if self.vis_config.show_stats:
                    add_statistics(img_array, "Image", self.vis_config)

                # Add kernel view in an expander
                with st.expander("Kernel View", expanded=True):
                    if selected_pixel:
                        x, y = selected_pixel
                        kernel = self._extract_kernel(
                            x, y, img_array, self.kernel_config.kernel_size
                        )
                        if kernel is not None:
                            self._render_kernel_view(
                                kernel, x, y, self.kernel_config.kernel_size
                            )

        except Exception as e:
            st.error(f"Error displaying image: {e}")

    def _extract_kernel(
        self, x: int, y: int, img_array: np.ndarray, kernel_size: int
    ) -> Optional[np.ndarray]:
        """Extract kernel around selected pixel."""
        half = kernel_size // 2
        try:
            if (
                half <= x < img_array.shape[1] - half
                and half <= y < img_array.shape[0] - half
            ):
                return img_array[y - half : y + half + 1, x - half : x + half + 1]
        except Exception as e:
            st.error(f"Error extracting kernel: {str(e)}")
        return None

    def _render_kernel_view(
        self, kernel: np.ndarray, x: int, y: int, kernel_size: int
    ) -> None:
        """Render kernel view with annotations."""
        fig, ax = setup_figure()  # Use utility function

        # Display kernel with same scaling as main image (0 to 1)
        im = ax.imshow(
            kernel,
            cmap=self.settings.colormap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            aspect="equal",
        )

        # Add value annotations using utility with settings values
        add_value_annotations(
            ax=ax,
            data=kernel,
            decimals=self.settings.decimals,
            color=self.settings.annotation_color,
            bg_color=self.settings.annotation_bg_color,
            bg_alpha=self.settings.annotation_bg_alpha,
        )

        # Create kernel overlay config with current settings
        kernel_config = KernelOverlayConfig(
            kernel_size=kernel_size,
            outline_color=st.session_state.get("kernel_color", "#FF0000"),
            outline_width=st.session_state.get("kernel_width", 2),
            grid_color=st.session_state.get("grid_color", "#FF0000"),
            grid_width=st.session_state.get("grid_width", 1),
            center_color=st.session_state.get("center_color", "#FF0000"),
            center_alpha=st.session_state.get("center_alpha", 0.5),
        )

        # Add kernel overlay (grid and outline)
        add_kernel_overlay(
            ax=ax,
            # Center of the kernel view
            center=(kernel_size // 2, kernel_size // 2),
            kernel_size=kernel_size,
            image_shape=kernel.shape,
            config=kernel_config,
        )

        # Set title
        ax.set_title(f"Kernel at ({x}, {y})")

        # Add colorbar if enabled
        if self.vis_config.show_colorbar:
            plt.colorbar(mappable=im, ax=ax, label="Intensity")

        plt.tight_layout()
        st.pyplot(fig)

        # Show kernel statistics if enabled
        if self.vis_config.show_stats:
            add_statistics(kernel, "Kernel", self.vis_config)
