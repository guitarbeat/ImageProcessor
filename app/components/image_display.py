"""
Component for image display with matplotlib integration.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

from components import Component
from utils.constants import DEFAULT_DISPLAY_SIZE, DEFAULT_COLORMAP
from utils.visualization import add_kernel_overlay, highlight_pixel


@dataclass
class ImageDisplayConfig:
    """Configuration for image display."""
    colormap: str = DEFAULT_COLORMAP
    title: str = ""
    show_colorbar: bool = True
    show_stats: bool = True
    figsize: Tuple[int, int] = DEFAULT_DISPLAY_SIZE


@contextmanager
def figure_context(*args, **kwargs):
    """Context manager for creating and closing matplotlib figures."""
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


class ImageDisplay(Component):
    """Component for displaying images with matplotlib."""

    def __init__(self, config: ImageDisplayConfig):
        self.config = config

    @staticmethod
    def _get_image_stats(img_array: np.ndarray) -> Dict[str, float]:
        """Calculate and return image statistics."""
        return {
            "Min": float(np.min(img_array)),
            "Max": float(np.max(img_array)),
            "Mean": float(np.mean(img_array)),
            "Std": float(np.std(img_array))
        }

    def render(self, image: Image.Image) -> None:
        """Render the image with optional overlays and kernel view."""
        try:
            # Convert image to grayscale and normalize
            if image.mode != 'L':
                image = image.convert('L')
            img_array = np.array(image, dtype=np.float32) / 255.0

            # Create figure and axis
            with figure_context(figsize=self.config.figsize) as fig:
                ax = fig.add_subplot(111)

                # Display image
                ax.imshow(
                    img_array,
                    cmap=self.config.colormap,
                    interpolation='nearest',
                    vmin=0,
                    vmax=1,
                    aspect='equal'
                )

                # Add overlays based on session state
                selected_pixel = st.session_state.get('selected_pixel')
                if self.config.title == "Input Image" and selected_pixel:
                    kernel_size = st.session_state.get('kernel_size', 7)
                    add_kernel_overlay(
                        ax=ax,
                        center=selected_pixel,
                        kernel_size=kernel_size,
                        image_shape=img_array.shape
                    )
                elif selected_pixel:
                    highlight_pixel(
                        ax=ax,
                        position=selected_pixel
                    )

                # Configure axes
                ax.axis('off')
                if self.config.show_colorbar:
                    plt.colorbar(
                        ax.images[0],
                        ax=ax,
                        orientation='vertical',
                        label='Intensity'
                    )
                if self.config.title:
                    ax.set_title(self.config.title)

                # Adjust layout and display plot
                plt.tight_layout()
                st.pyplot(fig)

                # Display image statistics if enabled
                if self.config.show_stats:
                    stats = self._get_image_stats(img_array)
                    st.write("Image Statistics:", stats)

            # Add kernel view in an expander
            with st.expander("View Kernel Around Center Pixel"):
                if selected_pixel:
                    x, y = selected_pixel
                    kernel_size = st.session_state.get('kernel_size', 7)
                    half_kernel = kernel_size // 2

                    # Check kernel boundaries
                    if (0 <= x - half_kernel and
                        0 <= y - half_kernel and
                        x + half_kernel < img_array.shape[1] and
                            y + half_kernel < img_array.shape[0]):

                        # Extract and plot the kernel
                        kernel = img_array[
                            y - half_kernel: y + half_kernel + 1,
                            x - half_kernel: x + half_kernel + 1
                        ]

                        with figure_context() as fig_kernel:
                            ax_kernel = fig_kernel.add_subplot(111)
                            ax_kernel.imshow(kernel, cmap='gray',
                                             interpolation='nearest',
                                             vmin=0,
                                             vmax=1)

                            # Add pixel value annotations
                            for i in range(kernel_size):
                                for j in range(kernel_size):
                                    value = kernel[i, j]
                                    decimals = st.session_state.get(
                                        'annotation_decimals', 3)
                                    ax_kernel.text(
                                        j, i, f'{value:.{decimals}f}',
                                        ha='center',
                                        va='center',
                                        color=st.session_state.get(
                                            'annotation_color', '#FFFFFF'),
                                        bbox=dict(
                                            facecolor=st.session_state.get(
                                                'annotation_bg_color', '#000000'),
                                            alpha=st.session_state.get(
                                                'annotation_bg_alpha', 0.5),
                                            edgecolor='none',
                                            pad=1
                                        )
                                    )

                            # Add grid and center highlight to kernel view
                            add_kernel_overlay(
                                ax=ax_kernel,
                                center=(half_kernel, half_kernel),
                                kernel_size=kernel_size,
                                image_shape=kernel.shape
                            )

                            ax_kernel.set_title(f"Kernel around ({x}, {y})")
                            ax_kernel.axis('off')
                            plt.tight_layout()
                            st.pyplot(fig_kernel)
                    else:
                        st.warning(
                            "Selected pixel is too close to the border to extract the kernel.")
                else:
                    st.info("No center pixel selected.")

        except (ValueError, TypeError, RuntimeError) as e:
            st.error(f"Error displaying image: {e}")
