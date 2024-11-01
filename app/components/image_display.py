"""
Component for image display with matplotlib integration.
"""
from dataclasses import dataclass
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from contextlib import contextmanager

from . import Component
from utils.constants import DEFAULT_DISPLAY_SIZE, DEFAULT_COLORMAP

@dataclass
class ImageDisplayConfig:
    """Configuration for image display."""
    colormap: str = DEFAULT_COLORMAP
    title: str = ""
    show_colorbar: bool = True
    show_stats: bool = True
    figsize: tuple[int, int] = DEFAULT_DISPLAY_SIZE

@contextmanager
def figure_context(*args, **kwargs):
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
    def _get_image_stats(_img_array: np.ndarray) -> dict:
        """Calculate and cache image statistics."""
        return {
            "Min": float(np.min(_img_array)),
            "Max": float(np.max(_img_array)),
            "Mean": float(np.mean(_img_array)),
            "Std": float(np.std(_img_array))
        }

    def render(self, image: Image.Image) -> None:
        """Render the image with optimized matplotlib display."""
        try:
            # Convert image to array
            if image.mode != 'L':
                image = image.convert('L')
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Create new figure for each render
            with figure_context(figsize=self.config.figsize) as fig:
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                # Display image with nearest neighbor interpolation and exact pixel boundaries
                image_plot = ax.imshow(
                    img_array,
                    cmap=st.session_state.colormap,
                    interpolation='nearest',  # Use nearest neighbor interpolation
                    vmin=0,
                    vmax=1,
                    aspect='equal',  # Maintain aspect ratio
                    filternorm=False,  # Disable pixel value normalization
                    resample=False,    # Disable resampling
                )
                
                # Add grid to show pixel boundaries
                ax.set_xticks(np.arange(-0.5, img_array.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, img_array.shape[0], 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                
                # Add colorbar if enabled
                if st.session_state.show_colorbar:
                    plt.colorbar(
                        image_plot, 
                        ax=ax,
                        orientation='vertical',
                        label='Intensity'
                    )
                
                ax.set_title(self.config.title)
                
                # Display the plot
                st.pyplot(fig)
                
                # Show statistics if enabled
                if st.session_state.show_stats:
                    stats = self._get_image_stats(img_array)
                    st.write("Image Statistics:", stats)
                
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")