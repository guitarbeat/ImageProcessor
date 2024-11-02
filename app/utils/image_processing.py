"""Image processing utilities."""
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import streamlit as st



@st.cache_data
def preprocess_image(_image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """
    Cache preprocessed images. Uses underscore prefix to exclude image from hashing.
    Since we're converting the image anyway, we don't need to hash the input.
    """
    if _image.mode != 'L':
        _image = _image.convert('L')
    img_array = np.array(_image, dtype=np.float32) / 255.0
    return _image, img_array


def get_valid_kernel_bounds(image_size: tuple[int, int], kernel_size: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Calculate valid coordinate ranges for kernel processing."""
    half_kernel = kernel_size // 2
    width, height = image_size

    x_bounds = (half_kernel, width - half_kernel - 1)
    y_bounds = (half_kernel, height - half_kernel - 1)

    return x_bounds, y_bounds


@st.cache_data(show_spinner=False)
def process_image_region(
    image: np.ndarray,
    kernel_size: int,
    filter_type: str,
    region: Optional[Tuple[int, int, int, int]] = None,
    max_pixel: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Cached image region processing.
    
    The cache key is based on:
    - Image content
    - Kernel size
    - Filter type (LSCI, Mean, Standard Deviation)
    - Processing region or max pixel
    """
    try:
        # Initialize result array with zeros
        result = np.zeros_like(image)
        
        # Get kernel boundaries
        half_kernel = kernel_size // 2
        
        # Calculate processing bounds
        if max_pixel is not None:
            max_x, max_y = max_pixel
            x_start, y_start = half_kernel, half_kernel
            x_end = min(max_x + 1, image.shape[1] - half_kernel)
            y_end = min(max_y + 1, image.shape[0] - half_kernel)
        else:
            x_start, y_start = half_kernel, half_kernel
            y_end, x_end = image.shape[0] - half_kernel, image.shape[1] - half_kernel
            
        # Process region using sliding window
        window_view = np.lib.stride_tricks.sliding_window_view(
            image, (kernel_size, kernel_size)
        )[y_start-half_kernel:y_end-half_kernel, x_start-half_kernel:x_end-half_kernel]
        
        # Normalize filter type for comparison
        filter_type = filter_type.upper()
        
        # Apply specific filter computation based on type
        if filter_type == "MEAN":
            # Simple mean filter
            result[y_start:y_end, x_start:x_end] = np.mean(window_view, axis=(2, 3))
            
        elif filter_type == "STANDARD DEVIATION":
            # Standard deviation filter
            result[y_start:y_end, x_start:x_end] = np.std(window_view, axis=(2, 3))
            
        elif filter_type == "LSCI":  # Explicit LSCI check
            means = np.mean(window_view, axis=(2, 3))
            stds = np.std(window_view, axis=(2, 3))
            # Avoid division by zero and handle edge cases
            mask = means > 1e-10
            result[y_start:y_end, x_start:x_end] = np.where(
                mask, stds / means, 0.0
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
        return result

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

@st.cache_data
def apply_colormap(image: np.ndarray, colormap: str = 'gray') -> Image.Image:
    """Apply colormap to image and convert to PIL Image."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Create figure without display
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Apply colormap
    cm = mpl.colormaps[colormap]
    colored_image = cm(image)
    
    # Convert to uint8
    colored_image = (colored_image * 255).astype(np.uint8)
    
    # Convert to PIL
    result = Image.fromarray(colored_image)
    
    # Cleanup
    plt.close(fig)
    
    return result
