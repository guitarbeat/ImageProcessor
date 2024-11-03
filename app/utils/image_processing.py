"""Image processing utilities."""
from typing import Tuple, Optional, Literal, Callable
import numpy as np
from PIL import Image
import streamlit as st

FilterType = Literal["lsci", "nlm", "mean", "std_dev", "standard deviation"]

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


def process_image_region(
    image: np.ndarray,
    kernel_size: int,
    filter_type: FilterType,
    region: Optional[Tuple[int, int, int, int]] = None,
    max_pixel: Optional[Tuple[int, int]] = None,
    _progress_callback: Optional[Callable[[float], None]] = None
) -> Optional[np.ndarray]:
    """
    Process image region with specified filter.
    """
    try:
        # Get kernel boundaries
        half_kernel = kernel_size // 2
        
        # Calculate processing bounds
        if max_pixel is not None:
            max_x, max_y = max_pixel
            x_start, y_start = half_kernel, half_kernel
            x_end = min(max_x + 1, image.shape[1] - half_kernel)
            y_end = min(max_y + 1, image.shape[0] - half_kernel)
        elif region is not None:
            x1, y1, x2, y2 = region
            x_start = max(half_kernel, x1)
            y_start = max(half_kernel, y1)
            x_end = min(x2 + 1, image.shape[1] - half_kernel)
            y_end = min(y2 + 1, image.shape[0] - half_kernel)
        else:
            x_start, y_start = half_kernel, half_kernel
            y_end, x_end = image.shape[0] - half_kernel, image.shape[1] - half_kernel

        # Create result array with same shape as input
        result = np.zeros_like(image)
        
        # Create window view for efficient computation
        window_view = np.lib.stride_tricks.sliding_window_view(
            image, (kernel_size, kernel_size)
        )
        
        # Get the valid portion of the window view
        valid_view = window_view[
            y_start-half_kernel:y_end-half_kernel,
            x_start-half_kernel:x_end-half_kernel
        ]
        
        # Normalize filter type for comparison
        filter_type = filter_type.lower()
        
        total_pixels = (y_end - y_start) * (x_end - x_start)
        
        # Apply specific filter computation based on type
        if filter_type == "lsci":
            means = np.mean(valid_view, axis=(2, 3))
            stds = np.std(valid_view, axis=(2, 3), ddof=1)
            mask = means > 1e-10
            result[y_start:y_end, x_start:x_end] = np.where(
                mask, stds / means, 0.0
            )
        elif filter_type in ["mean", "standard deviation", "std_dev"]:
            if filter_type == "mean":
                values = np.mean(valid_view, axis=(2, 3))
            else:
                values = np.std(valid_view, axis=(2, 3), ddof=1)
            result[y_start:y_end, x_start:x_end] = values
        elif filter_type == "nlm":
            # NLM is handled by SpatialFilterProcessor directly
            return None
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
        # Update progress if callback provided
        if _progress_callback:
            update_interval = max(1, total_pixels // 100)
            for i in range(0, total_pixels, update_interval):
                _progress_callback(min(1.0, i / total_pixels))
            _progress_callback(1.0)
            
        return result

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


@st.cache_data
def apply_colormap(image: np.ndarray, colormap: str = 'gray') -> Image.Image:
    """Apply colormap to image and convert to PIL Image."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Normalize image to [0, 1] if needed
    if image.dtype != np.float32 or image.min() < 0 or image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
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
