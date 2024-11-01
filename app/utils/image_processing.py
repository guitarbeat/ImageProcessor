"""Image processing utilities."""


from typing import Tuple, Optional
import numpy as np
from PIL import Image
from app.processors.base import ImageProcessor


def preprocess_image(_image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """
    Cache preprocessed images. Uses underscore prefix to exclude image from hashing.
    Since we're converting the image anyway, we don't need to hash the input.

    Args:
        _image: PIL Image to preprocess

    Returns:
        Tuple of (PIL Image, numpy array)
    """
    if _image.mode != 'L':
        _image = _image.convert('L')
    img_array = np.array(_image, dtype=np.float32) / 255.0
    return _image, img_array


def get_valid_kernel_bounds(image_size: tuple[int, int], kernel_size: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Calculate valid coordinate ranges for kernel processing.

    Args:
        image_size: Tuple of (width, height)
        kernel_size: Size of the processing kernel

    Returns:
        ((x_min, x_max), (y_min, y_max)): Valid coordinate ranges
    """
    half_kernel = kernel_size // 2
    width, height = image_size

    x_bounds = (half_kernel, width - half_kernel - 1)
    y_bounds = (half_kernel, height - half_kernel - 1)

    return x_bounds, y_bounds


def process_image_region(
    processor: ImageProcessor,
    image: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None,
    max_pixel: Optional[Tuple[int, int]] = None,
    region_key: Optional[str] = None,
    containing_region: Optional[str] = None
) -> Optional[np.ndarray]:
    """Process image region with proper error handling."""
    try:
        # Initialize result array with zeros (black)
        result = np.zeros_like(image)

        # Get kernel boundaries
        half_kernel = processor.kernel_size // 2

        # Calculate processing bounds
        if max_pixel is not None:
            max_x, max_y = max_pixel
            x_start, y_start = half_kernel, half_kernel
            x_end, y_end = min(max_x + 1, image.shape[1] - half_kernel)
            y_end = min(max_y + 1, image.shape[0] - half_kernel)
        else:
            x_start, y_start = half_kernel, half_kernel
            y_end, x_end = image.shape[0] - \
                half_kernel, image.shape[1] - half_kernel

        # Process region up to max_pixel
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                window = processor.extract_window(y, x, image)
                result[y, x] = processor._compute_filter(window)

        return result

    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")
