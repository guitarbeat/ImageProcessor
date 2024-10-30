"""Image processing utilities."""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
from processors.base import ImageProcessor
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner="Processing image...")
def process_image_region(
    processor: ImageProcessor,
    image: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None,
    region_key: Optional[str] = None,
    containing_region: Optional[str] = None
) -> Optional[np.ndarray]:
    """Process image region with proper error handling.
    
    Args:
        processor: Image processor instance
        image: Input image array
        region: Optional (x1, y1, x2, y2) region to process
        region_key: Optional key identifying the region
        containing_region: Optional key identifying a larger cached region
        
    Returns:
        Processed image array or None if processing fails
    """
    try:
        # Extract region if specified
        if region:
            x1, y1, x2, y2 = region
            # Expand region to account for kernel size
            half_kernel = processor.kernel_size // 2
            y1_proc = max(0, y1 - half_kernel)
            y2_proc = min(image.shape[0], y2 + half_kernel)
            x1_proc = max(0, x1 - half_kernel)
            x2_proc = min(image.shape[1], x2 + half_kernel)
            
            # Extract expanded region
            process_region = image[y1_proc:y2_proc, x1_proc:x2_proc]
            
            # Process the region
            result = processor.process(
                image=process_region,
                region_key=region_key,
                containing_region=containing_region
            )
            
            # Crop result back to original region size
            if result is not None:
                y_offset = y1 - y1_proc
                x_offset = x1 - x1_proc
                result = result[
                    y_offset:y_offset + (y2 - y1),
                    x_offset:x_offset + (x2 - x1)
                ]
        else:
            # Process full image
            result = processor.process(
                image=image,
                region_key=region_key,
                containing_region=containing_region
            )
        
        if result is None or result.size == 0:
            raise RuntimeError("Processing returned empty result")
            
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")