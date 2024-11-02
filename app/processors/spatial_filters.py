"""Spatial filtering processors for image analysis."""
from typing import Literal, Optional, Tuple, Callable
import numpy as np
import streamlit as st

from app.processors.base import ImageProcessor
from app.utils.image_processing import process_image_region

FilterType = Literal["LSCI", "Mean", "Standard Deviation"]

class SpatialFilterProcessor(ImageProcessor):
    """Processor for spatial filtering computations."""
    
    def __init__(
        self,
        kernel_size: int = 7,
        filter_type: FilterType = "LSCI",
        chunk_size: int = 1000
    ) -> None:
        """Initialize the spatial filter processor."""
        super().__init__(kernel_size=kernel_size, chunk_size=chunk_size)
        self.filter_type = filter_type

    def _compute_filter(self, window: np.ndarray) -> float:
        """Compute filter value for a window."""
        try:
            if window.size == 0:
                return 0.0
                
            mean = float(np.mean(window))
            if self.filter_type.upper() == "MEAN":
                return mean
            
            std = float(np.std(window))
            if self.filter_type.upper() == "STANDARD DEVIATION":
                return std
                
            # LSCI computation
            if mean < 1e-10:
                return 0.0
            return std / mean
            
        except Exception as e:
            print(f"Error in filter computation: {e}")
            return 0.0
            
    def process(self, image: np.ndarray,
                region: Optional[Tuple[int, int, int, int]] = None,
                max_pixel: Optional[Tuple[int, int]] = None,
                progress_callback: Optional[Callable[[float], None]] = None,
                **kwargs) -> Optional[np.ndarray]:
        """Process image using cached region processing."""
        try:
            # Use cached processing with correct parameter order
            result = process_image_region(
                image=image,
                kernel_size=self.kernel_size,
                filter_type=self.filter_type,  # No need to convert here, done in process_image_region
                region=region,
                max_pixel=max_pixel
            )
            
            # Handle progress callback if provided
            if progress_callback:
                progress_callback(1.0)
            
            if result is None:
                raise ValueError("Processing returned None")
                
            return result
            
        except Exception as e:
            st.error(f"Error in process method: {str(e)}")
            return None