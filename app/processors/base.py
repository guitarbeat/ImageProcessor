"""
Base class for image processors.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Protocol
import numpy as np
import streamlit as st


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(self, progress: float) -> None: ...


class ImageProcessor(ABC):
    """Abstract base class for image processors."""

    def __init__(
        self,
        kernel_size: int = 7,
        chunk_size: int = 1000
    ) -> None:
        """Initialize the base processor."""
        self.kernel_size = kernel_size
        self.chunk_size = chunk_size

    def extract_window(self, y: int, x: int, image: np.ndarray) -> np.ndarray:
        """Extract a window from the image centered at (y, x)."""
        half = self.kernel_size // 2
        return image[y - half:y + half + 1, x - half:x + half + 1]

    @abstractmethod
    def _compute_filter(self, window: np.ndarray) -> float:
        """Compute filter value for a window. Must be implemented by subclasses."""
        pass

    def process(self, image: np.ndarray,
                region: Optional[Tuple[int, int, int, int]] = None,
                max_pixel: Optional[Tuple[int, int]] = None,
                image_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Process the image using a sliding window approach.

        Args:
            image: Input image array
            region: Optional tuple of (x1, y1, x2, y2) defining the region to process (deprecated)
            max_pixel: Optional tuple of (x, y) defining maximum pixel to process
            image_id: Optional identifier for the image to handle caching
        """
        try:
            # Initialize result array with zeros (black)
            result = np.zeros_like(image)

            # Get kernel boundaries
            half = self.kernel_size // 2

            # Calculate processing bounds
            if max_pixel is not None:
                max_x, max_y = max_pixel
                x_start, y_start = half, half
                x_end, y_end = min(
                    max_x + 1, image.shape[1] - half), min(max_y + 1, image.shape[0] - half)
            else:
                # Process full image
                y_start, y_end = half, image.shape[0] - half
                x_start, x_end = half, image.shape[1] - half

            total_pixels = (y_end - y_start) * (x_end - x_start)
            processed_pixels = 0

            # Process in chunks for better performance
            chunk_size = min(self.chunk_size, y_end - y_start)

            for chunk_start in range(y_start, y_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, y_end)

                # Process chunk
                for y in range(chunk_start, chunk_end):
                    for x in range(x_start, x_end):
                        window = self.extract_window(y, x, image)
                        result[y, x] = self._compute_filter(window)

                    processed_pixels += (x_end - x_start)

                    # Update progress
                    if total_pixels > 0:
                        progress = processed_pixels / total_pixels
                        st.session_state.processing_progress = progress

            # Ensure progress shows completion
            st.session_state.processing_progress = 1.0

            return result

        except Exception as e:
            st.error(f"Error in process method: {str(e)}")
            return None
