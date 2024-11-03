"""Spatial filtering processors for image analysis."""

from typing import Callable, Literal, Optional

import numpy as np
import streamlit as st

from app.processors.base import ImageProcessor
from app.processors.filters.lsci import LSCIComputation
from app.processors.filters.nlm import NLMComputation
from app.ui.settings.display import DisplaySettings

FilterType = Literal["lsci", "nlm", "mean", "std_dev"]


class SpatialFilterProcessor(ImageProcessor):
    """Processor for spatial filtering computations."""

    def __init__(
        self,
        kernel_size: int = 7,
        filter_type: FilterType = "lsci",
        chunk_size: int = 1000,
        filter_strength: float = 10.0,
        search_window_size: Optional[int] = None,
    ) -> None:
        """Initialize the spatial filter processor."""
        super().__init__(kernel_size=kernel_size, chunk_size=chunk_size)
        self.filter_type = filter_type.lower()
        self.filter_strength = filter_strength
        self.search_window_size = search_window_size
        self.settings = DisplaySettings.from_session_state()

        # Create appropriate computation object
        if self.filter_type == "nlm":
            self.computation = NLMComputation(
                kernel_size=kernel_size,
                filter_strength=filter_strength,
                search_window_size=search_window_size,
            )
        else:
            self.computation = LSCIComputation(kernel_size=kernel_size)

    def _compute_filter(self, window: np.ndarray) -> float:
        """Compute filter value for a window."""
        if self.filter_type == "nlm":
            return self.computation.compute(window)
        elif self.filter_type == "lsci":
            mean = np.mean(window)
            if mean > 1e-10:
                std = np.std(window, ddof=1)
                return std / mean
            return 0.0
        elif self.filter_type == "mean":
            return float(np.mean(window))
        elif self.filter_type in ["std_dev", "standard deviation"]:
            return float(np.std(window, ddof=1))
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    def process(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Process image with specified filter."""
        try:
            # Input validation
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            if image.ndim != 2:
                raise ValueError("Image must be 2-dimensional")
            if not np.issubdtype(image.dtype, np.floating):
                raise ValueError("Image must be floating point type")

            # Store image for NLM computation
            if self.filter_type == "nlm":
                st.session_state.current_image_array = image
                # For NLM, delegate to the NLM computation object
                result = self.computation.process_image(
                    image=image, progress_callback=progress_callback
                )
                if result is None:
                    raise ValueError("NLM processing returned None")
                return result

            # For other filters (LSCI, Mean, Std Dev)
            half = self.kernel_size // 2
            height, width = image.shape
            result = np.zeros((height - 2 * half, width - 2 * half), dtype=np.float32)

            # Create sliding window view
            window_view = np.lib.stride_tricks.sliding_window_view(
                image, (self.kernel_size, self.kernel_size)
            )

            total_pixels = result.size
            processed = 0

            # Process each pixel
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    window = window_view[i, j]
                    result[i, j] = self._compute_filter(window)

                    processed += 1
                    if progress_callback and total_pixels > 0:
                        progress_callback(processed / total_pixels)

            return result

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return None
