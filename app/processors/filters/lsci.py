"""LSCI computation implementation."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from app.processors.filters.utils import FilterComputation


class LSCIComputation(FilterComputation):
    def compute(self, window: np.ndarray) -> float:
        self.validate_input(window)
        mean = float(np.mean(window))
        if mean < 1e-10:
            return 0.0
        std = float(np.std(window, ddof=1))
        return std / mean

    def process_image(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """Process entire image using LSCI computation."""
        half = self.kernel_size // 2
        height, width = image.shape

        # Create result array for valid region only
        result = np.zeros((height - 2 * half, width - 2 * half), dtype=np.float32)

        # Process valid region only
        total_pixels = (height - 2 * half) * (width - 2 * half)
        processed = 0

        # Create sliding window view
        window_view = np.lib.stride_tricks.sliding_window_view(
            image, (self.kernel_size, self.kernel_size)
        )

        # Process each pixel
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                window = window_view[i, j]
                result[i, j] = self.compute(window)

                processed += 1
                if progress_callback and total_pixels > 0:
                    progress_callback(processed / total_pixels)

        return result

    def get_intermediate_values(self, window: np.ndarray) -> Dict[str, float]:
        """Get all intermediate values used in LSCI computation."""
        mean = float(np.mean(window))
        std = float(np.std(window))
        original_value = float(window[window.shape[0] // 2, window.shape[1] // 2])
        sc = self.compute(window)

        return {
            "original_value": original_value,
            "mean": mean,
            "std": std,
            "sc": sc,
            "half_kernel": self.kernel_size // 2,
            "image_height": window.shape[0],
            "image_width": window.shape[1],
            "valid_height": window.shape[0] - self.kernel_size + 1,
            "valid_width": window.shape[1] - self.kernel_size + 1,
            "total_pixels": self.kernel_size**2,
        }

    def get_formula_config(self) -> Dict[str, Any]:
        """Get LSCI formula configuration."""
        from app.utils.latex import SPECKLE_FORMULA_CONFIG

        return SPECKLE_FORMULA_CONFIG
