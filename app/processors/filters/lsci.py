"""LSCI computation implementation."""

from typing import Any, Dict

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
