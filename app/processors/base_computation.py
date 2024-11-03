"""Base computation classes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseComputation(ABC):
    """Base class for all computations."""

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size
        self.half_kernel = kernel_size // 2

    def validate_input(self, data: np.ndarray) -> None:
        """Validate input data."""
        if data is None or data.size == 0:
            raise ValueError("Invalid input data")
        if data.ndim != 2:
            raise ValueError("Input must be 2-dimensional")

    def extract_patch(self, data: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
        """Extract patch with boundary checking."""
        try:
            if (
                self.half_kernel <= x < data.shape[1] - self.half_kernel
                and self.half_kernel <= y < data.shape[0] - self.half_kernel
            ):
                return data[
                    y - self.half_kernel : y + self.half_kernel + 1,
                    x - self.half_kernel : x + self.half_kernel + 1,
                ]
        except Exception:
            return None
        return None

    @abstractmethod
    def compute(self, window: np.ndarray) -> float:
        """Compute the result for a window."""

    @abstractmethod
    def get_intermediate_values(self, window: np.ndarray) -> Dict[str, Any]:
        """Get intermediate values for explanation."""
