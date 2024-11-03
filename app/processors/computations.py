"""Filter computation classes with integrated math explanations."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np


class FilterComputation(ABC):
    """Base class for filter computations and their mathematical explanations."""

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    @abstractmethod
    def compute(self, window: np.ndarray) -> float:
        """Compute the filter value."""
        raise NotImplementedError

    @abstractmethod
    def get_formula_config(self) -> Dict[str, Any]:
        """Get the mathematical explanation."""
        raise NotImplementedError

    def get_intermediate_values(self, window: np.ndarray) -> Dict[str, float]:
        """Get intermediate values for math explanation."""
        return {}  # Default implementation returns empty dict

    def process_image(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """Process entire image using this computation."""
        raise NotImplementedError
