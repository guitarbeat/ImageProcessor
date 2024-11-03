"""Utility functions and base classes for filters."""
from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

class FilterComputation(ABC):
    """Base class for filter computations."""
    
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size
        
    @abstractmethod
    def compute(self, window: np.ndarray) -> float:
        """Compute the filter value."""
        pass
    
    @abstractmethod
    def get_formula_config(self) -> Dict[str, Any]:
        """Get the mathematical explanation."""
        pass

    def validate_input(self, data: np.ndarray) -> None:
        """Validate input data."""
        if data is None or data.size == 0:
            raise ValueError("Invalid input data")
        if data.ndim != 2:
            raise ValueError("Input must be 2-dimensional")

def compute_local_stats(window: np.ndarray) -> Tuple[float, float, float]:
    """Compute local statistics for a window."""
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    center_value = float(window[window.shape[0]//2, window.shape[1]//2])
    return mean, std, center_value

def create_window_view(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Create sliding window view of image."""
    return np.lib.stride_tricks.sliding_window_view(
        image, (kernel_size, kernel_size)
    )