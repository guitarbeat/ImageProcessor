"""LSCI processor with optimizations."""
from typing import Literal, Optional, Tuple
import numpy as np
from .base import ImageProcessor

FilterType = Literal["mean", "std_dev", "lsci"]

class LSCIProcessor(ImageProcessor):
    """Processor for LSCI computation."""
    
    def __init__(
        self,
        kernel_size: int = 7,
        filter_type: FilterType = "lsci",
        chunk_size: int = 1000
    ) -> None:
        """Initialize the LSCI processor."""
        super().__init__(kernel_size=kernel_size, chunk_size=chunk_size)
        self.filter_type = filter_type.lower()

    def _compute_filter(self, window: np.ndarray) -> float:
        """
        Compute filter value for a window.
        
        Args:
            window: numpy array of shape (kernel_size, kernel_size)
            
        Returns:
            float: Computed filter value based on filter_type
        """
        if window.size == 0:
            return 0.0
            
        mean = float(np.mean(window))
        if self.filter_type == "mean":
            return mean
        
        std = float(np.std(window))
        if self.filter_type == "std_dev":
            return std
            
        # Avoid division by zero for LSCI
        if mean < 1e-10:
            return 0.0
        return std / mean