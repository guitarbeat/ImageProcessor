"""Base class for filter analysis."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class FilterAnalysis(ABC):
    """Base class for filter-specific analysis."""

    @abstractmethod
    def analyze_weights(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze weight distribution."""
        pass

    @abstractmethod
    def analyze_spatial_patterns(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze spatial patterns."""
        pass

    @abstractmethod
    def render_analysis(self, img_array: np.ndarray, x: int, y: int) -> None:
        """Render complete analysis."""
        pass
