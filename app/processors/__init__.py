"""Image processing module."""

from .computations import LSCIComputation, NLMComputation
from .filters.processor import SpatialFilterProcessor
from .filters.utils import compute_local_stats, create_window_view
from .processor_base import BaseImageProcessor

__all__ = [
    "BaseImageProcessor",
    "LSCIComputation",
    "NLMComputation",
    "SpatialFilterProcessor",
    "compute_local_stats",
    "create_window_view",
]
