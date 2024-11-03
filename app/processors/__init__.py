"""Image processing module."""
from .base import ImageProcessor
from .computations import LSCIComputation, NLMComputation
from .filters.processor import SpatialFilterProcessor
from .filters.utils import compute_local_stats, create_window_view

__all__ = [
    'ImageProcessor',
    'LSCIComputation',
    'NLMComputation',
    'SpatialFilterProcessor',
    'compute_local_stats',
    'create_window_view'
] 