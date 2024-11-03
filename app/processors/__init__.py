"""Image processing module."""

from .processor_base import BaseImageProcessor
from .filters.processor import SpatialFilterProcessor
from .filters.utils import compute_local_stats, create_window_view
from .filters.lsci import LSCIComputation
from .filters.nlm import NLMComputation

__all__ = [
    "BaseImageProcessor",
    "LSCIComputation",
    "NLMComputation",
    "SpatialFilterProcessor",
    "compute_local_stats",
    "create_window_view",
]
