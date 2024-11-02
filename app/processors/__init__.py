"""
Image processing algorithms.
"""
from .base import ImageProcessor
from .spatial_filters import SpatialFilterProcessor

__all__ = ['ImageProcessor', 'SpatialFilterProcessor'] 