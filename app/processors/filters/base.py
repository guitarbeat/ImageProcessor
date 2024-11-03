"""Base filter computation classes."""

from abc import abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np

from app.processors.base_computation import BaseComputation


class FilterBase(BaseComputation):
    """Base class for all filter computations."""

    def __init__(self, kernel_size: int):
        super().__init__(kernel_size)

    @abstractmethod
    def get_formula_config(self) -> Dict[str, Any]:
        """Get the mathematical explanation."""

    @abstractmethod
    def process_image(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """Process entire image using this computation."""
