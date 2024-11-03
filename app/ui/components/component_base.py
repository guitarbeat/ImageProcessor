"""Base component for UI elements."""

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class BaseUIComponent(ABC):
    """Base class for UI components."""

    @abstractmethod
    def render(self, image: Optional[Image.Image] = None) -> None:
        """Render the component in the Streamlit app."""
