"""
Reusable UI components for the application.
"""

from abc import ABC, abstractmethod

from PIL import Image


class Component(ABC):
    """Base class for UI components."""

    @abstractmethod
    def render(self, image: Image.Image = None) -> None:
        """Render the component in the Streamlit app."""
