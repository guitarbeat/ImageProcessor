"""
Reusable UI components for the application.
"""
from abc import ABC, abstractmethod
import streamlit as st

class Component(ABC):
    """Base class for UI components."""
    
    @abstractmethod
    def render(self) -> None:
        """Render the component in the Streamlit app."""
        pass
