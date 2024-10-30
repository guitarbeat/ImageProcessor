"""
Components for processing parameter controls.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Literal
import streamlit as st

from . import Component

FilterType = Literal["mean", "std_dev", "lsci"]

@dataclass
class LSCIParams:
    """Parameters for LSCI processing."""
    kernel_size: int = 7
    
    def validate(self) -> None:
        """Validate LSCI parameters."""
        if self.kernel_size % 2 != 1:
            raise ValueError("Kernel size must be odd")
        if self.kernel_size < 3:
            raise ValueError("Kernel size must be at least 3")

@dataclass
class ProcessingParams:
    """Processing parameters configuration."""
    lsci: LSCIParams = field(default_factory=LSCIParams)

class ProcessingParamsControl(Component):
    """Component for controlling processing parameters."""
    
    def __init__(self, params: ProcessingParams, 
                 on_change: Optional[Callable[[ProcessingParams], None]] = None):
        self.params = params
        self.on_change = on_change
    
    def render(self) -> None:
        """Render processing parameters controls."""
        st.subheader("Processing Parameters")
        
        # Only show kernel size control
        kernel_size = st.slider(
            "Kernel Size",
            min_value=3,
            max_value=15,
            value=st.session_state.get('kernel_size', 7),
            step=2,
            key='kernel_size_select',
            help="Size of the processing window (must be odd)"
        )
        
        # Update parameters
        self.params.lsci.kernel_size = kernel_size
        
        try:
            self.params.lsci.validate()
            if self.on_change:
                self.on_change(self.params)
        except ValueError as e:
            st.error(str(e)) 