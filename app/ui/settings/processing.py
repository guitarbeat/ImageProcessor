"""Processing settings configuration."""
from dataclasses import dataclass, field
import streamlit as st

@dataclass
class ProcessingSettings:
    """Processing-specific settings."""
    kernel_size: int = 7
    filter_type: str = "lsci"
    filter_strength: float = 10.0
    use_full_image: bool = True
    search_size: int = 21
    selected_filters: list[str] = field(
        default_factory=lambda: ["LSCI", "Mean", "Standard Deviation"]
    )

    @classmethod
    def from_session_state(cls) -> 'ProcessingSettings':
        """Create from session state."""
        return cls(
            kernel_size=st.session_state.get('kernel_size', 7),
            filter_type=st.session_state.get('filter_type', 'lsci'),
            filter_strength=st.session_state.get('filter_strength', 10.0),
            use_full_image=st.session_state.get('use_full_image', True),
            search_size=st.session_state.get('search_size', 21),
            selected_filters=st.session_state.get('selected_filters', 
                ["LSCI", "Mean", "Standard Deviation"])
        )
