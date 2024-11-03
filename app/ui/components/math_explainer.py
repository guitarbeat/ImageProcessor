"""
Component for displaying mathematical explanations of image processing algorithms.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import streamlit as st
import numpy as np

from app.ui.components.base import Component
from app.processors.computations import LSCIComputation, NLMComputation
from app.utils.latex import create_kernel_matrix_latex, get_search_window_bounds
from app.utils.visualization import create_visualization_config, create_kernel_overlay_config
from app.ui.settings import DisplaySettings


@dataclass
class MathExplainerConfig:
    """Configuration for math explainer component."""
    formula_config: Dict[str, Any]
    kernel_size: int = 7
    selected_pixel: Optional[tuple[int, int]] = None
    image_array: Optional[np.ndarray] = None


class MathExplainer(Component):
    """Component for displaying mathematical explanations."""

    def __init__(self, config: MathExplainerConfig):
        self.config = config
        self.settings = DisplaySettings.from_session_state()
        self.vis_config = self.settings.to_visualization_config()
        self.kernel_config = create_kernel_overlay_config()

    def render(self) -> None:
        """Render mathematical explanations."""
        if not all([self.config.selected_pixel, self.config.image_array is not None]):
            st.info("Select a pixel to view mathematical explanation.")
            return

        try:
            # Get input coordinates (x,y)
            x, y = self.config.selected_pixel
            half_kernel = self.config.kernel_size // 2
            
            # Validate input coordinates
            if not (0 <= x < self.config.image_array.shape[1] and 
                    0 <= y < self.config.image_array.shape[0]):
                st.error("Selected input coordinates (x,y) are out of bounds")
                return
            
            # Calculate output coordinates (i,j)
            i = x - half_kernel  # Output row coordinate
            j = y - half_kernel  # Output column coordinate
            
            # Get kernel in input coordinates
            kernel = self._extract_kernel(x, y)
            if kernel is None:
                st.warning("Cannot extract kernel for selected input coordinates (too close to border).")
                return

            # Get computation object and values
            try:
                current_filter = st.session_state.get('filter_type', 'lsci')
                computation = self._get_computation(current_filter)
                values = computation.get_intermediate_values(kernel)
                formula_config = computation.get_formula_config()
                
                # Create substitution dictionary with coordinate system mapping
                subs = self._create_substitution_dict(
                    input_coords=(x, y),
                    output_coords=(i, j),
                    kernel=kernel,
                    values=values,
                    computation=computation
                )
                
                # Add coordinate system explanation
                st.markdown("""
                ### Coordinate Systems
                - Input coordinates (x,y): Position in original image
                - Output coordinates (i,j): Position in processed image
                - Search coordinates (s,t): Positions within search window
                - Patch coordinates (p,q): Local coordinates within patches
                """)
                
                # Render formulas with consistent coordinate notation
                self._render_formulas(formula_config, subs)
                
            except Exception as e:
                st.error(f"Error computing values: {str(e)}")
                
        except Exception as e:
            st.error(f"Error rendering mathematical explanation: {str(e)}")

    def _get_computation(self, filter_type: str):
        """Get appropriate computation object."""
        if filter_type == "nlm":
            return NLMComputation(
                kernel_size=self.config.kernel_size,
                filter_strength=st.session_state.get('filter_strength', 10.0),
                search_window_size=None if st.session_state.get('use_full_image', True) 
                    else st.session_state.get('search_size', 21)
            )
        return LSCIComputation(kernel_size=self.config.kernel_size)

    def _create_substitution_dict(self, input_coords: Tuple[int, int], 
                                output_coords: Tuple[int, int],
                                kernel: np.ndarray, values: Dict[str, Any],
                                computation: Any) -> Dict[str, Any]:
        """Create substitution dictionary with consistent coordinate notation."""
        # Get input coordinates
        x, y = input_coords
        # Get output coordinates
        i, j = output_coords  # These are now properly passed in
        half_kernel = self.config.kernel_size // 2
        
        # Get search window bounds using input coordinates
        search_bounds = get_search_window_bounds(
            x=x, y=y,
            search_size=getattr(computation, 'search_window_size', None),
            image_width=self.config.image_array.shape[1],
            image_height=self.config.image_array.shape[0]
        )
        
        # Format values with consistent coordinate notation
        formatted_values = {}
        for k, v in values.items():
            if isinstance(v, (float, np.floating)):
                formatted_values[k] = float(v)
            elif isinstance(v, (int, np.integer)):
                formatted_values[k] = int(v)
            else:
                formatted_values[k] = v
        
        # Add processed coordinates explicitly
        processed_coords = {
            'processed_x': i,  # Use output coordinates
            'processed_y': j,  # Use output coordinates
            'valid_x_min': half_kernel,
            'valid_x_max': self.config.image_array.shape[1] - half_kernel,
            'valid_y_min': half_kernel,
            'valid_y_max': self.config.image_array.shape[0] - half_kernel,
            'total_pixels': self.config.kernel_size * self.config.kernel_size
        }
        
        return {
            # Input coordinates
            "x": x,
            "y": y,
            # Output coordinates
            "i": i,
            "j": j,
            # Configuration
            "decimals": self.vis_config.decimals,
            **search_bounds,
            "kernel_size": self.config.kernel_size,
            "half_kernel": half_kernel,
            # Kernel visualization
            "kernel_matrix_latex": create_kernel_matrix_latex(
                kernel, 
                float(values['original_value']),
                decimals=self.vis_config.decimals
            ),
            # Image dimensions
            "image_height": self.config.image_array.shape[0],
            "image_width": self.config.image_array.shape[1],
            "valid_height": self.config.image_array.shape[0] - self.config.kernel_size + 1,
            "valid_width": self.config.image_array.shape[1] - self.config.kernel_size + 1,
            # Processed coordinates
            **processed_coords,
            # Additional values
            **formatted_values
        }

    def _extract_kernel(self, x: int, y: int) -> Optional[np.ndarray]:
        """Extract kernel around selected pixel."""
        half = self.config.kernel_size // 2
        try:
            if (half <= x < self.config.image_array.shape[1] - half and
                half <= y < self.config.image_array.shape[0] - half):
                return self.config.image_array[
                    y - half:y + half + 1,
                    x - half:x + half + 1
                ]
        except Exception as e:
            st.error(f"Error extracting kernel: {str(e)}")
        return None

    def _render_formulas(self, formula_config: Dict[str, Any], subs: Dict[str, Any]) -> None:
        """Render formulas with substitutions."""
        try:
            # Create tabs for different formula sections
            main_tab, details_tab = st.tabs(["Main Formula", "Detailed Explanation"])
            
            with main_tab:
                # Main formula section
                st.markdown(f"### {formula_config['title']}")
                
                # Main formula with larger display
                try:
                    st.latex(formula_config["main_formula"].format(**subs))
                except KeyError as e:
                    st.error(f"Missing value in main formula: {e}")
                
                # Main explanation
                try:
                    st.markdown(formula_config["explanation"].format(**subs))
                except KeyError as e:
                    st.error(f"Missing value in main explanation: {e}")
            
            with details_tab:
                # Additional formulas in expandable sections
                if "additional_formulas" in formula_config:
                    for formula_info in formula_config["additional_formulas"]:
                        with st.expander(formula_info["title"], expanded=False):
                            try:
                                if "formula" in formula_info:
                                    st.latex(formula_info["formula"].format(**subs))
                                
                                if "explanation" in formula_info:
                                    st.markdown(formula_info["explanation"].format(**subs))
                                    
                            except KeyError as e:
                                st.error(f"Missing value in {formula_info['title']}: {e}")
                            except Exception as e:
                                st.error(f"Error rendering {formula_info['title']}: {str(e)}")
                                
        except Exception as e:
            st.error(f"Error rendering formulas: {str(e)}")
