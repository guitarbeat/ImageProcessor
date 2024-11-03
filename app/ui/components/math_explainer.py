"""
Component for displaying mathematical explanations of image processing algorithms.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st

from app.processors.computations import LSCIComputation, NLMComputation
from app.ui.components.base import Component
from app.ui.settings import DisplaySettings
from app.utils.latex import create_kernel_matrix_latex, get_search_window_bounds
from app.utils.visualization import (
    create_kernel_overlay_config,
)


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
        """Render mathematical explanations with integrated analysis."""
        if not all([self.config.selected_pixel, self.config.image_array is not None]):
            st.info("Select a pixel to view mathematical explanation.")
            return

        try:
            # Get coordinates and validate
            x, y = self.config.selected_pixel
            half_kernel = self.config.kernel_size // 2

            if not (
                0 <= x < self.config.image_array.shape[1]
                and 0 <= y < self.config.image_array.shape[0]
            ):
                st.error("Selected input coordinates (x,y) are out of bounds")
                return

            # Calculate output coordinates
            i = x - half_kernel
            j = y - half_kernel

            # Get kernel and computation objects
            kernel = self._extract_kernel(x, y)
            if kernel is None:
                st.warning("Cannot extract kernel (too close to border).")
                return

            # Get computation and values
            try:
                current_filter = st.session_state.get("filter_type", "lsci")
                computation = self._get_computation(current_filter)
                values = computation.get_intermediate_values(kernel)
                formula_config = computation.get_formula_config()

                # Create substitution dictionary
                subs = self._create_substitution_dict(
                    input_coords=(x, y),
                    output_coords=(i, j),
                    kernel=kernel,
                    values=values,
                    computation=computation,
                )

                # Create tabs for different aspects
                tabs = st.tabs(
                    ["🔍 Overview", "📐 Formulas", "📊 Analysis", "💡 Interpretation"]
                )

                with tabs[0]:
                    self._render_overview(subs, current_filter)

                with tabs[1]:
                    self._render_formulas(formula_config, subs)

                with tabs[2]:
                    self._render_analysis(computation, kernel, subs)

                with tabs[3]:
                    self._render_interpretation(current_filter, subs)

            except Exception as e:
                st.error(f"Error computing values: {str(e)}")

        except Exception as e:
            st.error(f"Error rendering explanation: {str(e)}")

    def _render_overview(self, subs: Dict[str, Any], filter_type: str) -> None:
        """Render overview with key information."""
        st.markdown("### Coordinate Systems")

        col1, col2 = st.columns(2)
        with col1:
            st.latex(
                r"""
            \begin{{aligned}}
            \text{{Input}} &: (x,y) = ({x}, {y}) \\
            I_{{{x},{y}}} &= {original_value:.{decimals}f}
            \end{{aligned}}
            """.format(
                    **subs
                )
            )

        with col2:
            st.latex(
                r"""
            \begin{{aligned}}
            \text{{Output}} &: (i,j) = ({i}, {j}) \\
            \text{{Result}} &= {nlm_value:.{decimals}f}
            \end{{aligned}}
            """.format(
                    **subs
                )
            )

        if filter_type == "nlm":
            st.markdown("### Search Configuration")
            st.latex(
                r"""
            \begin{{aligned}}
            \text{{Search Mode}} &: {search_window_description} \\
            \text{{Filter Strength}} &: h = {filter_strength:.2f}
            \end{{aligned}}
            """.format(
                    **subs
                )
            )

    def _render_analysis(
        self, computation: Any, kernel: np.ndarray, subs: Dict[str, Any]
    ) -> None:
        """Render integrated analysis."""
        if isinstance(computation, NLMComputation):
            similarity_map = computation.compute_similarity_map(
                self.config.image_array, subs["x"], subs["y"]
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                # Plot similarity map
                with figure_context() as fig:
                    ax = fig.add_subplot(111)
                    plot_similarity_map(
                        ax=ax,
                        similarity_map=similarity_map,
                        center=(subs["x"], subs["y"]),
                        kernel_config=self.kernel_config,
                        search_config=SearchWindowOverlayConfig(),
                        vis_config=self.vis_config,
                        title="Weight Distribution",
                        is_full_image=computation.search_window_size is None,
                    )
                    st.pyplot(fig)

            with col2:
                # Show statistics
                stats = self._compute_nlm_stats(similarity_map)
                st.markdown("#### Weight Statistics")
                st.latex(
                    r"""
                \begin{{aligned}}
                \text{{Mean}} &= {mean:.3f} \\
                \text{{Std}} &= {std:.3f} \\
                \text{{Active}} &= {active:.1f}\% \\
                \text{{Range}} &= [{min:.3f}, {max:.3f}]
                \end{{aligned}}
                """.format(
                        **stats
                    )
                )
        else:
            # LSCI Analysis
            stats = self._compute_lsci_stats(kernel)
            st.latex(
                r"""
            \begin{{aligned}}
            \mu &= {mean:.3f} \\
            \sigma &= {std:.3f} \\
            SC &= {sc:.3f}
            \end{{aligned}}
            """.format(
                    **stats
                )
            )

    def _render_interpretation(self, filter_type: str, subs: Dict[str, Any]) -> None:
        """Render interpretation of results."""
        if filter_type == "nlm":
            st.markdown(
                """
            ### NLM Interpretation
            
            #### Weight Distribution
            - Higher weights indicate more similar patches
            - Distribution width shows patch diversity
            - Active ratio indicates uniqueness level
            
            #### Spatial Effects
            - Center weight is always highest (self-similarity)
            - Decay rate shows similarity fall-off
            - Search area affects computation time
            """
            )
        else:
            st.markdown(
                """
            ### LSCI Interpretation
            
            #### Contrast Analysis
            - Higher SC indicates more speckle
            - Lower SC suggests more blurring
            - Local variations show flow patterns
            
            #### Statistical Meaning
            - Mean reflects average intensity
            - Standard deviation shows variation
            - SC normalizes for intensity differences
            """
            )

    def _compute_nlm_stats(self, similarity_map: np.ndarray) -> Dict[str, float]:
        """Compute NLM statistics."""
        return {
            "mean": float(np.mean(similarity_map)),
            "std": float(np.std(similarity_map)),
            "min": float(np.min(similarity_map)),
            "max": float(np.max(similarity_map)),
            "active": float(np.sum(similarity_map > 0) / similarity_map.size * 100),
        }

    def _compute_lsci_stats(self, kernel: np.ndarray) -> Dict[str, float]:
        """Compute LSCI statistics."""
        mean = float(np.mean(kernel))
        std = float(np.std(kernel))
        return {"mean": mean, "std": std, "sc": std / mean if mean > 0 else 0}

    def _get_computation(self, filter_type: str):
        """Get appropriate computation object."""
        if filter_type == "nlm":
            return NLMComputation(
                kernel_size=self.config.kernel_size,
                filter_strength=st.session_state.get("filter_strength", 10.0),
                search_window_size=(
                    None
                    if st.session_state.get("use_full_image", True)
                    else st.session_state.get("search_size", 21)
                ),
            )
        return LSCIComputation(kernel_size=self.config.kernel_size)

    def _create_substitution_dict(
        self,
        input_coords: Tuple[int, int],
        output_coords: Tuple[int, int],
        kernel: np.ndarray,
        values: Dict[str, Any],
        computation: Any,
    ) -> Dict[str, Any]:
        """Create substitution dictionary with consistent coordinate notation."""
        # Get input coordinates
        x, y = input_coords
        # Get output coordinates
        i, j = output_coords  # These are now properly passed in
        half_kernel = self.config.kernel_size // 2

        # Get search window bounds using input coordinates
        search_bounds = get_search_window_bounds(
            x=x,
            y=y,
            search_size=getattr(computation, "search_window_size", None),
            image_width=self.config.image_array.shape[1],
            image_height=self.config.image_array.shape[0],
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
            "processed_x": i,  # Use output coordinates
            "processed_y": j,  # Use output coordinates
            "valid_x_min": half_kernel,
            "valid_x_max": self.config.image_array.shape[1] - half_kernel,
            "valid_y_min": half_kernel,
            "valid_y_max": self.config.image_array.shape[0] - half_kernel,
            "total_pixels": self.config.kernel_size * self.config.kernel_size,
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
                float(values["original_value"]),
                decimals=self.vis_config.decimals,
            ),
            # Image dimensions
            "image_height": self.config.image_array.shape[0],
            "image_width": self.config.image_array.shape[1],
            "valid_height": self.config.image_array.shape[0]
            - self.config.kernel_size
            + 1,
            "valid_width": self.config.image_array.shape[1]
            - self.config.kernel_size
            + 1,
            # Processed coordinates
            **processed_coords,
            # Additional values
            **formatted_values,
        }

    def _extract_kernel(self, x: int, y: int) -> Optional[np.ndarray]:
        """Extract kernel around selected pixel."""
        half = self.config.kernel_size // 2
        try:
            if (
                half <= x < self.config.image_array.shape[1] - half
                and half <= y < self.config.image_array.shape[0] - half
            ):
                return self.config.image_array[
                    y - half : y + half + 1, x - half : x + half + 1
                ]
        except Exception as e:
            st.error(f"Error extracting kernel: {str(e)}")
        return None

    def _render_formulas(
        self, formula_config: Dict[str, Any], subs: Dict[str, Any]
    ) -> None:
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
                                    st.markdown(
                                        formula_info["explanation"].format(**subs)
                                    )

                            except KeyError as e:
                                st.error(
                                    f"Missing value in {formula_info['title']}: {e}"
                                )
                            except Exception as e:
                                st.error(
                                    f"Error rendering {formula_info['title']}: {str(e)}"
                                )

        except Exception as e:
            st.error(f"Error rendering formulas: {str(e)}")
