"""
Component for displaying mathematical explanations of image processing algorithms.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import streamlit as st
import numpy as np

from app.components import Component


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

    def _format_kernel_matrix(self, kernel: np.ndarray, x: int, y: int) -> str:
        """Format kernel matrix for LaTeX display."""
        matrix_str = r"\begin{bmatrix}"
        half_kernel = self.config.kernel_size // 2
        
        for i in range(kernel.shape[0]):
            row = []
            for j in range(kernel.shape[1]):
                value = kernel[i, j]
                # Bold the center value
                if i == half_kernel and j == half_kernel:
                    cell = r"\mathbf{" + f"{value:.3f}" + "}"
                else:
                    cell = f"{value:.3f}"
                row.append(cell)
            matrix_str += " & ".join(row)
            if i < kernel.shape[0] - 1:
                matrix_str += r"\\"
        matrix_str += r"\end{bmatrix}"
        return matrix_str

    def render(self) -> None:
        """Render mathematical explanations."""
        if not all([self.config.selected_pixel, self.config.image_array is not None]):
            st.info("Select a pixel to view mathematical explanation.")
            return

        x, y = self.config.selected_pixel
        kernel = self._extract_kernel(x, y)
        
        if kernel is None:
            st.warning("Cannot extract kernel for selected pixel (too close to border).")
            return

        # Calculate values for formula substitution with proper float formatting
        mean = float(np.mean(kernel))
        std = float(np.std(kernel))
        sc = std / mean if mean > 1e-10 else 0.0
        original_value = float(self.config.image_array[y, x])

        # Create substitution dictionary with formatted float values
        subs = {
            "x": int(x),  # Convert to int for coordinates
            "y": int(y),  # Convert to int for coordinates
            "kernel_size": int(self.config.kernel_size),  # Convert to int for dimensions
            "half_kernel": int(self.config.kernel_size // 2),  # Convert to int for dimensions
            "mean": f"{mean:.4f}",  # Format floats with fixed precision
            "std": f"{std:.4f}",
            "sc": f"{sc:.4f}",
            "original_value": f"{original_value:.4f}",
            "kernel_matrix_latex": self._format_kernel_matrix(kernel, x, y),
            "image_height": int(self.config.image_array.shape[0]),  # Convert to int for dimensions
            "image_width": int(self.config.image_array.shape[1]),
            "valid_height": int(self.config.image_array.shape[0] - self.config.kernel_size + 1),
            "valid_width": int(self.config.image_array.shape[1] - self.config.kernel_size + 1),
            "total_pixels": int(self.config.kernel_size ** 2)
        }

        # Render formulas with explanations
        st.markdown("### Mathematical Explanation")
        
        # Main formula
        st.latex(self.config.formula_config["main_formula"].format(**subs))
        st.markdown(self.config.formula_config["explanation"].format(**subs))

        # Additional formulas in tabs
        tabs = st.tabs([f["title"] for f in self.config.formula_config["additional_formulas"]])
        for tab, formula_info in zip(tabs, self.config.formula_config["additional_formulas"]):
            with tab:
                st.latex(formula_info["formula"].format(**subs))
                st.markdown(formula_info["explanation"].format(**subs))

    def _extract_kernel(self, x: int, y: int) -> Optional[np.ndarray]:
        """Extract kernel around selected pixel."""
        half = self.config.kernel_size // 2
        if (half <= x < self.config.image_array.shape[1] - half and
            half <= y < self.config.image_array.shape[0] - half):
            return self.config.image_array[
                y - half:y + half + 1,
                x - half:x + half + 1
            ]
        return None
