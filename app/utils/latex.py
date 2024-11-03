"""LaTeX formula configurations for mathematical explanations."""
from typing import Dict, Any, Optional, Tuple
import numpy as np

SPECKLE_FORMULA_CONFIG: Dict[str, Any] = {
    "title": "Speckle Contrast Calculation",
    "main_formula": (
        r"I_{{{x},{y}}} = {original_value:.{decimals}f} \quad \rightarrow \quad "
        r"SC_{{{i},{j}}} = \frac{{\sigma_{{{i},{j}}}}}{{\mu_{{{i},{j}}}}} = "
        r"\frac{{{std:.{decimals}f}}}{{{mean:.{decimals}f}}} = {sc:.{decimals}f}"
    ),
    "explanation": (
        "The transformation from input coordinates $(x,y)$ to output coordinates $(i,j)$:\n\n"
        "1. Input: Original intensity $I_{{{x},{y}}} = {original_value:.{decimals}f}$\n"
        "2. Output: Speckle Contrast $SC_{{{i},{j}}} = {sc:.{decimals}f}$\n"
        "3. Processing window: {kernel_size}×{kernel_size} neighborhood"
    ),
    "additional_formulas": [
        {
            "title": "Border Handling",
            "formula": (
                r"P_{{{x},{y}}}(i,j) = \begin{{cases}} "
                r"I_{{(x+i,y+j)}} & \text{{if }} |i| \leq {half_kernel} \text{{ and }} |j| \leq {half_kernel} \\ "
                r"\text{{undefined}} & \text{{otherwise}} "
                r"\end{{cases}}"
            ),
            "explanation": (
                "**Border Management Strategy:**\n\n"
                "1. A pixel at $(x,y)$ can only be processed if it has sufficient neighbors\n\n"
                "2. Valid processing region: {valid_width}×{valid_height} pixels\n\n"
                "3. Input coordinates: $(x,y)$ → Output: $({processed_x},{processed_y})$"
            )
        },
        {
            "title": "Mean Filter",
            "formula": (
                r"\mu_{{{processed_x},{processed_y}}} = "
                r"\frac{{1}}{{{total_pixels}}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = {mean:.4f}"
            ),
            "explanation": (
                "**Mean Calculation:**\n"
                "1. Sum all {total_pixels} pixels in the kernel\n"
                "2. Divide by total pixels for local average = {mean:.4f}"
            )
        },
        {
            "title": "Standard Deviation",
            "formula": (
                r"\sigma_{{{processed_x},{processed_y}}} = "
                r"\sqrt{{\frac{{1}}{{{total_pixels}}} \sum_{{i,j \in K_{{{x},{y}}}}} "
                r"(I_{{i,j}} - {mean:.4f})^2}} = {std:.4f}"
            ),
            "explanation": (
                "**Standard Deviation:**\n"
                "1. Subtract mean ({mean:.4f}) from each pixel\n"
                "2. Square differences and average\n"
                "3. Take square root for final value = {std:.4f}"
            )
        }
    ]
}

NLM_FORMULA_CONFIG: Dict[str, Any] = {
    "title": "Non-Local Means (NLM) Denoising",
    "main_formula": (
        r"I_{{{x},{y}}} = {original_value:.{decimals}f} \quad \rightarrow \quad "
        r"NLM_{{{i},{j}}} = "
        r"\frac{{1}}{{{norm_factor:.{decimals}f}}} "
        r"\sum\limits_{{(s,t) \in \Omega_{{{x},{y}}}}} "
        r"w_{{{x},{y}}}(s,t) \cdot I_{{s,t}} = {nlm_value:.{decimals}f}"
    ),
    "explanation": (
        "The Non-Local Means algorithm transforms input coordinates $(x,y)$ to output coordinates $(i,j)$:\n\n"
        "1. Input pixel: $I_{{{x},{y}}} = {original_value:.{decimals}f}$ at position $(x,y)$\n\n"
        "2. Output pixel: $NLM_{{{i},{j}}} = {nlm_value:.{decimals}f}$ at position $(i,j)$\n\n"
        "3. Search window $\Omega_{{{x},{y}}}$ centered at input position\n\n"
        "4. Weights $w_{{{x},{y}}}(s,t)$ measure similarity between patches"
    ),
    "additional_formulas": [
        {
            "title": "Coordinate Mapping",
            "formula": (
                r"\begin{{cases}} "
                r"i = x - {half_kernel} & \text{{output row}} \\"
                r"j = y - {half_kernel} & \text{{output column}} \\"
                r"(s,t) \in \Omega_{{{x},{y}}} & \text{{search positions}}"
                r"\end{{cases}}"
            ),
            "explanation": (
                "**Coordinate Systems:**\n\n"
                "1. Input coordinates $(x,y)$: Position in original image\n"
                "2. Output coordinates $(i,j)$: Position in processed image\n"
                "3. Search coordinates $(s,t)$: Positions within search window\n\n"
                "The kernel size {kernel_size}×{kernel_size} determines the offset between input and output coordinates."
            )
        },
        {
            "title": "Patch Similarity",
            "formula": (
                r"\begin{{aligned}} "
                r"d_{{{x},{y}}}(s,t) &= \sum_{{p,q}} (P_{{{x},{y}}}(p,q) - P_{{s,t}}(p,q))^2 \\"
                r"w_{{{x},{y}}}(s,t) &= \exp\left(-\frac{{d_{{{x},{y}}}(s,t)}}{{{filter_strength:.{decimals}f}^2}}\right)"
                r"\end{{aligned}}"
            ),
            "explanation": (
                "**Similarity Computation:**\n\n"
                "1. $P_{{{x},{y}}}$: Patch centered at input position $(x,y)$\n"
                "2. $P_{{s,t}}$: Patch centered at search position $(s,t)$\n"
                "3. $(p,q)$: Local coordinates within patches\n"
                "4. Filter strength $h = {filter_strength:.2f}$ controls similarity sensitivity"
            )
        },
        {
            "title": "Normalization",
            "formula": (
                r"\begin{{aligned}}"
                r"C_{{{x},{y}}} &= \sum\limits_{{(s,t) \in \Omega_{{{x},{y}}}}} w_{{{x},{y}}}(s,t) = {norm_factor:.{decimals}f} \\"
                r"w_{{{x},{y}}}^{{\text{{norm}}}}(s,t) &= \frac{{w_{{{x},{y}}}(s,t)}}{{{norm_factor:.{decimals}f}}}"
                r"\end{{aligned}}"
            ),
            "explanation": (
                "**Normalization Process:**\n"
                "1. Sum all weights $w_{{{x},{y}}}(s,t)$ to get $C_{{{x},{y}}} = {norm_factor:.{decimals}f}$\n"
                "2. Divide each weight by $C_{{{x},{y}}}$ to normalize\n"
                "3. Ensures $\sum w_{{{x},{y}}}^{{\text{{norm}}}}(s,t) = 1$"
            )
        },
        {
            "title": "Search Region",
            "formula": (
                r"\Omega_{{{x},{y}}} = \begin{{cases}} "
                r"[{search_y_min}, {search_y_max}] \times [{search_x_min}, {search_x_max}] & \text{{local search}} \\ "
                r"[0, {image_height}] \times [0, {image_width}] & \text{{full image}} "
                r"\end{{cases}}"
            ),
            "explanation": (
                "**Search Window Definition:**\n"
                "1. Local search: {kernel_size}×{kernel_size} patches centered at each $(i,j)$\n"
                "2. Search area: {search_window_description}\n"
                "3. Valid pixel positions: $({processed_x},{processed_y})$ in output\n"
                "4. Larger search area → more potential matches but slower computation"
            )
        }
    ]
}

def create_kernel_matrix_latex(kernel: np.ndarray, center_value: float, decimals: int = 3) -> str:
    """Create LaTeX matrix representation of kernel."""
    matrix = []
    for i in range(kernel.shape[0]):
        row = []
        for j in range(kernel.shape[1]):
            value = kernel[i, j]
            if i == kernel.shape[0]//2 and j == kernel.shape[1]//2:
                row.append(r"\mathbf{" + f"{value:.{decimals}f}" + "}")
            else:
                row.append(f"{value:.{decimals}f}")
        matrix.append(" & ".join(row))
    return r"\begin{bmatrix}" + r"\\" + r"\\".join(matrix) + r"\end{bmatrix}"

def get_search_window_bounds(x: int, y: int, search_size: Optional[int], image_width: int, image_height: int) -> Dict[str, int]:
    """Calculate search window bounds for LaTeX formulas."""
    if search_size is None:
        return {
            "search_x_min": 0,
            "search_x_max": image_width,
            "search_y_min": 0,
            "search_y_max": image_height
        }
    else:
        half_search = search_size // 2
        return {
            "search_x_min": max(0, x - half_search),
            "search_x_max": min(image_width, x + half_search),
            "search_y_min": max(0, y - half_search),
            "search_y_max": min(image_height, y + half_search)
        }

def _create_substitution_dict(self, input_coords: Tuple[int, int], 
                            output_coords: Tuple[int, int],
                            kernel: np.ndarray, values: Dict[str, Any],
                            computation: Any) -> Dict[str, Any]:
    """Create substitution dictionary with consistent coordinate notation."""
    x, y = input_coords
    i, j = output_coords  # Use output coordinates directly
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
