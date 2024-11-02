"""LaTeX formula configurations for mathematical explanations."""
from typing import Dict, Any


SPECKLE_FORMULA_CONFIG: Dict[str, Any] = {
    "title": "Speckle Contrast Calculation",
    "main_formula": r"I_{{{x},{y}}} = {original_value} \quad \rightarrow \quad SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std}}}{{{mean}}} = {sc}",
    "explanation": r"""
The transformation from raw intensity to Speckle Contrast involves several steps:
1. Starting with the original pixel intensity $I_{{{x},{y}}} = {original_value}$
2. Computing local statistics in a {kernel_size}×{kernel_size} neighborhood
3. Calculating the ratio of standard deviation to mean intensity
""",
    "additional_formulas": [
        {
            "title": "Border Handling",
            "formula": r"P_{{{x},{y}}}(i,j) = \begin{{cases}} I_{{{x}+i,{y}+j}} & \text{{if }} |i| \leq {half_kernel} \text{{ and }} |j| \leq {half_kernel} \\ \text{{undefined}} & \text{{otherwise}} \end{{cases}}",
            "explanation": r"""
**Border Management Strategy:**
1. A pixel at $({x},{y})$ can only be processed if it has sufficient neighbors in all directions
2. Valid processing region requirements:
   - Horizontal margin: ${half_kernel}$ pixels from left and right edges
   - Vertical margin: ${half_kernel}$ pixels from top and bottom edges
   - Results in processable area of {valid_width}×{valid_height} pixels

**Why Border Handling Matters:**
- Ensures consistent kernel size for all processed pixels
- Prevents edge artifacts in the final result
- Maintains statistical validity of the calculations
"""
        },
        {
            "title": "Neighborhood Analysis",
            "formula": r"\text{{Kernel Matrix }} K_{{{x},{y}}} = {kernel_matrix_latex}",
            "explanation": r"""
**Kernel Properties:**
1. Dimensions: ${kernel_size}\times{kernel_size}$ square matrix
2. Center element (in **bold**): Original pixel value at position $({x},{y})$
3. Surrounding elements: Neighboring pixel intensities
4. Total pixels considered: {total_pixels} values

This matrix represents the local neighborhood used for statistical calculations.
"""
        },
        {
            "title": "Mean Filter",
            "formula": r"\mu_{{{x},{y}}} = \frac{{1}}{{{total_pixels}}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = {mean}",
            "explanation": r"""
**Mean Calculation Details:**
1. Sum all pixel values in the {kernel_size}×{kernel_size} kernel
2. Divide by total number of pixels ({total_pixels})
3. Results in local average intensity of {mean}

The mean filter provides a measure of local brightness and helps normalize the contrast calculation.
"""
        },
        {
            "title": "Standard Deviation",
            "formula": r"\sigma_{{{x},{y}}} = \sqrt{{\frac{{1}}{{{total_pixels}}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - {mean})^2}} = {std}",
            "explanation": r"""
**Standard Deviation Process:**
1. Subtract mean ({mean}) from each pixel value
2. Square the differences
3. Average the squared differences
4. Take the square root
5. Results in local intensity variation of {std}
"""
        },
        {
            "title": "Speckle Contrast",
            "formula": r"SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std}}}{{{mean}}} = {sc}",
            "explanation": r"""
**Final Calculation Properties:**
1. Ratio of standard deviation ({std}) to mean ({mean})
2. Resulting contrast value: {sc}
3. Interpretation:
   - Low values: More uniform region
   - High values: More variable region
   - Zero: Either uniform area or undefined (μ = 0)
"""
        }
    ]
}
