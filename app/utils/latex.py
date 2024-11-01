"""LaTeX formula configurations for mathematical explanations."""
from typing import Dict, Any


SPECKLE_FORMULA_CONFIG: Dict[str, Any] = {
    "title": "Speckle Contrast Calculation",
    "main_formula": r"I_{{{x},{y}}} = {original_value:.2f} \quad \rightarrow \quad SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:.2f}}}{{{mean:.2f}}} = {sc:.3f}",
    "explanation": r"This formula shows the transition from the original pixel intensity $I_{{{x},{y}}}$ to the Speckle Contrast (SC) for the same pixel position.",
    "additional_formulas": [

        {
            "title": "Border Handling",
            "formula": r"P_{{x,y}}(i,j) = \begin{{cases}} I_{{x+i,y+j}} & \text{{if }} (x+i,y+j) \in \text{{valid region}} \\ 0 & \text{{otherwise}} \end{{cases}} \quad \text{{for }} i,j \in [-{half_kernel}, {half_kernel}]",
            "explanation": r"""
To avoid boundary issues, the algorithm only processes pixels within the valid region, which excludes pixels near the image border.

For a pixel at position $(x,y)$:
- The patch $P_{{x,y}}$ is defined using the kernel size {kernel_size} × {kernel_size}
- The valid processing region is determined by the kernel size:
  - Valid region: [{half_kernel}, {image_height} - {half_kernel}) × [{half_kernel}, {image_width} - {half_kernel})
  - Valid region size: {valid_height} × {valid_width}

The patch is constructed by considering pixels within the range $[-{half_kernel}, {half_kernel}]$ relative to the current pixel position.
"""
        },

        {
            "title": "Neighborhood Analysis",
            "formula": (
                r"\text{{Kernel Size: }} {kernel_size} \times {kernel_size}"
                r"\quad\quad\text{{Centered at pixel: }}({x}, {y})"
                r"\\"
                r"{kernel_matrix_latex}"
            ),
            "explanation": r"Analysis of a ${kernel_size}\times{kernel_size}$ neighborhood centered at pixel $({x},{y})$. The matrix shows pixel values, with the central value (in bold) being the processed pixel.",
        },

        {
            "title": "Mean Filter",
            "formula": r"\mu_{{{x},{y}}} = \frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = \frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} I_{{i,j}} = {mean:d}",
            "explanation": r"The mean intensity $\mu_{{{x},{y}}}$ at pixel $({x},{y})$ is calculated by summing the intensities $I_{{i,j}}$ of all pixels within the kernel $K$ centered at $({x},{y})$, and dividing by the total number of pixels $N = {kernel_size}^2 = {total_pixels}$ in the kernel.",
        },

        {
            "title": "Standard Deviation Calculation",
            "formula": r"\sigma_{{{x},{y}}} = \sqrt{{\frac{{1}}{{N}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - \mu_{{{x},{y}}})^2}} = \sqrt{{\frac{{1}}{{{kernel_size}^2}} \sum_{{i,j \in K_{{{x},{y}}}}} (I_{{i,j}} - {mean:d})^2}} = {std:d}",
            "explanation": r"The standard deviation $\sigma_{{{x},{y}}}$ at pixel $({x},{y})$ measures the spread of intensities around the mean $\mu_{{{x},{y}}}$. It is calculated by taking the square root of the average squared difference between each pixel intensity $I_{{i,j}}$ in the kernel $K$ and the mean intensity $\mu_{{{x},{y}}}$.",
        },

        {
            "title": "Speckle Contrast Calculation",
            "formula": r"SC_{{{x},{y}}} = \frac{{\sigma_{{{x},{y}}}}}{{\mu_{{{x},{y}}}}} = \frac{{{std:d}}}{{{mean:d}}} = {sc:.3f}",
            "explanation": r"Speckle Contrast (SC): ratio of standard deviation $\sigma_{{{x},{y}}}$ to mean intensity $\mu_{{{x},{y}}}$ within the kernel centered at $({x},{y})$.",
        },
    ],
}
