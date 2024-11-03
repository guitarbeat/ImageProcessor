"""Common visualization utilities."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def setup_figure(
    figsize: Optional[Tuple[int, int]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Create and setup figure with consistent styling."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    return fig, ax


def add_value_annotations(
    ax: plt.Axes,
    data: np.ndarray,
    decimals: int = 3,
    color: str = "black",
    bg_color: str = "white",
    bg_alpha: float = 0.7,
) -> None:
    """Add value annotations with consistent styling."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                color=color,
                bbox={"facecolor": bg_color, "alpha": bg_alpha, "edgecolor": "none"},
            )
