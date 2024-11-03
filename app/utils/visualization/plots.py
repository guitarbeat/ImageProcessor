"""Plotting utilities for visualization."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from app.utils.visualization.config import VisualizationConfig

def plot_weight_distribution(
    ax: plt.Axes,
    weights: np.ndarray,
    vis_config: VisualizationConfig,
    orientation: str = 'vertical',
    show_percentiles: bool = True
) -> None:
    """Plot weight distribution with consistent styling."""
    non_zero_weights = weights[weights > 0]
    
    # Plot histogram
    ax.hist(
        non_zero_weights,
        bins=50,
        orientation=orientation,
        color=st.session_state.get('center_color', '#FF0000'),
        alpha=0.7,
        label=f'n={len(non_zero_weights)}'
    )
    
    # Add statistics lines
    mean_val = np.mean(non_zero_weights)
    median_val = np.median(non_zero_weights)
    
    if orientation == 'vertical':
        ax.axvline(mean_val, color='r', linestyle='--',
                  label=f'Mean={mean_val:.{vis_config.decimals}f}')
        ax.axvline(median_val, color='g', linestyle=':',
                  label=f'Median={median_val:.{vis_config.decimals}f}')
        
        if show_percentiles:
            for p in [25, 75]:
                p_val = np.percentile(non_zero_weights, p)
                ax.axvline(p_val, color=f'C{p//25}', linestyle=':',
                         label=f'{p}th={p_val:.{vis_config.decimals}f}')
    else:
        ax.axhline(mean_val, color='r', linestyle='--',
                  label=f'Mean={mean_val:.{vis_config.decimals}f}')
        ax.axhline(median_val, color='g', linestyle=':',
                  label=f'Median={median_val:.{vis_config.decimals}f}')
    
    ax.legend()
