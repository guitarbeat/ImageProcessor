"""NLM-specific analysis and visualization."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.analysis.nlm_state import NLMState
from app.utils.context_managers import figure_context
from app.utils.visualization import (
    SearchWindowOverlayConfig,
    add_search_window_overlay,
    create_kernel_overlay_config,
    create_visualization_config,
    highlight_pixel,
    plot_similarity_map,
)

if TYPE_CHECKING:
    from scipy.optimize import curve_fit  # type: ignore


def exp_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential decay function for curve fitting.

    Args:
        x: Input values
        a: Amplitude
        b: Decay rate

    Returns:
        Exponentially decaying values
    """
    return a * np.exp(-b * x)


@dataclass
class NLMAnalysisConfig:
    """Configuration for NLM analysis."""

    filter_strength: float
    kernel_size: int
    search_window_size: Optional[int] = None
    decimals: int = 3


class NLMAnalysis:
    """Handles NLM-specific analysis and visualization."""

    def __init__(self, config: NLMAnalysisConfig):
        self.config = config
        self.vis_config = create_visualization_config()
        self.kernel_config = create_kernel_overlay_config()

    def analyze_weights(self, similarity_map: np.ndarray) -> Dict[str, float]:
        """Analyze weight distribution."""
        non_zero_weights = similarity_map[similarity_map > 0]
        percentiles = (
            np.percentile(non_zero_weights, [25, 50, 75])
            if len(non_zero_weights) > 0
            else [0, 0, 0]
        )

        return {
            "mean": float(np.mean(similarity_map)),
            "median": float(percentiles[1]),
            "max": float(np.max(similarity_map)),
            "min": float(np.min(similarity_map)),
            "std": float(np.std(similarity_map)),
            "active_ratio": float(np.sum(similarity_map > 0) / similarity_map.size),
            "q1": float(percentiles[0]),
            "q3": float(percentiles[2]),
            "iqr": float(percentiles[2] - percentiles[0]),
        }

    def analyze_spatial_patterns(self, similarity_map: np.ndarray) -> Dict[str, float]:
        """Analyze spatial patterns in weights."""
        row_means = np.mean(similarity_map, axis=1)
        col_means = np.mean(similarity_map, axis=0)
        center_y, center_x = similarity_map.shape[0] // 2, similarity_map.shape[1] // 2

        return {
            "horizontal_mean": float(np.mean(row_means)),
            "vertical_mean": float(np.mean(col_means)),
            "directional_ratio": float(np.mean(row_means) / np.mean(col_means)),
            "edge_center_ratio": float(
                np.mean(similarity_map[0, :])
                / (similarity_map[center_y, center_x] + 1e-10)
            ),
            "corner_center_ratio": float(
                similarity_map[0, 0] / (similarity_map[center_y, center_x] + 1e-10)
            ),
            "radial_decay": self._compute_radial_decay(similarity_map),
        }

    def _compute_radial_decay(self, similarity_map: np.ndarray) -> float:
        """Compute how quickly weights decay with distance from center."""
        center_y, center_x = similarity_map.shape[0] // 2, similarity_map.shape[1] // 2
        y_coords, x_coords = np.ogrid[
            : similarity_map.shape[0], : similarity_map.shape[1]
        ]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

        # Compute average weight at each unique distance
        unique_distances = np.unique(distances)
        avg_weights = [
            np.mean(similarity_map[distances == d]) for d in unique_distances
        ]

        # Fit exponential decay
        try:
            popt, _ = curve_fit(exp_decay, unique_distances, avg_weights)
            return float(popt[1])  # Return decay rate
        except ImportError:
            return float(np.nan)  # Specific exception for missing scipy
        except RuntimeError:
            # Specific exception for curve fitting failure
            return float(np.nan)

    def render_analysis(self, nlm_state: NLMState) -> None:
        """Integrated analysis rendering."""
        # Extract required data from nlm_state
        similarity_map = nlm_state.similarity_map
        img_array = nlm_state.image
        x, y = nlm_state.current_position
        search_range = nlm_state.search_range

        # Calculate statistics
        weight_stats = self.analyze_weights(similarity_map)
        spatial_stats = self.analyze_spatial_patterns(similarity_map)

        # Create consistent tabs
        analysis_tabs = st.tabs(
            ["🎯 Search Region", "📊 Weights", "🌐 Spatial", "💡 Interpretation"]
        )

        with analysis_tabs[0]:
            self._render_search_region(
                img_array=img_array,
                similarity_map=similarity_map,
                x=x,
                y=y,
                search_range=search_range,
            )

        with analysis_tabs[1]:
            self._render_weight_distribution(similarity_map, weight_stats)

        with analysis_tabs[2]:
            self._render_spatial_analysis(similarity_map, spatial_stats)

        with analysis_tabs[3]:
            self._render_interpretation(weight_stats, spatial_stats)

    def _render_coordinate_system(
        self, x: int, y: int, i: int, j: int, search_range: List[Tuple[int, int]]
    ) -> None:
        """Render coordinate system explanation."""
        st.markdown(
            """
        ### Coordinate Systems in NLM
        The NLM algorithm uses multiple coordinate systems:
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
            **Input Space (x,y):**
            - Current position: ({x}, {y})
            - Original intensity: $I_{{{x},{y}}}$
            - Kernel size: {self.config.kernel_size}×{self.config.kernel_size}
            """
            )

        with col2:
            st.markdown(
                f"""
            **Output Space (i,j):**
            - Processed position: ({i}, {j})
            - NLM result: $NLM_{{{i},{j}}}$
            - Offset: {self.config.kernel_size//2} pixels
            """
            )

        st.markdown(
            """
        **Search Space (s,t):**
        - Used for finding similar patches
        - Coordinates relative to search window
        
        **Patch Space (p,q):**
        - Local coordinates within patches
        - Used for patch comparison
        """
        )

        # Add search window information
        y_range, x_range = search_range
        if self.config.search_window_size is None:
            st.markdown(
                f"""
            **Full Image Search:**
            - X range: [{x_range[0]} - {x_range[1]}]
            - Y range: [{y_range[0]} - {y_range[1]}]
            - All valid patches are considered
            """
            )
        else:
            st.markdown(
                f"""
            **Limited Search Window:**
            - Size: {self.config.search_window_size}×{self.config.search_window_size}
            - X range: [{x_range[0]} - {x_range[1]}]
            - Y range: [{y_range[0]} - {y_range[1]}]
            """
            )

    def _render_weight_distribution(
        self, similarity_map: np.ndarray, stats: Dict[str, float]
    ) -> None:
        """Render weight distribution analysis."""
        col1, col2 = st.columns([2, 1])

        with col1, figure_context() as fig:
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

            # Plot histogram in top subplot
            ax1 = fig.add_subplot(gs[0])
            self._plot_weight_histogram(ax1, similarity_map)

            # Plot cumulative distribution in bottom subplot
            ax2 = fig.add_subplot(gs[1])
            self._plot_cumulative_distribution(ax2, similarity_map)

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown(
                f"""
            **Weight Statistics:**
            - Mean: {stats['mean']:.{self.config.decimals}f}
            - Median: {stats['median']:.{self.config.decimals}f}
            - Max: {stats['max']:.{self.config.decimals}f}
            - Active Pixels: {stats['active_ratio']*100:.1f}%
            """
            )

    def _plot_weight_histogram(self, ax: plt.Axes, similarity_map: np.ndarray) -> None:
        """Plot histogram of weight distribution."""
        non_zero_weights = similarity_map[similarity_map > 0]

        # Plot histogram
        n, bins, patches = ax.hist(
            non_zero_weights,
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            label=f"n={len(non_zero_weights)}",
        )

        # Add mean and median lines
        mean_val = np.mean(non_zero_weights)
        median_val = np.median(non_zero_weights)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            label=f"Mean={mean_val:.{self.config.decimals}f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle=":",
            label=f"Median={median_val:.{self.config.decimals}f}",
        )

        # Add percentile lines
        for p in [25, 75]:
            p_val = np.percentile(non_zero_weights, p)
            ax.axvline(
                p_val,
                color=f"C{p//25}",
                linestyle=":",
                label=f"{p}th={p_val:.{self.config.decimals}f}",
            )

        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density")
        ax.set_title("Weight Distribution")
        ax.legend()

    def _plot_cumulative_distribution(
        self, ax: plt.Axes, similarity_map: np.ndarray
    ) -> None:
        """Plot cumulative distribution of weights."""
        non_zero_weights = similarity_map[similarity_map > 0]
        sorted_weights = np.sort(non_zero_weights)
        cumulative = np.linspace(0, 1, len(sorted_weights))

        ax.plot(sorted_weights, cumulative, "b-", label="CDF")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Cumulative Fraction")
        ax.grid(True, alpha=0.3)

    def _render_search_region(
        self,
        img_array: np.ndarray,
        similarity_map: np.ndarray,
        x: int,
        y: int,
        search_range: List[Tuple[int, int]],
    ) -> None:
        """Render search region visualization."""
        with figure_context() as fig:
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

            # Image with search region overlay
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(img_array, cmap=self.vis_config.colormap)

            # Add search window overlay
            use_full_image = (
                search_range[1][1] - search_range[1][0] == img_array.shape[1]
            )
            if use_full_image:
                search_window_size = None
            else:
                search_window_size = search_range[1][1] - search_range[1][0]

            # Cast image shape to correct type
            height, width = img_array.shape[:2]  # Get first two dimensions
            image_shape = (height, width)  # Create tuple[int, int]

            add_search_window_overlay(
                ax=ax1,
                center=(x, y),
                search_window_size=search_window_size,
                image_shape=image_shape,  # Now correctly typed
                config=SearchWindowOverlayConfig(),
            )

            highlight_pixel(
                ax=ax1,
                position=(x, y),
                color=self.kernel_config.center_color,
                alpha=self.kernel_config.center_alpha,
            )
            ax1.set_title("Search Region")

            # Similarity map
            ax2 = fig.add_subplot(gs[1])
            plot_similarity_map(
                ax=ax2,
                similarity_map=similarity_map,
                center=(x, y),
                kernel_config=self.kernel_config,
                search_config=None,
                vis_config=self.vis_config,
                title="Similarity Weights",
                is_full_image=use_full_image,
            )

            plt.tight_layout()
            st.pyplot(fig)

    def _render_spatial_analysis(
        self, similarity_map: np.ndarray, stats: Dict[str, float]
    ) -> None:
        """Add more insightful visualizations."""
        # Add radial weight distribution
        self._plot_radial_weight_distribution(similarity_map)

        # Add directional analysis
        self._plot_directional_analysis(similarity_map)

    def _plot_radial_weight_distribution(self, similarity_map: np.ndarray) -> None:
        """Plot radial weight distribution."""
        center_y, center_x = similarity_map.shape[0] // 2, similarity_map.shape[1] // 2
        y_coords, x_coords = np.ogrid[
            : similarity_map.shape[0], : similarity_map.shape[1]
        ]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

        # Compute average weight at each unique distance
        unique_distances = np.unique(distances)
        avg_weights = [
            np.mean(similarity_map[distances == d]) for d in unique_distances
        ]

        # Plot radial weight distribution
        plt.figure(figsize=(8, 6))
        plt.plot(
            unique_distances, avg_weights, "o-", label="Radial Weight Distribution"
        )
        plt.xlabel("Distance")
        plt.ylabel("Average Weight")
        plt.title("Radial Weight Distribution")
        plt.legend()
        plt.show()

    def _plot_directional_analysis(self, similarity_map: np.ndarray) -> None:
        """Plot directional analysis."""
        center_y, center_x = similarity_map.shape[0] // 2, similarity_map.shape[1] // 2
        y_coords, x_coords = np.ogrid[
            : similarity_map.shape[0], : similarity_map.shape[1]
        ]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

        # Compute average weight at each unique distance
        unique_distances = np.unique(distances)
        avg_weights = [
            np.mean(similarity_map[distances == d]) for d in unique_distances
        ]

        # Plot directional analysis
        plt.figure(figsize=(8, 6))
        plt.plot(unique_distances, avg_weights, "o-", label="Directional Analysis")
        plt.xlabel("Distance")
        plt.ylabel("Average Weight")
        plt.title("Directional Analysis")
        plt.legend()
        plt.show()

    def _render_interpretation(
        self, weight_stats: Dict[str, float], spatial_stats: Dict[str, float]
    ) -> None:
        """Render interpretation of analysis results."""
        st.markdown("### Analysis Interpretation")

        st.markdown(
            """
        #### Weight Distribution
        - Higher weights indicate more similar patches
        - Wide distribution suggests diverse matches
        - Narrow distribution suggests uniform region
        """
        )

        st.markdown(
            """
        #### Spatial Patterns
        - Directional ratio ≈ 1: Isotropic similarity
        - Ratio > 1: Horizontal structures dominant
        - Ratio < 1: Vertical structures dominant
        """
        )

        st.markdown(
            """
        #### Boundary Effects
        - High edge ratios may indicate boundary artifacts
        - Corner/center ratio shows corner influence
        - Values near 1 suggest good boundary handling
        """
        )
