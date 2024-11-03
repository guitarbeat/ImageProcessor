"""
Component for controlling image processing behavior.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from scipy import stats as scipy_stats  # type: ignore

from app.analysis.nlm_analysis import NLMAnalysis, NLMAnalysisConfig, NLMState
from app.processors.filters import NLMComputation
from app.ui.components.common import create_pixel_selector
from app.ui.components.component_base import BaseUIComponent
from app.ui.settings.display import DisplaySettings
from app.utils.context_managers import figure_context
from app.utils.visualization import (
    SearchWindowOverlayConfig,
    add_kernel_overlay,
    add_search_window_overlay,
    add_value_annotations,
    create_kernel_overlay_config,
    create_visualization_config,
    highlight_pixel,
    plot_similarity_map,
    plot_weight_distribution,
)


@dataclass
class ProcessingControlConfig:
    """Configuration for processing control."""

    on_settings_changed: Callable[[Dict[str, Any]], None]
    initial_settings: Dict[str, Any] = field(default_factory=dict)
    display_settings: Optional[DisplaySettings] = None

    def __post_init__(self) -> None:
        """Initialize default settings if not provided."""
        if self.initial_settings is None:
            self.initial_settings = {}
        
        # Ensure all required settings have defaults
        defaults = {
            "process_full_image": True,
            "selected_region": None,
            "processing_progress": 0,
            "selected_pixel": None,
        }
        
        for key, value in defaults.items():
            if key not in self.initial_settings:
                self.initial_settings[key] = value

        if self.display_settings is None:
            self.display_settings = DisplaySettings()


class ProcessingControl(BaseUIComponent):
    """Component for controlling image processing behavior."""

    def __init__(self, config: ProcessingControlConfig):
        self.config = config
        self.settings = DisplaySettings.from_session_state()
        self.vis_config = create_visualization_config()
        self.kernel_config = create_kernel_overlay_config()

    def render(self, image: Optional[Image.Image] = None) -> None:
        """Render with better component integration."""
        if image is None:
            st.warning("No image loaded")
            return

        # Initialize session state
        if "selected_pixel" not in st.session_state:
            st.session_state.selected_pixel = None

        # Get state
        nlm_state = NLMState.from_session_state()

        # Create tabs with consistent styling
        tabs = st.tabs(["📊 Overview", "🔍 Analysis", "📐 Math", "⚙️ Settings"])

        with tabs[0]:
            self._render_overview(nlm_state)

        with tabs[1]:
            if nlm_state.show_analysis:
                self._render_analysis(nlm_state)

        with tabs[2]:
            if nlm_state.show_formulas:
                self._render_math_explanation(nlm_state)

        with tabs[3]:
            self._render_settings(nlm_state)

    def _render_overview(self, nlm_state: NLMState) -> None:
        """Render overview tab."""
        # Filter Selection
        filter_options = {
            "LSCI": "🌟 Speckle Contrast",
            "NLM": "🔍 Non-Local Means",
            "Mean": "📊 Mean",
            "Std": "📈 Std Dev",
        }

        selected_filters = st.multiselect(
            "Select Processing Methods",
            options=list(filter_options.keys()),
            default=st.session_state.get("selected_filters", ["LSCI"]),
            format_func=lambda x: filter_options[x],
        )

        if len(selected_filters) > 0:
            primary_filter = st.selectbox(
                "Primary Method",
                options=selected_filters,
                format_func=lambda x: filter_options[x],
                help="Select the primary method for detailed analysis",
            )
        else:
            primary_filter = None
            st.info("Select at least one method")

        # Update session state
        st.session_state.selected_filters = selected_filters
        st.session_state.filter_type = (
            primary_filter.lower() if primary_filter else "lsci"
        )

        # Get current settings
        st.session_state.filter_type
        kernel_size = st.session_state.get("kernel_size", 7)
        half_kernel = kernel_size // 2

        # Calculate valid pixel range
        img_array = st.session_state.get("image_array")
        if img_array is None:
            st.error("No image array found in session state")
            return

        # Process full image toggle
        process_full = st.toggle(
            "Full Image Processing",
            value=self.config.initial_settings["process_full_image"],
            help="Toggle between full image and pixel-wise analysis",
        )

        if not process_full:
            # Pixel selection
            x_min, x_max = half_kernel, img_array.shape[1] - half_kernel - 1
            y_min, y_max = half_kernel, img_array.shape[0] - half_kernel - 1

            pixel_x, pixel_y = create_pixel_selector(
                x_range=(x_min, x_max), y_range=(y_min, y_max)
            )

            st.session_state.selected_pixel = (pixel_x, pixel_y)

            # Render coordinate system
            self._render_coordinate_system(pixel_x, pixel_y, kernel_size, img_array)

    def _render_coordinate_system(
        self, x: int, y: int, kernel_size: int, img_array: np.ndarray
    ) -> None:
        """Render coordinate system information."""
        half_kernel = kernel_size // 2
        i, j = x - half_kernel, y - half_kernel

        st.latex(
            rf"""
        \begin{{aligned}}
        & \textbf{{Input Space}} & & \textbf{{Output Space}} \\
        & (x,y) = ({x}, {y}) & & (i,j) = ({i}, {j}) \\
        & I_{{{x},{y}}} = {img_array[y, x]:.3f} & & \text{{Kernel: }} {kernel_size} \times {kernel_size}
        \end{{aligned}}
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Valid Input Range")
            st.latex(
                rf"""
            \begin{{aligned}}
            x &\in [{half_kernel}, {img_array.shape[1] - half_kernel}] \\
            y &\in [{half_kernel}, {img_array.shape[0] - half_kernel}]
            \end{{aligned}}
            """
            )

        with col2:
            st.markdown("##### Valid Output Range")
            st.latex(
                rf"""
            \begin{{aligned}}
            i &\in [0, {img_array.shape[1] - kernel_size + 1}] \\
            j &\in [0, {img_array.shape[0] - kernel_size + 1}]
            \end{{aligned}}
            """
            )

    def _render_math_explanation(self, nlm_state: NLMState) -> None:
        """Render enhanced mathematical explanation."""
        # Get current image array
        img_array = st.session_state.get("image_array")
        if img_array is None:
            st.error("No image array found in session state")
            return

        # Get coordinates and filter type
        x, y = nlm_state.input_coords
        i, j = nlm_state.output_coords
        filter_type = st.session_state.get("filter_type", "lsci")
        kernel_size = nlm_state.kernel_size

        # Create tabs for different mathematical aspects
        tabs = st.tabs(
            [
                "🎯 Basic Concept",
                "📐 Formulation",
                "🔄 Transformation",
                "💡 Interpretation",
            ]
        )

        with tabs[0]:
            self._render_basic_concept(x, y, i, j, kernel_size, img_array, filter_type)

        with tabs[1]:
            self._render_mathematical_formulation(
                x, y, i, j, kernel_size, img_array, filter_type
            )

        with tabs[2]:
            self._render_coordinate_transformation(x, y, i, j, kernel_size, filter_type)

        with tabs[3]:
            self._render_mathematical_interpretation(filter_type)

    def _render_basic_concept(
        self,
        x: int,
        y: int,
        i: int,
        j: int,
        kernel_size: int,
        img_array: np.ndarray,
        filter_type: str,
    ) -> None:
        """Render basic mathematical concept."""
        if filter_type == "nlm":
            st.latex(
                rf"""
            \text{{Non-Local Means at }} (x,y) = ({x}, {y}):
            """
            )

            st.markdown(
                """
            1. **Patch Comparison**: Compare the neighborhood around $(x,y)$ with other patches
            2. **Weight Calculation**: Compute similarity weights based on patch differences
            3. **Weighted Average**: Combine pixel values using computed weights
            """
            )

            st.latex(
                rf"""
            NLM_{{{i},{j}}} = \frac{{\sum\limits_{{(s,t)}} w_{{{x},{y}}}(s,t) \cdot I_{{s,t}}}}{{\sum\limits_{{(s,t)}} w_{{{x},{y}}}(s,t)}}
            """
            )

        else:  # LSCI
            val = img_array[y, x]  # Get the actual pixel value
            st.latex(
                rf"""
            \text{{Speckle Contrast at }} (x,y) = ({x}, {y}), I_{{{x},{y}}} = {val:.3f}:
            """
            )

            st.markdown(
                """
            1. **Local Statistics**: Compute mean and standard deviation in neighborhood
            2. **Contrast Ratio**: Calculate ratio of std.dev to mean
            3. **Normalization**: Account for intensity variations
            """
            )

            st.latex(
                rf"""
            SC_{{{i},{j}}} = \frac{{\sigma_{{{i},{j}}}}}{{\mu_{{{i},{j}}}}}
            """
            )

    def _render_mathematical_formulation(
        self,
        x: int,
        y: int,
        i: int,
        j: int,
        kernel_size: int,
        img_array: np.ndarray,
        filter_type: str,
    ) -> None:
        """Render detailed mathematical formulation."""
        if filter_type == "nlm":
            # Weight calculation
            st.latex(
                r"""
            w_{{{x},{y}}}(s,t) = \exp\left(-\frac{{\sum\limits_{{p,q}} (P_{{{x},{y}}}(p,q) - P_{{s,t}}(p,q))^2}}{{h^2}}\right)
            """
            )

            # Normalization factor
            st.latex(
                r"""
            C_{{{x},{y}}} = \sum\limits_{{(s,t)}} w_{{{x},{y}}}(s,t)
            """
            )

            # Final computation
            st.latex(
                r"""
            NLM_{{{i},{j}}} = \frac{{1}}{{C_{{{x},{y}}}}} \sum\limits_{{(s,t)}} w_{{{x},{y}}}(s,t) \cdot I_{{s,t}}
            """
            )

        else:  # LSCI
            # Mean calculation
            st.latex(
                r"""
            \mu_{{{i},{j}}} = \frac{{1}}{{N^2}} \sum\limits_{{p=-h}}^h \sum\limits_{{q=-h}}^h I_{{x+p,y+q}}
            """
            )

            # Standard deviation
            st.latex(
                r"""
            \sigma_{{{i},{j}}} = \sqrt{{\frac{{1}}{{N^2}} \sum\limits_{{p=-h}}^h \sum\limits_{{q=-h}}^h 
            (I_{{x+p,y+q}} - \mu_{{{i},{j}}})^2}}
            """
            )

            # Speckle contrast
            st.latex(
                r"""
            SC_{{{i},{j}}} = \frac{{\sigma_{{{i},{j}}}}}{{\mu_{{{i},{j}}}}}
            """
            )

    def _render_coordinate_transformation(
        self, x: int, y: int, i: int, j: int, kernel_size: int, filter_type: str
    ) -> None:
        """Render coordinate transformation explanation."""
        half_kernel = kernel_size // 2

        st.latex(
            rf"""
        \begin{{cases}}
        i = x - {half_kernel} & \text{{output row}} \\
        j = y - {half_kernel} & \text{{output column}} \\
        (i,j) = ({i}, {j}) & \text{{processed coordinates}}
        \end{{cases}}
        """
        )

        if filter_type == "nlm":
            st.markdown(
                """
            **Coordinate Systems:**
            1. Input space $(x,y)$: Original image coordinates
            2. Output space $(i,j)$: Processed image coordinates
            3. Search space $(s,t)$: Relative search coordinates
            4. Patch space $(p,q)$: Local patch coordinates
            """
            )
        else:
            st.markdown(
                """
            **Coordinate Systems:**
            1. Input space $(x,y)$: Original image coordinates
            2. Output space $(i,j)$: Processed image coordinates
            3. Local space $(p,q)$: Kernel neighborhood coordinates
            """
            )

    def _render_mathematical_interpretation(self, filter_type: str) -> None:
        """Render mathematical interpretation."""
        if filter_type == "nlm":
            st.markdown(
                r"""
            ### Mathematical Properties
            
            1. **Weight Properties**:
            - $0 \leq w_{x,y}(s,t) \leq 1$: Normalized weights
            - $w_{x,y}(x,y) = 1$: Maximum self-similarity
            - Weights decay with patch dissimilarity
            
            2. **Filter Behavior**:
            - Preserves strong features
            - Adapts to local structure
            - Non-local averaging reduces noise
            
            3. **Parameter Effects**:
            - Filter strength (h) controls decay rate
            - Larger search windows find more matches
            - Patch size affects feature detection
            """
            )
        else:
            st.markdown(
                r"""
            ### Mathematical Properties
            
            1. **Contrast Properties**:
            - $SC \geq 0$: Non-negative ratio
            - Higher SC indicates more variation
            - Lower SC suggests more uniformity
            
            2. **Statistical Meaning**:
            - Normalized measure of variation
            - Independent of mean intensity
            - Sensitive to local structure
            
            3. **Kernel Effects**:
            - Larger kernels smooth variations
            - Captures local flow patterns
            - Boundary effects at edges
            """
            )

    def _render_analysis(self, nlm_state: NLMState) -> None:
        """Render analysis based on filter type."""
        # Get current filter type and image array
        filter_type = st.session_state.get("filter_type", "lsci")
        img_array = st.session_state.get("image_array")

        if img_array is None:
            st.error("No image array found in session state")
            return

        # Get coordinates and kernel size from state
        x, y = nlm_state.input_coords
        kernel_size = nlm_state.kernel_size

        if filter_type == "nlm":
            self._render_nlm_analysis(
                img_array=img_array,
                x=x,
                y=y,
                kernel_size=kernel_size,
                nlm_state=nlm_state,
            )
        else:
            self._render_lsci_analysis(
                img_array=img_array, x=x, y=y, kernel_size=kernel_size
            )

    def _render_nlm_analysis(
        self,
        img_array: np.ndarray,
        x: int,
        y: int,
        kernel_size: int,
        nlm_state: NLMState,
    ) -> None:
        """Render NLM-specific analysis."""
        nlm_comp = NLMComputation(
            kernel_size=kernel_size,
            filter_strength=nlm_state.filter_strength,
            search_window_size=(
                None if nlm_state.use_full_image else nlm_state.search_size
            ),
        )

        # Get search range
        if nlm_comp.search_window_size is None:
            search_range = [(0, img_array.shape[0]), (0, img_array.shape[1])]
        else:
            half_search = nlm_comp.search_window_size // 2
            search_range = [
                (max(0, y - half_search), min(img_array.shape[0], y + half_search + 1)),
                (max(0, x - half_search), min(img_array.shape[1], x + half_search + 1)),
            ]

        # Update session state with search range
        st.session_state.search_range = search_range

        # Compute and store similarity map
        similarity_map = nlm_comp.compute_similarity_map(img_array, x, y)
        st.session_state.similarity_map = similarity_map

        # Create analyzer and compute similarity map
        config = NLMAnalysisConfig(
            filter_strength=nlm_comp.filter_strength,
            kernel_size=kernel_size,
            search_window_size=nlm_comp.search_window_size,
        )
        analyzer = NLMAnalysis(config)

        # Create analysis tabs
        analysis_tabs = st.tabs(["🎯 Search Region", "📊 Weights", "🌐 Spatial"])

        with analysis_tabs[0]:
            analyzer._render_search_region(
                img_array, similarity_map, x, y, search_range
            )

        with analysis_tabs[1]:
            stats = analyzer.analyze_weights(similarity_map)
            analyzer._render_weight_distribution(similarity_map, stats)

        with analysis_tabs[2]:
            spatial_stats = analyzer.analyze_spatial_patterns(similarity_map)
            analyzer._render_spatial_analysis(similarity_map, spatial_stats)

    def _render_lsci_analysis(
        self, img_array: np.ndarray, x: int, y: int, kernel_size: int
    ) -> None:
        """Render LSCI-specific analysis."""
        # Extract kernel
        half_kernel = kernel_size // 2
        kernel = img_array[
            y - half_kernel : y + half_kernel + 1, x - half_kernel : x + half_kernel + 1
        ]

        # Create analysis tabs
        analysis_tabs = st.tabs(["🎯 Kernel Region", "📊 Statistics", "🌐 Spatial"])

        with analysis_tabs[0]:
            self._render_kernel_view(kernel, x, y, kernel_size)

        with analysis_tabs[1]:
            self._render_kernel_statistics(kernel)

        with analysis_tabs[2]:
            self._render_spatial_statistics(kernel)

    def _extract_patch(
        self, img_array: np.ndarray, x: int, y: int, kernel_size: int
    ) -> np.ndarray:
        """Extract a patch from the image."""
        half = kernel_size // 2
        return img_array[
            max(0, y - half) : min(img_array.shape[0], y + half + 1),
            max(0, x - half) : min(img_array.shape[1], x + half + 1),
        ]

    def _display_patch(self, patch: np.ndarray, title: str) -> None:
        """Display a patch using visualization context."""
        with figure_context() as fig:
            ax = fig.add_subplot(111)
            im = ax.imshow(patch, cmap=self.settings.colormap)
            add_value_annotations(ax, patch, decimals=self.settings.decimals)
            ax.set_title(title)
            if self.settings.show_colorbar:
                plt.colorbar(im)
            plt.tight_layout()
            st.pyplot(fig)

    def _plot_weight_distribution(
        self, similarity_map: np.ndarray, norm_factor: float, title: str
    ) -> None:
        """Enhanced weight distribution visualization."""
        with figure_context() as fig:
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

            # Get configs
            vis_config = create_visualization_config()

            # Plot using visualization utility
            ax1 = fig.add_subplot(gs[0])
            plot_weight_distribution(
                ax=ax1,
                weights=similarity_map,
                vis_config=vis_config,
                orientation="vertical",
                show_percentiles=True,
            )

            # Add cumulative distribution
            ax2 = fig.add_subplot(gs[1])
            plot_weight_distribution(
                ax=ax2,
                weights=similarity_map,
                vis_config=vis_config,
                orientation="horizontal",
                show_percentiles=False,
            )

            plt.tight_layout()
            st.pyplot(fig)

    def _plot_search_region(
        self,
        img_array: np.ndarray,
        similarity_map: np.ndarray,
        x: int,
        y: int,
        search_range: List[Tuple[int, int]],
        kernel_size: int,
    ) -> None:
        """Enhanced search region visualization."""
        with figure_context() as fig:
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

            # Get configs
            vis_config = create_visualization_config()
            kernel_config = create_kernel_overlay_config()
            search_config = SearchWindowOverlayConfig()

            # Image with search region overlay
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(img_array, cmap=vis_config.colormap)

            # Add search window overlay
            use_full_image = (
                search_range[1][1] - search_range[1][0] == img_array.shape[1]
            )
            if use_full_image:
                search_window_size = None
            else:
                search_window_size = search_range[1][1] - search_range[1][0]

            add_search_window_overlay(
                ax=ax1,
                center=(x, y),
                search_window_size=search_window_size,
                image_shape=(int(img_array.shape[1]), int(img_array.shape[0])),
                config=search_config,
            )

            highlight_pixel(
                ax=ax1,
                position=(x, y),
                color=kernel_config.center_color,
                alpha=kernel_config.center_alpha,
            )
            ax1.set_title("Search Region")

            # Similarity map
            ax2 = fig.add_subplot(gs[1])
            plot_similarity_map(
                ax=ax2,
                similarity_map=similarity_map,
                center=(x, y),
                kernel_config=kernel_config,
                search_config=None,
                vis_config=vis_config,
                title="Similarity Weights",
                is_full_image=use_full_image,
            )

            plt.tight_layout()
            st.pyplot(fig)

    def _get_top_similar_patches(
        self,
        img_array: np.ndarray,
        similarity_map: np.ndarray,
        x: int,
        y: int,
        kernel_size: int,
        top_k: int = 3,
    ) -> List[Tuple[np.ndarray, float, Tuple[int, int]]]:
        """Get the top-k most similar patches."""
        half = kernel_size // 2
        height, width = similarity_map.shape

        # Create list of (weight, position) tuples, excluding center pixel
        weights_and_positions = []
        for i in range(height):
            for j in range(width):
                # Convert similarity map coordinates to image coordinates
                img_y = i + y - height // 2
                img_x = j + x - width // 2

                # Skip center pixel and invalid positions
                if (img_y == y and img_x == x) or (
                    img_y < half
                    or img_y >= img_array.shape[0] - half
                    or img_x < half
                    or img_x >= img_array.shape[1] - half
                ):
                    continue

                weight = similarity_map[i, j]
                if weight > 0:  # Only consider non-zero weights
                    weights_and_positions.append((weight, (img_y, img_x)))

        # Sort by weight in descending order and get top-k
        weights_and_positions.sort(reverse=True)
        top_k_patches = []

        for weight, (patch_y, patch_x) in weights_and_positions[:top_k]:
            # Extract patch
            patch = img_array[
                patch_y - half : patch_y + half + 1, patch_x - half : patch_x + half + 1
            ]
            top_k_patches.append((patch, weight, (patch_x, patch_y)))

        return top_k_patches

    def _display_weight_statistics(
        self, similarity_map: np.ndarray, norm_factor: float
    ) -> None:
        """Display academic analysis of weight statistics."""
        non_zero_weights = similarity_map[similarity_map > 0]
        percentiles = np.percentile(non_zero_weights, [25, 50, 75])

        st.markdown(
            f"""
        **Distribution Metrics:**
        - Sample size: $n = {len(non_zero_weights)}$
        - Normalization: $C = {norm_factor:.{self.vis_config.decimals}f}$

        **Central Tendency:**
        - Mean ($\\mu$): ${np.mean(non_zero_weights):.{self.vis_config.decimals}f}$
        - Median: ${percentiles[1]:.{self.vis_config.decimals}f}$

        **Dispersion:**
        - Std ($\\sigma$): ${np.std(non_zero_weights):.{self.vis_config.decimals}f}$
        - IQR: ${(percentiles[2] - percentiles[0]):.{self.vis_config.decimals}f}$
        - Range: [{np.min(non_zero_weights):.{self.vis_config.decimals}f}, {np.max(non_zero_weights):.{self.vis_config.decimals}f}]
        """
        )

    def _display_search_parameters(
        self, search_range: List[Tuple[int, int]], nlm_comp: NLMComputation
    ) -> None:
        """Display academic analysis of search parameters."""
        y_range, x_range = search_range
        search_area = (y_range[1] - y_range[0]) * (x_range[1] - x_range[0])

        st.markdown(
            f"""
        **Search Window Parameters:**
        - Dimensions: ${y_range[1] - y_range[0]} \\times {x_range[1] - x_range[0]}$
        - Total area: ${search_area}$ pixels
        - Filter strength ($h$): ${nlm_comp.filter_strength:.{self.vis_config.decimals}f}$

        **Boundaries:**
        - X: $[{x_range[0]}, {x_range[1]})$
        - Y: $[{y_range[0]}, {y_range[1]})$
        """
        )

    def _render_pixel_info(
        self, x: int, y: int, kernel_size: int, img_array: np.ndarray
    ) -> None:
        """Render enhanced pixel information with LaTeX formatting."""
        half_kernel = kernel_size // 2

        # Calculate output coordinates and valid ranges
        i, j = x - half_kernel, y - half_kernel  # Output coordinates
        valid_input_x = [half_kernel, img_array.shape[1] - half_kernel]
        valid_input_y = [half_kernel, img_array.shape[0] - half_kernel]
        valid_output_x = [0, img_array.shape[1] - kernel_size + 1]
        valid_output_y = [0, img_array.shape[0] - kernel_size + 1]

        # Create two columns for coordinate systems
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📍 Coordinate System")
            st.latex(
                rf"""
            \begin{{aligned}}
            \text{{Input }} (x,y) &= ({x}, {y}) \\
            I_{{{x},{y}}} &= {img_array[y, x]:.3f} \\
            \text{{Valid }} x &\in [{valid_input_x[0]}, {valid_input_x[1]}] \\
            \text{{Valid }} y &\in [{valid_input_y[0]}, {valid_input_y[1]}]
            \end{{aligned}}
            """
            )

        with col2:
            st.markdown("### 🎯 Processing Region")
            st.latex(
                rf"""
            \begin{{aligned}}
            \text{{Output }} (i,j) &= ({i}, {j}) \\
            \text{{Kernel}} &= {kernel_size} \times {kernel_size} \\
            \text{{Valid }} i &\in [{valid_output_x[0]}, {valid_output_x[1]}] \\
            \text{{Valid }} j &\in [{valid_output_y[0]}, {valid_output_y[1]}]
            \end{{aligned}}
            """
            )

        # Add NLM-specific information if needed
        if st.session_state.get("filter_type") == "nlm":
            st.markdown("### 🔍 Search Configuration")
            search_size = (
                "Global"
                if st.session_state.get("use_full_image", True)
                else f"{
                    st.session_state.get('search_size', 21)}×{st.session_state.get('search_size', 21)}"
            )
            h = st.session_state.get("filter_strength", 10.0)

            st.latex(
                rf"""
            \begin{{aligned}}
            \text{{Search Mode}} &= \text{{{search_size}}} \\
            \text{{Filter Strength }} (h) &= {h:.2f} \\
            \text{{Patch Size}} &= {kernel_size} \times {kernel_size}
            \end{{aligned}}
            """
            )

    def _render_weight_distribution(
        self, similarity_map: np.ndarray, stats: Dict[str, float]
    ) -> None:
        """Render enhanced weight distribution analysis."""
        col1, col2 = st.columns([2, 1])

        with col1, figure_context() as fig:
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

            # Plot histogram in top subplot with better styling
            ax1 = fig.add_subplot(gs[0])
            non_zero_weights = similarity_map[similarity_map > 0]

            # Use better binning strategy
            n_bins = min(50, len(np.unique(non_zero_weights)))
            weights, bins, patches = ax1.hist(
                non_zero_weights,
                bins=n_bins,
                density=True,
                alpha=0.7,
                color="skyblue",
                label=f"n={len(non_zero_weights)}",
            )

            # Add kernel density estimate
            if len(non_zero_weights) > 1:
                kde = scipy_stats.gaussian_kde(non_zero_weights)
                x_range = np.linspace(bins[0], bins[-1], 200)
                ax1.plot(x_range, kde(x_range), "r-", lw=2, label="Density Estimate")

            # Add statistical markers
            ax1.axvline(
                stats["mean"],
                color="red",
                linestyle="--",
                label=f'Mean={stats["mean"]:.3f}',
            )
            ax1.axvline(
                stats["median"],
                color="green",
                linestyle=":",
                label=f'Median={stats["median"]:.3f}',
            )

            # Add IQR shading
            ax1.axvspan(
                stats["q1"],
                stats["q3"],
                alpha=0.2,
                color="gray",
                label=f'IQR={stats["iqr"]:.3f}',
            )

            ax1.set_title("Weight Distribution")
            ax1.set_xlabel("Weight Value")
            ax1.set_ylabel("Density")
            ax1.legend()

            # Plot cumulative distribution in bottom subplot
            ax2 = fig.add_subplot(gs[1])
            sorted_weights = np.sort(non_zero_weights)
            cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
            ax2.plot(sorted_weights, cumulative, "b-", label="Cumulative")
            ax2.set_xlabel("Weight Value")
            ax2.set_ylabel("Cumulative")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown(
                f"""
            ### Weight Statistics
            - **Mean:** {stats['mean']:.3f}
            - **Median:** {stats['median']:.3f}
            - **Std Dev:** {stats['std']:.3f}
            - **Active Ratio:** {stats['active_ratio']*100:.1f}%

            ### Distribution Shape
            - **IQR:** {stats['iqr']:.3f}
            - **Q1:** {stats['q1']:.3f}
            - **Q3:** {stats['q3']:.3f}
            - **Range:** [{stats['min']:.3f}, {stats['max']:.3f}]
            """
            )

    def _add_custom_styles(self):
        """Add custom CSS styles."""
        st.markdown(
            """
            <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 20px;
                background-color: #f0f2f6;
                border-radius: 4px 4px 0 0;
            }
            .stTabs [aria-selected="true"] {
                background-color: #e6e9ef;
            }
            .stat-box {
                padding: 1rem;
                border-radius: 0.5rem;
                background-color: #f0f2f6;
                margin-bottom: 1rem;
            }
            .metric-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.5rem;
            }
            .metric-label {
                color: #555;
                font-size: 0.9rem;
            }
            .metric-value {
                font-weight: bold;
                color: #0e1117;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

    def _render_metric_box(self, title: str, metrics: dict):
        """Render a styled metric box."""
        st.markdown(
            f"""
            <div class="stat-box">
                <h4>{title}</h4>
                {''.join([
                f'''
                    <div class="metric-row">
                        <span class="metric-label">{label}:</span>
                        <span class="metric-value">{value}</span>
                    </div>
                    '''
                for label, value in metrics.items()
            ])}
            </div>
        """,
            unsafe_allow_html=True,
        )

    def _render_kernel_view(
        self, kernel: np.ndarray, x: int, y: int, kernel_size: int
    ) -> None:
        """Render kernel view with annotations."""
        with figure_context() as fig:
            ax = fig.add_subplot(111)

            # Display kernel
            im = ax.imshow(
                kernel,
                cmap=self.settings.colormap,
                interpolation="nearest",
                vmin=0,
                vmax=1,
                aspect="equal",
            )

            # Add value annotations
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{kernel[i, j]:.{self.vis_config.decimals}f}",
                        ha="center",
                        va="center",
                        color="black",
                        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                    )

            # Add kernel overlay
            add_kernel_overlay(
                ax=ax,
                center=(kernel_size // 2, kernel_size // 2),
                kernel_size=kernel_size,
                image_shape=(int(kernel.shape[1]), int(kernel.shape[0])),
                config=self.kernel_config,
            )

            # Set title and add colorbar
            ax.set_title(f"Kernel at ({x}, {y})")
            if self.vis_config.show_colorbar:
                plt.colorbar(mappable=im, ax=ax, label="Intensity")

            plt.tight_layout()
            st.pyplot(fig)

            # Show kernel statistics if enabled
            if self.vis_config.show_stats:
                self._render_kernel_statistics(kernel)

    def _render_kernel_statistics(self, kernel: np.ndarray) -> None:
        """Render statistical analysis of kernel values."""
        # Validate kernel
        if kernel is None or kernel.size == 0:
            st.warning("No valid kernel data available")
            return

        try:
            stats = {
                "Mean": float(np.mean(kernel)),
                "Std Dev": float(np.std(kernel)),
                "Min": float(np.min(kernel)),
                "Max": float(np.max(kernel)),
                "Median": float(np.median(kernel)),
            }

            st.markdown("#### Kernel Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    "\n".join(
                        [
                            f"- **{key}:** {value:.{self.vis_config.decimals}f}"
                            for key, value in stats.items()
                        ]
                    )
                )

            with col2:
                if stats["Mean"] > 0:  # Avoid division by zero
                    st.markdown("#### Distribution")
                    st.markdown(
                        f"""
                    - **Range:** [{stats['Min']:.{self.vis_config.decimals}f}, {stats['Max']:.{self.vis_config.decimals}f}]
                    - **Spread:** {(stats['Max'] - stats['Min']):.{self.vis_config.decimals}f}
                    - **CV:** {(stats['Std Dev'] / stats['Mean']):.{self.vis_config.decimals}f}
                    """
                    )
                else:
                    st.warning("Cannot compute distribution metrics (mean is zero)")

        except Exception as e:
            st.error(f"Error computing kernel statistics: {str(e)}")

    def _render_spatial_statistics(self, kernel: np.ndarray) -> None:
        """Render spatial analysis of kernel values."""
        # Validate kernel
        if kernel is None or kernel.size == 0:
            st.warning("No valid kernel data available for spatial analysis")
            return

        try:
            st.markdown("#### Spatial Analysis")

            # Calculate directional statistics
            row_means = np.mean(kernel, axis=1)
            col_means = np.mean(kernel, axis=0)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Directional Means**")
                h_mean = np.mean(row_means)
                v_mean = np.mean(col_means)
                ratio = h_mean / v_mean if v_mean != 0 else 0

                st.markdown(
                    f"""
                - Horizontal: {h_mean:.{self.vis_config.decimals}f}
                - Vertical: {v_mean:.{self.vis_config.decimals}f}
                - Ratio (H/V): {ratio:.{self.vis_config.decimals}f}
                """
                )

            with col2:
                st.markdown("**Center vs Edges**")
                try:
                    center = kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]
                    edges = np.concatenate(
                        [
                            kernel[0, :],  # Top
                            kernel[-1, :],  # Bottom
                            kernel[1:-1, 0],  # Left
                            kernel[1:-1, -1],  # Right
                        ]
                    )
                    edge_mean = np.mean(edges)
                    edge_ratio = edge_mean / center if center != 0 else 0

                    st.markdown(
                        f"""
                    - Center: {center:.{self.vis_config.decimals}f}
                    - Edge Mean: {edge_mean:.{self.vis_config.decimals}f}
                    - Ratio (E/C): {edge_ratio:.{self.vis_config.decimals}f}
                    """
                    )
                except (IndexError, ValueError):
                    st.warning("Unable to compute center vs edges statistics")

        except Exception as e:
            st.error(f"Error computing spatial statistics: {str(e)}")

    def _render_settings(self, nlm_state: NLMState) -> None:
        """Render settings tab."""
        # Display settings
        st.markdown("### 🎨 Display Settings")
        col1, col2 = st.columns(2)

        with col1:
            show_colorbar = st.checkbox(
                "Show Colorbar",
                value=st.session_state.get("show_colorbar", True),
                key="proc_show_colorbar_select",
                help="Toggle colorbar visibility",
            )

            show_stats = st.checkbox(
                "Show Statistics",
                value=st.session_state.get("show_stats", True),
                key="proc_show_stats_select",
                help="Toggle statistics display",
            )

        with col2:
            colormap = st.selectbox(
                "Colormap",
                options=["viridis", "gray", "plasma", "inferno"],
                index=0,
                key="proc_colormap_select",
                help="Select colormap for visualization",
            )

            decimals = st.number_input(
                "Decimal Places",
                min_value=1,
                max_value=6,
                value=st.session_state.get("decimals", 3),
                key="proc_decimals_select",
                help="Number of decimal places to display",
            )

        # NLM-specific settings
        if nlm_state.filter_type == "nlm":
            st.markdown("### 🔍 NLM Settings")

            # Analysis settings
            show_analysis = st.checkbox(
                "Show Analysis",
                value=nlm_state.show_analysis,
                key="proc_show_analysis_select",
                help="Toggle analysis display",
            )

            show_formulas = st.checkbox(
                "Show Formulas",
                value=nlm_state.show_formulas,
                key="proc_show_formulas_select",
                help="Toggle formula display",
            )

            # Visualization mode
            display_mode = st.radio(
                "Display Mode",
                options=["side_by_side", "overlay"],
                index=0 if nlm_state.display_mode == "side_by_side" else 1,
                key="proc_display_mode_select",
                help="Select how to display input and processed images",
                horizontal=True,
            )

        # Update settings in session state
        if callable(self.config.on_settings_changed):
            settings_update = {
                "show_colorbar": show_colorbar,
                "show_stats": show_stats,
                "colormap": colormap,
                "decimals": decimals,
                "use_full_image": nlm_state.use_full_image,
                "process_full_image": nlm_state.use_full_image,
                "selected_region": st.session_state.get("selected_region", None),
            }

            # Add NLM-specific settings if applicable
            if nlm_state.filter_type == "nlm":
                settings_update.update({
                    "show_analysis": show_analysis,
                    "show_formulas": show_formulas,
                    "display_mode": display_mode,
                })

            self.config.on_settings_changed(settings_update)
