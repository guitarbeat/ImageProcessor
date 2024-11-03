"""
Component for controlling image processing behavior.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List, Tuple
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from app.ui.components.base import Component
from app.ui.components.common import create_pixel_selector
from app.ui.components.math_explainer import MathExplainer, MathExplainerConfig
from app.ui.settings.display import DisplaySettings
from app.processors.filters import LSCIComputation, NLMComputation
from app.utils.visualization import (
    create_visualization_config,
    create_kernel_overlay_config,
    plot_weight_distribution,
    add_kernel_overlay,
    highlight_pixel,
    add_value_annotations,
    KernelOverlayConfig,
    SearchWindowOverlayConfig,
    add_search_window_overlay,
    plot_similarity_map
)
from app.utils.context_managers import figure_context, visualization_context
from app.utils.latex import (
    SPECKLE_FORMULA_CONFIG,
    NLM_FORMULA_CONFIG
)
from app.analysis.nlm_analysis import NLMAnalysis, NLMAnalysisConfig


@dataclass
class ProcessingControlConfig:
    """Configuration for processing control."""
    on_settings_changed: Callable[[Dict[str, Any]], None]
    initial_settings: Dict[str, Any] = None
    display_settings: Optional[DisplaySettings] = None

    def __post_init__(self) -> None:
        """Initialize default settings if not provided."""
        if self.initial_settings is None:
            self.initial_settings = {
                "process_full_image": True,
                "selected_region": None,
                "processing_progress": 0
            }
        if self.display_settings is None:
            self.display_settings = DisplaySettings()


class ProcessingControl(Component):
    """Component for controlling image processing behavior."""

    def __init__(self, config: ProcessingControlConfig):
        self.config = config
        self.settings = DisplaySettings.from_session_state()
        self.vis_config = create_visualization_config()
        self.kernel_config = create_kernel_overlay_config()

    def render(self, image: Optional[Image.Image] = None) -> None:
        """Render the processing control interface."""
        if image is None:
            st.warning("No image loaded")
            return

        # Initialize session state variables
        if "clear_button_clicked" not in st.session_state:
            st.session_state.clear_button_clicked = False

        st.session_state.setdefault('processed_regions', set())
        st.session_state.setdefault('processing_progress', 0)
        st.session_state.setdefault('selected_points', [])
        st.session_state.setdefault('coordinates', None)
        st.session_state.setdefault('selected_region', None)
        st.session_state.setdefault('selected_filters', [])

        # Get current filter type from processing params
        current_filter = st.session_state.get('filter_type', 'lsci')
        kernel_size = st.session_state.get('kernel_size', 7)
        half_kernel = kernel_size // 2

        # Calculate valid pixel range based on kernel size
        img_array = st.session_state.get('image_array')
        if img_array is None:
            st.error("No image array found in session state")
            return

        # Define valid pixel ranges accounting for kernel boundaries
        x_min, x_max = half_kernel, img_array.shape[1] - half_kernel - 1
        y_min, y_max = half_kernel, img_array.shape[0] - half_kernel - 1

        # Define all available filters
        available_filters = ["LSCI", "NLM", "Mean", "Standard Deviation"]

        # Set default filters based on current mode
        if not st.session_state.selected_filters:
            if current_filter == "lsci":
                st.session_state.selected_filters = ["LSCI", "Mean", "Standard Deviation"]
            else:  # NLM mode
                # Include both LSCI and NLM by default when in NLM mode
                st.session_state.selected_filters = ["LSCI", "NLM", "Mean", "Standard Deviation"]

        # Filter selection
        selected_filters = st.multiselect(
            "Analysis Views",
            options=available_filters,
            default=st.session_state.selected_filters,
            help="Choose which analysis results to display"
        )
        
        # Update selected filters in session state
        st.session_state.selected_filters = selected_filters

        # Process full image toggle
        process_full = st.toggle(
            "Full Image",
            value=self.config.initial_settings["process_full_image"],
            help="Toggle between full/region processing",
            key="process_full_toggle"
        )

        if not process_full:
            # Use UI component for pixel selection
            pixel_x, pixel_y = create_pixel_selector(
                x_range=(x_min, x_max),
                y_range=(y_min, y_max)
            )
            
            # Store selected pixel
            st.session_state.selected_pixel = (pixel_x, pixel_y)

            # Show analysis in tabs
            tab1, tab2, tab3 = st.tabs(["📍 Pixel Info", "📐 Math Explanation", "📊 Analysis"])
            
            with tab1:
                self._render_pixel_info(pixel_x, pixel_y, kernel_size, img_array)
            
            with tab2:
                # Choose formula config based on filter type
                current_filter = st.session_state.get('filter_type', 'lsci')
                formula_config = NLM_FORMULA_CONFIG if current_filter == "nlm" else SPECKLE_FORMULA_CONFIG
                
                # Create math explainer
                math_explainer = MathExplainer(
                    MathExplainerConfig(
                        formula_config=formula_config,
                        kernel_size=kernel_size,
                        selected_pixel=st.session_state.selected_pixel,
                        image_array=img_array
                    )
                )
                math_explainer.render()

            with tab3:
                if current_filter == "nlm":
                    st.markdown("### NLM Analysis")
                    
                    # Create NLM computation object
                    nlm_comp = NLMComputation(
                        kernel_size=kernel_size,
                        filter_strength=st.session_state.get('filter_strength', 10.0),
                        search_window_size=None if st.session_state.get('use_full_image', True) 
                            else st.session_state.get('search_size', 21)
                    )
                    
                    # Get NLM results
                    x, y = st.session_state.selected_pixel
                    if nlm_comp.search_window_size is None:
                        search_range = [(0, img_array.shape[0]), (0, img_array.shape[1])]
                    else:
                        half_search = nlm_comp.search_window_size // 2
                        search_range = [
                            (max(0, y - half_search), min(img_array.shape[0], y + half_search + 1)),
                            (max(0, x - half_search), min(img_array.shape[1], x + half_search + 1))
                        ]
                    
                    self.render_nlm_analysis(img_array, nlm_comp, x, y, search_range)

                elif current_filter == "lsci":
                    st.markdown("### LSCI Analysis")
                    
                    # Create LSCI computation object
                    lsci_comp = LSCIComputation(kernel_size=kernel_size)
                    
                    # Get pixel coordinates and kernel
                    x, y = st.session_state.selected_pixel
                    kernel = img_array[y-half_kernel:y+half_kernel+1, x-half_kernel:x+half_kernel+1]
                    values = lsci_comp.get_intermediate_values(kernel)
                    
                    # Create analysis tabs
                    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                        "Local Statistics", "Distribution Analysis", "Spatial Analysis"
                    ])
                    
                    with analysis_tab1:
                        st.markdown("""
                        ### Local Statistical Analysis
                        The LSCI algorithm computes contrast as the ratio of standard deviation to mean 
                        intensity in a local neighborhood. This provides a measure of local speckle contrast.
                        """)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("#### Reference Values")
                            st.markdown(f"""
                            **Central Pixel:**
                            - Position: $({x}, {y})$
                            - Intensity: $I_{{{x},{y}}} = {float(values['original_value']):.{self.vis_config.decimals}f}$
                            
                            **Local Statistics:**
                            - Mean ($\\mu$): ${float(values['mean']):.{self.vis_config.decimals}f}$
                            - Std Dev ($\\sigma$): ${float(values['std']):.{self.vis_config.decimals}f}$
                            - Contrast ($\\sigma/\\mu$): ${float(values['sc']):.{self.vis_config.decimals}f}$
                            """)
                            
                            # Add theoretical context
                            st.markdown("""
                            **Interpretation:**
                            - Higher contrast indicates more speckle variation
                            - Lower contrast suggests more temporal averaging
                            - Values typically range from 0 to 1
                            """)
                            
                        with col2:
                            st.markdown("#### Kernel Visualization")
                            self._display_patch(kernel, f"Neighborhood at ({x}, {y})")
                    
                    with analysis_tab2:
                        st.markdown("""
                        ### Distribution Analysis
                        The distribution of intensity values within the kernel provides insight 
                        into the local speckle statistics.
                        """)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Enhanced histogram with statistical overlay
                            with figure_context() as fig:
                                ax = fig.add_subplot(111)
                                
                                # Plot histogram
                                counts, bins, _ = ax.hist(kernel.flatten(), bins=20, 
                                                        density=True, alpha=0.7,
                                                        color=st.session_state.get('center_color', '#FF0000'))
                                
                                # Add statistical markers
                                mean = float(values['mean'])
                                std = float(values['std'])
                                ax.axvline(mean, color='r', linestyle='--', 
                                          label=f'Mean: {mean:.{self.vis_config.decimals}f}')
                                ax.axvline(mean + std, color='g', linestyle=':', 
                                          label=f'Mean ± Std: {std:.{self.vis_config.decimals}f}')
                                ax.axvline(mean - std, color='g', linestyle=':')
                                
                                # Add theoretical Gaussian for comparison
                                x_range = np.linspace(bins[0], bins[-1], 100)
                                gaussian = np.exp(-(x_range - mean)**2 / (2*std**2)) / (std * np.sqrt(2*np.pi))
                                ax.plot(x_range, gaussian, 'k--', alpha=0.5, label='Gaussian Fit')
                                
                                ax.set_xlabel('Intensity')
                                ax.set_ylabel('Density')
                                ax.set_title('Kernel Intensity Distribution')
                                ax.legend()
                                st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### Statistical Analysis")
                            percentiles = np.percentile(kernel, [25, 50, 75])
                            st.markdown(f"""
                            **Distribution Metrics:**
                            - Sample size: $n = {kernel.size}$
                            - Range: [{kernel.min():.{self.vis_config.decimals}f}, {kernel.max():.{self.vis_config.decimals}f}]$
                            
                            **Central Tendency:**
                            - Mean ($\\mu$): ${mean:.{self.vis_config.decimals}f}$
                            - Median: ${percentiles[1]:.{self.vis_config.decimals}f}$
                            
                            **Dispersion:**
                            - Std ($\\sigma$): ${std:.{self.vis_config.decimals}f}$
                            - IQR: ${(percentiles[2] - percentiles[0]):.{self.vis_config.decimals}f}$
                            - CV ($\\sigma/\\mu$): ${float(values['sc']):.{self.vis_config.decimals}f}$
                            """)
                    
                    with analysis_tab3:
                        st.markdown("""
                        ### Spatial Analysis
                        The spatial distribution of intensities within the kernel reveals local 
                        patterns and potential artifacts.
                        """)
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Create 2D visualization of kernel with statistical overlays
                            with figure_context() as fig:
                                ax = fig.add_subplot(111)
                                im = ax.imshow(kernel, cmap=self.vis_config.colormap)
                                
                                # Add kernel overlay
                                add_kernel_overlay(
                                    ax=ax,
                                    center=(kernel_size//2, kernel_size//2),
                                    kernel_size=kernel_size,
                                    image_shape=kernel.shape,
                                    config=self.kernel_config
                                )
                                
                                # Highlight center pixel
                                highlight_pixel(
                                    ax=ax,
                                    position=(kernel_size//2, kernel_size//2),
                                    color=self.kernel_config.center_color,
                                    alpha=self.kernel_config.center_alpha
                                )
                                
                                ax.set_title('Spatial Intensity Distribution')
                                plt.colorbar(im, ax=ax, label='Intensity')
                                st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### Spatial Statistics")
                            # Calculate row and column statistics
                            row_means = np.mean(kernel, axis=1)
                            col_means = np.mean(kernel, axis=0)
                            st.markdown(f"""
                            **Directional Analysis:**
                            - Row Mean Range: [{np.min(row_means):.{self.vis_config.decimals}f}, {np.max(row_means):.{self.vis_config.decimals}f}]
                            - Col Mean Range: [{np.min(col_means):.{self.vis_config.decimals}f}, {np.max(col_means):.{self.vis_config.decimals}f}]
                            
                            **Boundary Effects:**
                            - Edge/Center Ratio: {np.mean(kernel[0,:])/(kernel[kernel_size//2,kernel_size//2] + 1e-10):.{self.vis_config.decimals}f}
                            - Corner/Center Ratio: {kernel[0,0]/(kernel[kernel_size//2,kernel_size//2] + 1e-10):.{self.vis_config.decimals}f}
                            """)
                            
                            # Add interpretation
                            st.markdown("""
                            **Interpretation:**
                            - Uniform ratios suggest isotropic speckle
                            - High edge ratios may indicate boundary effects
                            - Directional patterns could reveal flow direction
                            """)

        # Update settings
        self.config.on_settings_changed({
            "process_full_image": process_full,
            "selected_region": None
        })

    def render_nlm_analysis(self, img_array: np.ndarray, nlm_comp: NLMComputation, 
                           x: int, y: int, search_range: List[Tuple[int, int]]) -> None:
        """Render NLM analysis using dedicated analyzer."""
        # Add custom styles
        self._add_custom_styles()
        
        config = NLMAnalysisConfig(
            filter_strength=nlm_comp.filter_strength,
            kernel_size=nlm_comp.kernel_size,
            search_window_size=nlm_comp.search_window_size,
            decimals=self.vis_config.decimals
        )
        
        analyzer = NLMAnalysis(config)
        similarity_map = nlm_comp.compute_similarity_map(img_array, x, y)
        
        # Create main layout
        st.markdown("### 🔍 NLM Analysis")
        
        # Create two columns for main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main visualization area with tabs
            tabs = st.tabs([
                "🎯 Search Region",
                "📊 Weight Distribution",
                "🌐 Spatial Analysis"
            ])
            
            with tabs[0]:
                self._plot_search_region(
                    img_array=img_array,
                    similarity_map=similarity_map,
                    x=x, y=y,
                    search_range=search_range,
                    kernel_size=nlm_comp.kernel_size
                )
                
            with tabs[1]:
                stats = analyzer.analyze_weights(similarity_map)
                analyzer._render_weight_distribution(similarity_map, stats)
                
            with tabs[2]:
                spatial_stats = analyzer.analyze_spatial_patterns(similarity_map)
                analyzer._render_spatial_analysis(similarity_map, spatial_stats)
        
        with col2:
            # Coordinate information
            half_kernel = nlm_comp.kernel_size // 2
            self._render_metric_box("📍 Coordinates", {
                "Input (x,y)": f"({x}, {y})",
                "Output (i,j)": f"({x-half_kernel}, {y-half_kernel})"
            })
            
            # Search configuration
            search_size = "Global" if nlm_comp.search_window_size is None else f"{nlm_comp.search_window_size}×{nlm_comp.search_window_size}"
            self._render_metric_box("🔍 Search Config", {
                "Mode": search_size,
                "Filter Strength": f"{nlm_comp.filter_strength:.1f}",
                "Kernel Size": f"{nlm_comp.kernel_size}×{nlm_comp.kernel_size}"
            })
            
            # Weight statistics
            stats = analyzer.analyze_weights(similarity_map)
            self._render_metric_box("📊 Statistics", {
                "Mean Weight": f"{stats['mean']:.3f}",
                "Active Ratio": f"{stats['active_ratio']*100:.1f}%",
                "Range": f"[{stats['min']:.3f}, {stats['max']:.3f}]"
            })
            
            # Spatial analysis
            spatial_stats = analyzer.analyze_spatial_patterns(similarity_map)
            self._render_metric_box("🌐 Spatial Analysis", {
                "H/V Ratio": f"{spatial_stats['directional_ratio']:.2f}",
                "Edge/Center": f"{spatial_stats['edge_center_ratio']:.2f}"
            })

    def _extract_patch(self, img_array: np.ndarray, x: int, y: int, kernel_size: int) -> np.ndarray:
        """Extract a patch from the image."""
        half = kernel_size // 2
        return img_array[
            max(0, y-half):min(img_array.shape[0], y+half+1),
            max(0, x-half):min(img_array.shape[1], x+half+1)
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

    def _plot_weight_distribution(self, similarity_map: np.ndarray, norm_factor: float, title: str) -> None:
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
                orientation='vertical',
                show_percentiles=True
            )
            
            # Add cumulative distribution
            ax2 = fig.add_subplot(gs[1])
            plot_weight_distribution(
                ax=ax2,
                weights=similarity_map,
                vis_config=vis_config,
                orientation='horizontal',
                show_percentiles=False
            )
            
            plt.tight_layout()
            st.pyplot(fig)

    def _plot_search_region(self, img_array: np.ndarray, similarity_map: np.ndarray,
                           x: int, y: int, search_range: List[Tuple[int, int]], 
                           kernel_size: int) -> None:
        """Enhanced search region visualization."""
        with figure_context() as fig:
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
            
            # Get configs
            vis_config = create_visualization_config()
            kernel_config = create_kernel_overlay_config()
            search_config = SearchWindowOverlayConfig()
            
            # Image with search region overlay
            ax1 = fig.add_subplot(gs[0])
            im1 = ax1.imshow(img_array, cmap=vis_config.colormap)
            
            # Add search window overlay
            use_full_image = search_range[1][1] - search_range[1][0] == img_array.shape[1]
            if use_full_image:
                search_window_size = None
            else:
                search_window_size = search_range[1][1] - search_range[1][0]
            
            add_search_window_overlay(
                ax=ax1,
                center=(x, y),
                search_window_size=search_window_size,
                image_shape=img_array.shape,
                config=search_config
            )
            
            highlight_pixel(
                ax=ax1,
                position=(x, y),
                color=kernel_config.center_color,
                alpha=kernel_config.center_alpha
            )
            ax1.set_title('Search Region')
            
            # Similarity map
            ax2 = fig.add_subplot(gs[1])
            plot_similarity_map(
                ax=ax2,
                similarity_map=similarity_map,
                center=(x, y),
                kernel_config=kernel_config,
                search_config=None,
                vis_config=vis_config,
                title='Similarity Weights',
                is_full_image=use_full_image
            )
            
            plt.tight_layout()
            st.pyplot(fig)

    def _get_top_similar_patches(
        self, img_array: np.ndarray, similarity_map: np.ndarray, 
        x: int, y: int, kernel_size: int, top_k: int = 3
    ) -> List[Tuple[np.ndarray, float, Tuple[int, int]]]:
        """Get the top-k most similar patches."""
        half = kernel_size // 2
        height, width = similarity_map.shape
        
        # Create list of (weight, position) tuples, excluding center pixel
        weights_and_positions = []
        for i in range(height):
            for j in range(width):
                # Convert similarity map coordinates to image coordinates
                img_y = i + y - height//2
                img_x = j + x - width//2
                
                # Skip center pixel and invalid positions
                if (img_y == y and img_x == x) or \
                   (img_y < half or img_y >= img_array.shape[0] - half or \
                    img_x < half or img_x >= img_array.shape[1] - half):
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
                patch_y-half:patch_y+half+1,
                patch_x-half:patch_x+half+1
            ]
            top_k_patches.append((patch, weight, (patch_x, patch_y)))
        
        return top_k_patches

    def _display_weight_statistics(self, similarity_map: np.ndarray, norm_factor: float) -> None:
        """Display academic analysis of weight statistics."""
        non_zero_weights = similarity_map[similarity_map > 0]
        percentiles = np.percentile(non_zero_weights, [25, 50, 75])
        
        st.markdown(f"""
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
        """)

    def _display_search_parameters(self, search_range: List[Tuple[int, int]], nlm_comp: NLMComputation) -> None:
        """Display academic analysis of search parameters."""
        y_range, x_range = search_range
        search_area = (y_range[1] - y_range[0]) * (x_range[1] - x_range[0])
        
        st.markdown(f"""
        **Search Window Parameters:**
        - Dimensions: ${y_range[1] - y_range[0]} \\times {x_range[1] - x_range[0]}$
        - Total area: ${search_area}$ pixels
        - Filter strength ($h$): ${nlm_comp.filter_strength:.{self.vis_config.decimals}f}$
        
        **Boundaries:**
        - X: $[{x_range[0]}, {x_range[1]})$
        - Y: $[{y_range[0]}, {y_range[1]})$
        """)

    def _render_pixel_info(self, x: int, y: int, kernel_size: int, img_array: np.ndarray) -> None:
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
            st.latex(r"""
            \begin{{aligned}}
            \text{{Input }} (x,y) &= ({x}, {y}) \\
            I_{{{x},{y}}} &= {val:.3f} \\
            \text{{Valid }} x &\in [{x_min}, {x_max}] \\
            \text{{Valid }} y &\in [{y_min}, {y_max}]
            \end{{aligned}}
            """.format(
                x=x, y=y,
                val=img_array[y, x],
                x_min=valid_input_x[0],
                x_max=valid_input_x[1],
                y_min=valid_input_y[0],
                y_max=valid_input_y[1]
            ))
            
        with col2:
            st.markdown("### 🎯 Processing Region")
            st.latex(r"""
            \begin{{aligned}}
            \text{{Output }} (i,j) &= ({i}, {j}) \\
            \text{{Kernel}} &= {k} \times {k} \\
            \text{{Valid }} i &\in [{i_min}, {i_max}] \\
            \text{{Valid }} j &\in [{j_min}, {j_max}]
            \end{{aligned}}
            """.format(
                i=i, j=j,
                k=kernel_size,
                i_min=valid_output_x[0],
                i_max=valid_output_x[1],
                j_min=valid_output_y[0],
                j_max=valid_output_y[1]
            ))
        
        # Add NLM-specific information if needed
        if st.session_state.get('filter_type') == 'nlm':
            st.markdown("### 🔍 Search Configuration")
            search_size = "Global" if st.session_state.get('use_full_image', True) else f"{st.session_state.get('search_size', 21)}×{st.session_state.get('search_size', 21)}"
            h = st.session_state.get('filter_strength', 10.0)
            
            st.latex(r"""
            \begin{{aligned}}
            \text{{Search Mode}} &= \text{{{mode}}} \\
            \text{{Filter Strength }} (h) &= {strength:.2f} \\
            \text{{Patch Size}} &= {size} \times {size}
            \end{{aligned}}
            """.format(
                mode=search_size,
                strength=h,
                size=kernel_size
            ))

    def _render_weight_distribution(self, similarity_map: np.ndarray, stats: Dict[str, float]) -> None:
        """Render enhanced weight distribution analysis."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with figure_context() as fig:
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
                    color='skyblue',
                    label=f'n={len(non_zero_weights)}'
                )
                
                # Add kernel density estimate
                from scipy import stats
                if len(non_zero_weights) > 1:
                    kde = stats.gaussian_kde(non_zero_weights)
                    x_range = np.linspace(bins[0], bins[-1], 200)
                    ax1.plot(x_range, kde(x_range), 'r-', lw=2, 
                            label='Density Estimate')
                
                # Add statistical markers
                ax1.axvline(stats['mean'], color='red', linestyle='--',
                           label=f'Mean={stats["mean"]:.3f}')
                ax1.axvline(stats['median'], color='green', linestyle=':',
                           label=f'Median={stats["median"]:.3f}')
                
                # Add IQR shading
                ax1.axvspan(stats['q1'], stats['q3'], alpha=0.2, color='gray',
                           label=f'IQR={stats["iqr"]:.3f}')
                
                ax1.set_title('Weight Distribution')
                ax1.set_xlabel('Weight Value')
                ax1.set_ylabel('Density')
                ax1.legend()
                
                # Plot cumulative distribution in bottom subplot
                ax2 = fig.add_subplot(gs[1])
                sorted_weights = np.sort(non_zero_weights)
                cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
                ax2.plot(sorted_weights, cumulative, 'b-', label='Cumulative')
                ax2.set_xlabel('Weight Value')
                ax2.set_ylabel('Cumulative')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
        with col2:
            st.markdown(f"""
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
            """)

    def _add_custom_styles(self):
        """Add custom CSS styles."""
        st.markdown("""
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
        """, unsafe_allow_html=True)

    def _render_metric_box(self, title: str, metrics: dict):
        """Render a styled metric box."""
        st.markdown(f"""
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
        """, unsafe_allow_html=True)
