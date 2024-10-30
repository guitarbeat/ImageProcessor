"""
Component for controlling image processing behavior.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from . import Component
from utils.image_processing import preprocess_image
from .sidebar import DisplaySettings

@dataclass
class ProcessingControlConfig:
    """Configuration for processing control."""
    on_settings_changed: Callable[[Dict[str, Any]], None]
    initial_settings: Dict[str, Any] = None
    display_settings: Optional[DisplaySettings] = None
    
    def __post_init__(self):
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
        if "processed_regions" not in st.session_state:
            st.session_state.processed_regions = set()
    
    def render(self, image: Optional[Image.Image] = None) -> None:
        """Render processing control interface."""
        if image is None:
            st.warning("No image loaded")
            return

        if "max_processable_pixels" not in st.session_state:
            st.session_state.max_processable_pixels = self._calculate_max_pixels(image)
        if "selected_region" not in st.session_state:
            st.session_state.selected_region = None
        if "selected_filters" not in st.session_state:
            st.session_state.selected_filters = ["LSCI"]

        # Create two columns for better layout
        filter_col, mode_col = st.columns([0.7, 0.3])
        
        with filter_col:
            # Add multiple filter selection with validation
            available_filters = {
                "LSCI": "Laser Speckle Contrast Imaging",
                "Mean": "Average Intensity",
                "Standard Deviation": "Local Contrast"
            }
            selected = st.multiselect(
                "Select Processing Filters",
                options=list(available_filters.keys()),
                default=st.session_state.selected_filters,
                format_func=lambda x: f"{x} ({available_filters[x]})",
                help="Choose one or more filters to apply to the image"
            )
            
            # Ensure at least one filter is selected
            if not selected:
                st.warning("⚠️ Please select at least one filter")
                selected = ["LSCI"]  # Default to LSCI if none selected
            
            st.session_state.selected_filters = selected

        with mode_col:
            # Add processing mode selection with better styling
            process_full = st.toggle(
                "Process Full Image",
                value=self.config.initial_settings["process_full_image"],
                help="Toggle between full image processing and region selection"
            )
            
            if process_full:
                st.info("Processing entire image", icon="🔍")
            else:
                st.info("Select region below", icon="✂️")

        # Show region selection or progress
        if not process_full:
            self._render_selection_interface(image)
            
            # Show current selection info using columns instead of expander
            if st.session_state.selected_region:
                x1, y1, x2, y2 = st.session_state.selected_region
                st.markdown("#### Selection Details")
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Width", f"{x2-x1} px")
                with metric_cols[1]:
                    st.metric("Height", f"{y2-y1} px")
                with metric_cols[2]:
                    st.metric("Total Pixels", f"{(x2-x1)*(y2-y1):,}")
        else:
            # Show progress bar for full image processing
            if "processing_progress" in st.session_state:
                progress = st.session_state.processing_progress
                if progress > 0:
                    st.progress(progress, text=f"Processing... {progress:.1%}")

        # Update settings
        settings = {
            "process_full_image": process_full,
            "selected_region": st.session_state.get("selected_region"),
            "processing_progress": st.session_state.get("processing_progress", 0)
        }
        self.config.on_settings_changed(settings)

    def _render_selection_interface(self, image: Image.Image) -> None:
        """Render the image selection interface."""
        try:
            # Create help text container
            help_container = st.container()
            with help_container:
                st.markdown("""
                ### Region Selection
                - 🖱️ Click and drag to select a region
                - 🎯 Selection must be larger than kernel size
                - 🔄 Previous selection is cleared when changing images
                """)
            
            # Create a more robust unique key that includes colormap
            image_id = st.session_state.get('current_image_id', '')
            colormap = st.session_state.get('colormap', 'gray')
            image_key = f"process_coords_{image_id}_{colormap}"
            
            # Add underscore to prevent hashing
            gray_image, _ = preprocess_image(image)
            display_image = self._resize_for_display(gray_image)
            
            # Use BytesIO instead of temporary file
            buf = BytesIO()
            display_array = np.array(display_image)
            
            # Normalize array to 0-1 range
            display_array = (display_array - display_array.min()) / (display_array.max() - display_array.min())
            
            # Save with colormap to buffer
            plt.imsave(buf, display_array, cmap=colormap, format='png')
            buf.seek(0)
            display_image = Image.open(buf)
            
            coordinates = streamlit_image_coordinates(
                display_image,
                key=image_key,
                click_and_drag=True
            )
            
            # Reset selection when image or colormap changes
            current_state = f"{st.session_state.get('current_image_id', '')}_{colormap}"
            if "last_state" not in st.session_state or st.session_state.last_state != current_state:
                st.session_state.selected_region = None
                st.session_state.processed_regions.clear()
                st.session_state.last_state = current_state
            
            if coordinates:
                original_coords = self._scale_coordinates(
                    coordinates, 
                    image.size, 
                    display_image.size
                )
                
                # Handle region selection (drag)
                if all(key in original_coords for key in ['x1', 'y1', 'x2', 'y2']):
                    self._handle_region_selection(original_coords, image)
                
                # Handle single click (ignored)
                elif all(key in original_coords for key in ['x', 'y']):
                    st.warning("🖱️ Please drag to select a region instead of clicking")
                
        except Exception as e:
            st.error(f"Selection error: {str(e)}")

    def _handle_region_selection(self, coords: Dict[str, int], image: Image.Image) -> None:
        """Handle region selection logic."""
        x1, y1 = coords['x1'], coords['y1']
        x2, y2 = coords['x2'], coords['y2']
        
        # Get current kernel size from session state
        kernel_size = st.session_state.get('kernel_size', 7)
        half_kernel = kernel_size // 2
        
        # Ensure coordinates are within image bounds with kernel padding
        x1 = max(half_kernel, min(x1, image.size[0] - half_kernel))
        x2 = max(half_kernel, min(x2, image.size[0] - half_kernel))
        y1 = max(half_kernel, min(y1, image.size[1] - half_kernel))
        y2 = max(half_kernel, min(y2, image.size[1] - half_kernel))
        
        # Ensure minimum region size (at least one kernel)
        if abs(x2 - x1) < kernel_size or abs(y2 - y1) < kernel_size:
            st.warning(f"⚠️ Selected region must be at least {kernel_size}×{kernel_size} pixels (current kernel size)")
            return
        
        # Ensure maximum region size
        max_pixels = self._calculate_max_pixels(image)
        region_pixels = abs(x2 - x1) * abs(y2 - y1)
        if region_pixels > max_pixels:
            st.warning(f"⚠️ Selected region is too large. Maximum allowed: {max_pixels:,} pixels")
            return
        
        # Store normalized coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        st.session_state.selected_region = (x1, y1, x2, y2)
        
        # Add region to processed regions
        region_key = f"{x1},{y1},{x2},{y2}"
        if not self._is_region_processed(region_key):
            self._add_processed_region(region_key)

    def _resize_for_display(self, image: Image.Image) -> Image.Image:
        """Resize image for display while maintaining aspect ratio."""
        target_height = 150  # Compact height
        width, height = image.size
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        # Ensure width doesn't exceed reasonable bounds
        max_width = 400
        if new_width > max_width:
            new_width = max_width
            target_height = int(max_width / aspect_ratio)
        
        return image.resize((new_width, target_height), Image.Resampling.LANCZOS)

    def _scale_coordinates(self, coords: Dict[str, int], 
                         original_size: tuple[int, int], 
                         display_size: tuple[int, int]) -> Dict[str, int]:
        """Scale coordinates from display size to original size."""
        scale_x = original_size[0] / display_size[0]
        scale_y = original_size[1] / display_size[1]
        
        scaled_coords = {}
        if all(key in coords for key in ['x1', 'y1', 'x2', 'y2']):
            scaled_coords = {
                'x1': int(coords['x1'] * scale_x),
                'y1': int(coords['y1'] * scale_y),
                'x2': int(coords['x2'] * scale_x),
                'y2': int(coords['y2'] * scale_y)
            }
        elif all(key in coords for key in ['x', 'y']):
            scaled_coords = {
                'x': int(coords['x'] * scale_x),
                'y': int(coords['y'] * scale_y)
            }
        return scaled_coords

    def _calculate_max_pixels(self, image: Image.Image) -> int:
        """Calculate maximum processable pixels based on image size."""
        width, height = image.size
        return min(width * height, 1_000_000)

    @staticmethod
    def update_progress(progress: float) -> None:
        """Update processing progress."""
        st.session_state.processing_progress = progress

    def _add_processed_region(self, region_key: str) -> None:
        """Mark a region as processed."""
        st.session_state.processed_regions.add(region_key)

    def _is_region_processed(self, region_key: str) -> bool:
        """Check if a region has been processed."""
        return region_key in st.session_state.processed_regions