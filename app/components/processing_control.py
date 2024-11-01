"""
Component for controlling image processing behavior.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
import numpy as np

from . import Component
from .sidebar import DisplaySettings


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


def clear_selection_callback():
    """Callback to clear selection state"""
    st.session_state.selected_points = []
    st.session_state.coordinates = None
    st.session_state.selected_region = None
    # Add a key to track button clicks
    st.session_state.clear_button_clicked = True


def handle_point_selection(coordinates: dict) -> None:
    """Handle point selection with proper state management"""
    if not coordinates or 'x' not in coordinates or 'y' not in coordinates:
        return

    if len(st.session_state.selected_points) < 2:
        # Append new point only if it's not already in the list
        if not any(p['x'] == coordinates['x'] and p['y'] == coordinates['y']
                   for p in st.session_state.selected_points):
            st.session_state.selected_points.append(coordinates)
    else:
        # Reset points and add new one
        st.session_state.selected_points = [coordinates]


def handle_region_selection(coordinates: dict) -> None:
    """Handle region selection with proper state management"""
    if not coordinates or not all(k in coordinates for k in ['x1', 'y1', 'x2', 'y2']):
        return

    st.session_state["coordinates"] = coordinates
    x1, y1 = min(coordinates['x1'], coordinates['x2']), min(
        coordinates['y1'], coordinates['y2'])
    x2, y2 = max(coordinates['x1'], coordinates['x2']), max(
        coordinates['y1'], coordinates['y2'])
    st.session_state.selected_region = (x1, y1, x2, y2)
    st.session_state.selected_points = []


def render_selection_interface(image: Image.Image) -> None:
    """
    Render the image selection interface.

    Args:
        image: PIL Image to render selection interface for
    """
    try:
        # Convert to grayscale for display
        if image.mode != 'L':
            image = image.convert('L')

        # Create a copy for drawing
        display_image = image.copy()

        # Create a container for selection controls
        with st.container():
            # Top row: Selection mode and clear button side by side
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                selection_mode = st.radio(
                    "Selection Mode",
                    ["Click Points", "Drag Region"],
                    horizontal=True,
                    help="Choose how to select the region"
                )
            with col2:
                if st.session_state.get("selected_points") or st.session_state.get("coordinates"):
                    # Replace if st.button() with direct button call
                    st.button(
                        "Clear Selection",
                        on_click=clear_selection_callback,
                        key="clear_selection_button",  # Add unique key
                        use_container_width=True
                    )

        # Draw existing rectangle if any
        if selection_mode == "Click Points" and st.session_state.get("selected_points"):
            draw = ImageDraw.Draw(display_image)
            points = st.session_state["selected_points"]
            if len(points) == 2:
                x1, y1 = points[0]['x'], points[0]['y']
                x2, y2 = points[1]['x'], points[1]['y']
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )
                point_size = 5
                for x, y in [(x1, y1), (x2, y2)]:
                    draw.ellipse(
                        [x-point_size, y-point_size, x+point_size, y+point_size],
                        fill="red"
                    )

        elif selection_mode == "Drag Region" and st.session_state.get("coordinates"):
            draw = ImageDraw.Draw(display_image)
            coords = st.session_state["coordinates"]
            x1, y1 = coords['x1'], coords['y1']
            x2, y2 = coords['x2'], coords['y2']

            overlay = Image.new('RGBA', display_image.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )
            display_image = Image.alpha_composite(
                display_image.convert('RGBA'), overlay).convert('RGB')

        # Create container for image and instructions
        with st.container():
            # Show appropriate instructions above image
            if selection_mode == "Click Points":
                if not st.session_state.get("selected_points"):
                    st.caption("🖱️ Click first corner point")
                elif len(st.session_state.get("selected_points", [])) == 1:
                    st.caption("🖱️ Click second corner point")
            else:
                st.caption("🖱️ Click and drag to select region")

            # Handle selection based on mode
            if selection_mode == "Click Points":
                if "selected_points" not in st.session_state:
                    st.session_state.selected_points = []

                coordinates = streamlit_image_coordinates(
                    display_image,
                    key=f"process_coords_{
                        st.session_state.get('current_image_id', '')}",
                    click_and_drag=False
                )

                if coordinates:
                    handle_point_selection(coordinates)

            else:  # Drag Region mode
                coordinates = streamlit_image_coordinates(
                    display_image,
                    key=f"process_coords_drag_{
                        st.session_state.get('current_image_id', '')}",
                    click_and_drag=True
                )

                if coordinates:
                    handle_region_selection(coordinates)

    except Exception as e:
        st.error(f"Selection error: {str(e)}")


class ProcessingControl(Component):
    """Component for controlling image processing behavior."""

    def __init__(self, config: ProcessingControlConfig) -> None:
        """
        Initialize ProcessingControl component.

        Args:
            config: Configuration settings for the processing control
        """
        self.config = config

    def render(self, image: Optional[Image.Image] = None) -> None:
        """
        Render the processing control interface.

        Args:
            image: Optional PIL Image to process
        """
        if image is None:
            st.warning("No image loaded")
            return

        # Initialize all required session state variables
        if "clear_button_clicked" not in st.session_state:
            st.session_state.clear_button_clicked = False

        st.session_state.setdefault('processed_regions', set())
        st.session_state.setdefault('selected_filters', ["LSCI"])
        st.session_state.setdefault('processing_progress', 0)
        st.session_state.setdefault('selected_points', [])
        st.session_state.setdefault('coordinates', None)
        st.session_state.setdefault('selected_region', None)

        # Main processing control container
        with st.container():
            # First row: Filter selection and full image toggle
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                available_filters = {
                    "LSCI": "Laser Speckle Contrast Imaging",
                    "Mean": "Mean Filter",
                    "Standard Deviation": "Standard Deviation Filter"
                }
                selected = st.multiselect(
                    "Filters",
                    options=list(available_filters.keys()),
                    default=st.session_state.selected_filters,
                    format_func=lambda x: x,
                    help="Choose processing filters"
                )

                if not selected:
                    selected = ["LSCI"]
                st.session_state.selected_filters = selected

            with col2:
                # Add key to force re-render when toggling
                process_full = st.toggle(
                    "Full Image",
                    value=self.config.initial_settings["process_full_image"],
                    help="Toggle between full/region processing",
                    key="process_full_toggle"  # Add unique key
                )

                # Clear visualization-related state when switching modes
                if process_full != st.session_state.get('previous_process_full', None):
                    if 'selected_pixel' in st.session_state:
                        del st.session_state.selected_pixel
                    st.session_state.previous_process_full = process_full

            # Pixel selection when not processing full image
            if not process_full:
                image_width, image_height = image.size
                kernel_size = st.session_state.get('kernel_size', 7)
                half_kernel = kernel_size // 2

                # Calculate valid coordinate ranges
                x_min, y_min = half_kernel, half_kernel
                x_max = image_width - half_kernel - 1
                y_max = image_height - half_kernel - 1

                # Add pixel selection with columns for better layout
                col1, col2 = st.columns(2)
                with col1:
                    pixel_x = st.number_input(
                        "Pixel X Position",
                        min_value=x_min,
                        max_value=x_max,
                        value=x_min,
                        step=1,
                        help=f"X coordinate ({x_min} to {x_max}). Adjusted for {
                            kernel_size}×{kernel_size} kernel"
                    )
                with col2:
                    pixel_y = st.number_input(
                        "Pixel Y Position",
                        min_value=y_min,
                        max_value=y_max,
                        value=y_min,
                        step=1,
                        help=f"Y coordinate ({y_min} to {y_max}). Adjusted for {
                            kernel_size}×{kernel_size} kernel"
                    )

                # Store the selected pixel in session state
                st.session_state.selected_pixel = (pixel_x, pixel_y)

                # Show pixel info and math explanation in tabs
                tab1, tab2 = st.tabs(["📍 Pixel Info", "📐 Math Explanation"])
                
                with tab1:
                    st.info(
                        f"""
                        📍 Pixel Information:
                        • Center Position: ({pixel_x}, {pixel_y})
                        • Kernel Window: {kernel_size}×{kernel_size} pixels
                        • Valid Range: X [{x_min}-{x_max}], Y [{y_min}-{y_max}]
                        """,
                        icon="ℹ️"
                    )
                
                with tab2:
                    from .math_explainer import MathExplainer, MathExplainerConfig
                    from utils.latex import SPECKLE_FORMULA_CONFIG
                    
                    # Convert image to array if needed
                    if isinstance(image, Image.Image):
                        img_array = np.array(image.convert('L'), dtype=np.float32) / 255.0
                    else:
                        img_array = image
                        
                    math_explainer = MathExplainer(
                        MathExplainerConfig(
                            formula_config=SPECKLE_FORMULA_CONFIG,
                            kernel_size=kernel_size,
                            selected_pixel=st.session_state.selected_pixel,
                            image_array=img_array
                        )
                    )
                    math_explainer.render()

        # Update settings
        self.config.on_settings_changed({
            "process_full_image": process_full,
            "selected_region": None  # We're not using region selection anymore
        })
