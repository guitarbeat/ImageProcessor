# Standard library imports
from typing import Any, Callable, Dict, Literal, Optional, Tuple

# Third-party imports
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison  # type: ignore

# Processor imports
from app.processors.filters import SpatialFilterProcessor
from app.ui.components.image_display import ImageDisplay, ImageDisplayConfig

# Component imports
from app.ui.components.image_selector import ImageSelector, ImageSelectorConfig
from app.ui.components.processing_control import (
    ProcessingControl,
    ProcessingControlConfig,
)
from app.ui.components.processing_params import ProcessingParams
from app.ui.components.sidebar import DisplaySettings, Sidebar, SidebarConfig

# Local application imports
from app.utils.config import AppConfig
from app.utils.image_processing import apply_colormap, preprocess_image

# Load configuration
config = AppConfig.load()


class ImageProcessor:
    """Handles image processing logic."""

    @staticmethod
    def process_image(
        image: np.ndarray,
        kernel_size: int,
        filter_type: Literal["lsci", "nlm", "mean", "std_dev"],
        filter_strength: float = 10.0,
        search_window_size: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional[np.ndarray]:
        """Process image with specified filter."""
        try:
            processor = SpatialFilterProcessor(
                kernel_size=kernel_size,
                filter_type=filter_type,
                filter_strength=filter_strength,
                search_window_size=search_window_size,
            )

            result = processor.process(image=image, progress_callback=progress_callback)

            if result is None or result.size == 0:
                st.error("Processing returned empty result")
                return None

            return result

        except Exception as e:
            st.error(f"Error in process_image: {str(e)}")
            return None


class UIState:
    """Manages UI state and callbacks."""

    DEFAULT_STATE = {
        "filter_type": "lsci",
        "kernel_size": 7,
        "colormap": "gray",
        "process_full_image": True,
        "pixel_range": (0, 1000),
        "selected_coordinates": None,
        "selected_region": None,
        "filter_strength": 10.0,
        "use_full_image": True,
        "search_size": 21,
        "show_colorbar": True,
        "show_stats": True,
        "initialized": True,
        "selected_filters": [],
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if not hasattr(st.session_state, "initialized"):
            # Set defaults from config where applicable
            defaults = {**self.DEFAULT_STATE}
            
            # Add selected_region to defaults if not present
            if "selected_region" not in defaults:
                defaults["selected_region"] = None

            # Override with config values if they exist
            if "show_colorbar" in self.config.ui:
                defaults["show_colorbar"] = self.config.ui["show_colorbar"]
            if "show_stats" in self.config.ui:
                defaults["show_stats"] = self.config.ui["show_stats"]

            for key, value in defaults.items():
                st.session_state.setdefault(key, value)

    @staticmethod
    def on_image_selected(image: Image.Image, name: str) -> None:
        """Handle image selection."""
        # Preprocess and store image
        processed_image, img_array = preprocess_image(image)
        st.session_state.image = processed_image
        st.session_state.image_array = img_array
        st.session_state.image_name = name

    @staticmethod
    def on_colormap_changed(colormap: str) -> None:
        """Handle colormap changes."""
        st.session_state.colormap = colormap

    @staticmethod
    def on_display_settings_changed(settings: DisplaySettings) -> None:
        """Handle display settings changes."""
        if settings.show_colorbar != st.session_state.show_colorbar:
            st.session_state.show_colorbar = settings.show_colorbar
        if settings.show_stats != st.session_state.show_stats:
            st.session_state.show_stats = settings.show_stats

    @staticmethod
    def on_processing_settings_changed(settings: Dict[str, Any]) -> None:
        """Handle processing settings changes."""
        st.session_state.process_full_image = settings["process_full_image"]
        st.session_state.selected_region = settings["selected_region"]

    @staticmethod
    def on_params_changed(params: ProcessingParams) -> None:
        """Handle parameter changes."""
        # Update common parameters
        st.session_state.kernel_size = params.common.kernel_size
        st.session_state.filter_type = params.filter_type

        # Update NLM parameters if applicable
        if params.filter_type == "nlm":
            st.session_state.filter_strength = params.nlm.filter_strength
            st.session_state.search_window_size = params.nlm.search_window_size
            st.session_state.use_full_image = params.nlm.search_window_size is None


class DisplayMode:
    """Handle different display modes."""

    @staticmethod
    def create_progress_container(filter_type: str) -> tuple[Any, Any, Any]:
        """Create a container for progress bar and status text."""
        container = st.empty()
        with container:
            col1, col2 = st.columns([4, 1])
            with col1:
                progress_bar = st.progress(0, text=f"Processing {filter_type}...")
            with col2:
                status = st.empty()
        return progress_bar, status, container

    @staticmethod
    def render_side_by_side(
        input_display: ImageDisplay,
        processed_displays: Dict[str, ImageDisplay],
        input_image: Image.Image,
        processed_images: Dict[str, Image.Image],
    ) -> None:
        """Render all images side by side."""
        num_images = len(processed_images) + 1  # +1 for input image
        cols = st.columns(num_images)

        # Display input image
        with cols[0]:
            st.markdown("#### Input Image")
            input_display.render(input_image)

        # Display processed images
        for i, (filter_name, image) in enumerate(processed_images.items(), 1):
            with cols[i]:
                st.markdown(f"#### {filter_name}")
                processed_displays[filter_name].render(image)

    @staticmethod
    def render_comparison(
        input_image: Image.Image, processed_images: Dict[str, Image.Image]
    ) -> None:
        """Render interactive comparison between selected images."""
        if len(processed_images) > 1:
            # Allow user to select which images to compare
            image1_key = st.selectbox(
                "First Image", ["Input"] + list(processed_images.keys())
            )
            image2_key = st.selectbox(
                "Second Image", ["Input"] + list(processed_images.keys())
            )

            img1 = (
                input_image if image1_key == "Input" else processed_images[image1_key]
            )
            img2 = (
                input_image if image2_key == "Input" else processed_images[image2_key]
            )

            image_comparison(
                img1=img1,
                img2=img2,
                label1=image1_key,
                label2=image2_key,
                width=input_image.width,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )
        else:
            # Fall back to simple comparison if only one processed image
            first_image = list(processed_images.values())[0]
            image_comparison(
                img1=input_image,
                img2=first_image,
                label1="Input Image",
                label2=list(processed_images.keys())[0],
                width=input_image.width,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )


class ImageProcessingApp:
    """Main application class."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.ui_state = UIState(config)
        self.display_mode = DisplayMode()

    def setup_page(self) -> None:
        """Setup page configuration and styling."""
        st.set_page_config(
            page_title=self.config.title,
            page_icon=str(self.config.page_icon),
            layout=(
                "wide" if self.config.layout == "wide" else "centered"
            ),  # Must be 'centered' or 'wide'
            initial_sidebar_state="collapsed",
        )

    def render_sidebar(self) -> None:
        """Render sidebar components."""
        # Get current image from session state
        image = st.session_state.get("image")

        # Get colormaps from config or use defaults
        colormaps = self.config.ui.get(
            "colormaps", ["gray", "viridis", "plasma", "inferno"]
        )

        sidebar = Sidebar(
            SidebarConfig(
                title=self.config.title,
                colormaps=colormaps,  # Use the colormaps from config or defaults
                on_colormap_changed=self.ui_state.on_colormap_changed,
                on_display_settings_changed=self.ui_state.on_display_settings_changed,
                on_params_changed=self.ui_state.on_params_changed,
                initial_colormap=st.session_state.colormap,
            )
        )
        sidebar.render(image)  # Pass the image from session state

    def render_main_content(self) -> None:
        """Render main content area."""
        self._render_input_section()
        self._render_processing_section()
        self._render_display_section()

    def _render_input_section(self) -> None:
        """Render image input section."""
        with st.expander("Image Input", expanded=False):
            st.info("📸 Select an image to process or upload your own", icon="ℹ️")

            image_selector = ImageSelector(
                ImageSelectorConfig(
                    sample_images_path=self.config.sample_images_path,
                    on_image_selected=self.ui_state.on_image_selected,
                    thumbnail_size=(150, 150),
                )
            )
            image_selector.render()

    def _render_processing_section(self) -> None:
        """Render image processing section."""
        if "image" not in st.session_state:
            return

        # Create processing control
        processing_control = ProcessingControl(
            ProcessingControlConfig(
                on_settings_changed=self.ui_state.on_processing_settings_changed,
                initial_settings=self.ui_state.DEFAULT_STATE,
                display_settings=DisplaySettings.from_session_state(),
            )
        )

        # Check if processing is needed
        needs_processing = (
            "processed_image" not in st.session_state
            # Default to True
            or st.session_state.get("needs_processing", True)
        )

        if needs_processing:
            # Process image with current settings
            self._process_current_image()
            # Clear the processing flag
            st.session_state.needs_processing = False

        # Render processing control
        processing_control.render(st.session_state.image)

    def _process_current_image(self) -> None:
        """Process the current image with current settings."""
        if "image_array" not in st.session_state:
            return

        try:
            # Get current settings
            filter_type = st.session_state.get("filter_type", "lsci")
            kernel_size = st.session_state.get("kernel_size", 7)
            filter_strength = st.session_state.get("filter_strength", 10.0)
            search_window_size = (
                None
                if st.session_state.get("use_full_image", True)
                else st.session_state.get("search_size", 21)
            )

            # Process image
            result = ImageProcessor.process_image(
                image=st.session_state.image_array,
                kernel_size=kernel_size,
                filter_type=filter_type,
                filter_strength=filter_strength,
                search_window_size=search_window_size,
            )

            if result is not None:
                st.session_state.processed_image = result

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    def _render_display_section(self) -> None:
        """Render image display section."""
        if st.session_state.get("image") is not None:
            display_mode = st.radio(
                "Display Mode",
                ["Side by Side", "Interactive Comparison"],
                horizontal=True,
            )

            input_image = st.session_state.image
            if input_image is not None:
                processed_images = self._process_filters(input_image)

                if processed_images:
                    if display_mode == "Side by Side":
                        input_display, processed_displays = self._create_displays()

                        # Create columns for each image
                        num_images = len(processed_images) + 1  # +1 for input image
                        cols = st.columns(num_images)

                        # Display input image
                        with cols[0]:
                            st.markdown("#### Input Image")
                            input_display.render(input_image)

                        # Display processed images
                        for i, (filter_name, image) in enumerate(
                            processed_images.items(), 1
                        ):
                            with cols[i]:
                                st.markdown(f"#### {filter_name}")
                                processed_displays[filter_name].render(image)
                    else:
                        self.display_mode.render_comparison(
                            input_image, processed_images
                        )

    def _create_display_config(self, title: str) -> ImageDisplayConfig:
        """Create consistent display configuration."""
        return ImageDisplayConfig(
            colormap=st.session_state.colormap,
            title=title,  # Restore title
            show_colorbar=st.session_state.show_colorbar,
            show_stats=st.session_state.show_stats,
        )

    def _create_displays(self) -> Tuple[ImageDisplay, Dict[str, ImageDisplay]]:
        """Create input and processed image displays."""
        input_display = ImageDisplay(self._create_display_config("Input Image"))

        # Create displays for each selected filter
        processed_displays = {
            filter_type: ImageDisplay(
                self._create_display_config(f"{filter_type} Image")
            )
            for filter_type in st.session_state.selected_filters
        }

        return input_display, processed_displays

    def _process_filters(self, input_image: Image.Image) -> Dict[str, Image.Image]:
        """Process image with selected filters."""
        processed_images = {}
        progress_containers = {}

        # Get selected filters from session state
        selected_filters = st.session_state.get("selected_filters", [])
        if not selected_filters:
            st.warning("No filters selected. Please select at least one filter.")
            return {}

        # Create progress bars for each filter
        for filter_type in selected_filters:
            progress_bar, status, container = DisplayMode.create_progress_container(
                filter_type
            )
            progress_containers[filter_type] = (progress_bar, status, container)

        # Get input array once
        input_array = st.session_state.get("image_array")
        if input_array is None:
            st.error("No image array found in session state")
            return {}

        # Process each filter
        for filter_type in selected_filters:
            progress_bar, status, container = progress_containers[filter_type]

            def progress_callback(progress: float, filter_name: str = filter_type):
                progress_bar.progress(progress, text=f"Processing {filter_name}...")
                if progress >= 1.0:
                    status.success("✓")

            # Process image
            try:
                result = ImageProcessor.process_image(
                    image=input_array,
                    kernel_size=st.session_state.kernel_size,
                    filter_type=filter_type.lower(),
                    filter_strength=st.session_state.get("filter_strength", 10.0),
                    search_window_size=st.session_state.get("search_window_size"),
                    progress_callback=progress_callback,
                )

                if result is not None:
                    # Apply colormap and store result
                    processed_images[filter_type] = apply_colormap(
                        result, st.session_state.colormap
                    )
                else:
                    st.error(f"Failed to process {filter_type}")
            except Exception as e:
                st.error(f"Error processing {filter_type}: {str(e)}")
            finally:
                # Clear progress container
                container.empty()

        return processed_images


def main():
    """Main application entry point."""
    app = ImageProcessingApp(config)
    app.setup_page()
    app.render_sidebar()
    app.render_main_content()


if __name__ == "__main__":
    main()
