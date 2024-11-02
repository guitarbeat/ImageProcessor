# Standard library imports
from typing import Dict, Any, Optional, Tuple

# Third-party imports
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

# Local application imports
from app.utils.config import AppConfig

# Component imports
from app.components.image_selector import (
    ImageSelector,
    ImageSelectorConfig
)
from app.components.sidebar import (
    Sidebar,
    SidebarConfig,
    DisplaySettings
)
from app.components.image_display import (
    ImageDisplay,
    ImageDisplayConfig
)
from app.components.processing_control import (
    ProcessingControl,
    ProcessingControlConfig
)
from app.components.processing_params import ProcessingParams

# Processor imports
from app.processors.spatial_filters import SpatialFilterProcessor

# Load configuration
config = AppConfig.load()


class ImageProcessor:
    """Handles image processing logic."""

    @staticmethod
    def process_region(
        kernel_size: int,
        filter_type: str,
        image: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
        max_pixel: Optional[Tuple[int, int]] = None,
        image_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Process image region with caching."""
        try:
            processor = SpatialFilterProcessor(
                kernel_size=kernel_size,
                filter_type=filter_type,
                chunk_size=1000
            )

            result = processor.process(
                image=image,
                region=region,
                max_pixel=max_pixel,
                image_id=image_id
            )

            if result is None or result.size == 0:
                st.error("Processing returned empty result")
                return None

            return result

        except Exception as e:
            st.error(f"Error in process_region: {str(e)}")
            return None

    @staticmethod
    def process_with_handling(
        input_array: np.ndarray,
        kernel_size: int,
        filter_type: str,
        region: Optional[Tuple[int, int, int, int]] = None,
        max_pixel: Optional[Tuple[int, int]] = None,
        image_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Process image with error handling."""
        try:
            return ImageProcessor.process_region(
                kernel_size=kernel_size,
                filter_type=filter_type,
                image=input_array,
                region=region,
                max_pixel=max_pixel,
                image_id=image_id
            )
        except Exception as e:
            st.error(f"Error during image processing: {str(e)}")
            return None


class UIState:
    """Manages UI state and callbacks."""

    DEFAULT_STATE = {
        'filter_type': "lsci",
        'selected_filters': ["LSCI", "Mean", "Standard Deviation"],
        'kernel_size': 7,
        'colormap': "gray",
        'process_full_image': True,
        'pixel_range': (0, 1000),
        'selected_coordinates': None,
        'selected_region': None,
        'initialized': True
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if not hasattr(st.session_state, 'initialized'):
            # Set defaults from config where applicable
            defaults = {
                **self.DEFAULT_STATE,
                'show_colorbar': self.config.ui.show_colorbar,
                'show_stats': self.config.ui.show_stats
            }

            for key, value in defaults.items():
                st.session_state.setdefault(key, value)

    @staticmethod
    def on_image_selected(image: Image.Image, name: str) -> None:
        st.session_state.image = image
        st.session_state.image_name = name

    @staticmethod
    def on_colormap_changed(colormap: str) -> None:
        st.session_state.colormap = colormap

    @staticmethod
    def on_display_settings_changed(settings: DisplaySettings) -> None:
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
        st.session_state.kernel_size = params.lsci.kernel_size


class DisplayMode:
    """Handle different display modes."""

    @staticmethod
    def _create_progress_bar(total_filters: int) -> Tuple[st.progress, str]:
        """Create and return a progress bar with initial text."""
        progress_text = "Processing filters..."
        progress_bar = st.progress(0, text=progress_text)
        return progress_bar, progress_text

    @staticmethod
    def _update_progress(progress_bar: st.progress, idx: int, total: int, filter_type: str) -> None:
        """Update progress bar with current progress."""
        progress_bar.progress(
            (idx/total),
            text=f"Processing {filter_type}..."
        )

    @staticmethod
    def render_side_by_side(input_display: ImageDisplay,
                            processed_displays: Dict[str, ImageDisplay],
                            input_image: Image.Image,
                            processed_images: Dict[str, Image.Image]):
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
    def render_comparison(input_image: Image.Image,
                          processed_images: Dict[str, Image.Image]):
        """Render interactive comparison between selected images."""
        if len(processed_images) > 1:
            # Allow user to select which images to compare
            image1_key = st.selectbox(
                "First Image", ["Input"] + list(processed_images.keys()))
            image2_key = st.selectbox(
                "Second Image", ["Input"] + list(processed_images.keys()))

            img1 = input_image if image1_key == "Input" else processed_images[image1_key]
            img2 = input_image if image2_key == "Input" else processed_images[image2_key]

            image_comparison(
                img1=img1,
                img2=img2,
                label1=image1_key,
                label2=image2_key,
                width=input_image.width,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
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
                in_memory=True
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
            layout=self.config.layout,
            initial_sidebar_state="collapsed"
        )

    def render_sidebar(self) -> None:
        """Render sidebar components."""
        # Get current image from session state
        image = st.session_state.get('image')

        sidebar = Sidebar(SidebarConfig(
            title=self.config.title,
            colormaps=self.config.ui.colormaps,
            on_colormap_changed=self.ui_state.on_colormap_changed,
            on_display_settings_changed=self.ui_state.on_display_settings_changed,
            on_params_changed=self.ui_state.on_params_changed,
            initial_colormap=st.session_state.colormap
        ))
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

            image_selector = ImageSelector(ImageSelectorConfig(
                sample_images_path=self.config.sample_images_path,
                on_image_selected=self.ui_state.on_image_selected,
                thumbnail_size=(150, 150)
            ))
            image_selector.render()

    def _render_processing_section(self) -> None:
        """Render processing control section."""
        with st.expander("Processing Control", expanded=True):
            if st.session_state.image is not None:
                processing_control = ProcessingControl(ProcessingControlConfig(
                    on_settings_changed=self.ui_state.on_processing_settings_changed,
                    initial_settings={
                        "process_full_image": st.session_state.process_full_image,
                        "selected_region": st.session_state.selected_region
                    },
                    display_settings=DisplaySettings(
                        show_colorbar=st.session_state.show_colorbar,
                        show_stats=st.session_state.show_stats
                    )
                ))
                processing_control.render(st.session_state.image)
            else:
                st.info("Load an image to access processing controls", icon="ℹ️")

    def _render_display_section(self) -> None:
        """Render image display section."""
        if st.session_state.image is not None:
            display_mode = st.radio(
                "Display Mode",
                ["Side by Side", "Interactive Comparison"],
                horizontal=True
            )

            input_image = self._get_input_image()
            if input_image is not None:
                processed_images = self._process_filters(input_image)

                if processed_images:
                    if display_mode == "Side by Side":
                        input_display, processed_displays = self._create_displays()
                        self.display_mode.render_side_by_side(
                            input_display,
                            processed_displays,
                            input_image,
                            processed_images
                        )
                    else:
                        self.display_mode.render_comparison(
                            input_image,
                            processed_images
                        )

    def _create_display_config(self, title: str) -> ImageDisplayConfig:
        """Create consistent display configuration."""
        return ImageDisplayConfig(
            colormap=st.session_state.colormap,
            title=title,
            show_colorbar=st.session_state.show_colorbar,
            show_stats=st.session_state.show_stats
        )

    def _create_displays(self) -> Tuple[ImageDisplay, Dict[str, ImageDisplay]]:
        """Create input and processed image displays."""
        input_display = ImageDisplay(
            self._create_display_config("Input Image"))

        processed_displays = {
            filter_type: ImageDisplay(
                self._create_display_config(f"{filter_type} Image"))
            for filter_type in st.session_state.selected_filters
        }

        return input_display, processed_displays

    def _process_filters(self, input_image: Image.Image) -> Dict[str, Image.Image]:
        """Process all selected filters with progress tracking."""
        processed_images = {}
        total_filters = len(st.session_state.selected_filters)

        progress_bar, _ = DisplayMode._create_progress_bar(total_filters)

        for idx, filter_type in enumerate(st.session_state.selected_filters):
            DisplayMode._update_progress(
                progress_bar, idx, total_filters, filter_type)
            processed_result = self._process_image(input_image, filter_type)
            if processed_result is not None:
                processed_images[filter_type] = processed_result

        progress_bar.progress(1.0, text="Processing complete!")
        return processed_images

    def _get_input_image(self) -> Optional[Image.Image]:
        """Get the input image for processing."""
        if not st.session_state.process_full_image and st.session_state.selected_region:
            x1, y1, x2, y2 = st.session_state.selected_region
            return st.session_state.image.crop((x1, y1, x2, y2))
        return st.session_state.image

    def _process_image(self, input_image: Image.Image, filter_type: str) -> Optional[Image.Image]:
        """Process the input image with specified filter."""
        try:
            input_array = np.array(input_image.convert('L'), dtype=np.float32) / 255.0

            # Get max pixel if not processing full image
            max_pixel = None
            if not st.session_state.process_full_image:
                if st.session_state.get('selected_pixel'):
                    max_pixel = st.session_state.selected_pixel

            processed_array = ImageProcessor.process_with_handling(
                input_array=input_array,
                kernel_size=st.session_state.kernel_size,
                filter_type=filter_type.lower(),
                max_pixel=max_pixel,
                image_id=st.session_state.get('current_image_id', '')
            )

            if processed_array is not None:
                # Apply colormap before converting to PIL Image
                from app.utils.image_processing import apply_colormap
                return apply_colormap(processed_array, st.session_state.colormap)
                
            return None
        except Exception as e:
            st.error(f"Error in _process_image: {str(e)}")
            return None


def main():
    """Main application entry point."""
    app = ImageProcessingApp(config)
    app.setup_page()
    app.render_sidebar()
    app.render_main_content()


if __name__ == "__main__":
    main()
