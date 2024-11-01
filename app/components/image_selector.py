"""
Component for image selection (upload or preloaded).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import streamlit as st
from streamlit_image_select import image_select
from PIL import Image

from components import Component
from utils import get_image_files

@dataclass
class ImageSelectorConfig:
    """Configuration for image selector component."""
    sample_images_path: Path
    on_image_selected: Callable[[Image.Image, str], None]
    allowed_types: list[str] = None
    thumbnail_size: tuple[int, int] = (80, 80)  # Size for image grid thumbnails

    def __post_init__(self):
        self.allowed_types = self.allowed_types or ["png", "jpg", "jpeg", "tif", "tiff"]

class ImageSelector(Component):
    """Component for selecting or uploading images."""
    
    def __init__(self, config: ImageSelectorConfig):
        self.config = config
        
    def render(self) -> None:
        """Render the image selection interface."""
        sample_tab, upload_tab = st.tabs(["📸 Sample Images", "📤 Upload Image"])
        
        with sample_tab:
            self._render_sample_images()
            
        with upload_tab:
            self._render_upload_section()

    def _render_upload_section(self) -> None:
        """Render the upload image section."""
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=self.config.allowed_types,
            help="Upload your own image file"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                # Set unique image ID
                st.session_state.current_image_id = f"{uploaded_file.name}_{hash(uploaded_file.getvalue())}"
                # Clear previous selections
                st.session_state.selected_region = None
                st.session_state.processed_regions = set()
                self.config.on_image_selected(image, uploaded_file.name)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    @staticmethod
    def _create_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Create a thumbnail maintaining aspect ratio."""
        # Calculate aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        
        # Determine new dimensions maintaining aspect ratio
        max_size = size[0]  # Use the first dimension as max
        if width > height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
            
        # Create new image with white background
        thumb = Image.new('RGB', size, 'white')
        # Resize original image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Calculate position to center the image
        x = (size[0] - new_width) // 2
        y = (size[1] - new_height) // 2
        # Paste resized image onto white background
        thumb.paste(resized, (x, y))
        
        return thumb

    @staticmethod
    def _create_thumbnail_cached(image_path: str, size: tuple[int, int]) -> Image.Image:
        """Create and cache thumbnails."""
        with Image.open(image_path) as img:
            return ImageSelector._create_thumbnail(img, size)

    def _render_sample_images(self) -> None:
        """Render the sample images section."""
        try:
            image_files = self._get_sample_images()
            
            if not image_files:
                st.warning(f"No sample images found in {self.config.sample_images_path}")
                return
            
            # Create list of images and their thumbnails
            images = []
            captions = []
            
            # Use cached thumbnails
            for img_path in image_files:
                try:
                    thumb = self._create_thumbnail_cached(
                        str(img_path), 
                        self.config.thumbnail_size
                    )
                    images.append(thumb)
                    captions.append(img_path.name)
                except Exception as e:
                    st.error(f"Error loading image {img_path.name}: {str(e)}")
            
            if images:
                selected_idx = image_select(
                    "Select an image",
                    images=images,
                    captions=captions,
                    use_container_width=False,
                    return_value="index"
                )
                
                # Load full image only when selected
                if selected_idx is not None and selected_idx != st.session_state.get("last_selected_idx"):
                    selected_path = image_files[selected_idx]
                    try:
                        image = Image.open(selected_path)
                        st.session_state.last_selected_idx = selected_idx
                        # Set unique image ID
                        st.session_state.current_image_id = f"{selected_path.name}_{hash(selected_path.read_bytes())}"
                        # Clear previous selections
                        st.session_state.selected_region = None
                        st.session_state.processed_regions = set()
                        self.config.on_image_selected(image, selected_path.name)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
                    
        except FileNotFoundError as e:
            st.error(f"Sample images directory not found: {e}")
        except Exception as e:
            st.error(f"Error loading sample images: {e}")

    def _get_sample_images(self) -> list[Path]:
        """Get list of sample images."""
        return get_image_files(self.config.sample_images_path)