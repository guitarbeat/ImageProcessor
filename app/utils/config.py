"""
Configuration module for the Image Processing Explorer application.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml

@dataclass
class UIStyle:
    """UI style configuration."""
    kernel_outline_color: str
    kernel_center_pixel_color: str
    search_window_color: str
    pixel_text_color: str
    pixel_font_size: str

@dataclass
class UIConfig:
    """UI configuration."""
    max_image_size: int
    sidebar_width: str
    default_view: str
    colormaps: List[str]
    style: UIStyle
    show_colorbar: bool = False
    show_stats: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'UIConfig':
        return cls(
            max_image_size=config_dict["max_image_size"],
            sidebar_width=config_dict["sidebar_width"],
            default_view=config_dict["default_view"],
            colormaps=config_dict["colormaps"],
            style=UIStyle(**config_dict["style"]),
            show_colorbar=config_dict.get("display", {}).get("show_colorbar", False),
            show_stats=config_dict.get("display", {}).get("show_stats", False)
        )

@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Application settings
    title: str
    description: str
    theme: str
    layout: str
    page_icon: Path
    
    # Paths
    sample_images_path: Path
    latex_path: Path
    
    # UI settings
    ui: UIConfig

    @classmethod
    def load(cls) -> 'AppConfig':
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        # Create UI Style config
        style = UIStyle(**config_dict["ui"]["style"])
        
        # Create UI config
        ui_config = UIConfig.from_dict(config_dict["ui"])
        
        return cls(
            title=config_dict["app"]["title"],
            description=config_dict["app"]["description"],
            theme=config_dict["app"]["theme"],
            layout=config_dict["app"]["layout"],
            page_icon=Path(config_dict["app"]["page_icon"]),
            sample_images_path=Path(config_dict["paths"]["sample_images"]),
            latex_path=Path(config_dict["paths"]["latex"]),
            ui=ui_config
        ) 