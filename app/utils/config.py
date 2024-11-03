"""Configuration management for the application."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AppConfig:
    """Application configuration."""

    title: str
    description: str
    theme: str
    layout: str
    page_icon: str
    logo: str
    paths: dict
    algorithms: dict
    processing: dict
    ui: dict

    @classmethod
    def load(cls, config_path: str = "app/config.yaml") -> "AppConfig":
        """Load configuration from YAML file."""
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            return cls(
                title=config_data["app"]["title"],
                description=config_data["app"]["description"],
                theme=config_data["app"]["theme"],
                layout=config_data["app"]["layout"],
                page_icon=config_data["app"]["page_icon"],
                logo=config_data["app"]["logo"],
                paths=config_data["paths"],
                algorithms=config_data["algorithms"],
                processing=config_data["processing"],
                ui=config_data["ui"],
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise Exception(f"Error loading configuration: {e}")

    @property
    def sample_images_path(self) -> Path:
        """Get the path to sample images directory."""
        return Path(self.paths["sample_images"])

    @property
    def latex_path(self) -> Path:
        """Get the path to LaTeX assets directory."""
        return Path(self.paths["latex"])

    def get_algorithm_config(self, name: str) -> dict:
        """Get configuration for specific algorithm."""
        return self.algorithms.get(name, {})

    def get_ui_config(self) -> dict:
        """Get UI-specific configuration."""
        return self.ui

    def get_processing_config(self) -> dict:
        """Get processing-specific configuration."""
        return self.processing
