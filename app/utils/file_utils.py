"""File utility functions."""

from pathlib import Path
from typing import List

from app.utils.constants import ALLOWED_IMAGE_EXTENSIONS


def get_image_files(directory: Path) -> List[Path]:
    """Get list of image files from directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_files: List[Path] = []
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        image_files.extend(list(directory.glob(f"*{ext}")))

    # Sort for consistent ordering
    image_files.sort()
    return image_files
