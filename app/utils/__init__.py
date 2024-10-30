"""
Utility functions for the application.
"""
from pathlib import Path
from typing import List

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    # Make sure directory is a Path object
    directory = Path(directory)
    
    # Check if directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Get all image files with case-insensitive extensions
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    
    for ext in extensions:
        # Add both lowercase and uppercase variants
        image_files.extend(directory.glob(f"*{ext.lower()}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)  # Sort for consistent ordering
