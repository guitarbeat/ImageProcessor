"""Constants used throughout the application."""

# Processing constants
DEFAULT_KERNEL_SIZE = 7
DEFAULT_FILTER_TYPE = "lsci"
MAX_CACHE_SIZE = 1024

# UI constants
DEFAULT_THUMBNAIL_SIZE = (150, 150)
DEFAULT_DISPLAY_SIZE = (6, 6)
DEFAULT_COLORMAP = "gray"

# File extensions
ALLOWED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# Add more meaningful constants
DISPLAY_MODES = ["Side by Side", "Comparison Slider"]
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds
MAX_PROCESSABLE_PIXELS = 1_000_000
DEFAULT_DISPLAY_WIDTH = 700
DEFAULT_COMPARISON_POSITION = 50