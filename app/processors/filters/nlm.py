"""Non-Local Means (NLM) filter implementation.

The NLM algorithm works by:
1. For each pixel (x,y), look at its surrounding patch
2. Compare this patch with patches around other pixels in a search window
3. Calculate weights based on how similar the patches are
4. Take a weighted average of all pixel values using these weights
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import streamlit as st

from app.processors.filters.utils import FilterComputation


class NLMComputation(FilterComputation):
    def __init__(
        self,
        kernel_size: int,
        filter_strength: float = 10.0,
        search_window_size: Optional[int] = None,
    ):
        """Initialize NLM computation.

        Args:
            kernel_size: Size of patches to compare (e.g., 7x7)
            filter_strength: Controls how much difference between patches is allowed
                           Higher values = more smoothing
            search_window_size: Size of region to search for similar patches
                              None means search the entire image
        """
        super().__init__(kernel_size)
        self.filter_strength = filter_strength
        self.search_window_size = search_window_size

    def extract_patch(self, image: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
        """Extract a patch from the image centered at (x, y)."""
        half = self.kernel_size // 2
        try:
            if half <= x < image.shape[1] - half and half <= y < image.shape[0] - half:
                return image[y - half: y + half + 1, x - half: x + half + 1]
        except Exception as e:
            st.error(f"Error extracting patch: {str(e)}")
        return None

    def compute_similarity_map(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """Compute similarity map for a given pixel position."""
        half_kernel = self.kernel_size // 2
        h2 = self.filter_strength**2

        # Get reference patch
        ref_patch = self.extract_patch(image, x, y)
        if ref_patch is None:
            return np.zeros((1, 1))

        # Define search window boundaries
        if self.search_window_size is None:
            # Use full valid image region
            height, width = image.shape
            search_area = {
                "y_start": half_kernel,
                "y_end": height - half_kernel,
                "x_start": half_kernel,
                "x_end": width - half_kernel,
            }
            # Create similarity map for valid region
            similarity_map = np.zeros(
                (height - 2 * half_kernel, width - 2 * half_kernel), dtype=np.float32
            )
        else:
            # Use limited search window
            half_search = self.search_window_size // 2
            search_area = {
                "y_start": max(half_kernel, y - half_search),
                "y_end": min(image.shape[0] - half_kernel, y + half_search + 1),
                "x_start": max(half_kernel, x - half_search),
                "x_end": min(image.shape[1] - half_kernel, x + half_search + 1),
            }
            height = search_area["y_end"] - search_area["y_start"]
            width = search_area["x_end"] - search_area["x_start"]
            similarity_map = np.zeros((height, width), dtype=np.float32)

        # Compute similarities
        for i in range(search_area["y_start"], search_area["y_end"]):
            for j in range(search_area["x_start"], search_area["x_end"]):
                comp_patch = self.extract_patch(image, j, i)
                if comp_patch is not None:
                    # Calculate patch difference
                    diff = np.sum((ref_patch - comp_patch) ** 2)
                    # Compute weight
                    weight = np.exp(-diff / h2)
                    # Store in similarity map with proper indexing
                    if self.search_window_size is None:
                        # For full image, use relative coordinates to valid region
                        map_y = i - search_area["y_start"]
                        map_x = j - search_area["x_start"]
                    else:
                        # For limited window, use relative coordinates
                        map_y = i - search_area["y_start"]
                        map_x = j - search_area["x_start"]

                    # Check bounds before assignment
                    if (
                        0 <= map_y < similarity_map.shape[0]
                        and 0 <= map_x < similarity_map.shape[1]
                    ):
                        similarity_map[map_y, map_x] = weight

        # Normalize weights
        max_sim = np.max(similarity_map)
        if max_sim > 0:
            similarity_map /= max_sim

        return similarity_map

    def compute(self, window: np.ndarray) -> float:
        """Compute NLM value for a single pixel."""
        try:
            # Get current image and selected pixel with proper validation
            image = st.session_state.get("current_image_array")
            selected_pixel = st.session_state.get("selected_pixel")

            if image is None or selected_pixel is None:
                # Fallback to center pixel if no valid context
                return float(window[window.shape[0] // 2, window.shape[1] // 2])

            # Safely unpack coordinates after validation
            x, y = selected_pixel

            # Get similarity weights
            similarity_map = self.compute_similarity_map(image, x, y)

            # Extract corresponding region from window
            half_kernel = self.kernel_size // 2
            window_region = window[
                half_kernel: window.shape[0] - half_kernel,
                half_kernel: window.shape[1] - half_kernel,
            ]

            # Ensure shapes match
            if similarity_map.shape != window_region.shape:
                # Crop similarity map to match window region
                h, w = window_region.shape
                similarity_map = similarity_map[:h, :w]

            # Compute weighted average
            weights_sum = np.sum(similarity_map)
            if weights_sum > 0:
                weighted_sum = np.sum(similarity_map * window_region)
                return float(weighted_sum / weights_sum)

            # Fallback to center pixel if no valid weights
            return float(window[window.shape[0] // 2, window.shape[1] // 2])

        except Exception as e:
            st.error(f"Error in NLM computation: {str(e)}")
            # Log additional context for debugging
            st.error(f"Window shape: {window.shape}")
            st.error(
                f"Selected pixel: {st.session_state.get('selected_pixel')}")
            # Fallback to center pixel
            return float(window[window.shape[0] // 2, window.shape[1] // 2])

    def process_image(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """Process entire image using NLM computation."""
        half = self.kernel_size // 2
        height, width = image.shape

        # Create result array for valid region only
        result = np.zeros(
            (height - 2 * half, width - 2 * half), dtype=np.float32)

        # Store full image for patch comparisons
        st.session_state.current_image_array = image

        # Process valid region only
        total_pixels = (height - 2 * half) * (width - 2 * half)
        processed = 0

        for y in range(half, height - half):
            for x in range(half, width - half):
                # Extract window
                window = image[y - half: y + half + 1, x - half: x + half + 1]

                # Compute NLM value using compute method
                result[y - half, x - half] = self.compute(window)

                processed += 1
                if progress_callback:
                    progress_callback(processed / total_pixels)

        return result

    def get_intermediate_values(self, window: np.ndarray) -> Dict[str, Any]:
        """Get all intermediate values used in NLM computation."""
        center_value = float(
            window[window.shape[0] // 2, window.shape[1] // 2])

        # Compute similarity map and normalization factor
        similarity_map = self.compute_similarity_map(
            st.session_state.get("current_image_array"),
            st.session_state.get("selected_pixel", (0, 0))[0],
            st.session_state.get("selected_pixel", (0, 0))[1],
        )
        norm_factor = float(np.sum(similarity_map))
        nlm_value = self.compute(window)

        # Update search window description
        if self.search_window_size is None:
            search_desc = "Using full image search"
        else:
            search_desc = f"Using {self.search_window_size}×{self.search_window_size} search window"

        return {
            "original_value": center_value,
            "filter_strength": self.filter_strength,
            "search_window_size": self.search_window_size,
            "nlm_value": nlm_value,
            "norm_factor": norm_factor,  # Add normalization factor
            "half_kernel": self.kernel_size // 2,
            "search_window_description": search_desc,
        }

    def get_formula_config(self) -> Dict[str, Any]:
        """Get NLM formula configuration."""
        from app.utils.latex import NLM_FORMULA_CONFIG

        return NLM_FORMULA_CONFIG

    def compute_normalization_factors(
        self, image: np.ndarray, x: int, y: int
    ) -> np.ndarray:
        """Compute normalization factors for the entire image."""
        half = self.kernel_size // 2
        height, width = image.shape
        norm_map = np.zeros_like(image)

        # Process valid region only
        for i in range(half, height - half):
            for j in range(half, width - half):
                # Get reference patch
                ref_patch = image[i - half: i +
                                  half + 1, j - half: j + half + 1]

                # Compute weights for all pixels
                weights_sum = 0.0

                # Define search range based on search window size
                if self.search_window_size is None:
                    y_start, y_end = half, height - half
                    x_start, x_end = half, width - half
                else:
                    half_search = self.search_window_size // 2
                    y_start = max(half, i - half_search)
                    y_end = min(height - half, i + half_search + 1)
                    x_start = max(half, j - half_search)
                    x_end = min(width - half, j + half_search + 1)

                # Compute weights for all pixels in search range
                for ni in range(y_start, y_end):
                    for nj in range(x_start, x_end):
                        if ni == i and nj == j:
                            weights_sum += 1.0  # Self-similarity
                            continue

                        # Get comparison patch
                        comp_patch = image[
                            ni - half: ni + half + 1, nj - half: nj + half + 1
                        ]

                        # Compute weight using Gaussian-weighted patch distance
                        distance = self.calculate_patch_distance(
                            ref_patch, comp_patch)
                        weight = np.exp(-distance / (self.filter_strength**2))
                        weights_sum += weight

                norm_map[i, j] = weights_sum

        return norm_map

    @staticmethod
    @np.vectorize
    def _compute_weight(diff: float, h2: float) -> float:
        """Vectorized weight computation."""
        return np.exp(-diff / h2)

    """
    Coordinate Systems:
    - Input Space (x,y): Original image coordinates
    - Output Space (i,j): Processed image coordinates, offset by half_kernel
    - Search Space (s,t): Coordinates within search window
    - Patch Space (p,q): Local coordinates within patches
    """
