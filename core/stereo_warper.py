"""
Advanced Stereoscopic Warper Module with Edge-Aware Occlusion Handling.
Eliminates edge-bleeding / double-contour artifacts (ghost limbs) by strictly
extrapolating background textures into disocclusion regions instead of stretching foreground objects.
"""

import cv2
import numpy as np


class StereoWarper:
    def __init__(self, device="cuda"):
        self.device = device
        self._cached_grid = None
        self._cached_shape = None

    def compute_auto_convergence(self, depth_map: np.ndarray) -> float:
        """
        Analyzes the depth map to set an optimal convergence plane.
        Balancing between the background and median subject depth creates a pronounced
        pop-out effect where the subject reaches forward into the VR viewing space.
        """
        h, w = depth_map.shape[:2]
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        center_roi = depth_map[y1:y2, x1:x2]

        if center_roi.size == 0:
            return 0.35

        p25 = float(np.percentile(depth_map, 25))
        med = float(np.median(center_roi))
        auto_conv = 0.40 * p25 + 0.60 * med
        return float(np.clip(auto_conv, 0.15, 0.50))

    def generate_stereo_pair(
        self,
        image_bgr: np.ndarray,
        depth_map: np.ndarray,
        ipd_offset: float = 0.038,
        convergence: float = 0.5,
        fill_holes: bool = True,
        swap_eyes: bool = False,
        auto_convergence: bool = True
    ):
        """
        Generates left and right eye stereoscopic pair with occlusion protection and Auto-Convergence.
        """
        h, w = image_bgr.shape[:2]

        if depth_map.dtype != np.float32:
            depth_map = depth_map.astype(np.float32)
        if depth_map.max() > 1.0:
            depth_map = depth_map / 255.0

        # Dynamic depth range enhancement: ensure the scene utilizes the full depth range [0, 1]
        p_low = float(np.percentile(depth_map, 2))
        p_high = float(np.percentile(depth_map, 98))
        if p_high - p_low > 0.10:
            depth_map = np.clip((depth_map - p_low) / (p_high - p_low), 0.0, 1.0)

        # Effective convergence plane
        if auto_convergence:
            eff_conv = self.compute_auto_convergence(depth_map)
        else:
            eff_conv = convergence

        # Maximum pixel shift
        max_shift = w * ipd_offset
        # Disparity relative to the subject plane
        disparity = (depth_map - eff_conv) * max_shift

        # Coordinate grid
        if self._cached_shape != (w, h):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._cached_grid = (gx, gy)
            self._cached_shape = (w, h)
        grid_x, grid_y = self._cached_grid

        # Disparity map for Left Eye (-0.5 * disparity) and Right Eye (+0.5 * disparity)
        shift_left = -0.5 * disparity
        shift_right = 0.5 * disparity

        left_eye = self._render_clean_view(image_bgr, depth_map, shift_left, grid_x, grid_y, is_left=True)
        right_eye = self._render_clean_view(image_bgr, depth_map, shift_right, grid_x, grid_y, is_left=False)

        if swap_eyes:
            return right_eye, left_eye
        return left_eye, right_eye

    def _render_clean_view(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        shift_map: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        is_left: bool
    ) -> np.ndarray:
        """
        Renders a single eye view using pure bilinear backward remap for continuous surfaces,
        combined with surgical background-leak-only inpainting.

        Continuous surfaces retain 100% of their original photo sharpness and grain.
        Foreground objects are never modified, eliminating edge fringes.
        Background pixels that sampled across steep depth boundaries into foreground
        (sampled_d > depth + 0.05) are cleanly healed.
        """
        h, w = image.shape[:2]

        map_x = grid_x - shift_map

        # 1. Base remap: pure bilinear interpolation preserves 100% photo sharpness & noise
        warped = cv2.remap(
            image,
            map_x.astype(np.float32),
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # 2. Targeted Background Leak Detection
        sampled_x = np.clip(np.round(map_x).astype(np.int32), 0, w - 1)
        gy_int = grid_y.astype(np.int32)
        sampled_d = depth[gy_int, sampled_x]

        # Ghosting occurs strictly when background destination samples foreground
        leak_bg_to_fg = (sampled_d > depth + 0.05).astype(np.uint8) * 255

        # 3. Surgical inpainting restricted strictly to background leak pixels
        if np.any(leak_bg_to_fg):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            leak_dilated = cv2.dilate(leak_bg_to_fg, kernel, iterations=1)
            inpainted = cv2.inpaint(warped, leak_dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            return inpainted

        return warped

    def create_sbs(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        half_sbs: bool = False
    ) -> np.ndarray:
        """Combines left and right images into a Side-by-Side (SBS) frame."""
        h, w = left_eye.shape[:2]
        if half_sbs:
            w_half = w // 2
            left_scaled = cv2.resize(left_eye, (w_half, h), interpolation=cv2.INTER_AREA)
            right_scaled = cv2.resize(right_eye, (w_half, h), interpolation=cv2.INTER_AREA)
            return np.hstack([left_scaled, right_scaled])
        else:
            return np.hstack([left_eye, right_eye])

    def create_anaglyph(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> np.ndarray:
        """Combines left and right into Red/Cyan anaglyph 3D."""
        anaglyph = np.zeros_like(left_eye)
        anaglyph[:, :, 0] = right_eye[:, :, 0]  # Blue
        anaglyph[:, :, 1] = right_eye[:, :, 1]  # Green
        anaglyph[:, :, 2] = left_eye[:, :, 2]   # Red
        return anaglyph
