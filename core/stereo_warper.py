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
        ipd_offset: float = 0.032,
        convergence: float = 0.0,
        fill_holes: bool = True,
        swap_eyes: bool = False,
        auto_convergence: bool = True
    ):
        """
        Generates left and right eye stereoscopic pair with occlusion protection and Auto-Convergence.
        Anchoring the background disparity to zero eliminates backward sampling leaks,
        guaranteeing zero ghosting on backgrounds and silky smooth contours without inpainting bites.
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

        # Background anchor plane: anchoring disparity at the background prevents
        # the background from negative-sampling foreground objects
        if auto_convergence:
            bg_anchor = float(np.percentile(depth_map, 5))
        else:
            bg_anchor = convergence

        # Maximum pixel shift
        max_shift = w * ipd_offset
        # Non-negative disparity: background remains steady, subjects pop forward into natural 3D
        disparity = np.maximum(0.0, depth_map - bg_anchor) * max_shift

        # 2-Layer Disocclusion Synthesis:
        # Protect contours and disocclusion zones (elbows, limbs) from ghosting
        fg_mask = (depth_map > (bg_anchor + 0.08)).astype(np.uint8) * 255
        inpaint_mask = cv2.dilate(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        bg_inpainted = cv2.inpaint(image_bgr, inpaint_mask, 7, cv2.INPAINT_TELEA)
        fg_alpha = cv2.GaussianBlur(fg_mask.astype(np.float32) / 255.0, (5, 5), 0)[..., None]

        # Coordinate grid
        if self._cached_shape != (w, h):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._cached_grid = (gx, gy)
            self._cached_shape = (w, h)
        grid_x, grid_y = self._cached_grid

        # Disparity map for Left Eye (-0.5 * disparity) and Right Eye (+0.5 * disparity)
        shift_left = -0.5 * disparity
        shift_right = 0.5 * disparity

        left_eye = self._render_layered_view(image_bgr, bg_inpainted, fg_alpha, shift_left, grid_x, grid_y)
        right_eye = self._render_layered_view(image_bgr, bg_inpainted, fg_alpha, shift_right, grid_x, grid_y)

        if swap_eyes:
            return right_eye, left_eye
        return left_eye, right_eye

    def _render_layered_view(
        self,
        fg_image: np.ndarray,
        bg_image: np.ndarray,
        fg_alpha: np.ndarray,
        shift_map: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray
    ) -> np.ndarray:
        """
        Renders an eye view by warping the foreground layer over the complete background layer.
        Eliminates duplicate limb ghosts (elbows, shoulders, legs) while preserving silky subpixel
        anti-aliasing and razor-sharp original photo sharpness.
        """
        map_x = grid_x - shift_map

        warped_fg = cv2.remap(
            fg_image,
            map_x.astype(np.float32),
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        warped_alpha = cv2.remap(
            fg_alpha,
            map_x.astype(np.float32),
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        if warped_alpha.ndim == 2:
            warped_alpha = warped_alpha[..., None]

        composite = (warped_fg * warped_alpha + bg_image * (1.0 - warped_alpha)).astype(np.uint8)
        return composite

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
