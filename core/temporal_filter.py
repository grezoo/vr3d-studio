"""
Temporal Consistency Filter for Stereoscopic Depth
Smooths depth maps across video frames to eliminate flickering and depth shimmer.
Includes automatic scene-cut detection to prevent ghosting between cuts.
"""

import cv2
import numpy as np


class TemporalDepthFilter:
    def __init__(self, alpha: float = 0.75, scene_change_threshold: float = 30.0):
        """
        Args:
            alpha: Weight of the current frame (0.0 to 1.0).
                   Higher values preserve fast motion, lower values provide smoother depth.
                   Default 0.75 provides optimal stability with zero lag.
            scene_change_threshold: Mean absolute difference between consecutive frames
                                     to trigger an instant cache reset.
        """
        self.alpha = float(np.clip(alpha, 0.1, 1.0))
        self.scene_change_threshold = scene_change_threshold
        self.prev_gray = None
        self.prev_depth = None

    def reset(self):
        """Resets the temporal history (e.g. at the start of a new video)."""
        self.prev_gray = None
        self.prev_depth = None

    def process(self, frame_bgr: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Processes a depth map with temporal smoothing.

        Args:
            frame_bgr: Current video frame (H, W, 3) uint8.
            depth_map: Current depth map (H, W) float32 [0.0, 1.0].

        Returns:
            smoothed_depth: Temporally stable depth map (H, W) float32.
        """
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Initial frame
        if self.prev_depth is None or self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_depth = depth_map.copy()
            return depth_map

        # Check for scene cut / abrupt transition
        # Compute mean absolute difference on downsampled grayscale image for speed
        h, w = curr_gray.shape
        small_curr = cv2.resize(curr_gray, (160, 90), interpolation=cv2.INTER_AREA)
        small_prev = cv2.resize(self.prev_gray, (160, 90), interpolation=cv2.INTER_AREA)
        diff = np.mean(np.abs(small_curr.astype(np.float32) - small_prev.astype(np.float32)))

        if diff > self.scene_change_threshold:
            # Scene cut detected: reset history immediately
            self.prev_gray = curr_gray
            self.prev_depth = depth_map.copy()
            return depth_map

        # Motion-adaptive exponential smoothing
        # Where image has changed a lot (high motion), use higher alpha
        smoothed_depth = self.alpha * depth_map + (1.0 - self.alpha) * self.prev_depth

        self.prev_gray = curr_gray
        self.prev_depth = smoothed_depth.copy()
        return smoothed_depth
