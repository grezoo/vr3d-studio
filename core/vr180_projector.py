"""
VR180 Projector Module
Maps rectilinear stereoscopic pairs into 180-degree Equirectangular / Hemispherical VR format.
Outputs Side-by-Side (SBS) 180° VR frames compatible with Meta Quest, Pico, Skybox VR, DeoVR, and YouTube VR.
"""

import cv2
import numpy as np


class VR180Projector:
    def __init__(self, output_eye_size=(1920, 1920), h_fov_deg=110.0):
        """
        Args:
            output_eye_size: (width, height) per eye in the VR180 equirectangular canvas.
                             Standard VR180 SBS output will be (2 * width, height).
            h_fov_deg: Horizontal Field of View (in degrees) that the source image occupies
                       inside the 180° dome (typically 90° - 120° for natural perspective).
        """
        self.eye_width, self.eye_height = output_eye_size
        self.h_fov_deg = h_fov_deg
        self.cached_maps = None
        self.last_src_shape = None

    def _build_remap_tables(self, src_w: int, src_h: int):
        """
        Precomputes the equirectangular-to-rectilinear remap lookup tables for cv2.remap.
        """
        ew, eh = self.eye_width, self.eye_height

        # Equirectangular coordinates:
        # u in [0, 1] maps to longitude lambda in [-pi/2, +pi/2] (left to right: -90° to +90°)
        # v in [0, 1] maps to latitude phi in [+pi/2, -pi/2] (top to bottom: +90° to -90°)
        u = np.linspace(0.0, 1.0, ew, dtype=np.float32)
        v = np.linspace(0.0, 1.0, eh, dtype=np.float32)
        grid_u, grid_v = np.meshgrid(u, v)

        lon = (grid_u - 0.5) * np.pi      # -pi/2 to +pi/2 (left to right)
        lat = (0.5 - grid_v) * np.pi      # +pi/2 to -pi/2 (top to bottom: zenith to nadir)

        # 3D unit ray for each equirectangular pixel
        # X: Right, Y: Up, Z: Forward
        X = np.sin(lon) * np.cos(lat)
        Y = np.sin(lat)
        Z = np.cos(lon) * np.cos(lat)

        # Forward perspective camera:
        # Source image has aspect ratio src_w / src_h
        aspect = src_w / float(src_h)
        h_fov_rad = np.radians(self.h_fov_deg)
        focal_h = 1.0 / np.tan(h_fov_rad / 2.0)
        focal_v = focal_h * aspect

        # Perspective projection onto the rectilinear image plane:
        # Valid only in front of the camera (Z > 0)
        valid_z = Z > 0.001
        norm_x = np.zeros_like(X)
        norm_y = np.zeros_like(Y)

        norm_x[valid_z] = (X[valid_z] / Z[valid_z]) * focal_h
        norm_y[valid_z] = (Y[valid_z] / Z[valid_z]) * focal_v

        # Map normalized coordinates [-1, 1] to source image pixel indices:
        # norm_x in [-1, 1] -> [0, src_w - 1] (left to right)
        # norm_y in [+1, -1] -> [0, src_h - 1] (+1 is top row y=0, -1 is bottom row y=src_h-1)
        map_x = (norm_x + 1.0) * 0.5 * (src_w - 1)
        map_y = (1.0 - norm_y) * 0.5 * (src_h - 1)

        # Mark pixels outside the source field of view as out-of-bounds (-1)
        out_of_bounds = (~valid_z) | (norm_x < -1.0) | (norm_x > 1.0) | (norm_y < -1.0) | (norm_y > 1.0)
        map_x[out_of_bounds] = -1.0
        map_y[out_of_bounds] = -1.0

        self.cached_maps = (map_x.astype(np.float32), map_y.astype(np.float32))
        self.last_src_shape = (src_w, src_h)

    def project_eye(self, eye_img: np.ndarray) -> np.ndarray:
        """
        Projects a single eye image into a 180° equirectangular half-sphere canvas.
        """
        h, w = eye_img.shape[:2]
        if self.cached_maps is None or self.last_src_shape != (w, h):
            self._build_remap_tables(w, h)

        map_x, map_y = self.cached_maps
        projected = cv2.remap(
            eye_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return projected

    def project_vr180_sbs(self, left_eye: np.ndarray, right_eye: np.ndarray) -> np.ndarray:
        """
        Takes Left and Right eye images and produces a full VR180 SBS equirectangular frame.
        Resulting dimensions: (eye_height, 2 * eye_width, 3).
        """
        left_proj = self.project_eye(left_eye)
        right_proj = self.project_eye(right_eye)
        return np.hstack([left_proj, right_proj])
