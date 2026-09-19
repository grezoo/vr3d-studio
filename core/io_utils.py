import ctypes
import os
import cv2
import numpy as np


def get_safe_path(path: str) -> str:
    """Returns 8.3 short path on Windows to prevent Unicode/accent issues in native C libraries."""
    if os.name == "nt" and os.path.exists(path):
        buf = ctypes.create_unicode_buffer(1024)
        res = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
        if res > 0:
            return buf.value
    return path


def video_capture_safe(path: str) -> cv2.VideoCapture:
    """Opens a video file safely with FFmpeg backend and short-path fallback."""
    safe_path = get_safe_path(path)
    cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(safe_path)
    if not cap.isOpened() and safe_path != path:
        cap = cv2.VideoCapture(path)
    return cap


def imread_safe(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Reads an image safely even from paths with accented/Unicode characters."""
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_safe(path: str, img: np.ndarray, params=None) -> bool:
    """Writes an image safely even to paths with accented/Unicode characters."""
    try:
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = ".png"
        success, encoded = cv2.imencode(ext, img, params)
        if success:
            encoded.tofile(path)
            return True
        return False
    except Exception:
        return False
