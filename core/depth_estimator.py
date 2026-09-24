"""
Depth Estimator Module
Wraps Depth Anything V2 for single frame and batch video depth estimation.
Supports Small (fastest), Base (balanced), and Large (maximum depth fidelity).
Utilizes CUDA FP16 on NVIDIA RTX 5070 for maximum throughput.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np
import torch

# Ensure depth_anything_v2 can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
da_repo_dir = os.path.join(current_dir, "depth_anything_v2")
if da_repo_dir not in sys.path:
    sys.path.insert(0, da_repo_dir)

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "filename": "depth_anything_v2_vits.pth",
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "desc": "Small (Ultra-gyors, ~100+ FPS, élő előnézethez)",
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "filename": "depth_anything_v2_vitb.pth",
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
        "desc": "Base (Kiegyensúlyozott, nagy pontosság, ~50 FPS)",
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "filename": "depth_anything_v2_vitl.pth",
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
        "desc": "Large (Csúcsminőség, legélesebb mélységél, ~25-35 FPS)",
    },
}


class DepthEstimator:
    def __init__(self, model_size: str = "vits", weights_dir: str = None, device: str = None):
        """
        Args:
            model_size: 'vits', 'vitb', or 'vitl'
            weights_dir: Directory where model weights are cached
            device: 'cuda' or 'cpu' (defaults to cuda if available)
        """
        if model_size not in MODEL_CONFIGS and model_size != "marigold":
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys()) + ['marigold']}")

        self.model_size = model_size
        self.config = MODEL_CONFIGS.get(model_size, {})
        self.pipe = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_size == "marigold":
            self._load_marigold()
            return

        if weights_dir is None:
            base_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.weights_dir = os.path.join(base_project, "weights")
        else:
            self.weights_dir = weights_dir

        os.makedirs(self.weights_dir, exist_ok=True)
        self.weights_path = os.path.join(self.weights_dir, self.config["filename"])

        self._ensure_weights()
        self._load_model()

    def _load_marigold(self):
        """Loads Hugging Face Marigold LCM diffusion depth pipeline."""
        from diffusers import MarigoldDepthPipeline
        print("Loading Hugging Face Marigold LCM pipeline onto GPU...")
        self.pipe = MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0",
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe.to(self.device)
        print("Marigold pipeline ready!")

    def _ensure_weights(self):
        """Downloads model weights if not already present."""
        if not os.path.exists(self.weights_path) or os.path.getsize(self.weights_path) < 1000:
            print(f"Downloading {self.model_size} weights from {self.config['url']}...")
            urllib.request.urlretrieve(self.config["url"], self.weights_path)
            print(f"Weights downloaded to {self.weights_path}")

    def _load_model(self):
        """Initializes and loads the PyTorch model onto the target device."""
        self.model = DepthAnythingV2(
            encoder=self.config["encoder"],
            features=self.config["features"],
            out_channels=self.config["out_channels"],
        )
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Warm up if CUDA
        if self.device == "cuda":
            dummy = np.zeros((256, 256, 3), dtype=np.uint8)
            with torch.no_grad():
                self.model.infer_image(dummy, input_size=256)

    def estimate_depth(self, image_bgr: np.ndarray, input_size: int = 518) -> np.ndarray:
        """
        Estimates normalized depth map for a single BGR image.

        Args:
            image_bgr: BGR input image (H, W, 3) uint8.
            input_size: Processing resolution size for the vision transformer (multiple of 14).
                        Default 518 is the standard Depth Anything V2 resolution.

        Returns:
            Normalized depth map (H, W) float32 in range [0.0, 1.0].
            1.0 = closest to camera, 0.0 = furthest in background.
        """
        if self.model_size == "marigold":
            from PIL import Image
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            with torch.no_grad():
                out = self.pipe(pil_img, num_inference_steps=4)
            depth_pred = np.squeeze(out.prediction)
            d_min, d_max = depth_pred.min(), depth_pred.max()
            norm_depth = 1.0 - (depth_pred - d_min) / max(1e-6, (d_max - d_min))
            return norm_depth.astype(np.float32)

        raw_depth = self.model.infer_image(image_bgr, input_size=input_size)

        d_min = raw_depth.min()
        d_max = raw_depth.max()
        if d_max - d_min > 1e-6:
            norm_depth = (raw_depth - d_min) / (d_max - d_min)
        else:
            norm_depth = np.zeros_like(raw_depth, dtype=np.float32)

        return norm_depth.astype(np.float32)

    def depth_to_colormap(self, norm_depth: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """Helper to convert a [0, 1] depth map into an RGB heat map for visualization."""
        depth_u8 = (np.clip(norm_depth, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, colormap)
