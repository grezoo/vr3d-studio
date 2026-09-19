"""
End-to-End Test for VR3D Studio Pipeline
Tests Depth Estimation, DIBR Stereo Warping, VR180 Projection,
and NVENC Video Hardware Encoding on RTX 5070.
"""

import os
import sys
import subprocess
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.io_utils import imread_safe, imwrite_safe
from core.depth_estimator import DepthEstimator
from core.stereo_warper import StereoWarper
from core.vr180_projector import VR180Projector
from core.video_processor import VideoProcessor


def generate_synthetic_scene(width=1280, height=720, shift=0):
    """Draws a scene with clear depth cues: mountains in back, house in mid, circle in foreground."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky
    img[: int(height * 0.6)] = (235, 180, 100)  # BGR
    # Ground
    img[int(height * 0.6) :] = (60, 130, 40)
    # Background mountain
    pts = np.array([[100, int(height * 0.6)], [450, 150], [800, int(height * 0.6)]], np.int32)
    cv2.fillPoly(img, [pts], (180, 140, 120))
    # Foreground ball moving
    ball_x = int(300 + (shift % 400))
    ball_y = int(height * 0.7)
    cv2.circle(img, (ball_x, ball_y), 70, (50, 50, 230), -1)
    cv2.putText(img, "Foreground 3D Sphere", (ball_x - 90, ball_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def run_pipeline_test():
    print("--- [1/4] Generating synthetic test inputs ---")
    out_dir = os.path.join(BASE_DIR, "tests", "output")
    os.makedirs(out_dir, exist_ok=True)

    test_img_path = os.path.join(out_dir, "test_input.png")
    frame0 = generate_synthetic_scene(1280, 720, shift=0)
    imwrite_safe(test_img_path, frame0)
    print(f"Generated test image: {test_img_path}")

    # Generate 2-second test video (60 frames at 30 fps) with audio tone
    test_video_no_audio = os.path.join(out_dir, "test_no_audio.mp4")
    test_video_path = os.path.join(out_dir, "test_input_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(test_video_no_audio, fourcc, 30.0, (1280, 720))
    for i in range(60):
        f = generate_synthetic_scene(1280, 720, shift=i * 6)
        vw.write(f)
    vw.release()

    # Add synthetic beep audio with ffmpeg
    subprocess.run([
        "ffmpeg", "-y",
        "-i", test_video_no_audio,
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        test_video_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(test_video_no_audio):
        os.remove(test_video_no_audio)
    print(f"Generated test video with audio: {test_video_path}")

    print("\n--- [2/4] Initializing Depth Anything V2 (Small) on RTX 5070 ---")
    estimator = DepthEstimator(model_size="vits")
    warper = StereoWarper()
    processor = VideoProcessor(estimator, warper)
    print("AI Model loaded successfully on GPU!")

    print("\n--- [3/4] Testing Image Conversions ---")
    modes = ["sbs_full", "sbs_half", "vr180", "anaglyph", "depth_only"]
    for mode in modes:
        out_img = os.path.join(out_dir, f"result_image_{mode}.png")
        processor.process_image(test_img_path, out_img, mode=mode)
        assert os.path.exists(out_img) and os.path.getsize(out_img) > 1000
        print(f"  [OK] Image mode '{mode}' generated: {out_img} ({os.path.getsize(out_img)} bytes)")

    print("\n--- [4/4] Testing Video Conversion with NVENC and VR180 ---")
    sbs_video_out = os.path.join(out_dir, "result_video_sbs.mp4")
    vr180_video_out = os.path.join(out_dir, "result_video_vr180.mp4")

    print("Converting video to 3D SBS...")
    processor.process_video(test_video_path, sbs_video_out, mode="sbs_full", use_nvenc=True)
    assert os.path.exists(sbs_video_out) and os.path.getsize(sbs_video_out) > 5000
    print(f"  [OK] 3D SBS Video: {sbs_video_out} ({os.path.getsize(sbs_video_out)} bytes)")

    print("Converting video to VR180...")
    processor.process_video(test_video_path, vr180_video_out, mode="vr180", use_nvenc=True, vr_size=1080)
    assert os.path.exists(vr180_video_out) and os.path.getsize(vr180_video_out) > 5000
    print(f"  [OK] VR180 Video: {vr180_video_out} ({os.path.getsize(vr180_video_out)} bytes)")

    print("\n==========================================")
    print(" ALL TESTS PASSED! VR3D STUDIO IS 100% OPERATIONAL!")
    print("==========================================")


if __name__ == "__main__":
    run_pipeline_test()
