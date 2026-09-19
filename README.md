# VR3D Studio 🥽
### Open-Source 2D to 3D SBS & VR180 AI Converter (Owl3D Alternative)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU](https://img.shields.io/badge/GPU%20Acceleration-NVIDIA%20NVENC-76b900.svg)](https://developer.nvidia.com/video-encode-decode-gpu-support-matrix)

**VR3D Studio** is an open-source, 100% offline, watermark-free desktop application and CLI tool that converts regular 2D videos, photos, and photo albums into stereoscopic 3D Side-by-Side (SBS) and immersive VR180 formats for VR headsets (Pico 4, Meta Quest 2/3/Pro, Apple Vision Pro, Skybox VR).

Built as a high-performance, free alternative to proprietary subscription tools like Owl3D.

---

## ✨ Features

- 🦉 **Unlimited & Free (Owl3D Alternative):** No credits, no subscriptions, no cloud queue, no watermarks. Runs 100% locally on your machine.
- 🧠 **SOTA Depth AI:** Powered by **Depth Anything V2** (`vits`, `vitb`, and `vitl`) for monocular depth estimation with edge preservation.
- 🎯 **Auto-Convergence (Subject-Locking):** Dynamically locks convergence (zero-disparity plane) onto the primary foreground subject, eliminating double body contours and eye divergence strain.
- 👁️ **Wiggle 3D Preview (Glasses-Free):** Interactive GUI preview that oscillates between left and right eye views, allowing you to clearly see depth layers on a standard 2D PC monitor before converting.
- ⚡ **5-Second Sample Quick-Test:** Generate a quick 5-second 3D sample clip from any point in a video in seconds to test in your VR headset before committing to a full-length render.
- 📁 **Batch Photo Album Converter:** Process entire folders of family photos, travel albums, or wallpapers into VR180/SBS 3D in seconds (~0.1–0.2s per photo on modern GPUs).
- ⚡ **NVIDIA NVENC Hardware Encoding:** Direct raw NVENC streaming (`hevc_nvenc`) for export speeds with audio passthrough.
- 🌐 **Full Headset Compatibility:**
  - Auto-injects Google Spatial Media VR180 equirectangular spherical metadata tags.
  - Automatically appends standard naming tags (`_180_sbs`, `_3D_full_sbs`) so players like **Pico Video / Gallery**, **Meta Quest TV**, **Skybox VR**, **4X-VR**, and **DeoVR** detect and play them in 3D without manual menu adjustments.
- 🛡️ **Unicode / International Path Safe:** Robust Windows file handling supporting Hungarian and international accents.

---

## 🖥️ System Requirements

- **OS:** Windows 10 / 11 (64-bit) or Linux.
- **GPU:** NVIDIA GPU with CUDA support (RTX 30xx, 40xx, 50xx series recommended; tested on RTX 5070 12GB).
- **FFmpeg:** Installed and added to system PATH.
- **Python:** 3.10 - 3.14.

---

## 🚀 Quick Start

### 1. Clone repository
```bash
git clone https://github.com/your-username/vr3d-studio.git
cd vr3d-studio
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note for PyTorch CUDA:** Ensure you install the CUDA-enabled PyTorch build for your system (e.g. from [pytorch.org](https://pytorch.org/get-started/locally/)).

### 3. Launch GUI
Double-click `run_converter.bat` or run:
```bash
python main.py
```
Model weights (`depth_anything_v2_vits.pth`, etc.) are downloaded automatically into `weights/` on the first run.

---

## 🕹️ GUI Usage

1. **Select Input:** Click **📄 Egyedi fájl tallózása** for single video/photo, or **📁 Teljes Fotóalbum** for entire photo folders.
2. **Choose Mode:** Select **VR180 3D** (for VR headsets) or **Full SBS 3D** (for 3D monitors/TVs).
3. **Check Depth on Monitor:**
   - Click the **Wiggle 3D** tab to see depth layers moving without any 3D glasses!
   - Select a preset: **Lágy 3D** (Soft/Relaxed), **Természetes ✨** (Natural/Recommended 1.8%), or **Erős 3D** (Strong).
4. **Test Before Full Render (Videos):**
   - Click **⚡ 5 mp Minta Gyorsteszt** to render a 5-second sample from your current timeline position.
5. **Start:** Click **🚀 Teljes Konvertálás Indítása**.

---

## ⌨️ Command Line (CLI) Usage

For automated or headless processing:

```bash
# Convert a video to VR180 SBS with NVENC acceleration:
python main.py -i "input_video.mp4" -m vr180 --fov 110

# Batch convert an entire folder of photos to VR180 3D:
python main.py -i "C:/Photos/Vacation" -o "C:/Photos/Vacation_3D" -m vr180

# Convert image to Full Side-by-Side 3D:
python main.py -i "photo.jpg" -m sbs_full --ipd 0.018

# Options:
#   --mode, -m        vr180 | sbs_full | sbs_half | anaglyph | depth_only
#   --model           vits (fast) | vitb (balanced) | vitl (quality)
#   --ipd             Disparity strength (default: 0.018)
#   --fov             VR180 horizontal FOV (default: 110.0)
#   --swap-eyes       Swap Left and Right eye channels
#   --no-nvenc        Use CPU x264 instead of NVIDIA NVENC
```

---

## 🗂️ Project Structure

```
vr3d-studio/
├── run_converter.bat           # 1-click launcher for Windows
├── main.py                     # Entry point (GUI + CLI)
├── requirements.txt            # Python dependencies
├── core/
│   ├── depth_estimator.py      # Depth Anything V2 AI engine
│   ├── stereo_warper.py        # Disparity remap with Auto-Convergence
│   ├── vr180_projector.py      # Equirectangular 180° dome projector
│   ├── vr_metadata.py          # Spatial Media VR180 MP4 metadata injector
│   ├── temporal_filter.py      # Anti-flicker temporal consistency filter
│   ├── video_processor.py      # NVENC streaming pipeline & snippet cutter
│   ├── io_utils.py             # Unicode-safe image/video I/O
│   └── depth_anything_v2/      # Model architecture
├── gui/
│   └── app.py                  # CustomTkinter Dark UI (Wiggle 3D, Presets)
└── weights/                    # Cached AI weights (auto-downloaded)
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
Depth Anything V2 is licensed under the Apache 2.0 license.
