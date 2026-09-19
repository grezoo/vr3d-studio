"""
VR3D Studio - Main Application Entry Point
Can be launched directly as GUI or via CLI flags for headless batch processing.
"""

import argparse
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser(description="VR3D Studio - 2D to 3D SBS & VR180 AI Converter")
    parser.add_argument("--input", "-i", type=str, help="Input video or image file path")
    parser.add_argument("--output", "-o", type=str, help="Output video or image file path")
    parser.add_argument(
        "--mode", "-m", type=str, default="sbs_full",
        choices=["sbs_full", "sbs_half", "vr180", "anaglyph", "depth_only"],
        help="Stereo conversion mode"
    )
    parser.add_argument("--model", type=str, default="vits", choices=["vits", "vitb", "vitl"], help="AI model size")
    parser.add_argument("--ipd", type=float, default=0.018, help="Interpupillary disparity scale (default: 0.018)")
    parser.add_argument("--conv", type=float, default=0.5, help="Zero parallax convergence depth (default: 0.5)")
    parser.add_argument("--fov", type=float, default=110.0, help="VR180 horizontal FOV in degrees (default: 110)")
    parser.add_argument("--swap-eyes", action="store_true", help="Swap Left and Right eye views")
    parser.add_argument("--no-nvenc", action="store_true", help="Disable NVENC hardware encoding")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal depth consistency filter")

    args = parser.parse_args()

    # If an input file is provided via CLI, run in headless batch mode
    if args.input:
        from core.depth_estimator import DepthEstimator
        from core.stereo_warper import StereoWarper
        from core.video_processor import VideoProcessor

        if not os.path.exists(args.input):
            print(f"Error: Input file does not exist: {args.input}")
            sys.exit(1)

        suffix_map = {
            "vr180": "_180_sbs",
            "sbs_full": "_3D_full_sbs",
            "sbs_half": "_3D_half_sbs",
            "anaglyph": "_3D_anaglyph",
            "depth_only": "_depth"
        }
        suffix = suffix_map.get(args.mode, f"_{args.mode}")

        print("Initializing AI Depth Estimator on NVIDIA RTX 5070...")
        estimator = DepthEstimator(model_size=args.model)
        warper = StereoWarper()
        processor = VideoProcessor(estimator, warper)

        if os.path.isdir(args.input):
            out_dir = args.output if args.output else os.path.join(args.input, "_3D_album")
            os.makedirs(out_dir, exist_ok=True)
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            images = [
                f for f in os.listdir(args.input)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
            print(f"Batch processing album folder: {len(images)} images -> {out_dir}")
            for idx, img_name in enumerate(images):
                in_img = os.path.join(args.input, img_name)
                root, ext = os.path.splitext(img_name)
                out_img = os.path.join(out_dir, f"{root}{suffix}{ext}")
                print(f"[{idx+1}/{len(images)}] {img_name} -> {os.path.basename(out_img)}...")
                processor.process_image(
                    in_img,
                    out_img,
                    mode=args.mode,
                    ipd_offset=args.ipd,
                    convergence=args.conv,
                    h_fov=args.fov,
                    swap_eyes=args.swap_eyes
                )
            print("Done! Album batch conversion completed successfully.")
            return

        out_path = args.output
        if not out_path:
            name, ext = os.path.splitext(args.input)
            out_path = f"{name}{suffix}{ext}"

        ext = os.path.splitext(args.input)[1].lower()
        is_video = ext in [".mp4", ".mkv", ".mov", ".avi"]

        print(f"Converting: {args.input} -> {out_path} (Mode: {args.mode})")
        if is_video:
            def on_progress(cur, total, fps, eta):
                pct = (cur / max(1, total)) * 100
                print(f"\rProgress: {cur}/{total} ({pct:.1f}%) | Speed: {fps:.1f} FPS | ETA: {eta}", end="", flush=True)

            processor.process_video(
                args.input,
                out_path,
                mode=args.mode,
                ipd_offset=args.ipd,
                convergence=args.conv,
                use_temporal_filter=not args.no_temporal,
                use_nvenc=not args.no_nvenc,
                h_fov=args.fov,
                swap_eyes=args.swap_eyes,
                progress_callback=on_progress
            )
            print("\nDone! Video conversion completed successfully.")
        else:
            processor.process_image(
                args.input,
                out_path,
                mode=args.mode,
                ipd_offset=args.ipd,
                convergence=args.conv,
                h_fov=args.fov,
                swap_eyes=args.swap_eyes
            )
            print("Done! Image conversion completed successfully.")

    else:
        # Launch graphical user interface
        from gui.app import VR3DStudioApp
        app = VR3DStudioApp()
        app.mainloop()


if __name__ == "__main__":
    main()
