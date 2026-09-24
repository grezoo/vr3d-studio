"""
Video Processor Module
High-throughput video processing pipeline with FFmpeg stdin streaming,
NVENC hardware GPU encoding, audio passthrough, and VR180 spatial metadata integration.
"""

import os
import subprocess
import time
import cv2
import numpy as np
from core.io_utils import imread_safe, imwrite_safe, video_capture_safe, get_safe_path
from core.depth_estimator import DepthEstimator
from core.stereo_warper import StereoWarper
from core.vr180_projector import VR180Projector
from core.temporal_filter import TemporalDepthFilter
from core.vr_metadata import VRMetadataInjector


class VideoProcessor:
    def __init__(self, depth_estimator: DepthEstimator, stereo_warper: StereoWarper = None):
        self.depth_estimator = depth_estimator
        self.stereo_warper = stereo_warper if stereo_warper else StereoWarper()
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def process_image(
        self,
        input_image_path: str,
        output_image_path: str,
        mode: str = "sbs_full",
        ipd_offset: float = 0.035,
        convergence: float = 0.5,
        h_fov: float = 110.0,
        vr_size: int = 1920,
        swap_eyes: bool = False,
        auto_convergence: bool = True
    ) -> bool:
        """
        Converts a single image to 3D SBS / VR180 / Anaglyph.
        """
        img = imread_safe(input_image_path)
        if img is None:
            raise ValueError(f"Could not load image from {input_image_path}")

        depth = self.depth_estimator.estimate_depth(img)
        left, right = self.stereo_warper.generate_stereo_pair(
            img, depth, ipd_offset=ipd_offset, convergence=convergence,
            swap_eyes=swap_eyes, auto_convergence=auto_convergence
        )

        if mode == "sbs_full":
            result = self.stereo_warper.create_sbs(left, right, half_sbs=False)
        elif mode == "sbs_half":
            result = self.stereo_warper.create_sbs(left, right, half_sbs=True)
        elif mode == "anaglyph":
            result = self.stereo_warper.create_anaglyph(left, right)
        elif mode == "vr180":
            projector = VR180Projector(output_eye_size=(vr_size, vr_size), h_fov_deg=h_fov)
            result = projector.project_vr180_sbs(left, right)
        elif mode == "depth_only":
            result = self.depth_estimator.depth_to_colormap(depth)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ext = os.path.splitext(output_image_path)[1].lower()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 96] if ext in [".jpg", ".jpeg"] else None
        imwrite_safe(output_image_path, result, params=params)
        return True

    def process_video(
        self,
        input_path: str,
        output_path: str,
        mode: str = "sbs_full",
        ipd_offset: float = 0.03,
        convergence: float = 0.5,
        use_temporal_filter: bool = True,
        use_nvenc: bool = True,
        h_fov: float = 110.0,
        vr_size: int = 1920,
        swap_eyes: bool = False,
        auto_convergence: bool = True,
        progress_callback=None,
        start_frame: int = 0,
        max_frames: int = None
    ) -> bool:
        """
        Converts an entire video or a snippet to 3D SBS / VR180 with hardware encoding.
        """
        self.is_cancelled = False

        cap = video_capture_safe(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        target_total = min(max(0, total_frames - start_frame), max_frames) if max_frames else max(1, total_frames - start_frame)

        # Determine target resolution
        if mode == "sbs_full":
            out_w, out_h = src_w * 2, src_h
        elif mode == "sbs_half":
            out_w, out_h = src_w, src_h
        elif mode == "anaglyph":
            out_w, out_h = src_w, src_h
        elif mode == "vr180":
            out_w, out_h = vr_size * 2, vr_size
        elif mode == "depth_only":
            out_w, out_h = src_w, src_h
        else:
            raise ValueError(f"Unknown mode: {mode}")

        vr_projector = VR180Projector(output_eye_size=(vr_size, vr_size), h_fov_deg=h_fov) if mode == "vr180" else None
        temporal_filter = TemporalDepthFilter(alpha=0.75) if use_temporal_filter else None

        # Build FFmpeg command for raw frame piping via stdin
        temp_video = output_path + ".temp_video.mp4"
        if os.path.exists(temp_video):
            os.remove(temp_video)

        # Video encoder selection
        vcodec = "hevc_nvenc" if use_nvenc else "libx264"
        encoder_args = (
            ["-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "22", "-pix_fmt", "yuv420p"]
            if use_nvenc
            else ["-c:v", "libx264", "-crf", "19", "-preset", "fast", "-pix_fmt", "yuv420p"]
        )

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            *encoder_args,
            temp_video
        ]

        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            # Fallback to libx264 if NVENC fails
            if use_nvenc:
                encoder_args = ["-c:v", "libx264", "-crf", "19", "-preset", "fast", "-pix_fmt", "yuv420p"]
                ffmpeg_cmd[ffmpeg_cmd.index("hevc_nvenc")] = "libx264"
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                raise e

        frame_idx = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                if self.is_cancelled:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Depth estimation
                depth = self.depth_estimator.estimate_depth(frame)

                # 2. Temporal consistency
                if temporal_filter is not None:
                    depth = temporal_filter.process(frame, depth)

                # 3. Mode processing
                if mode == "depth_only":
                    out_frame = self.depth_estimator.depth_to_colormap(depth)
                else:
                    left, right = self.stereo_warper.generate_stereo_pair(
                        frame, depth, ipd_offset=ipd_offset, convergence=convergence,
                        swap_eyes=swap_eyes, auto_convergence=auto_convergence
                    )
                    if mode == "sbs_full":
                        out_frame = self.stereo_warper.create_sbs(left, right, half_sbs=False)
                    elif mode == "sbs_half":
                        out_frame = self.stereo_warper.create_sbs(left, right, half_sbs=True)
                    elif mode == "anaglyph":
                        out_frame = self.stereo_warper.create_anaglyph(left, right)
                    elif mode == "vr180":
                        out_frame = vr_projector.project_vr180_sbs(left, right)

                # Write frame to FFmpeg pipe
                ffmpeg_proc.stdin.write(out_frame.tobytes())

                frame_idx += 1
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                remaining_frames = max(0, target_total - frame_idx)
                eta_seconds = remaining_frames / current_fps if current_fps > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                if progress_callback:
                    progress_callback(frame_idx, target_total, current_fps, eta_str)

                if max_frames and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

        if self.is_cancelled:
            if os.path.exists(temp_video):
                os.remove(temp_video)
            return False

        # Mux original audio track and metadata into the final file
        start_sec = (start_frame / fps) if fps > 0 else 0.0
        dur_sec = (frame_idx / fps) if (max_frames and fps > 0) else None
        mux_ok = self._mux_audio_and_metadata(input_path, temp_video, output_path, mode=mode, start_sec=start_sec, duration_sec=dur_sec)

        # Safety check: ensure final file exists and is not empty
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        else:
            # Fallback: if audio muxing failed for any reason, preserve the rendered frames!
            print(f"[Warning] Muxing failed or output missing. Preserving encoded video by renaming to: {output_path}")
            if os.path.exists(temp_video):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_video, output_path)

        return True

    def _mux_audio_and_metadata(self, original_path: str, temp_video: str, final_path: str, mode: str, start_sec: float = 0.0, duration_sec: float = None) -> bool:
        """
        Transfers audio stream from original file and injects 3D/VR180 metadata.
        """
        meta_flags = VRMetadataInjector.get_ffmpeg_stereo_flags("vr180" if mode == "vr180" else "sbs")

        audio_in_args = []
        if start_sec > 0:
            audio_in_args.extend(["-ss", f"{start_sec:.3f}"])
        if duration_sec:
            audio_in_args.extend(["-t", f"{duration_sec:.3f}"])

        mux_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            *audio_in_args,
            "-i", original_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a?",  # copy audio if exists
            *meta_flags,
            "-movflags", "+faststart",
            final_path
        ]

        try:
            res = subprocess.run(mux_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            if res.returncode != 0:
                print(f"[FFmpeg Mux Error] Exit code {res.returncode}:\n{res.stderr}")
                return False
            return True
        except Exception as e:
            print(f"[FFmpeg Mux Exception]: {e}")
            return False
