"""
VR3D Studio - Modern Graphical User Interface
Built with CustomTkinter. Multi-threaded with live preview, video timeline scrubber,
interactive 3D view tabs, and hardware status monitor.
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
import webbrowser

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.io_utils import imread_safe, imwrite_safe
from core.depth_estimator import DepthEstimator, MODEL_CONFIGS
from core.stereo_warper import StereoWarper
from core.vr180_projector import VR180Projector
from core.video_processor import VideoProcessor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class VR3DStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VR3D Studio - 2D to 3D SBS & VR180 AI Converter (RTX 5070 Edition)")
        self.geometry("1280x820")
        self.minsize(1050, 700)

        # Core engines
        self.depth_estimator = None
        self.stereo_warper = StereoWarper()
        self.video_processor = None

        # State variables
        self.input_file_path = ""
        self.output_file_path = ""
        self.is_video = False
        self.is_folder = False
        self.album_files = []
        self.total_video_frames = 0
        self.current_frame_bgr = None
        self.cached_depth = None
        self.cached_left = None
        self.cached_right = None
        self.active_tab = "Wiggle 3D (Szemüveg nélkül)"
        self.is_processing = False
        self.wiggle_eye = 0
        self.wiggle_job = None

        self._build_ui()
        self._init_backend_async()

    def _init_backend_async(self):
        """Initializes AI model in a background thread so UI starts immediately."""
        self.status_label.configure(text="Állapot: Depth Anything V2 inicializálása (GPU betöltés)...")
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            model_key = self.MODEL_DISPLAY_MAP.get(self.model_var.get(), "vits")
            self.depth_estimator = DepthEstimator(model_size=model_key)
            self.video_processor = VideoProcessor(self.depth_estimator, self.stereo_warper)
            self.after(0, lambda: self.status_label.configure(text="Állapot: Kész. NVIDIA RTX 5070 CUDA aktív."))
            self.after(0, self._on_settings_change)
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Model load error] {err_msg}")
            self.after(0, lambda msg=err_msg: self.status_label.configure(text=f"Modell hiba: {msg}"))

    def _build_ui(self):
        # Grid layout: 2 columns (Sidebar and Main Area)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # 1. LEFT SIDEBAR: Controls & Settings
        # ==========================================
        sidebar = ctk.CTkScrollableFrame(self, width=340, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Logo / Title
        title_label = ctk.CTkLabel(
            sidebar, text="VR3D STUDIO", font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(padx=15, pady=(15, 2), anchor="w")
        subtitle = ctk.CTkLabel(
            sidebar, text="2D -> 3D SBS & VR180 AI Converter", font=ctk.CTkFont(size=12), text_color="gray"
        )
        subtitle.pack(padx=15, pady=(0, 15), anchor="w")

        # Hardware Badge
        hw_frame = ctk.CTkFrame(sidebar, fg_color=("#2B2B2B", "#1F1F1F"), corner_radius=6)
        hw_frame.pack(fill="x", padx=15, pady=(0, 15))
        hw_label = ctk.CTkLabel(
            hw_frame, text="⚡ NVIDIA RTX 5070 (12GB) • CUDA", font=ctk.CTkFont(size=11, weight="bold"), text_color="#2ECC71"
        )
        hw_label.pack(padx=10, pady=6)

        # --- File Selection ---
        sec1 = ctk.CTkLabel(sidebar, text="1. FORRÁSFÁJL KIVÁLASZTÁSA", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498DB")
        sec1.pack(padx=15, pady=(10, 5), anchor="w")

        btn_browse_file = ctk.CTkButton(sidebar, text="📄 Egyedi fájl tallózása (Videó / Kép)", command=self._browse_input)
        btn_browse_file.pack(fill="x", padx=15, pady=(5, 3))

        btn_browse_folder = ctk.CTkButton(
            sidebar, text="📁 Teljes Fotóalbum / Mappa tallózása",
            fg_color="#8E44AD", hover_color="#732D91",
            command=self._browse_folder
        )
        btn_browse_folder.pack(fill="x", padx=15, pady=(3, 5))

        self.lbl_input_path = ctk.CTkLabel(sidebar, text="Nincs fájl vagy mappa kiválasztva", text_color="gray", wraplength=300, font=ctk.CTkFont(size=11))
        self.lbl_input_path.pack(padx=15, pady=(0, 10), anchor="w")

        # --- 3D & VR Settings ---
        sec2 = ctk.CTkLabel(sidebar, text="2. TÉRHATÁS ÉS VR BEÁLLÍTÁSOK", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3498DB")
        sec2.pack(padx=15, pady=(10, 5), anchor="w")

        # Display mapping dictionaries
        self.MODEL_DISPLAY_MAP = {
            "Depth Anything V2 (Kis / Villámgyors)": "vits",
            "Depth Anything V2 (Közepes / Kiegyensúlyozott)": "vitb",
            "Depth Anything V2 (Nagy / Csúcsminőség)": "vitl",
            "Hugging Face: Marigold LCM (Diffúziós 3D)": "marigold"
        }
        self.MODEL_INTERNAL_MAP = {v: k for k, v in self.MODEL_DISPLAY_MAP.items()}

        self.MODE_DISPLAY_MAP = {
            "VR180 3D (180° Sztereó SBS - Pico / Quest)": "vr180",
            "Full SBS 3D (Normál képernyős 3D - Teljes szélesség)": "sbs_full",
            "Half SBS 3D (Normál képernyős 3D - Felezett szélesség)": "sbs_half",
            "Anaglif 3D (Piros-Cián szemüveges)": "anaglyph",
            "Csak Mélységtérkép (Hőtérkép)": "depth_only"
        }
        self.MODE_INTERNAL_MAP = {v: k for k, v in self.MODE_DISPLAY_MAP.items()}

        self.PROFILE_DISPLAY_MAP = {
            "[1] Természetes 3D (Kényelmes, 3.5%)": (0.035, 0.50),
            "[2] Mély Dinamikus 3D (Látványos, 5.0%)": (0.050, 0.35),
            "[3] Pop-Out (Kilóg a képből! 6.5%)": (0.065, 0.15),
            "[4] Extrém 3D Térhatás (8.0%)": (0.080, 0.10),
            "[0] Lágy 3D (Pihentető, 2.0%)": (0.020, 0.50),
        }

        # Model Selector
        ctk.CTkLabel(sidebar, text="AI Modell:", font=ctk.CTkFont(size=11, weight="bold")).pack(padx=15, anchor="w")
        self.model_var = ctk.StringVar(value=list(self.MODEL_DISPLAY_MAP.keys())[0])
        self.cmb_model = ctk.CTkOptionMenu(
            sidebar,
            values=list(self.MODEL_DISPLAY_MAP.keys()),
            variable=self.model_var,
            command=self._on_model_changed
        )
        self.cmb_model.pack(fill="x", padx=15, pady=(2, 10))

        # Mode Selector
        ctk.CTkLabel(sidebar, text="Kimeneti formátum:", font=ctk.CTkFont(size=11, weight="bold")).pack(padx=15, anchor="w")
        self.mode_var = ctk.StringVar(value="VR180 3D (180° Sztereó SBS - Pico / Quest)")
        self.cmb_mode = ctk.CTkOptionMenu(
            sidebar,
            values=list(self.MODE_DISPLAY_MAP.keys()),
            variable=self.mode_var,
            command=self._on_mode_changed
        )
        self.cmb_mode.pack(fill="x", padx=15, pady=(2, 10))

        # 3D Profile Selector
        ctk.CTkLabel(sidebar, text="3D Térhatás Karakter (Profil):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F39C12").pack(padx=15, anchor="w")
        self.profile_var = ctk.StringVar(value=list(self.PROFILE_DISPLAY_MAP.keys())[0])
        self.cmb_profile = ctk.CTkOptionMenu(
            sidebar,
            values=list(self.PROFILE_DISPLAY_MAP.keys()),
            variable=self.profile_var,
            command=self._on_profile_changed
        )
        self.cmb_profile.pack(fill="x", padx=15, pady=(2, 10))

        # IPD / Disparity separation slider
        self.lbl_ipd = ctk.CTkLabel(sidebar, text="3D Hatás: Természetes (Ajánlott) [3.5%]", font=ctk.CTkFont(size=11))
        self.lbl_ipd.pack(padx=15, anchor="w")
        self.slider_ipd = ctk.CTkSlider(sidebar, from_=0.010, to=0.090, number_of_steps=80, command=self._on_ipd_slide)
        self.slider_ipd.set(0.035)
        self.slider_ipd.pack(fill="x", padx=15, pady=(2, 10))

        # Convergence / Zero parallax plane slider
        self.lbl_conv = ctk.CTkLabel(sidebar, text="Térbeli fókusz (Konvergencia): 50%", font=ctk.CTkFont(size=11))
        self.lbl_conv.pack(padx=15, anchor="w")
        self.slider_conv = ctk.CTkSlider(sidebar, from_=0.1, to=0.9, number_of_steps=80, command=self._on_conv_slide)
        self.slider_conv.set(0.50)
        self.slider_conv.pack(fill="x", padx=15, pady=(2, 5))

        self.chk_auto_conv = ctk.CTkCheckBox(
            sidebar, text="✨ Auto Fókusz (Owl3D test-rögzítés)",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498DB",
            command=self._on_auto_conv_toggle
        )
        self.chk_auto_conv.select()
        self.chk_auto_conv.pack(padx=15, pady=(0, 10), anchor="w")

        # VR180 FOV slider (only shown in VR180 mode)
        self.lbl_fov = ctk.CTkLabel(sidebar, text="VR Látószög (FOV): 110°", font=ctk.CTkFont(size=11))
        self.lbl_fov.pack(padx=15, anchor="w")
        self.slider_fov = ctk.CTkSlider(sidebar, from_=80.0, to=150.0, number_of_steps=70, command=self._on_fov_slide)
        self.slider_fov.set(110.0)
        self.slider_fov.pack(fill="x", padx=15, pady=(2, 10))

        # Options checkboxes
        self.chk_swap_eyes = ctk.CTkCheckBox(sidebar, text="Szemek felcserélése (Bal / Jobb csere)", font=ctk.CTkFont(size=11), command=self._on_swap_eyes_toggle)
        self.chk_swap_eyes.pack(padx=15, pady=(5, 5), anchor="w")

        self.chk_temporal = ctk.CTkCheckBox(sidebar, text="Időbeli simítás (Villogásmentes)", font=ctk.CTkFont(size=11))
        self.chk_temporal.select()
        self.chk_temporal.pack(padx=15, pady=(5, 5), anchor="w")

        self.chk_nvenc = ctk.CTkCheckBox(sidebar, text="NVIDIA NVENC hardveres kódolás", font=ctk.CTkFont(size=11))
        self.chk_nvenc.select()
        self.chk_nvenc.pack(padx=15, pady=(5, 15), anchor="w")

        # --- Actions ---
        self.btn_preview = ctk.CTkButton(
            sidebar, text="🔄 Előnézet Frissítése", fg_color="#34495E", hover_color="#2C3E50", command=self._update_preview_manual
        )
        self.btn_preview.pack(fill="x", padx=15, pady=(5, 4))

        self.btn_quick_test = ctk.CTkButton(
            sidebar, text="⚡ 5 mp Minta Gyorsteszt", fg_color="#D35400", hover_color="#BA4A00", height=32,
            font=ctk.CTkFont(size=12, weight="bold"), command=self._start_quick_test
        )
        self.btn_quick_test.pack(fill="x", padx=15, pady=(2, 8))

        self.btn_start = ctk.CTkButton(
            sidebar, text="🚀 Teljes Konvertálás Indítása", fg_color="#27AE60", hover_color="#219955", height=38,
            font=ctk.CTkFont(size=13, weight="bold"), command=self._start_conversion
        )
        self.btn_start.pack(fill="x", padx=15, pady=(5, 12))

        self.btn_cancel = ctk.CTkButton(
            sidebar, text="⏹ Megszakítás", fg_color="#C0392B", hover_color="#962D22", state="disabled", command=self._cancel_conversion
        )
        self.btn_cancel.pack(fill="x", padx=15, pady=(0, 15))

        # --- Revolut Support Box ---
        donate_frame = ctk.CTkFrame(sidebar, fg_color=("#1A252F", "#141D26"), corner_radius=8, border_width=1, border_color="#2980B9")
        donate_frame.pack(fill="x", padx=15, pady=(0, 20))

        lbl_donate = ctk.CTkLabel(
            donate_frame, text="☕ Támogatás / Support:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498DB"
        )
        lbl_donate.pack(padx=10, pady=(6, 2))

        btn_revolut = ctk.CTkButton(
            donate_frame, text="💙 Revolut: @grezoo", fg_color="#0075EB", hover_color="#005BBB", height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: webbrowser.open("https://revolut.me/grezoo")
        )
        btn_revolut.pack(fill="x", padx=10, pady=(2, 8))

        # ==========================================
        # 2. MAIN AREA: Preview & Progress
        # ==========================================
        main_frame = ctk.CTkFrame(self, corner_radius=0)
        main_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Top Tabs
        tab_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        tab_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.tab_buttons = {}
        tabs = [
            "Wiggle 3D (Szemüveg nélkül)",
            "3D SBS",
            "VR180",
            "Anaglif 3D",
            "Mélységtérkép",
            "Bal Szem",
            "Jobb Szem",
            "2D Eredeti"
        ]
        for tab_name in tabs:
            btn = ctk.CTkButton(
                tab_frame,
                text=tab_name,
                height=28,
                fg_color="#1F6AA5" if tab_name == self.active_tab else "#2B2B2B",
                command=lambda name=tab_name: self._switch_tab(name)
            )
            btn.pack(side="left", padx=3)
            self.tab_buttons[tab_name] = btn

        # Canvas for Image / Preview
        self.canvas_frame = ctk.CTkFrame(main_frame, fg_color="#121212", corner_radius=8)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.lbl_image = tk.Label(self.canvas_frame, bg="#121212", text="Válassz ki egy videót vagy képet a kezdéshez", fg="gray")
        self.lbl_image.grid(row=0, column=0, sticky="nsew")
        self.canvas_frame.bind("<Configure>", lambda evt: self._render_current_tab())

        # Video Timeline Scrubber (only visible for videos)
        self.scrub_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.scrub_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.scrub_frame.grid_columnconfigure(1, weight=1)

        self.lbl_scrub_time = ctk.CTkLabel(self.scrub_frame, text="00:00 / 00:00", font=ctk.CTkFont(size=11), width=90)
        self.lbl_scrub_time.grid(row=0, column=0, padx=(0, 10))

        self.slider_scrub = ctk.CTkSlider(self.scrub_frame, from_=0, to=100, command=self._on_scrub)
        self.slider_scrub.set(0)
        self.slider_scrub.grid(row=0, column=1, sticky="ew")

        # Progress and Stats
        bottom_frame = ctk.CTkFrame(main_frame, corner_radius=6)
        bottom_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        bottom_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=15, pady=(10, 5))

        self.status_label = ctk.CTkLabel(
            bottom_frame, text="Állapot: Inicializálás...", font=ctk.CTkFont(size=11), text_color="#BDC3C7"
        )
        self.status_label.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 8))

        self.stats_label = ctk.CTkLabel(
            bottom_frame, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498DB"
        )
        self.stats_label.grid(row=1, column=1, sticky="e", padx=15, pady=(0, 8))

    # ==========================================
    # Event Handlers
    # ==========================================
    def _browse_input(self):
        file_path = filedialog.askopenfilename(
            title="Válassz videót vagy képet",
            filetypes=[
                ("Médiafájlok", "*.mp4 *.mkv *.mov *.avi *.jpg *.jpeg *.png *.webp"),
                ("Videók", "*.mp4 *.mkv *.mov *.avi"),
                ("Képek", "*.jpg *.jpeg *.png *.webp"),
                ("Minden fájl", "*.*")
            ]
        )
        if not file_path:
            return

        self.input_file_path = file_path
        self.is_folder = False
        self.album_files = []
        self.lbl_input_path.configure(text=os.path.basename(file_path))

        ext = os.path.splitext(file_path)[1].lower()
        self.is_video = ext in [".mp4", ".mkv", ".mov", ".avi"]

        if self.is_video:
            cap = cv2.VideoCapture(file_path)
            self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider_scrub.configure(to=max(1, self.total_video_frames - 1))
            self.slider_scrub.set(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.current_frame_bgr = frame
                self.scrub_frame.grid()
        else:
            self.current_frame_bgr = imread_safe(file_path)
            self.scrub_frame.grid_remove()
            # If mode is VR180, auto-switch to Full SBS 3D for optimal photo viewing
            if self.MODE_DISPLAY_MAP.get(self.mode_var.get()) == "vr180":
                sbs_label = "Full SBS 3D (Normál képernyős 3D - Teljes szélesség)"
                self.mode_var.set(sbs_label)
                self._on_mode_changed(sbs_label)

        self._invalidate_cache()
        self._trigger_preview_computation()

    def _browse_folder(self):
        folder_path = filedialog.askdirectory(title="Válassz ki egy fotóalbumot / képeket tartalmazó mappát")
        if not folder_path:
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        if not files:
            messagebox.showwarning("Üres mappa", "A kiválasztott mappában nem találhatók képek (.jpg, .png, .webp)!")
            return

        self.input_file_path = folder_path
        self.is_folder = True
        self.is_video = False
        self.album_files = sorted(files)
        self.lbl_input_path.configure(
            text=f"📁 Album: {os.path.basename(folder_path)} ({len(self.album_files)} db fotó)"
        )
        self.scrub_frame.grid_remove()

        # If mode is VR180, auto-switch to Full SBS 3D (_3DH_SBS) for photo albums
        if self.MODE_DISPLAY_MAP.get(self.mode_var.get()) == "vr180":
            sbs_label = "Full SBS 3D (Normál képernyős 3D - Teljes szélesség)"
            self.mode_var.set(sbs_label)
            self._on_mode_changed(sbs_label)

        # Load first photo as live preview
        self.current_frame_bgr = imread_safe(self.album_files[0])
        self._invalidate_cache()
        self._trigger_preview_computation()

    def _on_scrub(self, val):
        if not self.is_video or not self.input_file_path:
            return
        frame_idx = int(val)
        cap = cv2.VideoCapture(self.input_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if ret:
            self.current_frame_bgr = frame
            current_sec = int(frame_idx / fps)
            total_sec = int(self.total_video_frames / fps)
            cur_str = time.strftime("%M:%S", time.gmtime(current_sec))
            tot_str = time.strftime("%M:%S", time.gmtime(total_sec))
            self.lbl_scrub_time.configure(text=f"{cur_str} / {tot_str}")
            self._invalidate_cache()
            self._trigger_preview_computation()

    def _on_model_changed(self, choice):
        self.status_label.configure(text=f"Állapot: Modell váltása ({choice})...")
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _on_mode_changed(self, choice):
        mode_key = self.MODE_DISPLAY_MAP.get(choice, "vr180")
        if mode_key == "vr180":
            self.lbl_fov.pack(padx=15, anchor="w")
            self.slider_fov.pack(fill="x", padx=15, pady=(2, 10))
        else:
            self.lbl_fov.pack_forget()
            self.slider_fov.pack_forget()
        self._render_current_tab()

    def _on_profile_changed(self, choice):
        if choice in self.PROFILE_DISPLAY_MAP:
            ipd, conv = self.PROFILE_DISPLAY_MAP[choice]
            self.slider_ipd.set(ipd)
            self._on_ipd_slide(ipd)
            self.slider_conv.set(conv)
            self._on_conv_slide(conv)

    def _get_ipd_label(self, val):
        pct = int(val * 1000) / 10.0
        if val < 0.025:
            desc = "Lágy (Pihentető)"
        elif val <= 0.042:
            desc = "Természetes (Ajánlott)"
        elif val <= 0.058:
            desc = "Erős (Látványos)"
        else:
            desc = "Extrém (Kiemelkedő)"
        return f"3D Hatás: {desc} [{pct:.1f}%]"

    def _set_preset(self, val):
        self.slider_ipd.set(val)
        self._on_ipd_slide(val)

    def _on_ipd_slide(self, val):
        self.lbl_ipd.configure(text=self._get_ipd_label(val))
        if hasattr(self, "btn_preset_soft"):
            self.btn_preset_soft.configure(fg_color="#1F6AA5" if abs(val - 0.020) < 0.005 else "#34495E")
            self.btn_preset_norm.configure(fg_color="#1F6AA5" if abs(val - 0.035) < 0.005 else "#34495E")
            self.btn_preset_strong.configure(fg_color="#1F6AA5" if abs(val - 0.050) < 0.005 else "#34495E")
        self._invalidate_stereo_cache()
        self._trigger_preview_computation()

    def _on_conv_slide(self, val):
        pct = int(val * 100)
        self.lbl_conv.configure(text=f"Térbeli fókusz (Konvergencia): {pct}%")
        self._invalidate_stereo_cache()
        self._trigger_preview_computation()

    def _on_fov_slide(self, val):
        deg = int(val)
        self.lbl_fov.configure(text=f"VR Látószög (FOV): {deg}°")
        self._render_current_tab()

    def _on_auto_conv_toggle(self):
        self._invalidate_stereo_cache()
        self._trigger_preview_computation()

    def _on_swap_eyes_toggle(self):
        self._invalidate_stereo_cache()
        self._trigger_preview_computation()

    def _on_settings_change(self):
        if self.current_frame_bgr is not None:
            self._trigger_preview_computation()

    def _switch_tab(self, tab_name):
        self.active_tab = tab_name
        if self.wiggle_job is not None:
            self.after_cancel(self.wiggle_job)
            self.wiggle_job = None

        for name, btn in self.tab_buttons.items():
            btn.configure(fg_color="#1F6AA5" if name == tab_name else "#2B2B2B")
        self._render_current_tab()

    def _invalidate_cache(self):
        self.cached_depth = None
        self.cached_left = None
        self.cached_right = None

    def _invalidate_stereo_cache(self):
        self.cached_left = None
        self.cached_right = None

    def _update_preview_manual(self):
        self._invalidate_cache()
        self._trigger_preview_computation()

    def _trigger_preview_computation(self):
        if self.current_frame_bgr is None or self.depth_estimator is None:
            return
        threading.Thread(target=self._compute_preview_worker, daemon=True).start()

    def _compute_preview_worker(self):
        try:
            # Step 1: Depth estimation if not cached
            if self.cached_depth is None:
                self.after(0, lambda: self.status_label.configure(text="Állapot: Mélységbecslés generálása..."))
                self.cached_depth = self.depth_estimator.estimate_depth(self.current_frame_bgr)

            # Step 2: Stereo warping if not cached
            if self.cached_left is None or self.cached_right is None:
                self.after(0, lambda: self.status_label.configure(text="Állapot: Sztereó képpár renderelése..."))
                ipd = self.slider_ipd.get()
                conv = self.slider_conv.get()
                swap = self.chk_swap_eyes.get() == 1
                auto_conv = self.chk_auto_conv.get() == 1
                self.cached_left, self.cached_right = self.stereo_warper.generate_stereo_pair(
                    self.current_frame_bgr, self.cached_depth, ipd_offset=ipd, convergence=conv,
                    swap_eyes=swap, auto_convergence=auto_conv
                )

            self.after(0, lambda: self.status_label.configure(text="Állapot: Előnézet kész."))
            self.after(0, self._render_current_tab)
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Preview error] {err_msg}")
            self.after(0, lambda msg=err_msg: self.status_label.configure(text=f"Előnézeti hiba: {msg}"))

    def _render_current_tab(self):
        if self.current_frame_bgr is None:
            return

        if self.wiggle_job is not None:
            self.after_cancel(self.wiggle_job)
            self.wiggle_job = None

        out_bgr = None
        tab = self.active_tab

        if tab == "Wiggle 3D (Szemüveg nélkül)":
            if self.cached_left is not None and self.cached_right is not None:
                self._step_wiggle()
            return
        elif tab == "2D Eredeti":
            out_bgr = self.current_frame_bgr
        elif tab == "Mélységtérkép":
            if self.cached_depth is not None:
                out_bgr = self.depth_estimator.depth_to_colormap(self.cached_depth)
        elif tab == "Bal Szem":
            out_bgr = self.cached_left
        elif tab == "Jobb Szem":
            out_bgr = self.cached_right
        elif tab == "3D SBS":
            if self.cached_left is not None and self.cached_right is not None:
                out_bgr = self.stereo_warper.create_sbs(self.cached_left, self.cached_right, half_sbs=False)
        elif tab == "Anaglif 3D":
            if self.cached_left is not None and self.cached_right is not None:
                out_bgr = self.stereo_warper.create_anaglyph(self.cached_left, self.cached_right)
        elif tab == "VR180":
            if self.cached_left is not None and self.cached_right is not None:
                projector = VR180Projector(output_eye_size=(1080, 1080), h_fov_deg=self.slider_fov.get())
                out_bgr = projector.project_vr180_sbs(self.cached_left, self.cached_right)

        if out_bgr is not None:
            self._display_bgr_image(out_bgr)

    def _step_wiggle(self):
        if self.active_tab != "Wiggle 3D (Szemüveg nélkül)":
            return
        if self.cached_left is None or self.cached_right is None:
            return

        self.wiggle_eye = 1 - self.wiggle_eye
        frame = self.cached_left if self.wiggle_eye == 0 else self.cached_right
        self._display_bgr_image(frame)
        self.wiggle_job = self.after(130, self._step_wiggle)

    def _display_bgr_image(self, bgr_img: np.ndarray):
        # Resize maintaining aspect ratio to fit canvas
        canvas_w = max(100, self.canvas_frame.winfo_width())
        canvas_h = max(100, self.canvas_frame.winfo_height())

        img_h, img_w = bgr_img.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)

        self.lbl_image.configure(image=photo, text="")
        self.lbl_image.image = photo

    # ==========================================
    # Quick Sample Test (5 seconds)
    # ==========================================
    def _start_quick_test(self):
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            messagebox.showwarning("Figyelmeztetés", "Kérlek, válassz ki egy videófájlt a gyorsteszthez!")
            return

        if not self.is_video:
            messagebox.showinfo("Információ", "Fotóknál a konvertálás azonnali (1 kép ~0.2 mp), használd közvetlenül a Konvertálás gombot!")
            return

        base_dir, file_name = os.path.split(self.input_file_path)
        name, ext = os.path.splitext(file_name)
        mode = self.MODE_DISPLAY_MAP.get(self.mode_var.get(), "vr180")

        suffix_map = {
            "vr180": "_180_SBS",
            "sbs_full": "_3DH_SBS",
            "sbs_half": "_3DH_Half_SBS",
            "anaglyph": "_3D_Anaglyph",
            "depth_only": "_Depth"
        }
        suffix = suffix_map.get(mode, f"_{mode}")
        self.output_file_path = os.path.join(base_dir, f"{name}_MINTA_5mp{suffix}.mp4")

        self.is_processing = True
        self.btn_start.configure(state="disabled")
        if hasattr(self, "btn_quick_test"):
            self.btn_quick_test.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.progress_bar.set(0.0)

        threading.Thread(target=self._quick_test_worker, daemon=True).start()

    def _quick_test_worker(self):
        mode = self.MODE_DISPLAY_MAP.get(self.mode_var.get(), "vr180")
        ipd = self.slider_ipd.get()
        conv = self.slider_conv.get()
        fov = self.slider_fov.get()
        swap = self.chk_swap_eyes.get() == 1
        auto_conv = self.chk_auto_conv.get() == 1
        temp_filter = self.chk_temporal.get() == 1
        nvenc = self.chk_nvenc.get() == 1

        try:
            start_frame = int(self.slider_scrub.get()) if hasattr(self, "slider_scrub") else 0
            cap = cv2.VideoCapture(self.input_file_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            sample_frames = int(fps * 5)

            def on_progress(cur, total, fps_val, eta):
                ratio = cur / max(1, total)
                pct = int(ratio * 100)
                self.after(0, lambda r=ratio: self.progress_bar.set(r))
                self.after(0, lambda c=cur, t=total, p=pct: self.status_label.configure(text=f"5 mp minta generálása: {c}/{t} képkocka ({p}%)"))
                self.after(0, lambda f=fps_val, rem=eta: self.stats_label.configure(text=f"⚡ {f:.1f} FPS | ETA: {rem}"))

            success = self.video_processor.process_video(
                self.input_file_path, self.output_file_path,
                mode=mode, ipd_offset=ipd, convergence=conv,
                use_temporal_filter=temp_filter, use_nvenc=nvenc,
                h_fov=fov, swap_eyes=swap, auto_convergence=auto_conv,
                progress_callback=on_progress,
                start_frame=start_frame,
                max_frames=sample_frames
            )

            if success:
                self.after(0, lambda: messagebox.showinfo(
                    "Minta elkészült!",
                    f"Az 5 másodperces 3D minta videó elkészült:\n{self.output_file_path}\n\nAzonnal megnézheted a headsetben vagy a videólejátszóban a 3D hatást!"
                ))
            else:
                self.after(0, lambda: self.status_label.configure(text="Állapot: Minta generálás megszakítva."))

        except Exception as exc:
            err_msg = str(exc)
            print(f"[Quick test error] {err_msg}")
            self.after(0, lambda msg=err_msg: messagebox.showerror("Hiba történt", f"Nem sikerült a minta generálása:\n{msg}"))
        finally:
            self.is_processing = False
            self.after(0, lambda: self.btn_start.configure(state="normal"))
            if hasattr(self, "btn_quick_test"):
                self.after(0, lambda: self.btn_quick_test.configure(state="normal"))
            self.after(0, lambda: self.btn_cancel.configure(state="disabled"))
            self.after(0, lambda: self.status_label.configure(text="Állapot: Kész."))

    # ==========================================
    # Conversion Management
    # ==========================================
    def _start_conversion(self):
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            messagebox.showwarning("Figyelmeztetés", "Kérlek, válassz ki egy létező videót, képet vagy fotóalbumot!")
            return

        if self.is_folder:
            out_path = filedialog.askdirectory(
                title="Válassz célmappát a 3D fotók mentéséhez",
                initialdir=self.input_file_path
            )
            if not out_path:
                return

            self.output_file_path = out_path
            self.is_processing = True
            self.btn_start.configure(state="disabled")
            if hasattr(self, "btn_quick_test"):
                self.btn_quick_test.configure(state="disabled")
            self.btn_cancel.configure(state="normal")
            self.progress_bar.set(0.0)

            threading.Thread(target=self._conversion_worker, daemon=True).start()
            return

        base_dir, file_name = os.path.split(self.input_file_path)
        name, ext = os.path.splitext(file_name)
        mode = self.MODE_DISPLAY_MAP.get(self.mode_var.get(), "vr180")

        # VR Headset standard naming suffixes (Pico, Quest, Skybox VR recognition)
        suffix_map = {
            "vr180": "_180_SBS",
            "sbs_full": "_3DH_SBS",
            "sbs_half": "_3DH_Half_SBS",
            "anaglyph": "_3D_Anaglyph",
            "depth_only": "_Depth"
        }
        suffix = suffix_map.get(mode, f"_{mode}")
        default_out = os.path.join(base_dir, f"{name}{suffix}{ext}")

        out_path = filedialog.asksaveasfilename(
            title="Kimeneti fájl mentése",
            initialdir=base_dir,
            initialfile=os.path.basename(default_out),
            defaultextension=".mp4" if self.is_video else ".jpg",
            filetypes=[("Videó fájl", "*.mp4")] if self.is_video else [("Képfájl", "*.jpg *.png")]
        )
        if not out_path:
            return

        self.output_file_path = out_path
        self.is_processing = True
        self.btn_start.configure(state="disabled")
        if hasattr(self, "btn_quick_test"):
            self.btn_quick_test.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.progress_bar.set(0.0)

        threading.Thread(target=self._conversion_worker, daemon=True).start()

    def _cancel_conversion(self):
        self.is_processing = False
        if self.video_processor:
            self.video_processor.cancel()
        self.status_label.configure(text="Állapot: Megszakítás folyamatban...")

    def _conversion_worker(self):
        mode = self.MODE_DISPLAY_MAP.get(self.mode_var.get(), "vr180")
        ipd = self.slider_ipd.get()
        conv = self.slider_conv.get()
        fov = self.slider_fov.get()
        swap = self.chk_swap_eyes.get() == 1
        auto_conv = self.chk_auto_conv.get() == 1
        temp_filter = self.chk_temporal.get() == 1
        nvenc = self.chk_nvenc.get() == 1

        try:
            if self.is_folder:
                # Process photo album folder
                total = len(self.album_files)
                os.makedirs(self.output_file_path, exist_ok=True)
                start_time = time.time()

                suffix_map = {
                    "vr180": "_180_SBS",
                    "sbs_full": "_3DH_SBS",
                    "sbs_half": "_3DH_Half_SBS",
                    "anaglyph": "_3D_Anaglyph",
                    "depth_only": "_Depth"
                }
                suffix = suffix_map.get(mode, f"_{mode}")
                success_count = 0
                error_count = 0

                for idx, in_img_path in enumerate(self.album_files):
                    if not self.is_processing:
                        break

                    fname = os.path.basename(in_img_path)
                    root, ext = os.path.splitext(fname)
                    clean_root = root if root.endswith(suffix) else f"{root}{suffix}"
                    out_img_path = os.path.join(self.output_file_path, f"{clean_root}{ext}")

                    ratio = idx / max(1, total)
                    self.after(0, lambda r=ratio: self.progress_bar.set(r))
                    pct = int(ratio * 100)
                    self.after(0, lambda i=idx+1, t=total, p=pct: self.status_label.configure(
                        text=f"Fotóalbum feldolgozása: {i}/{t} kép ({p}%)"
                    ))

                    try:
                        self.video_processor.process_image(
                            in_img_path, out_img_path,
                            mode=mode, ipd_offset=ipd, convergence=conv, h_fov=fov,
                            swap_eyes=swap, auto_convergence=auto_conv
                        )
                        success_count += 1
                    except Exception as img_err:
                        print(f"[Album Error on {fname}]: {img_err}")
                        error_count += 1

                    elapsed = time.time() - start_time
                    fps = (idx + 1) / max(0.001, elapsed)
                    rem_sec = int((total - (idx + 1)) / max(0.001, fps))
                    eta = time.strftime("%M:%S", time.gmtime(rem_sec))
                    self.after(0, lambda f=fps, rem=eta: self.stats_label.configure(
                        text=f"⚡ {f:.1f} kép/mp | ETA: {rem}"
                    ))

                if self.is_processing:
                    self.after(0, lambda: self.progress_bar.set(1.0))
                    self.after(0, lambda: self.status_label.configure(text=f"Állapot: Kész ({success_count} fotó elkészült)."))
                    self.after(0, lambda: messagebox.showinfo(
                        "Siker", f"A fotóalbum ({success_count}/{total} kép) sikeresen elkészült a célmappában:\n{self.output_file_path}"
                    ))
                    try:
                        os.startfile(self.output_file_path)
                    except Exception:
                        pass
                else:
                    self.after(0, lambda: self.status_label.configure(text="Állapot: Mappa konvertálás megszakítva."))

            elif not self.is_video:
                # Process single image
                self.after(0, lambda: self.status_label.configure(text="Kép konvertálása..."))
                self.video_processor.process_image(
                    self.input_file_path, self.output_file_path,
                    mode=mode, ipd_offset=ipd, convergence=conv, h_fov=fov,
                    swap_eyes=swap, auto_convergence=auto_conv
                )
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, lambda: messagebox.showinfo("Siker", f"A 3D kép elkészült:\n{self.output_file_path}"))
            else:
                # Process video
                def on_progress(cur, total, fps_val, eta):
                    ratio = cur / max(1, total)
                    pct = int(ratio * 100)
                    self.after(0, lambda r=ratio: self.progress_bar.set(r))
                    self.after(0, lambda c=cur, t=total, p=pct: self.status_label.configure(text=f"Feldolgozás: {c}/{t} képkocka ({p}%)"))
                    self.after(0, lambda f=fps_val, rem=eta: self.stats_label.configure(text=f"⚡ {f:.1f} FPS | ETA: {rem}"))

                success = self.video_processor.process_video(
                    self.input_file_path, self.output_file_path,
                    mode=mode, ipd_offset=ipd, convergence=conv,
                    use_temporal_filter=temp_filter, use_nvenc=nvenc,
                    h_fov=fov, swap_eyes=swap, auto_convergence=auto_conv,
                    progress_callback=on_progress
                )

                if success:
                    self.after(0, lambda: messagebox.showinfo("Siker", f"A 3D videó sikeresen elkészült:\n{self.output_file_path}"))
                else:
                    self.after(0, lambda: self.status_label.configure(text="Állapot: Konvertálás megszakítva."))

        except Exception as exc:
            err_msg = str(exc)
            print(f"[Conversion error] {err_msg}")
            self.after(0, lambda msg=err_msg: messagebox.showerror("Hiba történt", f"Nem sikerült a konvertálás:\n{msg}"))
        finally:
            self.is_processing = False
            self.after(0, lambda: self.btn_start.configure(state="normal"))
            if hasattr(self, "btn_quick_test"):
                self.after(0, lambda: self.btn_quick_test.configure(state="normal"))
            self.after(0, lambda: self.btn_cancel.configure(state="disabled"))
            self.after(0, lambda: self.status_label.configure(text="Állapot: Kész."))


def main():
    app = VR3DStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
