"""Qwen-Image-2.1-Turbo desktop app (stable-diffusion.cpp edition).

Text-to-image and image editing with Qwen-Image-2.1-Turbo, running on
stable-diffusion.cpp (``sd-server.exe``) with GGUF weights from
``unsloth/Qwen-Image-2.1-Turbo-GGUF``, plus an optional Real-ESRGAN x4
upscale stage handled by sd.cpp's built-in ESRGAN upscaler.

The server is spawned on the first Generate click (with ``--eager-load``) so
the ~10 GB of weights stay pinned in VRAM until the app closes: each
generation is an HTTP job against the local server (127.0.0.1:8321) with no
per-run model loading, and VRAM is released on app exit (unload + terminate).
Server-affecting options (denoiser, text encoder, GPU, engine switches) are
edited in the settings window and applied on the next generate, when the
server respawns if its signature changed.

Usage:
    python app.py
"""

from __future__ import annotations

import base64
import configparser
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

# When frozen into an exe (PyInstaller), __file__ points at the temp
# extraction dir, so anchor to the executable's own location instead.
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = APP_DIR / "prompts.json"
SETTINGS_FILE = APP_DIR / "settings.ini"
IMAGES_DIR = APP_DIR / "images"
MODELS_DIR = APP_DIR / "models"
SERVER_EXE = APP_DIR / "sdcpp" / "sd-server.exe"

DENOISER_DIR = MODELS_DIR / "qwen-image-2.1-turbo"
TEXT_ENCODER_DIR = MODELS_DIR / "text_encoder"
VAE_FILE = MODELS_DIR / "vae" / "qwen_image_2.1_vae_bf16.safetensors"
MMPROJ_FILE = TEXT_ENCODER_DIR / "mmproj-BF16.gguf"
MMPROJ_Q8_FILE = TEXT_ENCODER_DIR / "mmproj-q8_0.gguf"
# Free space needed to create the q8_0 mmproj: the 1.1 GB BF16 file
# converts to ~744 MB; rounded up to 800 MB for headroom.
MMPROJ_Q8_NEED_MB = 800
# Extra reference images (the primary input image is "image 1"). The model
# supports 10, but the graph-cut cache scales with total token count, so on
# a 16 GB card 3 extras is the practical cap.
MAX_EXTRA_REFS = 3
# Relaxed cap when the "limit references to 3" switch is off: 8 extras keeps
# the total (primary + extras) within the model's 10-reference limit.
MAX_EXTRA_REFS_FULL = 8
# Extras are downscaled to this longest side before being sent: a 2048 px
# ref would add 65k DiT tokens and OOM the card.
REF_MAX_SIDE = 768
SD_CLI_EXE = APP_DIR / "sdcpp" / "sd-cli.exe"
ESRGAN_WEIGHTS = MODELS_DIR / "realesrgan" / "realesrnet_x4plus.pth"

ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (768, 512),
    "2:3": (512, 768),
    "16:9": (1024, 576),
    "9:16": (576, 1024),
}

# Chips always visible in their own row; everything else scrolls.
PINNED = ["Colorize", "Restore", "Remove marks"]

# VRAM gate (GB) — a courtesy check before a generation; the server already
# holds the weights in VRAM, so this mostly guards against other apps
# (e.g. LM Studio) eating the headroom the diffusion pass needs.
GEN_VRAM_HEADROOM_GB = 2.0

T2I_TOTAL_STEPS = 8  # the Turbo checkpoint's baked-in 8-step schedule

# The server is bound to loopback and lives for the whole app session.
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8321
# Fallback backend when no CUDA GPU can be enumerated (a single-GPU machine
# lands on cuda0). The GPU dropdown in the settings window normally
# overrides this at runtime.
SERVER_GPU = "cuda0"

# The Turbo checkpoint's fixed 8-step sigma schedule (9 values, final zero
# included). The JSON API takes custom_sigmas as an explicit array, so this
# spells out the named CLI preset "qwen-image-2.1-turbo".
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508,
                0.845148, 0.704534, 0.414568, 0.0]

LOG_FILE = APP_DIR / "app.log"
logger = logging.getLogger("qwen-image-app")
logger.setLevel(logging.DEBUG)
_log_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s"))
logger.addHandler(_log_handler)


def _clear_log() -> None:
    """Clear app.log so each generation starts from a clean log file.

    Truncates through the FileHandler's own stream (opened in append mode)
    rather than a separate handle, so the handler's write offset stays in
    sync: after truncate(0) the next append lands at offset 0.
    """
    _log_handler.stream.seek(0)
    _log_handler.stream.truncate(0)
    _log_handler.stream.flush()


def _edit_resolution(path, target_side) -> tuple[int, int]:
    """Derive -W/-H from the input image so the output keeps its aspect
    ratio. The longest side is scaled to target_side and both
    dimensions are rounded to a multiple of 32 (sd.cpp requirement)."""
    with Image.open(path) as img:
        w_in, h_in = img.size
    scale = target_side / max(w_in, h_in)
    w = max(32, round(w_in * scale / 32) * 32)
    h = max(32, round(h_in * scale / 32) * 32)
    return w, h


def _aspect_resolution(aspect_key: str, target_side: int) -> tuple[int, int]:
    """Scale the selected aspect ratio to a target longest side (rounded
    to a multiple of 32px). Used for forced-aspect editing mode."""
    w_base, h_base = ASPECTS[aspect_key]
    scale = target_side / max(w_base, h_base)
    w = max(32, round(w_base * scale / 32) * 32)
    h = max(32, round(h_base * scale / 32) * 32)
    return w, h


class _Tooltip:
    """Minimal hover tooltip. CustomTkinter 6.x dropped its CTkTooltip,
    so this re-implements the tiny bit we need: show a bordered label
    near the cursor after a short delay, hide on leave."""

    def __init__(self, widget: "tk.Widget", text: str, delay_ms: int = 400) -> None:
        self._widget = widget
        self._text = text
        self._delay_ms = delay_ms
        self._tip: "tk.Toplevel | None" = None
        self._after_id = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, _event: "tk.Event") -> None:
        self._hide()
        self._after_id = self._widget.after(self._delay_ms, self._show)

    def _on_leave(self, _event: "tk.Event") -> None:
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self) -> None:
        if self._tip is not None:
            return
        x = self._widget.winfo_rootx() + 14
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=self._text, background="#1e1e1e", foreground="#eaeaea",
            relief="solid", borderwidth=1, padx=6, pady=3,
            font=("Segoe UI", 9),
        ).pack()

    def _hide(self) -> None:
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def _nvidia_smi(query: str) -> list[list[str]] | None:
    """Run `nvidia-smi --query-gpu <query>` (CSV, no units) and return one
    row of stripped fields per GPU, or None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu", query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        rows = [[p.strip() for p in line.split(",")]
                for line in out.stdout.strip().splitlines()]
        return rows or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def vram_free_gb(index: int = 0) -> float | None:
    """Return free VRAM in GB on the given GPU index (via nvidia-smi), or
    None when unavailable."""
    rows = _nvidia_smi("index,memory.free")
    if not rows:
        return None
    for parts in rows:
        if int(parts[0]) == index:
            return int(parts[1]) / 1024  # MiB -> GB
    return None


def list_gpus() -> list[dict]:
    """Describe every CUDA GPU as {label, device, index, total_gb} in
    nvidia-smi index order. Returns [] when nvidia-smi is unavailable; the
    app then falls back to SERVER_GPU and disables the dropdown."""
    rows = _nvidia_smi("index,name,memory.total")
    if not rows:
        return []
    out = []
    for parts in rows:
        index, name, total_mb = int(parts[0]), parts[1], int(parts[2])
        for prefix in ("NVIDIA ", "GeForce "):
            if name.startswith(prefix):
                name = name[len(prefix):]
        out.append({
            "label": f"GPU {index}: {name} ({round(total_mb / 1024)} GB)",
            "device": f"cuda{index}",
            "index": index,
            "total_gb": round(total_mb / 1024),
        })
    return out


def load_prompts() -> dict:
    # Prefer an external prompts.json next to the app; when frozen, fall
    # back to the copy bundled inside the exe so it runs standalone.
    path = PROMPTS_FILE
    if not path.exists() and getattr(sys, "frozen", False):
        path = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "prompts.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Qwen-Image-2.1 Turbo (sd.cpp)")
        self.geometry("960x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.busy = False
        self._input_image: Image.Image | None = None
        self._input_path: str | None = None
        self._ref_images: list[str] = []  # extra reference paths (image 2+)
        self._ref_tooltips: list[_Tooltip] = []  # open/hover tooltips on the row
        self._img_tooltip: _Tooltip | None = None  # tooltip on the primary thumbnail
        self._photo = None  # keep a reference to the preview PhotoImage
        self.ui_queue: queue.Queue = queue.Queue()
        self._all_chips: list[tuple[ctk.CTkButton, str]] = []

        self._log_console: subprocess.Popen | None = None

        # Server selections: plain attributes (the source of truth), not
        # widget values — the dropdowns/switches that edit them live in the
        # settings window, which is destroyed on close.
        self._gpus = list_gpus()
        denoiser_names = self._gguf_choices(DENOISER_DIR)
        llm_names = self._gguf_choices(TEXT_ENCODER_DIR, prefix_to_skip="mmproj")
        self.denoiser = denoiser_names[0] if denoiser_names else ""
        self.llm = llm_names[0] if llm_names else ""
        self.gpu_label = (max(self._gpus, key=lambda g: g["total_gb"])["label"]
                          if self._gpus else "")
        self.verbose = True
        self.settings = {"sage_attn": False, "te_cpu": False,
                         "mmproj_q8": False, "mmproj_lazy": False,
                         "ref_cap3": True, "ref_downscale": True}
        self._load_settings()  # switch values from settings.ini, if present
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Settings window (destroyed on close; state lives on self).
        self._settings_win: ctk.CTkToplevel | None = None
        self._set_denoiser_menu = None
        self._set_llm_menu = None
        self._set_gpu_menu = None
        self._set_sage_switch = None
        self._set_te_switch = None
        self._set_mmproj_switch = None
        self._set_lazy_switch = None
        self._set_refcap_switch = None
        self._set_refdown_switch = None

        # Server lifecycle: spawned lazily on the first Generate click (with
        # --eager-load) and kept pinned in VRAM until app exit; _ensure_server()
        # spawns it on first use and respawns it if the settings change or
        # the process dies.
        self._server: subprocess.Popen | None = None
        self._server_ready = False
        self._server_sig: tuple[str, str, str, bool, bool, bool, bool] | None = None
        # (denoiser, llm, gpu, sage_attn, te_cpu, mmproj_q8, need_mmproj)
        # the running server was started with.

        self.prompts = load_prompts()
        self._build_ui()
        self._refresh_model_status()
        self.after(100, self._poll)
        logger.info("app started (engine: %s)", SERVER_EXE)

    # ------------------------------------------------------------------ UI

    def _gguf_choices(self, directory: Path, prefix_to_skip: str | None = None) -> list[str]:
        """List *.gguf files in a models subfolder (by name), sorted."""
        if not directory.exists():
            return []
        names = [p.name for p in directory.glob("*.gguf")]
        if prefix_to_skip is not None:
            names = [n for n in names if not n.lower().startswith(prefix_to_skip)]
        return sorted(names)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(9, weight=1)

        # Row 0: model status bar + settings button. The denoiser / text
        # encoder / GPU pickers live in the settings window (this row was
        # cramped); their selections live on self.denoiser / self.llm /
        # self.gpu_label.
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            status_frame, text="Idle — models load on first generate", anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(10, 6), pady=8)

        self.settings_btn = ctk.CTkButton(
            status_frame, text="⚙ Settings", width=110,
            fg_color="#3a3a3a", hover_color="#2e2e2e",
            command=self._open_settings,
        )
        self.settings_btn.grid(row=0, column=1, sticky="e", padx=(0, 10), pady=8)

        # Row 1: prompt textbox.
        self.prompt_box = ctk.CTkTextbox(self, height=110, wrap="word")
        self.prompt_box.grid(row=1, column=0, sticky="ew", padx=10, pady=4)

        # Row 2: collapsible header for the chips + boosters section.
        # Collapsed by default.
        self.chips_collapsed = True
        self.chip_toggle_btn = ctk.CTkButton(self, text="Chips & Boosters  ▶", anchor="w",
                                             height=30, fg_color="#3a3a3a", hover_color="#2e2e2e",
                                             command=self._toggle_chips)
        self.chip_toggle_btn.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 0))

        # Row 3: the chips + boosters content (hidden when collapsed).
        # The previously always-visible "pinned" chips live here too —
        # first in the flow.
        self.chip_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.chip_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.chip_frame.grid_remove()  # start collapsed

        rest = [n for n in self.prompts.get("scenarios", {}) if n not in PINNED]
        boosters = list(self.prompts.get("boosters", {}))
        # Widgets are created unplaced; _layout_chips() flows them into
        # left-aligned rows that wrap to the frame width: the three
        # previously-pinned chips first, then the remaining scenarios,
        # then the boosters.
        self._chip_widgets: list[object] = []
        for name in PINNED + rest + boosters:
            chip = self._make_chip(self.chip_frame, name)
            self._all_chips.append((chip, name))
            self._chip_widgets.append(chip)
        self.chip_frame.bind("<Configure>", lambda e: self._layout_chips())
        self._sync_chips()

        # Row 4: input image picker ("image 1"). Same slot treatment as the
        # references row below: 48 px thumbnail with a full-path tooltip and
        # a "1 ×" caption button. height=28 on the frames stops CTkFrame's
        # 200x200 empty-default from stretching the row while no image is
        # set (thumbnails take over sizing once one is chosen).
        self._img_frame = ctk.CTkFrame(self, fg_color="transparent", height=28)
        self._img_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=2)
        self._img_top = ctk.CTkFrame(self._img_frame, fg_color="transparent", height=28)
        self._img_top.pack(side="top", fill="x")
        self.input_label = ctk.CTkLabel(self._img_top, text="No input image (text-to-image mode)", anchor="w")
        self.input_label.pack(side="left", padx=(0, 8))
        self._img_slot_frame = ctk.CTkFrame(self._img_top, fg_color="transparent", height=28)
        self._img_slot_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(self._img_top, text="Choose image…", width=130, command=self._choose_image).pack(side="left", padx=4)
        ctk.CTkButton(self._img_top, text="Clear", width=80, fg_color="#3a3a3a", hover_color="#2e2e2e",
                       command=self._clear_image).pack(side="left", padx=4)

        # Row 5: extra reference images (the primary above is "image 1").
        # Two lines: label/thumbnails/buttons, then the VRAM hint.
        # height=28 on the frames stops CTkFrame's 200x200 empty-default
        # from stretching the row (it only applies while a frame has no
        # children; thumbnails take over sizing once references exist).
        # The hint is packed below the full-width top row so it spans the
        # whole row and stays left-anchored.
        self._ref_frame = ctk.CTkFrame(self, fg_color="transparent", height=28)
        self._ref_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=2)
        self._ref_top = ctk.CTkFrame(self._ref_frame, fg_color="transparent", height=28)
        self._ref_top.pack(side="top", fill="x")
        self._ref_label = ctk.CTkLabel(self._ref_top, text="References: none", anchor="w")
        self._ref_label.pack(side="left", padx=(0, 8))
        self._ref_slot_frame = ctk.CTkFrame(self._ref_top, fg_color="transparent", height=28)
        self._ref_slot_frame.pack(side="left", fill="x", expand=True)
        self._ref_add_btn = ctk.CTkButton(self._ref_top, text="+ Add", width=80,
                                          command=self._choose_ref)
        self._ref_add_btn.pack(side="left", padx=4)
        self._ref_clear_btn = ctk.CTkButton(self._ref_top, text="Clear", width=70,
                                            fg_color="#3a3a3a", hover_color="#2e2e2e",
                                            command=self._clear_refs)
        self._ref_clear_btn.pack(side="left", padx=4)
        self._ref_hint_label = ctk.CTkLabel(self._ref_frame,
                                            text="1024 base res recommended with references",
                                            font=ctk.CTkFont(size=11), text_color="gray",
                                            anchor="w")
        self._ref_hint_label.pack(side="top", fill="x", pady=(2, 0))

        # Row 6: generation options. Preserve-input-aspect toggle is first
        # (leftmost); it disables the aspect menu when checked. The base
        # resolution dropdown drives both the forced aspect size and the
        # dynamic (input-derived) size.
        opt_frame = ctk.CTkFrame(self, fg_color="transparent")
        opt_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=2)
        self.preserve_aspect_check = ctk.CTkCheckBox(opt_frame, text="Preserve input aspect",
                                                     command=self._on_preserve_aspect_toggle)
        self.preserve_aspect_check.pack(side="left", padx=(0, 12))
        ctk.CTkLabel(opt_frame, text="Aspect:").pack(side="left", padx=(0, 4))
        self.aspect_menu = ctk.CTkOptionMenu(opt_frame, values=list(ASPECTS.keys()), width=90,
                                              command=lambda v: self._log_menu("aspect", v))
        self.aspect_menu.pack(side="left", padx=(0, 12))
        ctk.CTkLabel(opt_frame, text="Base res:").pack(side="left", padx=(0, 4))
        # 4096 is not feasible on a 16 GB GPU: the diffusion pass alone
        # needs ~22 GB at that size (attention scales O(n^2) with latent
        # tokens), so the menu is capped at 2048.
        self.resolution_menu = ctk.CTkOptionMenu(opt_frame, values=["1024", "2048"], width=90,
                                                  command=lambda v: self._log_menu("base resolution", v))
        self.resolution_menu.pack(side="left", padx=(0, 12))
        self.resolution_menu.set("1024")
        ctk.CTkLabel(opt_frame, text="Seed:").pack(side="left", padx=(0, 4))
        self.seed_entry = ctk.CTkEntry(opt_frame, width=90)
        self.seed_entry.insert(0, "42")
        self.seed_entry.pack(side="left", padx=(0, 12))


        # Row 7: upscale options (moved to their own row).
        upscale_frame = ctk.CTkFrame(self, fg_color="transparent")
        upscale_frame.grid(row=7, column=0, sticky="ew", padx=10, pady=2)
        self.upscale_check = ctk.CTkCheckBox(upscale_frame, text="Upscale (Real-ESRGAN)",
                                              command=self._on_upscale_toggle)
        self.upscale_check.pack(side="left", padx=(0, 8))
        self.factor_menu = ctk.CTkOptionMenu(upscale_frame, values=["x4", "x2"], width=70, state="disabled",
                                              command=lambda v: self._log_menu("upscale factor", v))
        self.factor_menu.pack(side="left", padx=(0, 4))
        # ESRGAN tile size (per-request field, no server restart): larger
        # tiles mean fewer seams and much faster upscaling (~30-60 s vs
        # ~2 min at 128 px for a 2048^2 image).
        ctk.CTkLabel(upscale_frame, text="Tile:").pack(side="left", padx=(12, 4))
        self.tile_menu = ctk.CTkOptionMenu(upscale_frame, values=["128", "256"], width=70,
                                            state="disabled",
                                            command=lambda v: self._log_menu("upscale tile", v))
        self.tile_menu.pack(side="left", padx=(0, 4))
        self.tile_menu.set("256")

        # Row 8: generate + progress.
        gen_frame = ctk.CTkFrame(self, fg_color="transparent")
        gen_frame.grid(row=8, column=0, sticky="ew", padx=10, pady=(4, 6))
        self.generate_btn = ctk.CTkButton(gen_frame, text="Generate", width=140, height=36,
                                           command=self._generate, state="disabled")
        self.generate_btn.pack(side="left")
        self.progress = ctk.CTkProgressBar(gen_frame, width=320)
        self.progress.set(0)
        self.progress.pack(side="left", padx=10)
        self.progress_label = ctk.CTkLabel(gen_frame, text="Idle", width=340, anchor="w")
        self.progress_label.pack(side="left", padx=(0, 10))
        ctk.CTkButton(gen_frame, text="View Outputs", width=170,
                      command=self._open_folder).pack(side="right")

        # Row 9: preview.
        self.preview_label = ctk.CTkLabel(self, text="Generated images appear here.",
                                          fg_color="#141414", corner_radius=8)
        self.preview_label.grid(row=9, column=0, sticky="nsew", padx=10, pady=(4, 10))

        # Sync the preserve-aspect toggle now that every widget exists
        # (it needs the aspect menu). With no input image the toggle is
        # disabled + FALSE and the aspect menu is selectable.
        self._sync_preserve_aspect()
        self._refresh_buttons()

    def _make_chip(self, parent, name: str) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent, text=name, width=len(name) * 8 + 28, height=28,
            command=lambda n=name: self._toggle_chip(n),
        )

    def _chip_text(self, name: str) -> str:
        return (
            self.prompts.get("scenarios", {}).get(name)
            or self.prompts.get("boosters", {}).get(name)
            or ""
        )

    def _toggle_chip(self, name: str) -> None:
        text = self._chip_text(name)
        if not text:
            return
        content = self.prompt_box.get("1.0", "end").strip()
        if text.lower() in content.lower():
            new = content.replace(text, "").strip()
            action = "removed"
        else:
            new = (content + " " + text).strip() if content else text
            action = "added"
        logger.info(f"chip: {name} ({action})")
        self._set_prompt(new)

    def _set_prompt(self, text: str) -> None:
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", text)
        self._sync_chips()

    def _sync_chips(self) -> None:
        content = self.prompt_box.get("1.0", "end").lower()
        for btn, name in self._all_chips:
            active = self._chip_text(name).lower() in content
            btn.configure(
                fg_color="#1f6aa5" if active else "#2b2b2b",
                hover_color="#2a7cb8" if active else "#3a3a3a",
            )

    def _layout_chips(self) -> None:
        """Flow-layout the chips + boosters into left-aligned rows that wrap
        to fit the frame width. Runs on resize and when the section expands."""
        if self.chip_frame.winfo_width() < 50:
            return  # hidden or not sized yet; <Configure> fires once it is
        row = col = 0
        # Fixed 6 chips per row — the window is wide enough for six at
        # every chip length we use, and a count-based wrap can never
        # overflow the right edge the way width estimates can.
        for widget in self._chip_widgets:
            if col >= 6:
                row += 1
                col = 0
            widget.grid(row=row, column=col, sticky="w", padx=(8, 8), pady=3)
            col += 1

    def _toggle_chips(self) -> None:
        """Collapse/expand the chips + boosters section."""
        self.chips_collapsed = not self.chips_collapsed
        logger.info(f"chips section: {'collapsed' if self.chips_collapsed else 'expanded'}")
        if self.chips_collapsed:
            self.chip_frame.grid_remove()
            self.chip_toggle_btn.configure(text="Chips & Boosters  ▶")
        else:
            self.chip_frame.grid()
            # Re-flow once the frame has been mapped and sized.
            self.chip_frame.after(0, self._layout_chips)
            self.chip_toggle_btn.configure(text="Chips & Boosters  ▼")

    # ------------------------------------------------- image / folder pick

    def _choose_image(self) -> None:
        if not MMPROJ_FILE.exists():
            logger.warning("image picker: mmproj missing, editing unavailable")
            messagebox.showwarning(
                "Vision weights missing",
                f"Image editing needs {MMPROJ_FILE.name} in {TEXT_ENCODER_DIR}\n\n"
                "Download it (see README, Setup step 3). Text-to-image still works.",
            )
            return
        path = filedialog.askopenfilename(
            title="Choose input image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")],
        )
        if not path:
            logger.info("image picker: cancelled")
            return
        self._input_image = Image.open(path).convert("RGBA")
        self._input_path = path
        self._refresh_img_row()
        logger.info(f"image picker: {path}")
        self._sync_preserve_aspect()

    def _clear_image(self) -> None:
        self._input_image = None
        self._input_path = None
        self._refresh_img_row()
        logger.info("input image cleared")
        self._sync_preserve_aspect()

    def _refresh_img_row(self) -> None:
        """Rebuild the primary image slot (the references row's cousin):
        label text, 48 px thumbnail with a full-path tooltip, and the
        "1 ×" caption button, which removes the primary like Clear."""
        if self._img_tooltip is not None:  # close an open tooltip first
            self._img_tooltip._hide()
        self._img_tooltip = None
        for child in self._img_slot_frame.winfo_children():
            child.destroy()
        # Tk frames latch onto the largest size their children ever
        # requested, so reset the height option on every rebuild: it lets
        # the row shrink back to 28 px after the image is cleared (a new
        # image re-stretches it via its children).
        self._img_slot_frame.configure(height=28)
        self._img_top.configure(height=28)
        self._img_frame.configure(height=28)
        if self._input_path is None:
            self.input_label.configure(text="No input image (text-to-image mode)")
            return
        self.input_label.configure(text="Input:")
        slot = ctk.CTkFrame(self._img_slot_frame, fg_color="transparent")
        slot.pack(side="left", padx=4, pady=2)
        try:
            thumb = Image.open(self._input_path).convert("RGB")
            thumb.thumbnail((48, 48), Image.LANCZOS)
            lbl = ctk.CTkLabel(slot, image=ctk.CTkImage(thumb, size=(48, 48)))
        except OSError:
            lbl = ctk.CTkLabel(slot, text="img")
        lbl.pack(padx=2, pady=(2, 0))
        self._img_tooltip = _Tooltip(lbl, self._input_path)
        ctk.CTkButton(slot, text="1  ×", width=34, height=18,
                      fg_color="#3a3a3a", hover_color="#2e2e2e",
                      command=self._clear_image).pack(pady=(1, 2))

    # --------------------------------------------- extra reference images

    def _max_extra_refs(self) -> int:
        """3 with the 'limit references to 3' switch on, 8 with it off."""
        return MAX_EXTRA_REFS if self.settings["ref_cap3"] else MAX_EXTRA_REFS_FULL

    def _choose_ref(self) -> None:
        if len(self._ref_images) >= self._max_extra_refs():
            return
        if not MMPROJ_FILE.exists():
            logger.warning("reference picker: mmproj missing, references unavailable")
            messagebox.showwarning(
                "Vision weights missing",
                f"Reference images need {MMPROJ_FILE.name} in {TEXT_ENCODER_DIR}\n\n"
                "Download it (see README, Setup step 3). Text-to-image still works.",
            )
            return
        path = filedialog.askopenfilename(
            title="Choose reference image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")],
        )
        if not path:
            logger.info("reference picker: cancelled")
            return
        if path in self._ref_images or path == self._input_path:
            logger.info(f"reference picker: {Path(path).name} already added, skipped")
            return
        self._ref_images.append(path)
        self._refresh_ref_row()
        logger.info(f"reference {len(self._ref_images) + 1} added: {path}")

    def _remove_ref(self, index: int) -> None:
        path = self._ref_images.pop(index)
        self._refresh_ref_row()
        logger.info(f"reference {index + 2} removed: {path}")

    def _clear_refs(self) -> None:
        if not self._ref_images:
            return
        self._ref_images.clear()
        self._refresh_ref_row()
        logger.info("references: all cleared")

    def _refresh_ref_row(self) -> None:
        """Rebuild the reference row from self._ref_images: the label, the
        thumbnail slots (each with its own remove button), and the + Add
        button, which hides at the cap."""
        n = len(self._ref_images)
        self._ref_label.configure(text="References: none" if n == 0
                                  else f"References: {n}")
        for tip in self._ref_tooltips:  # close any open tooltips first
            tip._hide()
        self._ref_tooltips = []
        for child in self._ref_slot_frame.winfo_children():
            child.destroy()
        # Same frame latch as the primary row: reset so the row shrinks
        # back to 28 px when the last reference is removed.
        self._ref_slot_frame.configure(height=28)
        self._ref_top.configure(height=28)
        self._ref_frame.configure(height=28)
        for i, path in enumerate(self._ref_images):
            slot = ctk.CTkFrame(self._ref_slot_frame, fg_color="transparent")
            slot.pack(side="left", padx=4, pady=2)
            try:
                thumb = Image.open(path).convert("RGB")
                thumb.thumbnail((48, 48), Image.LANCZOS)
                lbl = ctk.CTkLabel(slot, image=ctk.CTkImage(thumb, size=(48, 48)))
            except OSError:
                lbl = ctk.CTkLabel(slot, text="img")
            lbl.pack(padx=2, pady=(2, 0))
            self._ref_tooltips.append(_Tooltip(lbl, path))  # full path on hover
            ctk.CTkButton(slot, text=f"{i + 2}  ×", width=34, height=18,
                          fg_color="#3a3a3a", hover_color="#2e2e2e",
                          command=lambda i=i: self._remove_ref(i)).pack(pady=(1, 2))
        if n >= self._max_extra_refs():
            self._ref_add_btn.pack_forget()
        else:
            self._ref_add_btn.pack(side="left", padx=4)
        self._update_ref_hint()

    def _best_aspect_for_refs(self, paths: list[str]) -> str | None:
        """The aspect option that best fits the reference images. The
        engine auto-resizes each ref to the target's pixel area while
        preserving the ref's own aspect, so the option closest to the
        geometric mean of the ref ratios minimizes what the model has to
        invent. None when no ref size can be read."""
        logs = []
        for path in paths:
            try:
                with Image.open(path) as im:
                    w, h = im.size
                logs.append(math.log(w / h))
            except (OSError, ValueError):
                continue
        if not logs:
            return None
        mean = sum(logs) / len(logs)
        best_key, best_dist = None, None
        for key, (w, h) in ASPECTS.items():
            dist = abs(math.log(w / h) - mean)
            if best_dist is None or dist < best_dist:
                best_key, best_dist = key, dist
        return best_key  # exact ties keep the first dropdown option

    def _update_ref_hint(self) -> None:
        """Refresh the refs hint line: the 1024 base-res recommendation,
        plus a suggested output aspect when refs are set and the aspect
        menu is actually selectable (suppressed while preserve-input-
        aspect is on, since the output then follows the primary)."""
        text = "1024 base res recommended with references"
        if self._ref_images and not bool(self.preserve_aspect_check.get()):
            best = self._best_aspect_for_refs(self._ref_images)
            if best:
                text += f" — best output aspect: {best}"
        self._ref_hint_label.configure(text=text)

    def _ref_data_urls(self, w: int, h: int) -> list[str] | None:
        """Base64 PNG data-URLs for the img_gen body's `ref_images`: the
        primary (resized to the target canvas) first, then the extras
        downscaled to REF_MAX_SIDE longest side (while the 'downscale
        references' switch is on). None when no image at all is set."""
        urls: list[str] = []
        if self._input_path is not None:
            ref = Image.open(self._input_path).convert("RGB")
            ref = ref.resize((w, h), Image.LANCZOS)
            buf = BytesIO()
            ref.save(buf, format="PNG")
            urls.append("data:image/png;base64,"
                        + base64.b64encode(buf.getvalue()).decode("ascii"))
        for path in self._ref_images:
            ref = Image.open(path).convert("RGB")
            if self.settings["ref_downscale"]:
                ref.thumbnail((REF_MAX_SIDE, REF_MAX_SIDE), Image.LANCZOS)
            buf = BytesIO()
            ref.save(buf, format="PNG")
            urls.append("data:image/png;base64,"
                        + base64.b64encode(buf.getvalue()).decode("ascii"))
        return urls or None

    def _open_folder(self) -> None:
        IMAGES_DIR.mkdir(exist_ok=True)
        subprocess.Popen(["explorer", str(IMAGES_DIR)])
        logger.info("view outputs clicked")

    # ------------------------------------------------------------ callbacks

    def _on_upscale_toggle(self) -> None:
        on = bool(self.upscale_check.get())
        self.factor_menu.configure(state="normal" if on else "disabled")
        self.tile_menu.configure(state="normal" if on else "disabled")
        logger.info(f"upscale: {'on (factor ' + self.factor_menu.get()
                    + ', tile ' + self.tile_menu.get() + ')' if on else 'off'}")

    def _sync_preserve_aspect(self) -> None:
        """Sync the preserve-aspect toggle with the input image. With an
        input image it is enabled and TRUE (preserving the aspect, which
        disables the aspect menu); with no input image it is disabled and
        FALSE (so the aspect menu is selectable)."""
        if self._input_image is None:
            self.preserve_aspect_check.deselect()
            self.preserve_aspect_check.configure(state="disabled")
            self.aspect_menu.configure(state="normal")
        else:
            self.preserve_aspect_check.select()
            self.preserve_aspect_check.configure(state="normal")
            self.aspect_menu.configure(state="disabled")
        self._update_ref_hint()  # aspect tail depends on the menu state

    def _on_preserve_aspect_toggle(self) -> None:
        on = bool(self.preserve_aspect_check.get())
        # When preserving the input aspect, the aspect menu is irrelevant,
        # so disable it to avoid a misleading selection.
        self.aspect_menu.configure(state="disabled" if on else "normal")
        logger.info(f"preserve input aspect: {'on' if on else 'off'}")
        self._update_ref_hint()

    # ------------------------------------------------------------- workers

    def _model_readiness(self) -> list[str]:
        """Return a list of human-readable problems; empty when ready."""
        problems = []
        if not SERVER_EXE.exists():
            problems.append(f"sd-server.exe not found — extract the release zip into sdcpp/ (expected {SERVER_EXE})")
        denoiser = DENOISER_DIR.glob("*.gguf")
        if not any(denoiser):
            problems.append(f"No *.gguf in {DENOISER_DIR}")
        llm = [p for p in TEXT_ENCODER_DIR.glob("*.gguf")
                if not p.name.lower().startswith("mmproj")]
        if not llm:
            problems.append(f"No text-encoder *.gguf (non-mmproj) in {TEXT_ENCODER_DIR}")
        if not VAE_FILE.exists():
            problems.append(str(VAE_FILE))
        if not ESRGAN_WEIGHTS.exists():
            problems.append(str(ESRGAN_WEIGHTS))
        return problems

    def _generate(self) -> None:
        if self.busy:
            logger.info("generate clicked: ignored (already busy)")
            return
        prompt = self.prompt_box.get("1.0", "end").strip()
        if not prompt:
            logger.info("generate clicked: no prompt, ignored")
            messagebox.showwarning(
                "No prompt",
                "Please enter a prompt before generating.\n\n"
                "Type a description in the prompt box, or click one of the "
                "quick-add chips (Colorize, Restore, Remove marks) to start.",
            )
            self.progress_label.configure(text="Enter a prompt first.")
            return
        free = vram_free_gb(self._gpu_index())
        if free is not None and free < GEN_VRAM_HEADROOM_GB:
            logger.warning(f"generate clicked: VRAM gate blocked ({free:.1f} GB free < "
                           f"{GEN_VRAM_HEADROOM_GB} GB headroom)")
            self.progress_label.configure(
                text=f"Only {free:.1f} GB VRAM free — not enough headroom to generate."
            )
            return
        # Start every generation from a clean log file (the "generate clicked"
        # line below becomes the first line of the new log).
        _clear_log()
        params = {
            "prompt": prompt,
            "aspect": self.aspect_menu.get(),
            "seed": self._get_seed(),
            "input_image": self._input_image,
            "input_path": self._input_path,
            "refs": list(self._ref_images),
            "upscale": bool(self.upscale_check.get()),
            "factor": self.factor_menu.get(),
            "tile": self.tile_menu.get(),
            "preserve_aspect": bool(self.preserve_aspect_check.get()),
            "base_resolution": self.resolution_menu.get(),
        }
        logger.info(f"generate clicked: aspect={params['aspect']} seed={params['seed']} "
                    f"input={params['input_path']} refs={len(params['refs'])} "
                    f"upscale={params['upscale']} "
                    f"factor={params['factor']} tile={params['tile']} "
                    f"preserve_aspect={params['preserve_aspect']} "
                    f"prompt={params['prompt'][:100]!r}")
        self._open_log_console()
        self._set_busy(True)
        self.progress.set(0)
        threading.Thread(target=self._generate_worker, args=(params,), daemon=True).start()

    def _get_seed(self) -> int | None:
        raw = self.seed_entry.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _generate_worker(self, params: dict) -> None:
        t0 = time.time()
        try:
            # Editing mode: either preserve the input image's aspect ratio
            # (dynamic) or use the selected aspect (forced). T2I also scales
            # the selected aspect to the base resolution.
            base_res = int(params["base_resolution"])
            if params["input_image"] is not None:
                if params["preserve_aspect"]:
                    w, h = _edit_resolution(params["input_path"], base_res)
                    logger.info(f"editing mode (dynamic): derived {w}x{h} from input image aspect")
                else:
                    w, h = _aspect_resolution(params["aspect"], base_res)
                    logger.info(f"editing mode (forced): selected aspect {params['aspect']} at {w}x{h}")
            else:
                w, h = _aspect_resolution(params["aspect"], base_res)
            self._ensure_server()  # respawns if the settings changed

            IMAGES_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{params['factor']}" if params["upscale"] else ""
            out_path = IMAGES_DIR / f"qwen_{stamp}{suffix}.png"

            body: dict = {
                "prompt": params["prompt"],
                "width": w,
                "height": h,
                "batch_count": 1,
                "sample_params": {
                    "sample_method": "euler",
                    "sample_steps": T2I_TOTAL_STEPS,
                    # The Turbo checkpoint's fixed schedule (the named CLI
                    # preset "qwen-image-2.1-turbo" spelled out for JSON).
                    "custom_sigmas": list(TURBO_SIGMAS),
                    "guidance": {"txt_cfg": 1.0},
                },
                # Proactive VAE tiling: decodes in 1024px spatial tiles so
                # the full-res 2048^2 decode no longer OOMs. For images
                # smaller than 1024px the tile is larger than the image,
                # so this is a no-op.
                "vae_tiling_params": {
                    "enabled": True,
                    "tile_size_w": 1024,
                    "tile_size_h": 1024,
                    "target_overlap": 0.5,
                },
                "preview": "proj",
                "preview_interval": 1,
                "output_format": "png",
                "output_compression": 100,
            }
            if params["seed"] is not None:
                body["seed"] = params["seed"]
            # Reference images: the primary (resized to the target canvas)
            # first, then the extras (downscaled to REF_MAX_SIDE while the
            # 'downscale references' switch is on). The engine auto-detects
            # the qwen multi-ref preset from the weights.
            urls = self._ref_data_urls(w, h)
            if urls:
                body["ref_images"] = urls

            logger.info(f"POST /sdcpp/v1/img_gen {w}x{h} seed={params['seed']} "
                        f"refs={len(urls) if urls else 0} "
                        f"prompt={params['prompt'][:100]!r}")
            job_id = self._server_http("POST", "/sdcpp/v1/img_gen", body)["id"]
            logger.info(f"job {job_id} accepted")

            # One overall percentage for the whole run: sampling maps onto
            # 10-85% from the job preview's step counter, decode creeps
            # 85 -> 97 at 0.8%/s, save is 98%, upscale runs past that, and
            # 100% once everything is on disk. The bar never goes backwards.
            self.ui_queue.put(("progress", (0.10, "10% — sampling…")))
            max_pct = 10.0
            last_nudge = time.time()
            while True:
                job = self._server_http("GET", f"/sdcpp/v1/jobs/{job_id}")
                state = job.get("status")
                if state in ("failed", "cancelled"):
                    err = job.get("error") or {}
                    detail = err.get("message") or err.get("code") or "no detail"
                    raise RuntimeError(f"job {state}: {detail}")
                if state == "completed":
                    break
                # queued / generating — map the sampling step onto 10-85%.
                preview = job.get("preview") or {}
                step, total = preview.get("step"), preview.get("total_steps")
                if step and total and total > 1:
                    pct = 10.0 + 75.0 * (step - 1) / (total - 1)
                elif step:
                    pct = 10.0
                else:
                    pct = max_pct
                if pct >= 85.0:
                    # Sampling done; decode creep 85 -> 97 at 0.8%/s.
                    pct = min(97.0, max_pct + 0.8 * max(0.0, time.time() - last_nudge))
                    last_nudge = time.time()
                if pct > max_pct + 0.05:
                    max_pct = pct
                    logger.info(f"progress: {max_pct:.0f}%")
                    self.ui_queue.put(("progress", (max_pct / 100.0, f"{max_pct:.0f}%")))
                time.sleep(1.0)

            images = (job.get("result") or {}).get("images") or []
            if not images or not images[0].get("b64_json"):
                raise RuntimeError(f"job {job_id} completed without image data")
            out_path.write_bytes(base64.b64decode(images[0]["b64_json"]))
            max_pct = 98.0
            logger.info(f"image saved: {out_path}")
            self.ui_queue.put(("progress", (0.98, "98% — saved")))

            final_path = out_path
            if params["upscale"]:
                self.ui_queue.put(
                    ("progress", (max(0.9, max_pct / 100.0), "Upscaling (Real-ESRGAN)…"))
                )
                t_up = time.time()
                self._upscale_image(out_path, int(params["tile"]))
                logger.info(f"upscale done in {time.time() - t_up:.1f}s (tile {params['tile']})")
                if params["factor"] == "x2":
                    # x2 = native x4 output downscaled with Lanczos.
                    up = Image.open(out_path)
                    final_path = out_path.with_name(f"{out_path.stem}_x2.png")
                    up.save(final_path, resize=(up.width // 2, up.height // 2), resample=Image.LANCZOS)
                    logger.info(f"downscaled x4 output to x2: {final_path}")

            self.ui_queue.put(("progress", (1.0, "100%")))
            logger.info(f"generation complete: {final_path} (total {time.time() - t0:.1f}s)")
            # Decode + thumbnail HERE, off the UI thread: the final file can
            # be an 8192^2 PNG and decoding it in _poll froze the window.
            # Only the cheap PhotoImage conversion happens on the UI thread.
            img = Image.open(final_path)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (30, 30, 30))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert("RGB")
            img.thumbnail((560, 560), Image.LANCZOS)
            self.ui_queue.put(("image", (img, Path(final_path).name)))
            self.ui_queue.put(("idle", None))
        except Exception as e:
            logger.error(f"generation failed: {e}")
            self.ui_queue.put(("error", f"Generation failed: {e}"))
            self.ui_queue.put(("idle", None))

    # ---------------------------------------------------------------- server

    def _selected_gpu(self) -> dict | None:
        for g in self._gpus:
            if g["label"] == self.gpu_label:
                return g
        return None

    def _gpu_device(self) -> str:
        g = self._selected_gpu()
        return g["device"] if g else SERVER_GPU

    def _gpu_index(self) -> int:
        g = self._selected_gpu()
        return g["index"] if g else 0

    def _mmproj_file(self) -> Path:
        """Vision-model file for --llm_vision: the q8_0 convert when the
        switch is on and the file exists (created on demand by
        _ensure_mmproj_q8), otherwise the BF16 original."""
        if self.settings["mmproj_q8"] and MMPROJ_Q8_FILE.exists():
            return MMPROJ_Q8_FILE
        return MMPROJ_FILE

    def _mmproj_q8_available(self) -> tuple[bool, str]:
        """Whether the q8_0 vision model can be used: the file already
        exists, or there is enough free disk space to create it.
        Returns (ok, reason); reason is non-empty only when not ok."""
        if MMPROJ_Q8_FILE.exists():
            return True, ""
        free_mb = shutil.disk_usage(TEXT_ENCODER_DIR).free / 2**20
        if free_mb >= MMPROJ_Q8_NEED_MB:
            return True, ""
        return False, (f"needs ~{MMPROJ_Q8_NEED_MB} MB free, "
                       f"~{free_mb:.0f} MB available")

    def _server_args(self) -> list[str]:
        denoiser = self.denoiser
        llm = self.llm
        dev = self._gpu_device()
        # --backend places the COMPUTE (activations), --params-backend the
        # WEIGHTS. te=cpu needs both: without the --backend override the TE
        # forward pass still runs on the GPU and its ~7.8 GB compute buffer
        # spikes VRAM back up (that is what "clip-on-cpu" did pre-rename).
        backend = f"diffusion={dev},vae={dev},te=cpu" if self.settings["te_cpu"] else dev
        args = [
            str(SERVER_EXE),
            "--model-dir", str(MODELS_DIR),
            "--diffusion-model", str(DENOISER_DIR / denoiser),
            "--llm", str(TEXT_ENCODER_DIR / llm),
            "--vae", str(VAE_FILE),
            # Modern form of the deprecated --diffusion-fa.
            "--flash-attn", "off,diffusion=on",
            # Pin every component to the selected GPU (settings window).
            "--backend", backend,
            # Load the weights at startup and keep them pinned in VRAM for
            # the lifetime of the app.
            "--eager-load",
            "--upscale-model", str(ESRGAN_WEIGHTS),
            "--hires-upscalers-dir", str(ESRGAN_WEIGHTS.parent),
            "--listen-ip", SERVER_HOST,
            "--listen-port", str(SERVER_PORT),
            "--log-level", "debug",
        ]
        # Vision model (mmproj) only when this spawn needs it: always,
        # unless the on-demand switch is on and no input image is set.
        if self._need_mmproj():
            args += ["--llm_vision", str(self._mmproj_file())]
        # Optional engine switches (settings window). Each one changes the
        # server signature, so toggling one respawns the server on the next
        # generate.
        if self.settings["sage_attn"]:
            args.append("--sage-attn")
        if self.settings["te_cpu"]:
            args += ["--params-backend", "te=cpu"]
        return args

    def _server_http(self, method: str, path: str, body: dict | None = None,
                     timeout: float = 120.0) -> dict:
        """One HTTP round-trip against the local sd-server (parsed JSON)."""
        url = f"http://{SERVER_HOST}:{SERVER_PORT}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                detail = ""
            raise RuntimeError(f"server HTTP {e.code} for {method} {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"server unreachable at {url}: {e.reason}") from e

    def _spawn_server(self) -> None:
        args = self._server_args()
        logger.info("server command: " + subprocess.list2cmdline(args))
        # Server stdout/stderr is piped to app.log at DEBUG level (gated by
        # the verbose toggle), like the old sd-cli lines.
        self._server = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(APP_DIR),
        )
        self._server_sig = self._server_sig_current()
        threading.Thread(target=self._server_stdout_reader, daemon=True).start()
        logger.info(f"server spawned (pid {self._server.pid}); loading ~10 GB of weights into VRAM")

    def _server_stdout_reader(self) -> None:
        server = self._server
        if server is None or server.stdout is None:
            return
        for line in server.stdout:
            logger.debug("server: %s", line.rstrip())
        rc = server.wait()
        logger.info(f"server process exited with code {rc}")
        # If the server died on its own (not via _stop_server), surface it.
        # The next Generate will respawn it (_ensure_server sees no live proc).
        if self._server is server:
            self._server = None
            self._server_ready = False
            self.ui_queue.put(
                ("server_status", "Server process exited — it will restart on the next generate.")
            )
            self.ui_queue.put(("refresh_buttons", None))

    def _wait_server_ready(self, timeout: float = 300.0) -> None:
        """Poll /sdcpp/v1/models/status until the loaded models report ready."""
        t0 = time.time()
        while True:
            if self._server is None or self._server.poll() is not None:
                raise RuntimeError("server process exited while loading models")
            try:
                status = self._server_http("GET", "/sdcpp/v1/models/status", timeout=10)
                state = status.get("state", "")
            except RuntimeError:
                status, state = {}, ""  # server not accepting connections yet
            if state == "error":
                raise RuntimeError(f"server model load failed: {status.get('error') or 'no detail'}")
            if state == "ready":
                elapsed = int(time.time() - t0)
                self._server_ready = True
                logger.info(f"server ready after {elapsed}s — models pinned in VRAM")
                self.ui_queue.put(("server_status", f"Model ready — pinned in VRAM ({elapsed}s)"))
                self.ui_queue.put(("refresh_buttons", None))
                return
            self.ui_queue.put(("server_status", f"Loading models… ({int(time.time() - t0)}s)"))
            if time.time() - t0 > timeout:
                raise RuntimeError(f"server did not become ready within {timeout:.0f}s")
            time.sleep(1.0)

    def _need_mmproj(self) -> bool:
        """Whether this spawn needs the vision model (--llm_vision):
        always, unless the on-demand switch is on and no image of any
        kind is set (text-to-image never touches the vision tower).
        Reference images count too — they are passed to the VLM as well
        as the DiT, so without mmproj the model would silently ignore
        them."""
        return ((not self.settings["mmproj_lazy"])
                or (self._input_image is not None)
                or (len(self._ref_images) > 0))

    def _server_sig_current(self) -> tuple[str, str, str, bool, bool, bool, bool]:
        """The settings the running server must have been started with:
        (denoiser, llm, gpu, sage_attn, te_cpu, mmproj_q8, need_mmproj)."""
        return (self.denoiser, self.llm, self._gpu_device(),
                self.settings["sage_attn"], self.settings["te_cpu"],
                self.settings["mmproj_q8"], self._need_mmproj())

    def _ensure_mmproj_q8(self) -> None:
        """Create the q8_0 mmproj (one-time, a few seconds) when the switch
        is on and the file is missing. Runs sd-cli's convert mode on CPU.
        Falls back to the BF16 original on any failure so a conversion
        problem never blocks a generation."""
        if not self.settings["mmproj_q8"] or MMPROJ_Q8_FILE.exists():
            return
        if not (MMPROJ_FILE.exists() and SD_CLI_EXE.exists()):
            logger.warning("mmproj q8_0: source file or sd-cli missing; using BF16")
            self.ui_queue.put(("server_status",
                               "q8_0 vision model unavailable — using BF16 (see log)"))
            return
        self.ui_queue.put(("server_status",
                           "Creating q8_0 vision model (one-time)…"))
        logger.info("converting mmproj to q8_0: %s -> %s",
                    MMPROJ_FILE.name, MMPROJ_Q8_FILE.name)
        try:
            proc = subprocess.run(
                [str(SD_CLI_EXE), "-M", "convert",
                 "-m", str(MMPROJ_FILE), "-o", str(MMPROJ_Q8_FILE),
                 "--type", "q8_0", "--log-level", "warn"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=600, cwd=str(APP_DIR),
            )
        except Exception as e:
            logger.warning(f"mmproj q8_0 conversion failed: {e}; using BF16")
            self.ui_queue.put(("server_status",
                               "q8_0 vision model unavailable — using BF16 (see log)"))
            return
        if proc.returncode != 0 or not MMPROJ_Q8_FILE.exists():
            tail = " | ".join((proc.stderr or proc.stdout or "").strip().splitlines()[-3:])
            logger.warning("mmproj q8_0 conversion failed (rc=%s): %s; using BF16",
                           proc.returncode, tail)
            self.ui_queue.put(("server_status",
                               "q8_0 vision model unavailable — using BF16 (see log)"))
            return
        logger.info("mmproj q8_0 ready: %.0f MB",
                    MMPROJ_Q8_FILE.stat().st_size / 1e6)

    def _ensure_server(self) -> None:
        """Make sure a server matching the current settings is running.
        Spawns + loads on first use; respawns (one-time load cost) if the
        user changed anything server-affecting (denoiser, text encoder, GPU,
        engine switches) since it was started, or if the process died."""
        sig = self._server_sig_current()
        if self._server is not None and self._server.poll() is None and sig == self._server_sig:
            return
        logger.info("server settings changed — respawning the server")
        self._stop_server()
        self._ensure_mmproj_q8()
        self._spawn_server()
        self._wait_server_ready()

    def _upscale_image(self, path: Path, tile: int) -> None:
        """Upscale a PNG via the server's ESRGAN endpoint, overwriting it.
        ``tile`` is the ESRGAN tile size in px (per-request, no restart)."""
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        body = {
            "image": f"data:image/png;base64,{b64}",
            "upscaler": ESRGAN_WEIGHTS.stem,
            "repeats": 1,
            "tile_size": tile,
        }
        resp = self._server_http("POST", "/sdcpp/v1/upscale", body, timeout=1800)
        images = resp.get("images") or []
        if not images or not images[0].get("b64_json"):
            raise RuntimeError("upscale returned no image data")
        path.write_bytes(base64.b64decode(images[0]["b64_json"]))

    def _stop_server(self) -> None:
        """Unload the models (best effort) and terminate the server."""
        server = self._server
        if server is None:
            return
        self._server = None
        self._server_ready = False
        try:
            self._server_http("POST", "/sdcpp/v1/models/unload", timeout=30)
            logger.info("server: models unloaded")
        except Exception as e:
            logger.warning(f"server: unload failed ({e}); terminating anyway")
        try:
            server.terminate()
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("server: terminate timed out, killing")
            server.kill()
            server.wait()
        logger.info("server stopped")

    # ------------------------------------------------------------- plumbing

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._refresh_buttons()
        self._sync_settings_busy()

    def _sync_settings_busy(self) -> None:
        """Disable the settings-window controls that change the server
        signature while a generation runs (the server cannot be restarted
        mid-run). The verbose switch and the Settings button stay live."""
        win = self._settings_win
        if win is None or not win.winfo_exists():
            return
        if self.busy:
            for w in (self._set_denoiser_menu, self._set_llm_menu,
                      self._set_gpu_menu, self._set_sage_switch,
                      self._set_te_switch, self._set_mmproj_switch,
                      self._set_lazy_switch):
                if w is not None:
                    w.configure(state="disabled")
            return
        # Not busy: re-enable, except menus with no choices (no .gguf files /
        # no CUDA GPU), which stay disabled.
        denoiser_names = self._gguf_choices(DENOISER_DIR)
        llm_names = self._gguf_choices(TEXT_ENCODER_DIR, prefix_to_skip="mmproj")
        if self._set_denoiser_menu is not None:
            self._set_denoiser_menu.configure(state="normal" if denoiser_names else "disabled")
        if self._set_llm_menu is not None:
            self._set_llm_menu.configure(state="normal" if llm_names else "disabled")
        if self._set_gpu_menu is not None:
            self._set_gpu_menu.configure(state="normal" if self._gpus else "disabled")
        for w in (self._set_sage_switch, self._set_te_switch,
                  self._set_mmproj_switch, self._set_lazy_switch):
            if w is not None:
                w.configure(state="normal")

    def _refresh_buttons(self) -> None:
        # Generate is enabled as soon as the model files are in place; the
        # server itself spawns on first use (_ensure_server in the worker).
        self.generate_btn.configure(
            state="disabled" if (self.busy or self._model_readiness()) else "normal",
        )

    def _refresh_model_status(self) -> None:
        problems = self._model_readiness()
        if SERVER_EXE.exists():
            txt = "Model files ready."
        else:
            txt = f"Engine missing — expected {SERVER_EXE}"
        missing = [p for p in problems if p != str(SERVER_EXE)]
        if missing:
            txt += " Missing: " + ", ".join(Path(p).name for p in missing)
        self.status_label.configure(text=txt)
        self._refresh_buttons()

    def _open_log_console(self) -> None:
        """Open a console window tailing app.log (fresh each generation).

        Terminate any previous console and reopen it so the window always
        reflects the clean log of the current run, rather than relying on
        Get-Content -Wait noticing the truncation.
        """
        if self._log_console is not None and self._log_console.poll() is None:
            self._log_console.terminate()
            self._log_console = None
        self._log_console = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command",
             f"$Host.UI.RawUI.WindowTitle = 'qwen-image-app — app.log'; "
             f"Get-Content '{LOG_FILE}' -Wait"],
            cwd=str(APP_DIR),
        )
        logger.info("log console opened")

    def _log_menu(self, what: str, value: str) -> None:
        logger.info(f"menu: {what} = {value}")

    # --------------------------------------------------------- settings win

    def _open_settings(self) -> None:
        """Open (or refocus) the settings window. The window is destroyed
        on close; the selections it edits live on app attributes, so state
        survives closing and reopening."""
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.lift()
            self._settings_win.focus_force()
            return

        win = ctk.CTkToplevel(self)
        win.title("Settings")
        win.geometry("480x560")
        win.resizable(False, False)
        self._settings_win = win

        # Windows does not guarantee a fresh top-level is mapped above the
        # main window (the main window may still own the foreground when
        # the Toplevel is created), and at 480x560 inside the 960x780 main
        # window that means fully covered. The window is still unmapped
        # here, so the topmost flag must survive the map: it is released
        # by a delayed callback, giving the window manager time to
        # process the raise while the window is topmost.
        win.attributes("-topmost", True)
        win.lift()
        win.focus_force()

        def _drop_topmost() -> None:
            if win.winfo_exists():
                win.attributes("-topmost", False)

        win.after(250, _drop_topmost)

        # Footer is pinned to the window (outside the scroll area) so it
        # stays visible no matter how many sections there are.
        self._settings_footer = ctk.CTkLabel(
            win,
            text=("Server-affecting settings are disabled while a "
                  "generation is running.\nChanges apply on the next "
                  "generate — the server restarts automatically."),
            font=ctk.CTkFont(size=11), text_color="gray", justify="left",
        )
        self._settings_footer.pack(side="bottom", fill="x", padx=16, pady=(6, 10))

        # The sections scroll when they no longer fit in the window.
        body = ctk.CTkScrollableFrame(win, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self._settings_body = body

        denoiser_names = self._gguf_choices(DENOISER_DIR)
        llm_names = self._gguf_choices(TEXT_ENCODER_DIR, prefix_to_skip="mmproj")
        gpu_labels = [g["label"] for g in self._gpus]

        def section(parent, title: str) -> ctk.CTkFrame:
            frame = ctk.CTkFrame(parent)
            frame.pack(fill="x", padx=12, pady=(10, 0))
            ctk.CTkLabel(frame, text=title,
                         font=ctk.CTkFont(size=13, weight="bold")).pack(
                anchor="w", padx=10, pady=(8, 2))
            return frame

        # -- Model: denoiser / text encoder / GPU (all change the server
        #    signature -> respawn on the next generate) --
        model = section(body, "Model")
        self._set_denoiser_menu = ctk.CTkOptionMenu(
            model, values=denoiser_names or ["(no .gguf found)"],
            state="normal" if denoiser_names else "disabled",
            command=self._on_settings_denoiser)
        self._set_denoiser_menu.set(self.denoiser or "(no .gguf found)")
        self._set_denoiser_menu.pack(fill="x", padx=10, pady=(2, 6))

        self._set_llm_menu = ctk.CTkOptionMenu(
            model, values=llm_names or ["(no .gguf found)"],
            state="normal" if llm_names else "disabled",
            command=self._on_settings_llm)
        self._set_llm_menu.set(self.llm or "(no .gguf found)")
        self._set_llm_menu.pack(fill="x", padx=10, pady=(0, 6))

        self._set_gpu_menu = ctk.CTkOptionMenu(
            model, values=gpu_labels or ["(no CUDA GPU detected)"],
            state="normal" if gpu_labels else "disabled",
            command=self._on_settings_gpu)
        self._set_gpu_menu.set(self.gpu_label or "(no CUDA GPU detected)")
        self._set_gpu_menu.pack(fill="x", padx=10, pady=(0, 10))

        # -- Engine switches (all change the server signature) --
        engine = section(body, "Engine")

        def engine_switch(frame, text: str, desc: str, command) -> ctk.CTkSwitch:
            sw = ctk.CTkSwitch(frame, text=text, command=command)
            sw.pack(anchor="w", padx=10, pady=(8, 0))
            ctk.CTkLabel(frame, text=desc, font=ctk.CTkFont(size=11),
                         text_color="gray").pack(anchor="w", padx=42)
            return sw

        self._set_sage_switch = engine_switch(
            engine, "Sage Attention", "INT8 attention, faster diffusion",
            self._on_settings_sage)
        self._set_te_switch = engine_switch(
            engine, "Text encoder on CPU",
            "Frees ~6 GB VRAM; encoding runs on CPU (slower)",
            self._on_settings_te)
        # The q8 switch is always created enabled; the free-disk-space check
        # happens when the user turns it on (_on_settings_mmproj), not here.
        self._set_mmproj_switch = engine_switch(
            engine, "Vision model q8_0 (mmproj)",
            "Reduces the 1.1 GB vision model to ~0.7 GB; created "
            "once on first use",
            self._on_settings_mmproj)
        self._set_lazy_switch = engine_switch(
            engine, "Load vision model on demand",
            "Skip the ~1.1 GB vision model for text-to-image; loads "
            "automatically once an input image is set",
            self._on_settings_lazy)

        # -- Reference images (per-request; no server restart) --
        refsec = section(body, "Reference images")
        self._set_refcap_switch = engine_switch(
            refsec, "Limit references to 3",
            "Off accepts up to 8 extras (much more VRAM)",
            self._on_settings_refcap)
        self._set_refdown_switch = engine_switch(
            refsec, "Downscale references to ~768p",
            "Shrinks each extra to a 768 px longest side before sending",
            self._on_settings_refdown)

        # -- Logging (no server restart) --
        logsec = section(body, "Logging")
        self._set_verbose_switch = ctk.CTkSwitch(
            logsec, text="Verbose logging", command=self._on_settings_verbose)
        self._set_verbose_switch.pack(anchor="w", padx=10, pady=(8, 10))

        # Reflect the current attribute values, then apply the busy state
        # (the window may be opened mid-generation).
        for sw, key in ((self._set_sage_switch, "sage_attn"),
                        (self._set_te_switch, "te_cpu"),
                        (self._set_mmproj_switch, "mmproj_q8"),
                        (self._set_lazy_switch, "mmproj_lazy"),
                        (self._set_refcap_switch, "ref_cap3"),
                        (self._set_refdown_switch, "ref_downscale")):
            (sw.select() if self.settings[key] else sw.deselect())
        (self._set_verbose_switch.select() if self.verbose
         else self._set_verbose_switch.deselect())
        self._sync_settings_busy()

        win.protocol("WM_DELETE_WINDOW", self._close_settings)

    def _close_settings(self) -> None:
        if self._settings_win is not None:
            self._settings_win.destroy()
            self._settings_win = None

    def _notify_settings_changed(self) -> None:
        self.status_label.configure(
            text="Settings changed — server will restart on the next generate")

    def _on_settings_denoiser(self, value: str) -> None:
        self.denoiser = value
        logger.info(f"settings: denoiser = {value}")
        self._notify_settings_changed()

    def _on_settings_llm(self, value: str) -> None:
        self.llm = value
        logger.info(f"settings: text encoder = {value}")
        self._notify_settings_changed()

    def _on_settings_gpu(self, value: str) -> None:
        self.gpu_label = value
        logger.info(f"settings: gpu = {value}")
        self._notify_settings_changed()

    def _load_settings(self) -> None:
        """Restore the persisted switch values from settings.ini. The file
        is only ever created by _save_settings, so a missing file is the
        normal first-run case. Any read/parse problem falls back to the
        defaults — the file must never be able to block app startup."""
        if not SETTINGS_FILE.exists():
            return
        try:
            cp = configparser.ConfigParser()
            cp.read(SETTINGS_FILE, encoding="utf-8")
            for key in self.settings:
                if cp.has_option("engine", key):
                    self.settings[key] = cp.getboolean("engine", key)
            if cp.has_option("ui", "verbose"):
                self.verbose = cp.getboolean("ui", "verbose")
            logger.info(f"settings loaded from {SETTINGS_FILE.name}: "
                        f"{self.settings}, verbose={self.verbose}")
        except (configparser.Error, OSError) as e:
            logger.warning(f"settings.ini unreadable ({e}); using defaults")

    def _save_settings(self) -> None:
        """Persist the switch values to settings.ini. Written atomically
        (temp file + os.replace) so a crash mid-write cannot truncate the
        file; a write failure only logs — in-session state is unaffected."""
        try:
            cp = configparser.ConfigParser()
            cp["engine"] = {k: str(v).lower() for k, v in self.settings.items()}
            cp["ui"] = {"verbose": str(self.verbose).lower()}
            tmp = SETTINGS_FILE.with_suffix(".ini.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                cp.write(f)
            os.replace(tmp, SETTINGS_FILE)
            logger.info("settings saved to settings.ini")
        except OSError as e:
            logger.warning(f"settings.ini not saved: {e}")

    def _on_settings_sage(self) -> None:
        self.settings["sage_attn"] = bool(self._set_sage_switch.get())
        logger.info(f"settings: sage-attn = {self.settings['sage_attn']}")
        self._notify_settings_changed()
        self._save_settings()

    def _on_settings_te(self) -> None:
        self.settings["te_cpu"] = bool(self._set_te_switch.get())
        logger.info(f"settings: text encoder on CPU = {self.settings['te_cpu']}")
        self._notify_settings_changed()
        self._save_settings()

    def _on_settings_mmproj(self) -> None:
        want = bool(self._set_mmproj_switch.get())
        if want:
            ok, reason = self._mmproj_q8_available()
            if not ok:
                # No room for the one-time conversion: keep the switch off
                # and tell the user why (checked on enable, not at open).
                self._set_mmproj_switch.deselect()
                logger.warning(f"settings: vision model q8_0 denied — {reason}")
                messagebox.showwarning(
                    "q8_0 vision model unavailable",
                    f"The switch stays off: {reason}\n\n"
                    "Free up disk space and try again, or keep the BF16 "
                    "vision model.",
                )
                return
        self.settings["mmproj_q8"] = want
        logger.info(f"settings: vision model q8_0 = {self.settings['mmproj_q8']}")
        self._notify_settings_changed()
        self._save_settings()

    def _on_settings_lazy(self) -> None:
        self.settings["mmproj_lazy"] = bool(self._set_lazy_switch.get())
        logger.info(f"settings: load vision model on demand = {self.settings['mmproj_lazy']}")
        self._notify_settings_changed()
        self._save_settings()

    def _on_settings_refcap(self) -> None:
        self.settings["ref_cap3"] = bool(self._set_refcap_switch.get())
        logger.info(f"settings: reference cap = "
                    f"{'3 (switch on)' if self.settings['ref_cap3'] else '8 (switch off)'}")
        # Per-request setting: no server restart, so no status nudge. The
        # row refresh makes + Add (re)appear if the cap loosened.
        self._refresh_ref_row()
        self._save_settings()

    def _on_settings_refdown(self) -> None:
        self.settings["ref_downscale"] = bool(self._set_refdown_switch.get())
        logger.info(f"settings: reference downscale = "
                    f"{'on' if self.settings['ref_downscale'] else 'off'}")
        self._save_settings()

    def _on_settings_verbose(self) -> None:
        self.verbose = bool(self._set_verbose_switch.get())
        # Verbose ON = DEBUG (every server output line, etc.); OFF = INFO
        # (progress, command line, clicks — no per-line server output).
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.info(f"verbose logging: {'on' if self.verbose else 'off'}")
        self._save_settings()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "progress":
                    frac, text = payload
                    self.progress.set(frac)
                    self.progress_label.configure(text=text)
                elif kind == "image":
                    img, name = payload
                    self._show_preview(img)
                    self.progress_label.configure(text=f"Saved: {name}")
                elif kind == "error":
                    self.progress_label.configure(text=f"Error: {payload}")
                elif kind == "server_status":
                    self.status_label.configure(text=payload)
                elif kind == "refresh_buttons":
                    self._refresh_buttons()
                elif kind == "idle":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _show_preview(self, img: Image.Image) -> None:
        """Install a pre-decoded, pre-thumbed image. The heavy decode +
        thumbnail work happens in the generate worker; only the cheap
        PhotoImage conversion runs on the UI thread, so an 8192^2 PNG no
        longer freezes the window."""
        self._photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._photo, text="")

    def on_close(self) -> None:
        if self.busy:
            if not messagebox.askokcancel(
                "Quit",
                "A job is still running. Quit anyway?\n"
                "(VRAM is freed when the server process is terminated.)",
            ):
                logger.info("quit requested while busy: cancelled")
                return
            logger.warning("quit while generation still running")
        else:
            logger.info("app closed")
        self._stop_server()
        if self._log_console is not None and self._log_console.poll() is None:
            self._log_console.terminate()
            self._log_console = None
        self.destroy()


def main() -> None:
    # Suppress the console developer warning where the API exists.
    # customtkinter 6.0.0 removed this feature entirely, so guard the call.
    if hasattr(ctk, "developer_warning"):
        ctk.developer_warning(False)
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
