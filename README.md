# Qwen-Image-2.1 Turbo — desktop app (stable-diffusion.cpp edition)

Text-to-image + image editing with **Qwen-Image-2.1-Turbo** running on
**[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)**
(GGUF weights from [`unsloth/Qwen-Image-2.1-Turbo-GGUF`](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF)),
plus optional **Real-ESRGAN x4** upscaling via sd.cpp's built-in ESRGAN
upscaler — in one customtkinter window. Built for a 16 GB NVIDIA GPU
(tested target: RTX 5060 Ti).

No Python runtime is needed for inference: the app spawns the prebuilt
`sd-server.exe` **on the first Generate click** and talks to it over HTTP
(`127.0.0.1:8321`). The `--eager-load` flag keeps the ~10 GB of weights
pinned in VRAM until the app closes, so generations after the first have no
per-run cold start — VRAM is released when the app quits (unload + terminate).
There are no torch/diffusers wheels to manage.

## Folder layout

```
qwen-image-app/
├── app.py                       # the GUI (customtkinter) — drives sd-server.exe over HTTP
├── prompts.json                 # scenario + booster chips (edit freely)
├── requirements.txt             # GUI-only deps: customtkinter, pillow (GPU queries use nvidia-smi)
├── README.md                    # this plan
├── app.log                      # app + engine log (server stdout is piped here at DEBUG)
├── sdcpp/                       # stable-diffusion.cpp runtime (manual download — Setup 1)
│   ├── sd-server.exe            # the resident engine the app spawns
│   ├── sd-cli.exe               # (still in the release zip; the app no longer uses it)
│   ├── cudart64_12.dll          # CUDA runtime — from the cudart-sd zip (required)
│   ├── cublas64_12.dll          # CUDA BLAS — same zip
│   ├── cublasLt64_12.dll        # CUDA BLAS-Lt — same zip
│   └── …                        # (+ ggml*.dll etc.)
├── models/                      # weights, all placed manually (~10 GB total)
│   ├── qwen-image-2.1-turbo\
│   │   └── qwen-image-2.1-turbo-Q4_K_M.gguf     4.20 GB — denoiser (GGUF)
│   ├── text_encoder\
│   │   ├── Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf 5.15 GB — text encoder (GGUF)
│   │   └── mmproj-BF16.gguf                   1.16 GB — vision projector (image editing + references only)
│   ├── vae\
│   │   └── qwen_image_2.1_vae_bf16.safetensors 0.68 GB — VAE
│   └── realesrgan\
│       └── realesrnet_x4plus.pth              0.06 GB — Real-ESRGAN x4 (kept)
└── images/                      # outputs (auto-created, timestamped filenames)
```

## Setup (one time — the app does no downloading)

1. **Engine:** download **both** Windows CUDA assets from the
   [stable-diffusion.cpp releases](https://github.com/leejet/stable-diffusion.cpp/releases)
   page (latest as of writing: `master-995-fd21ecf`):
   - `sd-master-fd21ecf-bin-win-cuda12-x64.zip` (~344 MB) — the engine
     (contains both `sd-server.exe` and `sd-cli.exe`)
   - `cudart-sd-bin-win-cu12-x64.zip` (~537 MB) — **required** CUDA runtime
     DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`)
   Extract both into `sdcpp/` (each zip extracts flat) so that
   `sdcpp\sd-server.exe` and the three CUDA DLLs sit side by side. Without
   the cudart DLLs, the server logs "no GPU devices" and silently runs on
   CPU.

2. **GUI dependencies** (system Python — no embedded runtime needed):
   ```
   python -m pip install -r requirements.txt
   ```

3. **Weights (manual):** download the files yourself and place them in
   `models/`. The GGUF is the denoiser only; it needs a text encoder, a VAE
   and (for editing) a vision projector alongside it:

   | File | Size | Source |
   |---|---:|---|
   | `qwen-image-2.1-turbo/qwen-image-2.1-turbo-Q4_K_M.gguf` | 4.20 GB | [unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) — recommended; `Q8_0` (7.64 GB) for closest-to-bf16, `Q3_K_XL` (3.61 GB) if you must go below 4 GB |
   | `text_encoder/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` | 5.15 GB | [unsloth/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF) — Dynamic 2.0 rung unsloth recommends for this model (alt: `Qwen3VL-8B-Instruct-Q4_K_M.gguf`, 5.03 GB, from [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)) |
   | `text_encoder/mmproj-BF16.gguf` | 1.16 GB | same unsloth repo — **only needed for image editing / reference images** (registered with the server via `--llm_vision`) |
   | `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) (also in `unsloth/Qwen-Image-2.1-FP8`) — must be the 2.1 VAE, not interchangeable with older Qwen-Image/Wan VAEs |
   | `realesrgan/realesrnet_x4plus.pth` | 0.06 GB | already in place; from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) (v0.1.0, file `RealESRGAN_x4plus.pth`) |

## Running

```
python app.py
```

The app starts with no server running (zero VRAM cost). On the first
**Generate** click it:

1. spawns `sdcpp\sd-server.exe` with the selected weights and `--eager-load`,
2. polls `GET /sdcpp/v1/models/status` until it reports `ready` — the status
   bar shows `Loading models… (Ns)` while the progress bar stays at 0%, then
   `Model ready — pinned in VRAM (Ns)`,
3. posts the generation job. Later generations reuse the warm server with no
   load step; closing the app unloads + terminates it, releasing all VRAM.

- **Generate** — prompt box is the source of truth; quick-add chips append
  their template (click again to remove). An input image switches to
  editing mode and appears as a 48 px thumbnail with a full-path hover
  tooltip and a `1 ×` caption (removes it, like the row's Clear). The
  whole run is one HTTP job
  (`POST /sdcpp/v1/img_gen` → poll `GET /sdcpp/v1/jobs/{id}` every second);
  the job's sampling step counter is mapped onto one overall 0→100 % bar.
- **References row** — the model supports up to 10 reference images; the
  app caps it at **3 extras** by default (the primary input image is
  "image 1"; Settings → *Limit references to 3* off raises the cap to 8)
  because each reference adds tokens to the diffusion pass and the graph-cut
  cache scales with total token count — beyond ~4 total is OOM territory on
  a 16 GB card. Extras are thumbnails with a number caption (② ×, ③ ×, ④ ×)
  and a hover tooltip with the full path; **+ Add** is hidden at the cap and
  **Clear** removes the extras only (the primary row's Clear is separate).
  Extras are downscaled to 768 px longest side before being sent (Settings →
  *Downscale references to ~768p*; off sends them at full size). The row's
  hint recommends **1024 base resolution** with references and, while refs
  are set and the aspect menu is selectable, appends the **best output
  aspect** — the dropdown option closest to the geometric mean of the ref
  ratios (the engine resizes each ref to the target's pixel area while
  keeping the ref's own aspect, so a matching target minimizes what the
  model has to invent). In the prompt, address the images by position: "image 1", "image 2", …
- **Upscale (Real-ESRGAN)** — optional second stage via the server's
  synchronous `POST /sdcpp/v1/upscale` endpoint (the ESRGAN weights are
  registered with the server when it starts). x4 native; x2 = x4 output
  downscaled with Lanczos by `app.py`. The **Tile** dropdown (128/256,
  default 256) sets the ESRGAN tile size per request — 256 px tiles are
  ~2-4x faster with fewer seams. (The old denoise slider is gone —
  sd.cpp's upscaler has no denoise-strength flag.)
- **Settings window** (⚙ Settings button) holds the denoiser, text encoder
  and GPU dropdowns, the engine switches (Sage Attention, Text encoder on
  CPU, Vision model q8_0, Load vision model on demand) and the
  verbose-logging switch. The
  sections scroll if they overflow; the footer note is always visible. Changing anything
  server-affecting respawns the server on the next generation (one-time
  load cost per change — the models are pinned, so there is no per-run
  load to skip). The server-affecting controls are disabled while a
  generation is running; verbose logging and per-run options stay live.
- **Closing the app** unloads the models (best effort) and terminates the
  server, releasing all VRAM.

## What `app.py` does

Server command (once, at app start):

```
sdcpp\sd-server.exe
    --model-dir       models
    --diffusion-model models\qwen-image-2.1-turbo\qwen-image-2.1-turbo-Q4_K_M.gguf
    --llm             models\text_encoder\Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf
    --llm_vision      models\text_encoder\mmproj-BF16.gguf   # or mmproj-q8_0.gguf (see below);
                                                             # omitted in T2I with the on-demand switch on
    --vae             models\vae\qwen_image_2.1_vae_bf16.safetensors
    --flash-attn      off,diffusion=on
    --backend         cuda<selected>   # from the GPU dropdown (settings window); defaults to the largest card
    --eager-load
    --upscale-model   models\realesrgan\realesrnet_x4plus.pth
    --hires-upscalers-dir models\realesrgan
    --listen-ip 127.0.0.1 --listen-port 8321
    --log-level debug
```

Optional — appended when the matching switch is on in the settings window.
Each one changes the server signature, so the server respawns on the next
generate after a change:

```
    --sage-attn                    # Sage Attention (INT8 attention)
    --params-backend te=cpu        # text-encoder weights pinned in RAM
```

The **Vision model q8_0** switch doesn't append a flag — it changes
`--llm_vision` to `models\text_encoder\mmproj-q8_0.gguf`. That file is
created automatically on first use by converting the BF16 original with
`sdcpp\sd-cli.exe -M convert --type q8_0` (one-time, a few seconds, CPU);
if the conversion ever fails, the app falls back to the BF16 file. It
reduces the vision model's 1.1 GB footprint to ~0.7 GB — pinned RAM with
"Text encoder on CPU" on, VRAM with it off. The switch is always enabled
when the settings window opens; when you turn it on, the app checks there
is ~800 MB of free disk space to create the q8_0 file (~744 MB) — if the
q8 file is already present that check is a single file lookup. Without
enough space the switch reverts to off and a warning explains why.

The **Load vision model on demand** switch (off by default) omits
`--llm_vision` from the spawn command while no image of any kind is set —
text-to-image never uses the vision tower, so its ~1.1 GB of weights stay
out of memory. The moment an input image **or a reference image** is
chosen, the server signature changes and the server respawns **with**
`--llm_vision` on the next generation (references are passed to the VLM as
well as the DiT, so without the vision model the model would silently
ignore them); clearing all images drops it again.
Each transition costs the one-time ~8 s eager load. Once loaded, the
"Text encoder on CPU" switch governs where the vision weights live, as
with everything else in the text encoder.

Each generation then posts one job to `POST /sdcpp/v1/img_gen`:

```json
{
  "prompt": "<prompt>", "width": <w>, "height": <h>, "seed": 42,
  "sample_params": {
    "sample_method": "euler", "sample_steps": 8,
    "custom_sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508,
                      0.845148, 0.704534, 0.414568, 0.0],
    "guidance": {"txt_cfg": 1.0}
  },
  "vae_tiling_params": {"enabled": true, "tile_size_w": 1024,
                        "tile_size_h": 1024, "target_overlap": 0.5},
  "preview": "proj", "output_format": "png"
}
```

`custom_sigmas` is the Turbo checkpoint's baked-in 8-step schedule — the
named CLI preset `qwen-image-2.1-turbo` spelled out, because the JSON API
takes an explicit array (count minus one sets the step count). Any image
adds `ref_images`: the primary input first, resized client-side to the
target canvas (the server keeps reference images at their own dimensions
otherwise), then the extras (downscaled to 768 px longest side while the
*Downscale references* switch is on). The engine
auto-detects the Qwen multi-reference preset from the weights (each ref is
VAE-encoded into the diffusion pass **and** passed to the VLM), so no
extra spawn flags are needed.
Progress comes from the job's `preview.step` / `preview.total_steps`; the
finished `result.images[0].b64_json` is written to `images\`.

## Notes

- **VRAM is pinned from first generate until app exit:** opening the app
  costs nothing; after the first generation expect ~10 GB resident. Other GPU
  apps (e.g. LM Studio) will only find the remaining headroom while the models
  are loaded — close the app to give the VRAM back.
- Turbo runs its baked-in 8-step schedule; CFG=1, no negative prompts
  (boosters are positive phrasing for this reason).
- Base resolution is capped at 2048: the diffusion pass alone needs ~22 GB
  at 4096² on a 16 GB GPU, so 4096 was removed from the dropdown. The
  default is **1024** — safe with reference images (see below) and half the
  VRAM of 2048; pick 2048 for detail in reference-free runs.
- **References use VRAM in proportion to token count.** Each reference is
  VAE-encoded and concatenated onto the diffusion token sequence, and the
  graph-cut cache scales with total tokens (~267 KB/token measured). A
  768² ref adds ~9 216 tokens (~2.4 GB of cache). At 2048 base, a single
  edit already sits near the 16 GB ceiling, so extra references can OOM —
  that's why the app recommends 1024 base with references (the row shows
  the hint) and defaults to 3 × 768 px extras (the cap can be raised to 8
  and the downscale disabled in Settings — at your own VRAM risk).
- Supported aspect ratios: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16 — all already
  divisible by 32 as sd.cpp requires.
- Outputs can be RGBA (transparent background via the official prompt format);
  keep them as `.png`/`.webp` to preserve alpha.
- `prompts.json` is plain JSON — add your own chips under `scenarios`
  or `boosters`, restart the app.
- **Switch settings persist to `settings.ini`** (project root): the seven
  settings-window switches (Sage Attention, Text encoder on CPU, Vision
  model q8_0, Load vision model on demand, Limit references to 3, Downscale
  references to ~768p, Verbose logging) are saved on every change and
  restored at startup. The two reference switches are per-request (no
  server restart); the rest respawn the server on the next generate. The file is created on the first
  switch change only; a missing or corrupt file just means defaults
  (switches off, verbose on). The dropdowns (denoiser, text encoder, GPU)
  and the tile size are **not** persisted — on restart they return to
  first .gguf in each folder, largest GPU, tile 256.
- Real-ESRGAN is CC BY-NC 4.0: fine for personal use. Qwen-Image weights are
  under the Qwen Research License.

## Troubleshooting

- **`sd-server.exe` not found** — check that `sdcpp\sd-server.exe` exists
  after extracting the release zip into `sdcpp/`.
- **Runs on the wrong GPU / OOM at load** — the server is pinned to the
  card chosen in the **GPU dropdown** in the settings window (defaults
  to the card with the most VRAM; the `SERVER_GPU` constant is only the
  fallback when no GPU can be enumerated). Pick the 16 GB card — the 8 GB one cannot
  hold the ~10 GB model set. Dropdown labels use the `nvidia-smi -L` index
  ordering.
- **`Loading models…` (on first generate) never becomes ready** — the status
  label and `app.log` carry the server's own output (it is piped in at DEBUG);
  a missing weight file, a port already in use (8321), or a CUDA problem all
  show up there. A model-load failure flips the status state to `error` and
  the app reports it immediately; the next Generate retries from scratch.
- **CUDA backend not selected / slow CPU runs** — the log shows
  `auto-fit: no GPU devices; using the default backend` when the CUDA
  runtime DLLs are missing: the `cuda12` Windows build needs
  `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` from the
  **separate** `cudart-sd-bin-win-cu12-x64.zip` release asset sitting next
  to `sd-server.exe`. Without them the server silently falls back to CPU
  (minutes per image instead of ~1 minute).
- **OOM during generation** — VAE tiling (1024px) keeps the decode under
  control; the diffusion pass itself is the memory hog at large sizes.
- **Editing mode fails without vision weights** — you must have
  `models\text_encoder\mmproj-BF16.gguf` in place; it is not needed for
  plain text-to-image.

## Removed vs. the old plans

- `python/` embedded Python 3.12 (only existed because of torch wheel pins)
- `torch`, `diffusers`, `transformers`, `accelerate`, `realesrgan`,
  `huggingface_hub` from requirements — inference no longer uses any of them
- old weight layout `models/qwen-image-2.1-turbo/{model_index.json, transformer/, text_encoder/, vae/, scheduler/, processor/}`
- the "Load model / Unload" buttons and the "Text encoder in RAM" toggle
- the per-generation `sd-cli.exe` subprocess (replaced by the resident
  `sd-server.exe` with `--eager-load`)
