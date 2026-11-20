# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-20

### Added
- Initial release: Qwen-Image-2.1-Turbo desktop app (stable-diffusion.cpp edition).
- Text-to-image + image-editing GUI (customtkinter) that drives a resident
  `sd-server.exe` over HTTP (`127.0.0.1:8321`) with `--eager-load`.
- Optional Real-ESRGAN x4 upscale stage via the server's built-in ESRGAN upscaler.
- Settings window (denoiser / text encoder / GPU / engine switches) with
  persistence to `settings.ini`.
- Standalone Windows `.exe` build (PyInstaller) — see the release assets.
