@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ============================================================
rem  setup.bat - one-time setup for qwen-image-app
rem
rem  Downloads the stable-diffusion.cpp engine (sdcpp\) and the
rem  Qwen-Image-2.1-Turbo weights (models\). ~13 GB total.
rem
rem  * Re-runnable: files whose size matches the source are skipped;
rem    any other file is re-downloaded from scratch (a size mismatch
rem    means an older build or a partial download - never resume onto it).
rem  * Requires Windows 10+ (curl and tar ship with the OS).
rem  * All sources are public. If a weight is ever gated, set
rem    HF_TOKEN before running and re-run.
rem
rem  If stable-diffusion.cpp ships a new build, update SDTAG.
rem ============================================================

set "ROOT=%~dp0"
set "SD=%ROOT%sdcpp"
set "MD=%ROOT%models"
set "SDTAG=master-995-fd21ecf"
set "AUTH="
if defined HF_TOKEN set "AUTH=-H "Authorization:Bearer %HF_TOKEN%""

echo === qwen-image-app setup ===
echo.

rem --- 1. folders ---
echo [1/3] folders
for %%D in ("%SD%" "%MD%\qwen-image-2.1-turbo" "%MD%\text_encoder" "%MD%\vae" "%MD%\realesrgan") do if not exist %%D mkdir %%D

rem --- 2. engine ---
echo.
echo [2/3] engine   sdcpp\   (~880 MB)
if exist "%SD%\sd-server.exe" if exist "%SD%\cudart64_12.dll" (
    echo   [skip] engine already present
) else (
    call :zip "https://github.com/leejet/stable-diffusion.cpp/releases/download/%SDTAG%/sd-master-fd21ecf-bin-win-cuda12-x64.zip"
    call :zip "https://github.com/leejet/stable-diffusion.cpp/releases/download/%SDTAG%/cudart-sd-bin-win-cu12-x64.zip"
)

rem --- 3. weights ---
echo.
echo [3/3] weights   models\   (~11 GB)
call :get "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/resolve/main/qwen-image-2.1-turbo-Q4_K_M.gguf" "%MD%\qwen-image-2.1-turbo\qwen-image-2.1-turbo-Q4_K_M.gguf"
call :get "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf" "%MD%\text_encoder\Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
call :get "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-BF16.gguf" "%MD%\text_encoder\mmproj-BF16.gguf"
call :get "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors" "%MD%\vae\qwen_image_2.1_vae_bf16.safetensors"
call :get "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" "%MD%\realesrgan\realesrnet_x4plus.pth"

rem --- verify ---
echo.
echo [check] expected files
set "MISSING=0"
for %%F in (
    "%SD%\sd-server.exe"
    "%SD%\cudart64_12.dll"
    "%MD%\qwen-image-2.1-turbo\qwen-image-2.1-turbo-Q4_K_M.gguf"
    "%MD%\text_encoder\Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
    "%MD%\text_encoder\mmproj-BF16.gguf"
    "%MD%\vae\qwen_image_2.1_vae_bf16.safetensors"
    "%MD%\realesrgan\realesrnet_x4plus.pth"
) do if not exist %%F (
    echo   [MISSING] %%F
    set "MISSING=1"
)
if "%MISSING%"=="1" (
    echo.
    echo Some files are missing - re-run setup.bat to resume.
    endlocal & exit /b 1
)
echo   all present.

echo.
echo Done. Run:   qwen-image-app.exe     (or:  python app.py)
endlocal
exit /b 0

rem ============================================================
rem  :get <url> <dest>
rem  Download one file. Skips if already complete, resumes if partial.
:get
set "URL=%~1"
set "DEST=%~2"
set "RMT="
set "LOC="
if exist "%DEST%" (
    for /f "tokens=2 delims=: " %%A in ('curl -sIL %AUTH% "%URL%" ^| findstr /i "content-length"') do set "RMT=%%A"
    for %%A in ("%DEST%") do set "LOC=%%~zA"
    if defined RMT if "!RMT!"=="!LOC!" (
        echo   [skip] %DEST%   (complete)
        goto :eof
    )
    rem Size differs - the local file is not the current remote file
    rem (an older build, or a partial download). Resuming would append
    rem the remote tail onto the wrong bytes and corrupt it, so start
    rem over from scratch.
    echo   [redo] %DEST%   (have !LOC!, need !RMT!)
    del "%DEST%"
)
curl -L %AUTH% --fail --retry 3 -o "%DEST%" "%URL%"
if errorlevel 1 (
    echo   [FAIL] %URL%
    exit /b 1
)
goto :eof

rem ============================================================
rem  :zip <url>
rem  Download a zip, extract flat into sdcpp\, delete the zip.
:zip
set "URL=%~1"
set "ZIP=%SD%\.download.zip"
if exist "%ZIP%" del "%ZIP%"
echo   [get ] %URL%
curl -L -C - %AUTH% --fail --retry 3 -o "%ZIP%" "%URL%"
if errorlevel 1 (
    echo   [FAIL] %URL%
    exit /b 1
)
echo   [ext ] to %SD%
tar -xf "%ZIP%" -C "%SD%"
del "%ZIP%"
goto :eof
