
## Architecture

```
sayso/ (transcriber)
├── src-tauri/             # Rust desktop host (Tauri 2)
│   ├── src/main.rs        # Native desktop window entry point
│   ├── src/lib.rs         # Tauri commands and application lifecycle
│   ├── tauri.conf.json    # Window & build configurations
│   └── Cargo.toml         # Rust dependencies
├── src/                   # Responsive minimalistic dark UI (Vite + TypeScript)
│   ├── index.html         # Meeting view, VAD visualizer, History & Settings
│   ├── main.ts            # UI state, audio player, WebSocket coordination
│   ├── api.ts             # REST & WebSocket client
│   ├── api.test.ts        # Vitest frontend tests
│   └── styles.css         # Dark theme styling
├── backend/               # Python ML & Audio Capture Sidecar
│   ├── main.py            # FastAPI server & WebSocket /ws broadcaster
│   ├── audio_devices.py   # Windows PyAudioWPatch mic & WASAPI loopback discovery
│   ├── vad_engine.py      # Silero VAD (16kHz 512 frame) & phrase boundary detector
│   ├── recorder.py        # Dual-stream capture & resampling engine
│   ├── stt_engine.py      # Nano-Cohere model lifecycle & 2-pass inference
│   ├── storage.py         # Session recordings & transcript management
│   └── config.py          # Production parameters (25.0s clips, 16kHz mono)
├── tests/                 # Fast test suites
│   ├── test_backend.py    # Python unit tests (devices, VAD, storage, API)
│   └── test_e2e.py        # Python E2E integration test
├── docs/
│   ├── TODO.md            # Multi-stage product roadmap (Stages 1 - 5)
│   ├── CLANKER_DOCS.md    # Architecture and AI developer guide
│   └── CLANKER_STT.md     # Cohere vs Whisper benchmark results
├── start.bat              # One-click desktop app launcher
├── start-web.bat          # Web browser dev server launcher
└── test.bat               # Runs backend & frontend test suites
```

---

## STT Engine & Model Lifecycle Notes

- **Model**: `CohereLabs/cohere-transcribe-03-2026` via `nano_cohere_transcribe`.
- **Supported Languages & "Auto" Constraint**: `nano_cohere_transcribe` enforces explicit language tokens from `SUPPORTED_LANGUAGES = ('en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'el', 'ar', 'ja', 'zh', 'vi', 'ko')`. Passing `"auto"` raises a `ValueError`. Native multilingual auto-detection is not supported by Cohere Transcribe; explicit language selection (defaulting to `"en"`) is required.
- **On-Demand VRAM Lifecycle**:
  - `auto_load=False` by default; weights are not held in VRAM at startup.
  - Model status is `"not_loaded"` when weights exist on disk but are not loaded in memory.
  - Starting a recording or requesting re-transcription initiates model loading asynchronously in the background (`ensure_model_loading()`) while audio capture begins immediately.
  - Once recording finishes and the high-quality pass completes (or fails), `unload_model()` cleans up PyTorch CUDA cache (`torch.cuda.empty_cache()`), releases model memory, and reverts status to `"not_loaded"` to free VRAM.
- **Local Cache Resolution**: In [`backend/stt_engine.py`](file:///D:/projects/transcriber/backend/stt_engine.py), `STTEngine.get_cached_model_path()` resolves the local snapshot directory (`models--CohereLabs--cohere-transcribe-03-2026\snapshots\<hash>`). Passing the direct folder path to `from_pretrained()` avoids remote HTTP revision checks on `huggingface.co` and loads in ~3s on CUDA.
- **Asynchronous Loading**: `ensure_model_loading()` spawns a daemon loader thread (`_load_model_worker`). `status` progresses: `not_downloaded` -> `loading` -> `ready`.
- **Chunked High-Quality Pass & WebSocket Progress**:
  - `split_audio_chunks_energy()` divides captured session audio into energy-bounded speech chunks.
  - `run_high_quality_pass()` reports progress via `on_hq_progress(processed, total, session_id)` callback.
  - WebSocket broadcasts `hq_pass_progress` events with `session_id`, `processed_chunks`, `total_chunks`, and `percent`.
  - Frontend displays `#hq-progress-modal` with a progress bar and automatically navigates to the session in History upon `hq_pass_completed`.
- **Process Hygiene**: Do not leave the backend (`python -m backend.main`), dev server (`npm run dev`), or Tauri host running upon task completion. Ensure port 48653 is closed.

## Reliability and UI State Contracts

- Recording admission and stop transitions are serialized by `_recording_transition_lock`. Duplicate starts/stops return HTTP 409 and mutation requests are never automatically replayed by the frontend API client.
- `/api/status` is the recovery source of truth after refresh or a missed WebSocket event. It includes `model_status` (including `"not_loaded"`), `is_model_loaded`, `device`, `cuda_device_name`, active session, recording mode, elapsed duration, and audio-capture error.
- **Frameless Window Shell & Integrated Titlebar**:
  - The native OS window chrome is disabled (`"decorations": false` and `"shadow": true` in `tauri.conf.json`).
  - Sleek top titlebar (`#app-titlebar`) features the Sayso brand logo (`/icon.svg`), title, native window drag region (`data-tauri-drag-region`, `-webkit-app-region: drag`), and custom window controls (`#btn-win-minimize`, `#btn-win-maximize`, `#btn-win-close`).
  - Window buttons communicate directly with Tauri 2 IPC commands (`minimize_window`, `toggle_maximize_window`, `close_window`, `is_window_maximized`) with graceful web fallbacks.
  - Double-clicking the titlebar toggles maximize state.
- **Left Navigation Sidebar**:
  - Main app navigation moved to a dedicated left sidebar (`#app-sidebar`):
    - Top view buttons: Record (`#tab-record`) and History (`#tab-history`) with badge counter (`#recordings-count`).
    - Bottom footer controls: Status Circle (`#btn-status-indicator`) and Settings (`#btn-open-settings`).
  - Unified status dot glows green (ready/loaded), yellow (busy/transcribing), or red (offline/attention) and opens the `#diagnostics-popover` extending into the main view. The `#app-sidebar` container must preserve `overflow: visible` (with relative positioning and `z-index: 100`) so the popover is never clipped by sidebar bounds. Popover can be dismissed via outside click, close button, or Escape key.
- **Instant Speaking Skeleton & Top-First Transcript Stream**:
  - Voice activity detection instantly displays an animated speaking placeholder row (`.live-speaking-skeleton`) at the top of the transcript list as soon as speech begins.
  - When the speech segment finishes, `phrase_pending` smoothly adopts the active skeleton without layout shifts, and `phrase_transcribed` reveals the final text in place.
  - Live transcripts are prepended (`.prepend()`) to the top of the stream so the latest words are immediately visible without manual scrolling.
- **Phrase Presentation, Live Editing & Clipboard Sync**:
  - Live phrases during recording display as clean `You: <phrase>` or `Them: <phrase>` rows without timestamps; microphone audio is `You` and computer/system audio is `Them`.
  - History transcript displays as a continuous list of phrases (`[timestamp] You: phrase`) without card borders, with auto-resizing textareas that expand naturally without nested scrollbars.
  - Session title, full transcript, and individual phrases all support live in-memory synchronization on `input` keystrokes plus debounced (600ms) background auto-saving on typing and immediate flush on `blur`.
  - The Copy action (`#btn-copy-transcript`), TXT export, and SRT export read directly from live DOM values before copying/saving, ensuring that newly typed edits are immediately copied even before an auto-save finishes.
- **Interrupted Session Recovery & Continuous Audio Flushing**:
  - Audio capture is incrementally saved to `audio.wav` every 5 seconds and on every transcribed phrase.
  - If the application is abruptly closed during a recording, the backend recovers the session on next launch with status `"interrupted"`, preserves all transcribed phrases, and keeps `audio.wav` playable and retryable.
  - The history UI marks abruptly closed sessions as `"Closed abruptly"` (`.transcript-tag.warn`) and offers a `"Transcribe recording"` button to run full high-quality transcription on the preserved audio.
  - Successful high-quality completions (`status: "completed"`) automatically hide the Re-transcribe button to avoid redundant work.
- **App Splash Screen & Clean Startup Sequence**:
  - A sleek, minimalist dark launch screen (`#splash-screen`) displays brand audio wave animations and status text while connecting to the local backend.
  - Fades out smoothly into the main application once WebSocket connection and initial diagnostics are confirmed. Includes a 10s safety timeout to avoid blocking the UI if startup hangs.
  - Initial load only fetches devices and history if the backend is already online, and suppresses error toasts while the backend is still launching so users never see a transient "History unavailable" banner when launching the app. Successful history fetch immediately clears any active history error banner.
- A stopped session is durably marked `processing_hq` before HQ inference is dispatched. Completion/failure is committed only when the callback's `job_id` still owns that session. A transcript edited after the job starts is not overwritten by its late result.
- Only one HQ job per session can be reserved. Re-transcription is rejected while recording, while its session already has an HQ job, or while the model is not downloaded.
- STT failures never create simulated/fabricated transcript text. Phrase events include an `error` and the UI keeps the segment visibly retryable via the post-recording HQ pass.
- Frontend long operations expose disabled and `aria-busy` controls, inline progress/status messages, recoverable history/detail loading states, and processing/error states. UI state is reconciled by both WebSocket events and a non-overlapping status poll.
- Phrase lifecycle events are ordered as `phrase_pending` (audio segment completed, before STT queueing) followed by `phrase_transcribed`. Both carry the same `phrase_id`, allowing the frontend to render an animated skeleton and replace it in place without duplicate rows. Pending phrases are UI-only and are not persisted until transcription returns.
- Saved device/language settings live in `localStorage` under `sayso.settings`.
- Tests set `SAYSO_DISABLE_MODEL_AUTOLOAD=1` in `tests/conftest.py`. Backend tests must use fake models and simulated audio; they must never download/load the real model or require audio hardware.

## Process Ownership and Shutdown

- The Rust/Tauri host is the owner of the Python backend on `127.0.0.1:48653`.
  `start.bat` and `start.ps1` deliberately do not start Python themselves.
- All command invocations (`netstat`, `taskkill`, `python --version`) in the Tauri host
  use `silent_command` configured with Windows `CREATE_NO_WINDOW` (`0x08000000`)
  so process management and shutdown never produce flashing console windows.
- In `cleanup_processes()`, candidate listener PIDs across all reserved ports are gathered
  in a single `netstat` call, speeding up shutdown and eliminating redundant executions.
- On `ExitRequested`/`Exit`, the host forcefully terminates the backend process tree
  with Windows `taskkill /T /F`. It also scans the app-reserved Vite ports
  `41765`/`41766` so stale dev servers from an older/crashed launch are removed.
- If a backend is already listening when the app starts, its listener PID is adopted
  and cleaned up on exit rather than being left behind. This is intentional because
  those localhost ports are reserved for Sayso.
- The web-only launcher (`start-web.bat`) is separate from the Tauri lifecycle and
  should be stopped with Ctrl+C; the desktop launcher is the supported path when
  shutdown cleanup is required.

## Validation Commands

```powershell
npm test
python -m pytest -q
npm run build
cargo check --manifest-path src-tauri/Cargo.toml
```

## Portable Windows Packaging

`package-portable.ps1` creates `Sayso-Portable/Sayso/`, containing `Sayso.exe`,
`backend/`, an embedded Python runtime, writable `recordings/`, and
`models/huggingface/`. The Cohere cache is included by default. Use
`-NoModel` to ship an empty model directory and let the app download it on first
run. The package output is deliberately not listed in `.gitignore`.

A normal build requires network access for the embedded Python archive, pip
packages, and (unless `-NoModel` is used) the local Hugging Face model cache.
The `nano_cohere_transcribe` dependency is installed from a pinned GitHub
source revision because it is not published on PyPI. PyTorch and torchaudio are
pinned to matching CUDA 12.6 wheels from the PyTorch index so packaged NVIDIA
builds expose CUDA to the backend; the backend selects `cuda` whenever
`torch.cuda.is_available()` is true.
The script uses the cache under `$env:HF_HUB_CACHE`, `$env:HF_HOME/hub`, or the
standard `%USERPROFILE%\\.cache\\huggingface\\hub` location. Run
`powershell -ExecutionPolicy Bypass -File .\\package-portable.ps1` on a Windows x64
development machine. Each normal run removes and recreates the existing
`Sayso-Portable/Sayso` package automatically; `-Clean` remains accepted for
backward compatibility. `-ValidateOnly` checks the packaging inputs without
building or copying large artifacts.

The native host first looks for `python\\pythonw.exe` beside the packaged
executable, sets `HF_HOME`, `HF_HUB_CACHE`, and `TORCH_HOME` below `models/`,
and only then falls back to a Python executable on PATH for development.
