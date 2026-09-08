
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
- **Unified Status Indicator & Diagnostics**: Replaced dual pills with a single status circle dot in header:
  - Green (`.status-good`): backend connected and model ready (either in memory or ready on disk).
  - Yellow (`.status-warn`): model loading, downloading, or transcribing.
  - Red (`.status-bad` / `.status-offline`): backend offline or model error / not downloaded.
  - Clicking opens `#diagnostics-popover` detailing backend connectivity, model status, compute device (CUDA / CPU), selected audio inputs, and active session.
- **3-Bar VAD Visualizer**: Replaced horizontal progress meters with matching 3 white vertical bars for microphone (`#vad-bar-1..3`) and computer/system audio (`#system-vad-bar-1..3`) that smoothly expand upward when speech is detected and contract down to 4px when silent.
- **Phrase Presentation & Auto-Save**:
  - Live phrases during recording display as clean `You: <phrase>` or `Them: <phrase>` rows without timestamps; microphone audio is `You` and computer/system audio is `Them`.
  - History transcript displays as a continuous list of phrases (`[timestamp] You: phrase`) without card borders, with auto-resizing textareas that expand naturally without nested scrollbars.
  - Session title editing auto-saves immediately on blur and debounced (600ms) on typing, without requiring an explicit Save button.
- A stopped session is durably marked `processing_hq` before HQ inference is dispatched. Completion/failure is committed only when the callback's `job_id` still owns that session. A transcript edited after the job starts is not overwritten by its late result.
- Only one HQ job per session can be reserved. Re-transcription is rejected while recording, while its session already has an HQ job, or while the model is not downloaded.
- STT failures never create simulated/fabricated transcript text. Phrase events include an `error` and the UI keeps the segment visibly retryable via the post-recording HQ pass.
- Frontend long operations expose disabled and `aria-busy` controls, inline progress/status messages, recoverable history/detail loading states, and processing/error states. UI state is reconciled by both WebSocket events and a non-overlapping status poll.
- Phrase lifecycle events are ordered as `phrase_pending` (audio segment completed, before STT queueing) followed by `phrase_transcribed`. Both carry the same `phrase_id`, allowing the frontend to render an animated skeleton and replace it in place without duplicate rows. Pending phrases are UI-only and are not persisted until transcription returns.
- Saved device/language settings live in `localStorage` under `sayso.settings`.
- Tests set `SAYSO_DISABLE_MODEL_AUTOLOAD=1` in `tests/conftest.py`. Backend tests must use fake models and simulated audio; they must never download/load the real model or require audio hardware.

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
