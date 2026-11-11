
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
├── build.bat              # One-click production desktop & portable build script
├── start.bat              # One-click desktop app launcher
├── start-web.bat          # Web browser dev server launcher
└── test.bat               # Runs backend & frontend test suites
```

---

## STT Engine & Model Lifecycle Notes

- **Model**: `CohereLabs/cohere-transcribe-03-2026` via `nano_cohere_transcribe`.
- **Supported Languages & Dynamic Switching**:
  - `SUPPORTED_LANGUAGES = ('en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'el', 'ar', 'ja', 'zh', 'vi', 'ko')`.
  - Language can be switched dynamically before or *during* an active recording session via `POST /api/recording/language` (or `PATCH /api/recording/active`), broadcasted live over WebSocket (`recording_language_changed`), and selected from the sidebar flag dropdown.
- **On-Demand VRAM Lifecycle & Cache Clearance**:
  - `auto_load=False` by default; weights are not held in VRAM at startup.
  - Model status is `"not_loaded"` when weights exist on disk but are not loaded in memory.
  - Starting a recording or requesting re-transcription initiates model loading asynchronously in the background (`ensure_model_loading()`) while audio capture begins immediately.
  - While model weights are loading into memory (`status === "loading"`), a non-intrusive gray popup (`#model-loading-popup`) is displayed. Once loaded (`status === "ready"`), the popup hides and nothing is shown (no persistent header badges or clutter).
  - **Critical VRAM Fix**: `nano_cohere_transcribe.api._MODEL_CACHE` maintains a module-level global dict caching `(path, device, dtype, tokenizer) -> model`. Even when `del self.model` is executed, the model remains pinned in memory by `_MODEL_CACHE`. `unload_model()` explicitly executes `cohere_api._MODEL_CACHE.clear()`, runs `gc.collect()`, and calls `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()`. This drops VRAM from ~4 GB down to baseline (~9 MB) immediately after the post-meeting high-quality pass completes or fails.
- **VAD Sensitivity, Turn-Taking & Hallucination Suppression**:
  - `PhraseSegmenter` implements adaptive silence pausing: speech shorter than 1.5s requires `short_phrase_silence_s` (1.8s) of silence before cutting, while speech >= 1.5s cuts at `silence_duration_s` (0.85s).
  - `MIN_ENERGY_RMS` is configured to `0.0012` (in `backend/config.py`) to accommodate lower-gain headsets and Windows microphone inputs without dropping real speech.
  - Silero VAD internal RNN hidden states are periodically reset during extended idle silence (>= 40 frames) and on phrase completion so long periods of silence do not saturate the model or suppress subsequent speech detection.
  - **Conversational Turn-Taking**: When partner speech onset occurs (`on_partner_speech_onset`), if the other channel is currently in a pause (`silence_frames_after_speech > 0`) with sufficient speech (`speech_dur >= min_speech_duration_s`), the phrase is immediately finalized at that turn boundary. No sticky 0.25s pause state is retained for subsequent phrases.
  - **Transcript Hygiene & Error UI**:
    - Suppressed hallucinations or empty transcriptions are discarded: they are never persisted to storage and never rendered as fake cards saying "This segment could not be transcribed."
    - If an error occurs (transcription error, audio capture error, HQ failure), it is **never written into the transcript**. Any pending skeleton card is removed, and a prominent red popup (`#global-error-popup`) appears with a **Copy** button to allow copying the exact error message.
  - In `run_high_quality_pass`, cross-speaker temporal overlaps are resolved by splitting the encompassing phrase around the interjecting turn.
- **Multi-Track Audio Storage & Folder Exploration**:
  - When recording both mic and system audio, Sayso durably writes separate audio files to `recordings/<session_id>/`:
    - `audio.wav`: combined mixed audio for standard playback.
    - `mic.wav`: isolated microphone stream (`Me`).
    - `system.wav`: isolated system / loopback stream (`Them`).
  - High-quality 2nd-pass transcription reads `system.wav` for `Them` phrases and `mic.wav` for `Me` phrases, eliminating crosstalk and user speech bleed into the other speaker's transcription.
  - The recording detail view features a **Folder** button (`#btn-open-folder`) calling `POST /api/recordings/{session_id}/open_folder`, which opens the session's folder directly in File Explorer.
- **Asynchronous Loading & Model Download**:
  - `ensure_model_loading()` spawns a daemon loader thread (`_load_model_worker`). `status` progresses: `not_downloaded` -> `loading` -> `ready`.
  - When downloading model files in `_download_worker`, `try_to_load_from_cache(COHERE_MODEL_ID, fname)` checks local cache (without `token`, as `try_to_load_from_cache` is strictly a local disk lookup). Remote download uses `hf_hub_download(..., token=hf_token)` which correctly passes authentication tokens from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN`.
  - **Environment Variables & Token Persistence**:
    - `backend.config.load_env_file()` automatically loads `.env` key-value pairs from `BASE_DIR / ".env"` into `os.environ` on startup, with a built-in fallback parser if `python-dotenv` is absent. A template is provided at `.env.example`.
    - `POST /api/settings/token` and `save_hf_token_to_env()` persist user tokens directly to `.env` and `os.environ`, allowing token configuration via the settings UI (`#input-hf-token`).
    - If a 401 error or GatedRepoError occurs (since `CohereLabs/cohere-transcribe-03-2026` is restricted), `_download_worker` formats clear actionable guidance advising the user to accept the license agreement and configure `HF_TOKEN`.
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
    - Top view buttons: Record (`#tab-record`), Upload (`#tab-upload`), and History (`#tab-history`) with badge counter (`#recordings-count`).
    - Bottom footer controls: Status Circle (`#btn-status-indicator`) and Settings (`#btn-open-settings`).
  - Unified status dot glows green (ready/loaded), yellow (busy/transcribing), or red (offline/attention) and opens the `#diagnostics-popover` extending into the main view. The `#app-sidebar` container must preserve `overflow: visible` (with relative positioning and `z-index: 100`) so the popover is never clipped by sidebar bounds. Popover can be dismissed via outside click, close button, or Escape key.
- **Audio File Upload & FFmpeg Pipeline**:
  - Dedicated Upload tab (`#view-upload`) with drag-and-drop zone (`#upload-dropzone`) and file picker supporting any audio/video format (MP3, WAV, M4A, FLAC, AAC, OGG, WMA, MP4, MKV, WebM, etc.).
  - Backend verifies FFmpeg via `is_ffmpeg_installed()` (`GET /api/system/ffmpeg`). If FFmpeg is missing, a prominent alert banner is displayed and uploads fail with `"ffmpeg not installed, please install."`.
  - On upload (`POST /api/upload/transcribe`), the file is converted into 16kHz mono 16-bit PCM WAV via FFmpeg (`convert_audio_to_wav_16k`), saved to `recordings/<session_id>/audio.wav`, and transcribed using chunked high-quality Cohere inference with real-time WebSocket progress updates (`hq_pass_progress`).
  - On completion, the frontend automatically opens the session in the History view.
- **Persistent Device Matching by Name**:
  - Device selections in settings (`localStorage` under `sayso.settings`) save `mic_name` and `system_name` rather than volatile integer PortAudio indices.
  - On application startup, device enumeration, or hardware disconnect/reconnect, `resolve_device_index()` matches devices by name (exact match with fuzzy fallback).
  - If a previously selected device is unplugged or disconnected, it never silently falls back to an arbitrary index (such as a monitor or secondary display). Instead, the UI preserves the preference as `<device_name> (Disconnected)` and the backend falls back safely to the system default device.
- **Custom White-Gray Rounded Scrollbars**:
  - Global custom scrollbar styling across all views (History, Transcript details, live stream, sessions scroll list, modal dialogues):
    - Track: matches black / default app background (`var(--bg-app)` / `#000000`).
    - Thumb: white-gray custom rounded corners rectangle (`#b0b0b0`, hover `#d8d8d8`, active `#ffffff`) with border inset floating inside the dark track.
    - Standard `scrollbar-color: #b0b0b0 var(--bg-app)` and `scrollbar-width: thin` for cross-browser consistency.
- **Instant Speaking Skeleton & Top-First Transcript Stream**:
  - Voice activity detection instantly displays an animated speaking placeholder row (`.live-speaking-skeleton`) at the top of the transcript list as soon as speech begins.
  - When the speech segment finishes, `phrase_pending` smoothly adopts the active skeleton without layout shifts, and `phrase_transcribed` reveals the final text in place.
  - Live transcripts are prepended (`.prepend()`) to the top of the stream so the latest words are immediately visible without manual scrolling.
- **Phrase Presentation, Live Editing & Bidirectional Sync**:
  - Live phrases during recording display as clean `You: <phrase>` or `Them: <phrase>` rows without timestamps; microphone audio is `You` and computer/system audio is `Them`.
  - History transcript displays as a continuous list of phrases (`[timestamp] You: phrase`) without card borders, with auto-resizing textareas that expand naturally without nested scrollbars.
  - **Bidirectional Automatic Synchronization (`src/transcript-sync.ts`)**:
    - Editing any phrase row (in `interactive-phrases-container`) automatically updates the "Full Transcript" text field (`#final-transcript-text`) in real-time on `input`, on `blur`, and on Enter.
    - Editing the "Full Transcript" text field (`#final-transcript-text`) automatically updates the interactive phrase rows (`interactive-phrases-container`) in real-time on `input`, preserving phrase IDs, audio jump timestamps, and speaker assignments using monotonic DP sequence alignment.
    - Reconciled edits auto-save with a 600ms debounce timer on typing and flush immediately on blur, committing both `final_transcript` and `phrases` to `backend/storage.py` atomically.
  - The Copy action (`#btn-copy-transcript`), TXT export, and SRT export read directly from live DOM values before copying/saving, ensuring that newly typed edits are immediately copied even before an auto-save finishes.
- **Interrupted Session Recovery & Continuous Audio Flushing**:
  - Audio capture is incrementally saved to `audio.wav` every 5 seconds and on every transcribed phrase.
  - If the application is abruptly closed during a recording, the backend recovers the session on next launch with status `"interrupted"`, preserves all transcribed phrases, and keeps `audio.wav` playable and retryable.
  - The history UI marks abruptly closed sessions as `"Closed abruptly"` (`.transcript-tag.warn`) and offers a `"Transcribe recording"` button to run full high-quality transcription on the preserved audio.
  - Successful high-quality completions (`status: "completed"`) automatically hide the Re-transcribe button to avoid redundant work.
- **App Splash Screen & Sub-Second Fast Startup Sequence**:
  - A sleek, minimalist dark launch screen (`#splash-screen`) displays brand audio wave animations and status text while connecting to the local backend.
  - **Fast Non-Blocking Backend Boot**: `backend.main` starts Uvicorn and binds `127.0.0.1:48653` without blocking on heavy ML imports. Heavy PyTorch C++/CUDA DLL loading is deferred and warmed asynchronously in a daemon background thread (`_detect_torch_device`), dropping backend cold-boot latency by over 50%. Audio capture can initialize immediately on record request while PyTorch completes background warmup.
  - **Adaptive Fast WebSocket Reconnect**: `api.connectWebSocket()` uses adaptive initial retry delays (50ms, 110ms, 170ms... up to 1000ms cap) before first connection, locking onto the backend socket within 50ms of the port opening rather than idling on a fixed 2s delay.
  - **Parallelized Initial Sync**: Status and audio device discovery run concurrently via `Promise.all([refreshStatus(), refreshDevices()])`. As soon as status and devices are verified, `dismissSplashScreen()` triggers immediately without blocking on past session history. Past session history is loaded asynchronously in the background.
  - **Snappy Splash Dismissal**: Splash fade-out transition is tuned to 180ms ease-out, making the main interface interactive and ready to record in ~0.8–1.0 second.
  - Includes a 10s safety timeout to avoid blocking the UI if startup hangs. Initial load suppresses error toasts while the backend is still launching so users never see a transient "History unavailable" banner. Successful history fetch immediately clears any active history error banner.
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
- **Single Instance Enforcement (`tauri-plugin-single-instance`)**:
  - The application uses `tauri-plugin-single-instance` to prevent multiple concurrent instances from running.
  - When the app is closed to tray and the user re-launches the `.exe` (or accidentally double-clicks it multiple times), the secondary process passes its focus command to the existing instance and terminates immediately.
  - The primary instance unminimizes, shows, and focuses the main window.
  - `ensure_backend_running()` runs inside `.setup()` so that secondary launching processes never attempt to connect to, spawn, or adopt the backend process, preventing duplicate tray icons and preventing secondary exits from shutting down the active backend.
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

## Speaker Diarization, Multi-Speaker Phrase Separation & Custom Speaker Names

- **Offline Diarization Engine (`backend/diarization.py`)**:
  - Automatically activates for single-file uploads, multi-speaker loopback captures, and 2nd-pass post-meeting processing.
  - Leverages locally cached Pyannote 3.1 (`models--pyannote--speaker-diarization-3.1` and `wespeaker-voxceleb-resnet34-LM` in `~/.cache/huggingface/hub`).
  - Passes in-memory waveform tensors (`{"waveform": tensor, "sample_rate": 16000}`) to avoid `torchcodec` dependency issues on Windows.
  - Falls back to heuristic acoustic feature clustering when Pyannote weights are not present.
  - Automatically merges contiguous turns of the same speaker separated by short pauses (`_merge_contiguous_turns`) and maps them to clean identifiers (`Speaker 1`, `Speaker 2`, etc.).
- **Multi-Speaker Separation within VAD Phrases (`split_phrases_by_speaker_turns`)**:
  - Silero VAD detects speech onset and pause boundaries (silence pauses), which can encompass multiple speakers talking back-to-back in the same audio track without an 0.85s pause.
  - `SpeakerDiarizer.split_phrases_by_speaker_turns` checks if an incoming VAD phrase contains multiple distinct speaker turns. If so, it splits the phrase at the speaker transition boundary into distinct sub-phrases (`phrase_id`, `phrase_id_s1`, etc.).
  - `run_high_quality_pass` slices the audio corresponding to each speaker's sub-interval and transcribes each separately, ensuring speaker attribution is clean and no text from different speakers is fused into a single phrase.
  - In continuous audio / file uploads, `SpeakerDiarizer.segment_audio_by_turns` segments the audio directly using diarization turn boundaries (merging only contiguous turns of the same speaker up to 25s), completely preventing cross-speaker chunking.
  - Live phrases on CUDA are also checked via `_transcribe_phrase_item`, emitting separate `phrase_transcribed` events for each detected speaker sub-segment.
- **App Simulation Evaluation Framework (`eval_dataset/evaluate_app_simulation.py`)**:
  - Simulates end-to-end transcriber behavior on evaluation datasets:
    - Mode `vad_stream`: Streams 512-sample frames through `PhraseSegmenter` (Silero VAD) to produce realistic live phrases, then executes `run_high_quality_pass`.
    - Mode `upload`: Directly processes audio files through `run_high_quality_pass` turn-based segmentation.
  - Computes Hungarian 1:1 speaker alignment, Global WER, Speaker-Attributed WER, CER, RTF, and detects speaker fusion errors where a single phrase spans across distinct ground-truth speaker turns.
  - Generates JSON and Markdown reports in `eval_dataset/benchmark_outputs/`.
- **User Name & Speaker Customization**:
  - User can configure their personal name in Settings (`#input-user-name`), persisted in `localStorage["sayso.settings"]`.
  - When starting a recording, `user_name` is transmitted to `POST /api/record/start` and stored on the session. Single-speaker tracks and microphone audio are labeled with the user's name rather than generic "Me".
  - If a recording has only 1 speaker, full text is formatted as natural paragraphs without repetitive "Speaker 1:" prefixes.
- **In-Place Speaker Renaming (`POST /api/recordings/{session_id}/rename_speaker`)**:
  - In the history transcript view, every phrase row features a speaker tag with an edit button (`.speaker-rename-btn`).
  - Clicking opens `#rename-speaker-modal` to rename that speaker across the entire meeting.
  - `StorageManager.rename_speaker` updates all in-memory and persisted phrases, adjusts `final_transcript` speaker prefixes with regex, increments `transcript_revision`, and updates exported `.txt` and `.srt` files.

## History Search

- Dedicated search bar in the History sidebar (`#history-search-input`, `#btn-clear-search`).
- Filters recordings in real-time across meeting titles, full transcript text, individual phrase texts, and speaker names.
- Renders matching substring highlights (`.search-match-highlight`) in meeting titles.
- Pressing Escape or clicking ✕ clears the search query and restores the full list.

## System Tray, Background Running & Floating Mini Widget

- **Tauri System Tray (`src-tauri/src/lib.rs`)**:
  - Configured with `TrayIconBuilder` and context menu: "Open Sayso", dynamic record item ("Start recording" / "Stop recording"), and "Exit Sayso".
  - Managed state `TrayState` updates the menu item text in real time via `update_tray_recording_state(is_recording)`.
  - Double-clicking the tray icon restores, unminimizes, and focuses the main window.
  - Clicking the recording tray button emits `tray_toggle_record` to the frontend, handled exclusively by the main window to toggle capture and show the floating mini widget.
- **Background Running Behavior**:
  - Closing the main window (`CloseRequested`) intercepts window closing, runs `api.prevent_close()`, and hides the window to the tray. The app and background recording continue uninterrupted.
  - To fully terminate the application, the user clicks "Quit Sayso" (`#btn-quit-app` in Settings) or "Exit Sayso" in the system tray menu.
  - `quit_app` calls `cleanup_processes()`, forcefully terminating child Python processes and freeing port `48653`.
- **Floating Mini Widget (`?view=mini`)**:
  - Configured as a secondary lightweight window (`tauri.conf.json`) with `width: 150`, `height: 54`, `transparent: true`, `decorations: false`, and `alwaysOnTop: true`.
  - **Visibility Rule**: Only shown if recording is active AND the main window is closed/hidden to the tray (`recording && !main_visible`). When the main application is open, the floating widget is automatically hidden.
  - Seamless mouse dragging supported via both CSS `-webkit-app-region: drag` and native `getCurrentWindow().startDragging()` on primary `mousedown`. Action buttons are excluded via `-webkit-app-region: no-drag`.
  - Displays:
    - Dynamic 3-bar audio activity visualizer (`#mini-vad-meter`) matching the main app's `.vad-bar` animation (flex-end alignment, `0.12s cubic-bezier(0.2, 0.8, 0.2, 1)` transition, and `.speaking` glow), combining mic and loopback RMS into a single unified meter.
    - "Open App" button (`#btn-mini-open-app`) to focus and unminimize the main window while hiding the mini widget.
    - "Stop Recording" button (`#btn-mini-stop-rec`).
  - Stopping through the floating widget emits `request_stop_recording`, closes the mini widget, opens the main window directly on the newly finished recording in the History view (`#tab-history`), displays the high-quality re-transcription progress modal (`#hq-progress-modal`), and updates in place with the finalized transcript and audio player once complete.

## Validation Commands

```powershell
npm test
python -m pytest -q
npm run build
cargo check --manifest-path src-tauri/Cargo.toml
cmd /c build.bat --no-bundle
```

## Python Environment Setup (Option B)

- **Supported Python**: Python 3.10 through 3.13 on PATH.
- **Dependencies**: All packages listed in `requirements-portable.txt` plus test packages (`pytest`, `httpx`).
- **GPU Acceleration**: NVIDIA GPU acceleration via `torch==2.11.0+cu126` and `torchaudio==2.11.0+cu126` (via extra index `https://download.pytorch.org/whl/cu126`). On launch, `_detect_torch_device()` automatically detects CUDA capability.
- **Native Host Interaction**: The Tauri host discovers Python via `local_python(&root)` (embedded `python/` folder) and falls back to system `python` on `PATH`. It spawns `python -m backend.main` using `CREATE_NO_WINDOW`.

## Portable Windows Packaging

`package-portable.ps1` creates `Sayso-Portable/Sayso/`, containing `Sayso.exe`,
`backend/`, an embedded Python runtime, writable `recordings/`, and
`models/huggingface/`. The Cohere cache is included by default. Use
`-NoModel` to ship an empty model directory and let the app download it on first
run. The package output is deliberately not listed in `.gitignore`.

A normal build requires network access for the embedded Python archive, pip
packages, and (unless `-NoModel` is used) the local Hugging Face model cache.
The `nano_cohere_transcribe` dependency is installed from a pinned GitHub
source revision because it is not published on PyPI. `python-multipart` is
required for FastAPI upload form parsing (`/api/upload/transcribe`). PyTorch and torchaudio are
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
