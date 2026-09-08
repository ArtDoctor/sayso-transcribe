# Sayso - Development Roadmap & Architecture

## Overview & Vision

A lightweight, reliable, high-performance local meeting transcription desktop application inspired by **Meetily** and built with:
- **Rust Desktop Shell & Frontend**: Tauri 2 + Vite/TypeScript for lightning-fast launch times, ultra-low resource usage, and a minimalistic dark-theme UI.
- **Python ML Backend**: Local asynchronous server powering real-time audio capture (Mic + Windows WASAPI Loopback), Silero VAD voice detection, and `nano-cohere-transcribe` with production-tuned 25.0s chunking.
- **Dual-Pass Transcription**:
  1. *Fast Live Transcription*: Low-latency phrase-based streaming transcription as you speak (VAD pause detection or 25s threshold).
  2. *High-Quality Post-Recording Pass*: Batched GPU inference across complete recorded audio for maximum accuracy and zero hallucination drift.
- **Full Privacy & Local Storage**: Audio recordings and editable transcripts stored locally with playback and export.

---

## Stage-by-Stage Plan

### Stage 1: Core Foundation & UI Shell (CURRENT STAGE - IMPLEMENTED)
*Goal: Working desktop app shell, Python backend connectivity, live Windows audio device enumeration with auto-checked defaults, real-time Silero VAD level streaming, and responsive dark-mode UI.*

- [x] **Project Structure**:
  - Tauri 2 Rust desktop shell (`src-tauri/`) with native window management and process coordination.
  - Vite + TypeScript frontend (`src/`) with modern dark theme and responsive layout.
  - Python backend (`backend/`) with FastAPI, WebSockets, and PyAudioWPatch.
- [x] **Audio Device Discovery (`audio_devices.py`)**:
  - Enumerate all microphone input devices and Windows WASAPI loopback devices (system audio).
  - Dynamically query and select current Windows default devices on every launch/settings check.
  - Fallback mechanisms for virtual audio cables and multi-channel devices.
- [x] **Real-Time Silero VAD Engine (`vad_engine.py`)**:
  - Load Silero VAD model (512-sample frames at 16kHz).
  - Compute voice probability and RMS energy for visual audio meters.
  - Emit real-time voice activity indicators for both Mic and PC/System audio.
- [x] **Dual-Mode Audio Recorder Engine (`recorder.py`)**:
  - Support two distinct recording modes:
    - **Mode 1**: Microphone Only (voice notes, solo dictation).
    - **Mode 2**: Mic + Windows Audio (meetings, remote calls on Zoom/Teams/Discord).
  - Synchronized dual-stream audio capture using `PyAudioWPatch`.
  - Phrase boundary segmentation (silence detection >= 800ms or max 25s duration).
- [x] **Minimalistic Dark UI**:
  - Sleek, intuitive Meetily-style interface.
  - Live dual-channel audio meters (Mic and System) with voice activity glowing indicators.
  - Active recording timer and status badges (Backend, Model, Audio).
  - Settings drawer for device selection and language preferences.
- [x] **Developer Ergonomics & Tests**:
  - Fast unit tests (`tests/test_backend.py`) verifying device discovery, VAD chunking, API endpoints, and storage.
  - One-click launch script (`start.bat`) running backend and frontend seamlessly.

---

### Stage 2: Live Phrase-Based Fast Transcription
*Goal: Seamless speech-to-text during meetings with asynchronous background model loading and live phrase streaming.*

- [x] **Background Model Initialization (`stt_engine.py`)**:
  - As soon as a meeting starts, spawn background thread to load `nano_cohere_transcribe`.
  - Audio recording begins with zero latency while model initializes.
  - In-memory phrase queue buffers audio chunks until model state becomes `Ready`.
- [x] **Live Phrase Streaming Pipeline**:
  - When Silero VAD detects a completed phrase or reaches the 25.0s ceiling:
    - Extract normalized 16kHz mono chunk.
    - Submit to STT worker thread.
    - Transcribe with greedy decoding (`beam=1`) for instantaneous speed.
  - Broadcast live transcription events (`phrase_transcribed`) via WebSocket:
    - Start timestamp, end timestamp, speaker tag (`You` vs `Remote/System`), and transcript text.
- [x] **Interactive Live Meeting UI**:
  - Live auto-scrolling phrase timeline.
  - Minimalist B&W speaker labels.
  - Clean real-time VU and VAD speech activity indicators.
  - Real-time timer and stop recording workflow.

---

### Stage 3: High-Quality Post-Recording 2nd-Pass Transcription
*Goal: Zero-compromise accuracy pass executed right after meeting finishes, leveraging batched Cohere inference on the entire audio track.*

- [x] **Full-Session Audio Serialization (`storage.py`)**:
  - Stream multi-channel or mixed WAV audio to disk (`recordings/<session_id>/audio.wav`).
  - Ensure uncompressed 16kHz 16-bit PCM format for clean playback and archiving.
- [x] **Post-Recording Batched 2nd Pass**:
  - When recording is stopped, send the full continuous audio waveform directly to `model.transcribe()`:
    - Nano-Cohere handles long audio splitting automatically via its internal energy-based boundary detector.
    - Executes parallel GPU chunk batching (`batch_size = 8`) with greedy decoding.
    - Joins decoded chunk texts seamlessly into a unified, high-accuracy transcript with zero prompt drift.
  - Reconcile and update the session transcript with the finalized high-quality result.
  - Notify UI via WebSocket (`hq_pass_completed`).

---

### Stage 4: Meeting History, Interactive Audio Playback & Transcript Editing
*Goal: Complete local meeting archive where users can search, replay audio, and fine-tune transcripts.*

- [x] **Session Archive Management**:
  - Persistent structured JSON storage for session metadata:
    - Session ID, title, date, duration, recording mode, audio file path, language.
  - REST endpoints for listing, fetching, updating, and deleting sessions.
- [x] **Interactive Audio Player**:
  - Built-in player with scrubber, play/pause, current/total time, and 1x/1.25x/1.5x playback speed.
- [x] **Synchronized Click-to-Seek Transcript**:
  - Clicking any phrase timestamp seeks the audio player to that exact timestamp.
- [x] **In-Place Transcript Editing**:
  - Edit phrase text directly with auto-save to disk via PATCH `/api/recordings/{session_id}`.
  - Editable meeting title.
- [x] **Multi-Format Export**:
  - Export to Plain Text (`.txt`), Subtitles (`.srt`), and clipboard copy.

---

### Stage 4.5: Model Download Gate & Minimalist Monochrome UI (NEW)
*Goal: Zero-failure model enforcement with real-time download progress and high-end black & white minimalist UI.*

- [x] **Model Download Gating & Verification**:
  - Meeting start is strictly disabled until the model is fully downloaded and loaded in memory.
  - Backend `POST /api/record/start` rejects recording requests with `400 Bad Request` if `model_status != 'ready'`.
  - Background `POST /api/model/download` downloads required model files (`config.json`, `tokenizer.model`, `model.safetensors`).
  - Real-time progress events (`model_download_progress`) streamed over WebSocket with %, downloaded MB / total MB, speed, and ETA.
- [x] **Ultra-Minimalist Black & White UI**:
  - Pure monochrome aesthetic (`#000000` deep black, `#121212` cards, `#242424` borders, `#ffffff` accents).
  - Minimal text and distraction-free typography.
  - Audio device selection completely relocated from the main view into the Settings modal.

---

### Stage 5: Meeting Intelligence, Local Summarization, Hotkeys & Polish
*Goal: Executive summaries, global shortcuts, standalone packaging, and quality-of-life enhancements.*

- [ ] **Local Meeting Summarization**:
  - Optional integration with local LLMs (e.g. Ollama or llama.cpp) to generate:
    - Executive Summary
    - Action Items & Owner assignments
    - Key Discussion Points
- [ ] **Global Hotkeys**:
  - System-wide hotkeys (e.g. `Ctrl+Alt+R` to toggle recording, `Ctrl+Alt+M` for mute/push-to-talk).
  - Floating mini-widget / system tray icon when recording.
- [ ] **Audio Pre-processing**:
  - Optional noise suppression and automatic gain control (AGC) on mic input.
  - Acoustic echo suppression between system loopback and mic.
- [ ] **Production Packaging**:
  - Self-contained Windows installer / executable bundling Tauri app and embedded Python environment.
