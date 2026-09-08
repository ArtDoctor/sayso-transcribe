# Sayso

An ultra-minimalist desktop transcription app inspired by Meetily.

I like using Meetily, but the lack of Cohere support, a few features I don’t need, some UI/UX quirks, and - most importantly - the absence of a dark theme pushed me to vibe-code my own version: **Sayso**.

I built this primarily for personal use, but if anyone else finds it useful, feel free to use it, fork it, or suggest improvements.

Built with a Rust desktop shell & UI (Tauri 2 + Vite/TypeScript) and a Python backend with Silero VAD and nano-cohere-transcribe. On my GPU (RTX 4070 12GB VRAM), the RTFx factor was 100-200 (~180 seconds of audio transcribed in 1 second of GPU time), while reaching very good word error rate (compare to e.g. whisper).

---

## Getting Started

### Prerequisites

- **Python 3.10+** (with `torch`, `silero-vad`, `pyaudiowpatch`, `nano-cohere-transcribe`)
- **Node.js 18+** & `npm`
- **Rust toolchain** (for compiling Tauri desktop binary)

### Running the App

Double-click [`start.bat`](file:///D:/projects/transcriber/start.bat) or run:

```cmd
start.bat
```

This will:
1. Verify Python and launch the backend server on `127.0.0.1:8765`.
2. Install frontend dependencies if needed (`npm install`).
3. Launch the native Tauri 2 desktop app.

For browser-only preview:
```cmd
start-web.bat
```

---

## Running Tests

Execute [`test.bat`](file:///D:/projects/transcriber/test.bat):

```cmd
test.bat
```

Runs:
1. `python -m pytest tests/test_backend.py -v` (Backend device, VAD, storage, and API tests)
2. `npm test` (Vitest frontend tests)
