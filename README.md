# Sayso

An ultra-minimalist desktop transcription app inspired by [Meetily](https://github.com/Zackriya-Solutions/meetily).

https://github.com/user-attachments/assets/52a862f3-a0b7-4dbd-938b-e03ba5edaa7e

I like using [Meetily](https://github.com/Zackriya-Solutions/meetily), but the lack of [Cohere](https://cohere.com/) support, a few features I don’t need, some UI/UX quirks, and - most importantly - the absence of a dark theme pushed me to vibe-code my own version: **Sayso**.

I built this primarily for personal use, but if anyone else finds it useful, feel free to use it, fork it, or suggest improvements.

Built with a Rust desktop shell & UI ([Tauri 2](https://v2.tauri.app/) + Vite/TypeScript) and a Python backend with [Silero VAD](https://github.com/snakers4/silero-vad) and [nano-cohere-transcribe](https://github.com/Deep-unlearning/nano-cohere-transcribe). On my GPU (RTX 4070 12GB VRAM), the RTFx factor was 100-200 (~180 seconds of audio transcribed in 1 second of GPU time), while reaching very good word error rate (compare to e.g. [Whisper](https://github.com/openai/whisper)).

This theoretically should work with 6-8GB VRAM GPUs too. But lower GPUs probably aren't enough. It can transcribe on CPU, but will be quite slow.

---

## Getting Started

### Prerequisites

- **Python 3.10+** (with [`torch`](https://pytorch.org/), [`silero-vad`](https://github.com/snakers4/silero-vad), [`pyaudiowpatch`](https://github.com/s0d3s/PyAudioWPatch), [`nano-cohere-transcribe`](https://github.com/Deep-unlearning/nano-cohere-transcribe))
- **Node.js 18+** & **npm**
- **Rust toolchain** (for compiling the [Tauri](https://v2.tauri.app/) desktop binary)

### Running the App

Double-click `start.bat` or run:

```cmd
start.bat
```

This will:
1. Verify Python and launch the backend server on `127.0.0.1:48653`.
2. Install frontend dependencies if needed (`npm install`).
3. Launch the native Tauri 2 desktop app.

For browser-only preview:
```cmd
start-web.bat
```

---

## Running Tests

Execute `test.bat`:

```cmd
test.bat
```

Runs:
1. `python -m pytest tests/test_backend.py -v` (Backend device, VAD, storage, and API tests)
2. `npm test` (Vitest frontend tests)
