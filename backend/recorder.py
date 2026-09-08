import time
import uuid
import logging
import threading
from typing import Optional, Dict, Any, Callable, List
import numpy as np

from .config import (
    SAMPLE_RATE,
    VAD_FRAME_SIZE,
    FORMAT_WIDTH,
)
from .vad_engine import PhraseSegmenter
from .audio_devices import PORTAUDIO_LOCK

logger = logging.getLogger(__name__)

try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    try:
        import pyaudio
        HAS_PYAUDIO = True
    except ImportError:
        pyaudio = None
        HAS_PYAUDIO = False


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Fast, dependency-free resampling to 16kHz mono float32."""
    if orig_sr == SAMPLE_RATE or len(audio) == 0:
        return audio.astype(np.float32)
    target_len = int(round(len(audio) * SAMPLE_RATE / orig_sr))
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x_orig = np.linspace(0, 1, len(audio), endpoint=False)
    x_target = np.linspace(0, 1, target_len, endpoint=False)
    return np.interp(x_target, x_orig, audio).astype(np.float32)


class AudioRecorder:
    """
    Dual-channel audio recorder for Windows:
    - Captures Microphone input and/or Windows WASAPI Loopback (PC audio).
    - Normalizes streams to 16kHz mono float32.
    - Feeds real-time Silero VAD for live voice activity meters.
    - Detects phrase boundaries and submits segments to STTEngine.
    - Accumulates full-session audio for storage and high-quality 2nd pass.
    """

    def __init__(
        self,
        on_vad_level: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_phrase_ready: Optional[Callable[[str, str, float, float, np.ndarray], None]] = None,
        on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.on_vad_level = on_vad_level
        self.on_phrase_ready = on_phrase_ready
        self.on_error = on_error
        self.last_error: Optional[str] = None

        self.is_recording = False
        self.session_id: Optional[str] = None
        self.recording_mode: str = "mic_only"  # mic_only | mic_and_system
        self.start_time: float = 0.0

        self._stop_event = threading.Event()
        self._mic_thread: Optional[threading.Thread] = None
        self._loopback_thread: Optional[threading.Thread] = None

        # Full audio accumulators
        self._mic_audio_chunks: List[np.ndarray] = []
        self._loopback_audio_chunks: List[np.ndarray] = []

        # Real-time VAD states
        self._last_mic_level: Dict[str, Any] = {"rms": 0.0, "prob": 0.0, "is_speaking": False}
        self._last_system_level: Dict[str, Any] = {"rms": 0.0, "prob": 0.0, "is_speaking": False}

        # Phrase segmenters
        self._mic_segmenter: Optional[PhraseSegmenter] = None
        self._system_segmenter: Optional[PhraseSegmenter] = None

        # Thread lock
        self._lock = threading.Lock()

    def start(
        self,
        session_id: str,
        mode: str = "mic_only",
        mic_device_index: Optional[int] = None,
        system_device_index: Optional[int] = None,
    ) -> bool:
        with self._lock:
            if self.is_recording:
                logger.warning("Recorder already running.")
                return False

            self.session_id = session_id
            self.recording_mode = mode
            self.start_time = time.time()
            self.last_error = None
            self._stop_event.clear()
            self._mic_audio_chunks.clear()
            self._loopback_audio_chunks.clear()

            # Initialize phrase segmenters
            self._mic_segmenter = PhraseSegmenter(
                stream_name="You",
                on_phrase_completed=self._handle_phrase_completed,
            )
            if mode == "mic_and_system":
                self._system_segmenter = PhraseSegmenter(
                    stream_name="System / Remote",
                    on_phrase_completed=self._handle_phrase_completed,
                )
            else:
                self._system_segmenter = None

            self.is_recording = True

            if not HAS_PYAUDIO or pyaudio is None:
                logger.warning("PyAudio not available, running simulation mode.")
                self._mic_thread = threading.Thread(target=self._simulate_capture_worker, daemon=True)
                self._mic_thread.start()
                return True

            # Start Microphone capture thread
            self._mic_thread = threading.Thread(
                target=self._capture_mic_worker,
                args=(mic_device_index,),
                daemon=True,
            )
            self._mic_thread.start()

            # Start System Audio Loopback capture thread if requested
            if mode == "mic_and_system":
                self._loopback_thread = threading.Thread(
                    target=self._capture_loopback_worker,
                    args=(system_device_index,),
                    daemon=True,
                )
                self._loopback_thread.start()

            logger.info(f"Recorder started session {session_id} in {mode} mode.")
            return True

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self.is_recording:
                return {"duration": 0.0, "audio": np.zeros(0, dtype=np.float32)}

            self.is_recording = False
            self._stop_event.set()

        # Wait for threads to finish gracefully
        if self._mic_thread and self._mic_thread.is_alive():
            self._mic_thread.join(timeout=1.5)
        if self._loopback_thread and self._loopback_thread.is_alive():
            self._loopback_thread.join(timeout=1.5)

        current_t = time.time() - self.start_time

        # Flush pending phrases
        if self._mic_segmenter:
            self._mic_segmenter.flush(current_t)
        if self._system_segmenter:
            self._system_segmenter.flush(current_t)

        # Combine audio
        full_mic = np.concatenate(self._mic_audio_chunks) if self._mic_audio_chunks else np.zeros(0, dtype=np.float32)
        full_sys = np.concatenate(self._loopback_audio_chunks) if self._loopback_audio_chunks else np.zeros(0, dtype=np.float32)

        # Mix mic and system audio for full playback
        if len(full_mic) > 0 and len(full_sys) > 0:
            max_len = max(len(full_mic), len(full_sys))
            mixed = np.zeros(max_len, dtype=np.float32)
            mixed[:len(full_mic)] += full_mic * 0.9
            mixed[:len(full_sys)] += full_sys * 0.9
            final_audio = np.clip(mixed, -1.0, 1.0)
        elif len(full_mic) > 0:
            final_audio = full_mic
        elif len(full_sys) > 0:
            final_audio = full_sys
        else:
            final_audio = np.zeros(int(current_t * SAMPLE_RATE), dtype=np.float32)

        stopped_session_id = self.session_id
        self.session_id = None
        duration = len(final_audio) / SAMPLE_RATE
        logger.info(f"Recorder stopped session {stopped_session_id}: {duration:.2f}s audio captured.")

        return {
            "session_id": stopped_session_id,
            "duration": duration,
            "audio": final_audio,
        }

    def _report_capture_error(self, source: str, error: str):
        self.last_error = error
        payload = {
            "type": "recording_error",
            "session_id": self.session_id,
            "source": source,
            "error": error,
        }
        logger.error("%s capture error: %s", source, error)
        if self.on_error:
            self.on_error(payload)

    def _handle_phrase_completed(self, stream_name: str, start_time: float, end_time: float, audio_np: np.ndarray):
        if self.on_phrase_ready and self.session_id:
            phrase_id = f"p_{int(start_time*1000)}_{uuid.uuid4().hex[:4]}"
            self.on_phrase_ready(self.session_id, stream_name, start_time, end_time, audio_np)

    def _capture_mic_worker(self, device_index: Optional[int]):
        p = None
        stream = None
        try:
            with PORTAUDIO_LOCK:
                p = pyaudio.PyAudio()
                # Resolve device info
                if device_index is not None and device_index >= 0:
                    try:
                        dev_info = p.get_device_info_by_index(device_index)
                    except Exception:
                        dev_info = p.get_default_input_device_info()
                else:
                    dev_info = p.get_default_input_device_info()

                if not dev_info:
                    from .audio_devices import get_audio_devices
                    devs = get_audio_devices()
                    def_mic = devs.get("default_mic")
                    if def_mic and def_mic.get("index", -1) >= 0:
                        dev_info = p.get_device_info_by_index(def_mic["index"])

                if not dev_info:
                    self._report_capture_error("microphone", "No microphone input device was found")
                    return

                idx = dev_info["index"]
                dev_sr = int(dev_info.get("defaultSampleRate", SAMPLE_RATE))
                dev_channels = min(2, max(1, dev_info.get("maxInputChannels", 1)))

                # Read in frames corresponding to ~32ms chunks
                frames_per_buffer = int(dev_sr * (VAD_FRAME_SIZE / SAMPLE_RATE))
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=dev_channels,
                    rate=dev_sr,
                    input=True,
                    input_device_index=idx,
                    frames_per_buffer=frames_per_buffer,
                )
            logger.info(f"Mic capture open on '{dev_info['name']}' at {dev_sr}Hz ({dev_channels}ch).")

            remainder_buf = np.zeros(0, dtype=np.float32)
            mic_samples_processed = 0

            while not self._stop_event.is_set():
                try:
                    data = stream.read(frames_per_buffer, exception_on_overflow=False)
                except Exception as e:
                    logger.debug(f"Mic read buffer overflow/error: {e}")
                    continue

                if not data:
                    continue

                # Convert int16 PCM bytes to float32 (-1.0 to 1.0)
                raw_int16 = np.frombuffer(data, dtype=np.int16)
                if dev_channels > 1:
                    raw_int16 = raw_int16.reshape(-1, dev_channels)
                    raw_mono = raw_int16.mean(axis=1)
                else:
                    raw_mono = raw_int16

                audio_f32 = raw_mono.astype(np.float32) / 32768.0

                # Resample to 16kHz
                if dev_sr != SAMPLE_RATE:
                    audio_16k = resample_to_16k(audio_f32, dev_sr)
                else:
                    audio_16k = audio_f32

                self._mic_audio_chunks.append(audio_16k)

                # Append to rolling VAD buffer
                remainder_buf = np.concatenate([remainder_buf, audio_16k])
                while len(remainder_buf) >= VAD_FRAME_SIZE:
                    frame = remainder_buf[:VAD_FRAME_SIZE]
                    remainder_buf = remainder_buf[VAD_FRAME_SIZE:]

                    timestamp_s = mic_samples_processed / SAMPLE_RATE
                    mic_samples_processed += len(frame)
                    rms, prob, is_speech = self._mic_segmenter.process_frame(frame, timestamp_s)
                    self._last_mic_level = {"rms": rms, "prob": prob, "is_speaking": is_speech}

                self._emit_vad_status()

        except Exception as e:
            logger.error(f"Error in mic capture worker: {e}", exc_info=True)
            self._report_capture_error("microphone", str(e))
        finally:
            with PORTAUDIO_LOCK:
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                if p is not None:
                    try:
                        p.terminate()
                    except Exception:
                        pass

    def _capture_loopback_worker(self, device_index: Optional[int]):
        p = None
        stream = None
        try:
            with PORTAUDIO_LOCK:
                p = pyaudio.PyAudio()
                # Resolve WASAPI loopback device
                if device_index is not None and device_index >= 0:
                    try:
                        dev_info = p.get_device_info_by_index(device_index)
                    except Exception:
                        dev_info = p.get_default_wasapi_loopback() if hasattr(p, "get_default_wasapi_loopback") else None
                else:
                    dev_info = p.get_default_wasapi_loopback() if hasattr(p, "get_default_wasapi_loopback") else None

                if not dev_info:
                    from .audio_devices import get_audio_devices
                    devs = get_audio_devices()
                    def_sys = devs.get("default_system")
                    if def_sys and def_sys.get("index", -1) >= 0:
                        dev_info = p.get_device_info_by_index(def_sys["index"])

                if not dev_info:
                    self._report_capture_error("system", "No Windows system-audio loopback device was found")
                    return

                idx = dev_info["index"]
                dev_sr = int(dev_info.get("defaultSampleRate", 48000))
                dev_channels = max(1, dev_info.get("maxInputChannels", 2))

                frames_per_buffer = int(dev_sr * (VAD_FRAME_SIZE / SAMPLE_RATE))
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=dev_channels,
                    rate=dev_sr,
                    input=True,
                    input_device_index=idx,
                    frames_per_buffer=frames_per_buffer,
                )
            logger.info(f"WASAPI loopback open on '{dev_info['name']}' at {dev_sr}Hz ({dev_channels}ch).")

            remainder_buf = np.zeros(0, dtype=np.float32)
            sys_samples_processed = 0

            while not self._stop_event.is_set():
                try:
                    data = stream.read(frames_per_buffer, exception_on_overflow=False)
                except Exception as e:
                    logger.debug(f"Loopback read overflow/error: {e}")
                    continue

                if not data:
                    continue

                raw_int16 = np.frombuffer(data, dtype=np.int16)
                if dev_channels > 1:
                    raw_int16 = raw_int16.reshape(-1, dev_channels)
                    raw_mono = raw_int16.mean(axis=1)
                else:
                    raw_mono = raw_int16

                audio_f32 = raw_mono.astype(np.float32) / 32768.0

                if dev_sr != SAMPLE_RATE:
                    audio_16k = resample_to_16k(audio_f32, dev_sr)
                else:
                    audio_16k = audio_f32

                self._loopback_audio_chunks.append(audio_16k)

                remainder_buf = np.concatenate([remainder_buf, audio_16k])
                while len(remainder_buf) >= VAD_FRAME_SIZE:
                    frame = remainder_buf[:VAD_FRAME_SIZE]
                    remainder_buf = remainder_buf[VAD_FRAME_SIZE:]

                    timestamp_s = sys_samples_processed / SAMPLE_RATE
                    sys_samples_processed += len(frame)
                    if self._system_segmenter:
                        rms, prob, is_speech = self._system_segmenter.process_frame(frame, timestamp_s)
                        self._last_system_level = {"rms": rms, "prob": prob, "is_speaking": is_speech}

                self._emit_vad_status()

        except Exception as e:
            logger.error(f"Error in loopback worker: {e}", exc_info=True)
            self._report_capture_error("system", str(e))
        finally:
            with PORTAUDIO_LOCK:
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                if p is not None:
                    try:
                        p.terminate()
                    except Exception:
                        pass

    def _simulate_capture_worker(self):
        """Simulation worker when PyAudio is not available."""
        logger.info("Running simulation audio loop.")
        while not self._stop_event.is_set():
            time.sleep(0.05)
            # Generate simulated background noise and periodic speech
            elapsed = time.time() - self.start_time
            is_speaking = (int(elapsed) % 6) < 3
            rms = 0.4 if is_speaking else 0.05
            prob = 0.85 if is_speaking else 0.02

            self._last_mic_level = {"rms": rms, "prob": prob, "is_speaking": is_speaking}
            if self.recording_mode == "mic_and_system":
                self._last_system_level = {"rms": 0.05, "prob": 0.02, "is_speaking": False}

            self._emit_vad_status()

    def _emit_vad_status(self):
        if self.on_vad_level and self.is_recording:
            elapsed = time.time() - self.start_time
            self.on_vad_level({
                "type": "vad_meter",
                "mic": self._last_mic_level,
                "system": self._last_system_level,
                "duration": round(elapsed, 1),
            })
