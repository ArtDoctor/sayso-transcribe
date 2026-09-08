import math
import time
import logging
from typing import Optional, Tuple, Callable
import numpy as np
import torch

from .config import (
    SAMPLE_RATE,
    VAD_FRAME_SIZE,
    VAD_SPEECH_THRESHOLD,
    VAD_SILENCE_DURATION_S,
    MAX_PHRASE_DURATION_S,
    MIN_PHRASE_DURATION_S,
    VAD_PRE_ROLL_S,
    VAD_POST_ROLL_S,
)

logger = logging.getLogger(__name__)


class SileroVADWrapper:
    """Wrapper around Silero VAD Torch model with graceful fallback."""

    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad()
            self.model.eval()
            logger.info("Silero VAD model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}. Running energy fallback.")
            self.model = None

    def predict_chunk(self, chunk_f32: np.ndarray) -> float:
        """
        Accepts 512 float32 samples (-1.0 to 1.0) at 16kHz.
        Returns speech probability between 0.0 and 1.0.
        """
        if self.model is not None:
            try:
                # Silero expects 1D tensor of length 512
                tensor = torch.from_numpy(chunk_f32).float()
                if tensor.ndim == 1 and tensor.shape[0] == VAD_FRAME_SIZE:
                    with torch.no_grad():
                        prob = self.model(tensor, SAMPLE_RATE).item()
                        return float(prob)
            except Exception as e:
                logger.debug(f"Silero VAD inference error: {e}")

        # Fallback to RMS energy thresholding if model unavailable
        rms = np.sqrt(np.mean(chunk_f32 ** 2) + 1e-9)
        return float(min(1.0, rms * 15.0))


class PhraseSegmenter:
    """
    Stateful phrase boundary detector for a single audio stream (Mic or System Audio).
    Splits continuous stream into coherent phrases based on VAD pauses or max clip limit.
    Includes pre-roll and post-roll audio padding (overlap margins) around detected speech.
    """

    def __init__(
        self,
        stream_name: str,
        speech_threshold: float = VAD_SPEECH_THRESHOLD,
        silence_duration_s: float = VAD_SILENCE_DURATION_S,
        max_phrase_duration_s: float = MAX_PHRASE_DURATION_S,
        min_phrase_duration_s: float = MIN_PHRASE_DURATION_S,
        pre_roll_s: float = VAD_PRE_ROLL_S,
        post_roll_s: float = VAD_POST_ROLL_S,
        on_phrase_completed: Optional[Callable[[str, float, float, np.ndarray], None]] = None,
    ):
        self.stream_name = stream_name
        self.speech_threshold = speech_threshold
        self.silence_duration_s = silence_duration_s
        self.max_phrase_duration_s = max_phrase_duration_s
        self.min_phrase_duration_s = min_phrase_duration_s
        self.pre_roll_s = pre_roll_s
        self.post_roll_s = post_roll_s
        self.on_phrase_completed = on_phrase_completed

        self.vad = SileroVADWrapper()

        # State
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.stream_offset_s: float = 0.0

        # Pre-roll & post-roll frame counts
        frame_dur = VAD_FRAME_SIZE / SAMPLE_RATE  # 512 / 16000 = 0.032s
        self.pre_roll_frames = max(1, int(round(self.pre_roll_s / frame_dur)))
        self.post_roll_frames = max(1, int(round(self.post_roll_s / frame_dur)))

        # Audio buffers
        self.phrase_chunks: list[np.ndarray] = []
        self.pre_buffer: list[np.ndarray] = []
        self.silence_frames_after_speech: int = 0

    def process_frame(self, frame_f32: np.ndarray, timestamp_s: float) -> Tuple[float, float, bool]:
        """
        Process a 512-sample float32 frame (32ms).
        Returns: (rms_level, speech_probability, is_voice_active)
        """
        frame_dur = VAD_FRAME_SIZE / SAMPLE_RATE

        # Calculate RMS for visual audio meter (0.0 to 1.0)
        rms = float(np.sqrt(np.mean(frame_f32 ** 2) + 1e-9))
        rms_norm = min(1.0, rms * 8.0)  # scale for UI visibility

        # Compute speech probability
        prob = self.vad.predict_chunk(frame_f32)
        is_speech = prob >= self.speech_threshold

        if is_speech:
            self.last_speech_time = timestamp_s
            self.silence_frames_after_speech = 0

            if not self.is_speaking:
                # Speech onset
                self.is_speaking = True
                actual_pre_roll_s = len(self.pre_buffer) * frame_dur
                self.speech_start_time = max(0.0, timestamp_s - actual_pre_roll_s)
                # Seed phrase with pre-buffer frames (pre-roll margin)
                self.phrase_chunks = list(self.pre_buffer)
                self.phrase_chunks.append(frame_f32)
            else:
                self.phrase_chunks.append(frame_f32)
                # Check max phrase duration limit
                start_ref = self.speech_start_time if self.speech_start_time is not None else timestamp_s
                curr_duration = (timestamp_s + frame_dur) - start_ref
                if curr_duration >= self.max_phrase_duration_s:
                    overlap = (
                        self.phrase_chunks[-self.pre_roll_frames:]
                        if len(self.phrase_chunks) >= self.pre_roll_frames
                        else list(self.phrase_chunks)
                    )
                    self._finalize_phrase(timestamp_s, keep_overlap=overlap)

        elif self.is_speaking:
            self.silence_frames_after_speech += 1
            # Include post-roll frames up to post_roll_frames
            if self.silence_frames_after_speech <= self.post_roll_frames:
                self.phrase_chunks.append(frame_f32)

            # Update rolling pre_buffer during silence
            self.pre_buffer.append(frame_f32)
            if len(self.pre_buffer) > self.pre_roll_frames:
                self.pre_buffer.pop(0)

            # Check pause duration
            last_ref = self.last_speech_time if self.last_speech_time is not None else timestamp_s
            start_ref = self.speech_start_time if self.speech_start_time is not None else timestamp_s
            silence_gap = timestamp_s - last_ref
            phrase_duration = (timestamp_s + frame_dur) - start_ref

            if silence_gap >= self.silence_duration_s or phrase_duration >= self.max_phrase_duration_s:
                self._finalize_phrase(timestamp_s)

        else:
            # Idle silence: update rolling pre_buffer
            self.pre_buffer.append(frame_f32)
            if len(self.pre_buffer) > self.pre_roll_frames:
                self.pre_buffer.pop(0)

        return (rms_norm, prob, is_speech)

    def _finalize_phrase(self, current_time_s: float, keep_overlap: Optional[list[np.ndarray]] = None):
        if not self.phrase_chunks or self.speech_start_time is None:
            self._reset_state(keep_overlap, current_time_s)
            return

        total_samples = sum(len(c) for c in self.phrase_chunks)
        duration_s = total_samples / SAMPLE_RATE

        if duration_s >= self.min_phrase_duration_s:
            phrase_audio = np.concatenate(self.phrase_chunks)
            start_t = max(0.0, self.speech_start_time)
            end_t = start_t + duration_s
            logger.info(f"[{self.stream_name}] Phrase detected with overlap: {start_t:.2f}s -> {end_t:.2f}s ({duration_s:.2f}s)")
            if self.on_phrase_completed:
                self.on_phrase_completed(self.stream_name, start_t, end_t, phrase_audio)

        self._reset_state(keep_overlap, current_time_s)

    def flush(self, current_time_s: float):
        """Force complete any pending phrase upon recording stop."""
        if self.is_speaking and self.phrase_chunks:
            self._finalize_phrase(current_time_s)
        self._reset_state()

    def _reset_state(self, keep_overlap: Optional[list[np.ndarray]] = None, current_time_s: Optional[float] = None):
        if keep_overlap and current_time_s is not None:
            self.is_speaking = True
            overlap_dur = (len(keep_overlap) * VAD_FRAME_SIZE) / SAMPLE_RATE
            self.speech_start_time = max(0.0, current_time_s - overlap_dur)
            self.last_speech_time = current_time_s
            self.phrase_chunks = list(keep_overlap)
            self.silence_frames_after_speech = 0
        else:
            self.is_speaking = False
            self.speech_start_time = None
            self.last_speech_time = None
            self.phrase_chunks = []
            self.silence_frames_after_speech = 0
