import math
import time
import logging
import threading
from typing import Optional, Tuple, Callable
import numpy as np

from .config import (
    SAMPLE_RATE,
    VAD_FRAME_SIZE,
    VAD_SPEECH_THRESHOLD,
    VAD_SILENCE_DURATION_S,
    VAD_SHORT_PHRASE_SILENCE_S,
    MAX_PHRASE_DURATION_S,
    MIN_PHRASE_DURATION_S,
    MIN_SPEECH_DURATION_S,
    MIN_ENERGY_RMS,
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
            import torch
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
                import torch
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

    def reset_states(self):
        if self.model is not None and hasattr(self.model, "reset_states"):
            try:
                self.model.reset_states()
            except Exception:
                pass


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
        short_phrase_silence_s: Optional[float] = None,
        max_phrase_duration_s: float = MAX_PHRASE_DURATION_S,
        min_phrase_duration_s: float = MIN_PHRASE_DURATION_S,
        min_speech_duration_s: float = MIN_SPEECH_DURATION_S,
        min_energy_rms: float = MIN_ENERGY_RMS,
        pre_roll_s: float = VAD_PRE_ROLL_S,
        post_roll_s: float = VAD_POST_ROLL_S,
        on_phrase_completed: Optional[Callable[[str, float, float, np.ndarray], None]] = None,
    ):
        self.stream_name = stream_name
        self.speech_threshold = speech_threshold
        self.silence_duration_s = silence_duration_s
        if short_phrase_silence_s is not None:
            self.short_phrase_silence_s = short_phrase_silence_s
        else:
            self.short_phrase_silence_s = VAD_SHORT_PHRASE_SILENCE_S if silence_duration_s == VAD_SILENCE_DURATION_S else silence_duration_s
        self.max_phrase_duration_s = max_phrase_duration_s
        self.min_phrase_duration_s = min_phrase_duration_s
        self.min_speech_duration_s = min_speech_duration_s
        self.min_energy_rms = min_energy_rms
        self.pre_roll_s = pre_roll_s
        self.post_roll_s = post_roll_s
        self.on_phrase_completed = on_phrase_completed

        self.vad = SileroVADWrapper()

        # State
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.stream_offset_s: float = 0.0

        # Cross-channel turn-taking coordination
        self.partner_segmenter: Optional["PhraseSegmenter"] = None
        self._lock = threading.Lock()

        # Pre-roll & post-roll frame counts
        frame_dur = VAD_FRAME_SIZE / SAMPLE_RATE  # 512 / 16000 = 0.032s
        self.pre_roll_frames = max(1, int(round(self.pre_roll_s / frame_dur)))
        self.post_roll_frames = max(1, int(round(self.post_roll_s / frame_dur)))

        # Audio buffers
        self.phrase_chunks: list[np.ndarray] = []
        self.pre_buffer: list[np.ndarray] = []
        self.silence_frames_after_speech: int = 0
        self.speech_frames_count: int = 0
        self.idle_silence_frames: int = 0

    def set_partner(self, partner: "PhraseSegmenter"):
        with self._lock:
            self.partner_segmenter = partner

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

        notify_partner_onset = False
        partner_start_ref = 0.0

        with self._lock:
            if is_speech:
                self.last_speech_time = timestamp_s
                self.silence_frames_after_speech = 0
                self.speech_frames_count += 1
                self.idle_silence_frames = 0

                if not self.is_speaking:
                    # Speech onset
                    self.is_speaking = True
                    actual_pre_roll_s = len(self.pre_buffer) * frame_dur
                    self.speech_start_time = max(0.0, timestamp_s - actual_pre_roll_s)
                    # Seed phrase with pre-buffer frames (pre-roll margin)
                    self.phrase_chunks = list(self.pre_buffer)
                    self.phrase_chunks.append(frame_f32)
                    notify_partner_onset = True
                    partner_start_ref = self.speech_start_time
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
                        self._finalize_phrase_locked(timestamp_s, keep_overlap=overlap)

            elif self.is_speaking:
                self.silence_frames_after_speech += 1
                # Include post-roll frames up to post_roll_frames
                if self.silence_frames_after_speech <= self.post_roll_frames:
                    self.phrase_chunks.append(frame_f32)

                # Update rolling pre_buffer during silence
                self.pre_buffer.append(frame_f32)
                if len(self.pre_buffer) > self.pre_roll_frames:
                    self.pre_buffer.pop(0)

                # Pause threshold calculation:
                # Short utterances (< 1.5s speech) allow extended pause so consecutive words collect.
                # Normal speech finalizes after standard silence gap.
                last_ref = self.last_speech_time if self.last_speech_time is not None else timestamp_s
                start_ref = self.speech_start_time if self.speech_start_time is not None else timestamp_s
                silence_gap = timestamp_s - last_ref
                phrase_duration = (timestamp_s + frame_dur) - start_ref
                speech_dur = self.speech_frames_count * frame_dur

                if speech_dur < 1.5:
                    target_silence = self.short_phrase_silence_s
                else:
                    target_silence = self.silence_duration_s

                if silence_gap >= target_silence or phrase_duration >= self.max_phrase_duration_s:
                    self._finalize_phrase_locked(timestamp_s)

            else:
                # Idle silence: update rolling pre_buffer
                self.pre_buffer.append(frame_f32)
                if len(self.pre_buffer) > self.pre_roll_frames:
                    self.pre_buffer.pop(0)

                # Periodically reset Silero RNN state during extended idle silence so it does not saturate
                self.idle_silence_frames += 1
                if self.idle_silence_frames >= 40:
                    self.vad.reset_states()
                    self.idle_silence_frames = 0

        # Notify partner segmenter outside our own lock to prevent lock inversion
        if notify_partner_onset and self.partner_segmenter is not None:
            try:
                self.partner_segmenter.on_partner_speech_onset(partner_start_ref)
            except Exception as e:
                logger.debug("Failed notifying partner segmenter: %s", e)

        return (rms_norm, prob, is_speech)

    def on_partner_speech_onset(self, partner_start_t: float):
        """
        Called when the other audio stream starts speaking.
        If this stream was in a pause/silence, finalize the phrase immediately
        because the other party took the conversational turn.
        """
        with self._lock:
            if not self.is_speaking:
                return
            # If this stream already stopped active speech and is currently in a pause:
            if self.silence_frames_after_speech > 0:
                speech_dur = self.speech_frames_count * (VAD_FRAME_SIZE / SAMPLE_RATE)
                if speech_dur >= self.min_speech_duration_s:
                    logger.info(
                        f"[{self.stream_name}] Turn switch detected: partner began speaking at {partner_start_t:.2f}s "
                        f"while in pause ({self.silence_frames_after_speech} silence frames). Finalizing phrase."
                    )
                    self._finalize_phrase_locked(partner_start_t)

    def _finalize_phrase_locked(self, current_time_s: float, keep_overlap: Optional[list[np.ndarray]] = None):
        """Must be called while holding self._lock."""
        if not self.phrase_chunks or self.speech_start_time is None:
            self._reset_state(keep_overlap, current_time_s)
            return

        total_samples = sum(len(c) for c in self.phrase_chunks)
        duration_s = total_samples / SAMPLE_RATE
        speech_duration_s = self.speech_frames_count * (VAD_FRAME_SIZE / SAMPLE_RATE)

        # Only emit if there was sufficient actual speech and audible energy (filter transients/clicks/noise)
        emit_data = None
        if duration_s >= self.min_phrase_duration_s and speech_duration_s >= self.min_speech_duration_s:
            phrase_audio = np.concatenate(self.phrase_chunks)
            rms = float(np.sqrt(np.mean(phrase_audio ** 2) + 1e-9))
            if rms >= self.min_energy_rms:
                start_t = max(0.0, self.speech_start_time)
                end_t = start_t + duration_s
                logger.info(
                    f"[{self.stream_name}] Phrase detected with overlap: {start_t:.2f}s -> {end_t:.2f}s "
                    f"({duration_s:.2f}s, speech={speech_duration_s:.2f}s, rms={rms:.4f})"
                )
                emit_data = (self.stream_name, start_t, end_t, phrase_audio)

        self._reset_state(keep_overlap, current_time_s)

        if emit_data and self.on_phrase_completed:
            try:
                self.on_phrase_completed(*emit_data)
            except Exception as e:
                logger.error(f"[{self.stream_name}] Error in on_phrase_completed callback: {e}", exc_info=True)

    def _finalize_phrase(self, current_time_s: float, keep_overlap: Optional[list[np.ndarray]] = None):
        with self._lock:
            self._finalize_phrase_locked(current_time_s, keep_overlap)

    def flush(self, current_time_s: float):
        """Force complete any pending phrase upon recording stop."""
        with self._lock:
            if self.is_speaking and self.phrase_chunks:
                self._finalize_phrase_locked(current_time_s)
            self._reset_state()

    def _reset_state(self, keep_overlap: Optional[list[np.ndarray]] = None, current_time_s: Optional[float] = None):
        if keep_overlap and current_time_s is not None:
            self.is_speaking = True
            overlap_dur = (len(keep_overlap) * VAD_FRAME_SIZE) / SAMPLE_RATE
            self.speech_start_time = max(0.0, current_time_s - overlap_dur)
            self.last_speech_time = current_time_s
            self.phrase_chunks = list(keep_overlap)
            self.silence_frames_after_speech = 0
            self.speech_frames_count = len(keep_overlap)
        else:
            self.is_speaking = False
            self.speech_start_time = None
            self.last_speech_time = None
            self.phrase_chunks = []
            self.silence_frames_after_speech = 0
            self.speech_frames_count = 0
            self.idle_silence_frames = 0
            self.vad.reset_states()

