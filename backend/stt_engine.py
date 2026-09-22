import threading
import queue
import time
import logging
from typing import Optional, Dict, Any, Callable, List
import numpy as np

from .config import (
    COHERE_MODEL_ID,
    DEFAULT_LANGUAGE,
    BATCH_SIZE,
    MAX_AUDIO_CLIP_S,
    OVERLAP_CHUNK_SECOND,
    PUNCTUATION,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class PhraseItem:
    def __init__(
        self,
        session_id: str,
        phrase_id: str,
        speaker: str,
        start_time: float,
        end_time: float,
        audio_np: np.ndarray,
        language: str = DEFAULT_LANGUAGE,
    ):
        self.session_id = session_id
        self.phrase_id = phrase_id
        self.speaker = speaker
        self.start_time = start_time
        self.end_time = end_time
        self.audio_np = audio_np
        self.language = language


class STTEngine:
    """
    Manages nano-cohere-transcribe model lifecycle:
    - Pre-meeting model verification & download with progress reporting.
    - Model loading into VRAM/RAM.
    - Live phrase queue processing for immediate phrase transcription once loaded.
    - Post-meeting high-quality batched transcription pass.
    """

    REQUIRED_FILES = ("config.json", "tokenizer.model", "model.safetensors")

    def __init__(
        self,
        on_status_change: Optional[Callable[[str, Optional[str]], None]] = None,
        on_phrase_transcribed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_hq_pass_completed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_download_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_hq_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_load: bool = False,
    ):
        self.on_status_change = on_status_change
        self.on_phrase_transcribed = on_phrase_transcribed
        self.on_hq_pass_completed = on_hq_pass_completed
        self.on_download_progress = on_download_progress
        self.on_hq_progress = on_hq_progress

        self.model = None
        self.download_progress: Dict[str, Any] = {
            "percent": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "filename": "",
            "speed_mb_s": 0.0,
            "eta_s": 0,
        }

        self.status = "not_downloaded"
        self.error_message: Optional[str] = None

        self._queue: queue.Queue[Optional[PhraseItem]] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._loader_thread: Optional[threading.Thread] = None
        self._download_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._hq_jobs_lock = threading.Lock()
        self._active_hq_jobs: Dict[str, str] = {}

        # Start phrase processing worker
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._process_queue_worker, daemon=True)
        self._worker_thread.start()

        # If already downloaded, set not_loaded (or trigger auto_load if explicitly requested)
        if self.is_model_downloaded():
            if auto_load:
                self.ensure_model_loading()
            else:
                self.status = "not_loaded"
        else:
            self.status = "not_downloaded"

    def get_cached_model_path(self) -> Optional[str]:
        """Return path to cached snapshot directory if all required files exist locally."""
        from pathlib import Path
        import os

        # 1. Check current configured HF cache
        try:
            from huggingface_hub import try_to_load_from_cache
            config_path = try_to_load_from_cache(COHERE_MODEL_ID, "config.json")
            if isinstance(config_path, str):
                snapshot_dir = Path(config_path).parent
                if all((snapshot_dir / f).exists() for f in self.REQUIRED_FILES):
                    return str(snapshot_dir)
        except Exception:
            pass

        # 2. Check default user HF cache (e.g. ~/.cache/huggingface/hub)
        try:
            from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
            from huggingface_hub import try_to_load_from_cache
            config_path = try_to_load_from_cache(COHERE_MODEL_ID, "config.json", cache_dir=HUGGINGFACE_HUB_CACHE)
            if isinstance(config_path, str):
                snapshot_dir = Path(config_path).parent
                if all((snapshot_dir / f).exists() for f in self.REQUIRED_FILES):
                    return str(snapshot_dir)
        except Exception:
            pass

        # 3. Direct inspection of standard user cache snapshot directory
        try:
            repo_folder = f"models--{COHERE_MODEL_ID.replace('/', '--')}"
            candidate_dirs = [
                Path.home() / ".cache" / "huggingface" / "hub" / repo_folder / "snapshots",
                Path(os.environ.get("USERPROFILE", "")) / ".cache" / "huggingface" / "hub" / repo_folder / "snapshots",
            ]
            for cand in candidate_dirs:
                if cand.is_dir():
                    for snap in cand.iterdir():
                        if snap.is_dir() and all((snap / f).exists() for f in self.REQUIRED_FILES):
                            return str(snap)
        except Exception:
            pass

        return None

    def is_model_downloaded(self) -> bool:
        """Check if all required Cohere model files are present in the Hugging Face cache (lock-free)."""
        return self.get_cached_model_path() is not None

    def set_status(self, new_status: str, error_msg: Optional[str] = None):
        self.status = new_status
        self.error_message = error_msg
        logger.info(f"STTEngine status: {new_status} (err: {error_msg})")
        if self.on_status_change:
            self.on_status_change(new_status, error_msg)

    def start_model_download(self) -> bool:
        """Start exactly one download/load transition; allow retry after an error."""
        with self._lifecycle_lock:
            if self.model is not None or self.status in ("downloading", "loading", "ready", "transcribing"):
                return False
            self.download_progress = {
                "percent": 0.0,
                "downloaded_mb": 0.0,
                "total_mb": 3940.4,
                "filename": "model.safetensors",
                "speed_mb_s": 0.0,
                "eta_s": 0,
            }
            self.set_status("downloading")
            if self.on_download_progress:
                self.on_download_progress(self.download_progress)
            self._download_thread = threading.Thread(target=self._download_worker, daemon=True)
            self._download_thread.start()
            return True

    def _download_worker(self):
        try:
            import os
            import io
            # Disable xet to use reliable direct streaming with transparent chunk progress
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            from huggingface_hub import hf_hub_download, try_to_load_from_cache
            from tqdm.auto import tqdm

            engine_self = self

            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            for fname in self.REQUIRED_FILES:
                # If file is already cached, skip downloading
                cached_file = try_to_load_from_cache(COHERE_MODEL_ID, fname, token=hf_token)
                if isinstance(cached_file, str):
                    logger.info(f"Model file '{fname}' already cached at {cached_file}")
                    continue

                logger.info(f"Downloading model file: {fname}...")
                engine_self.download_progress["filename"] = fname

                class ProgressTracker(tqdm):
                    def __init__(self, *args, **kwargs):
                        # Suppress terminal prints but keep disable=False so update() increments self.n
                        kwargs["file"] = io.StringIO()
                        kwargs["disable"] = False
                        super().__init__(*args, **kwargs)
                        self._last_emit = 0.0
                        self._start_time = time.perf_counter()
                        self._initial_n = self.n

                    def update(self, n=1):
                        super().update(n)
                        now = time.perf_counter()
                        if (now - self._last_emit >= 0.15) or (self.total and self.n >= self.total):
                            self._last_emit = now
                            elapsed = max(0.001, now - self._start_time)
                            downloaded_this_session = max(0, self.n - self._initial_n)
                            curr_mb = self.n / (1024 * 1024)
                            tot_mb = (self.total / (1024 * 1024)) if self.total else 0.0
                            speed_mb = (downloaded_this_session / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
                            rem_bytes = max(0, (self.total or 0) - self.n)
                            eta = int(rem_bytes / (speed_mb * 1024 * 1024)) if speed_mb > 0 else 0
                            pct = round((self.n / self.total * 100), 1) if self.total else 0.0

                            p_data = {
                                "filename": fname,
                                "percent": pct,
                                "downloaded_mb": round(curr_mb, 1),
                                "total_mb": round(tot_mb, 1),
                                "speed_mb_s": round(speed_mb, 1),
                                "eta_s": eta,
                            }
                            engine_self.download_progress = p_data
                            if engine_self.on_download_progress:
                                engine_self.on_download_progress(p_data)

                hf_hub_download(
                    repo_id=COHERE_MODEL_ID,
                    filename=fname,
                    tqdm_class=ProgressTracker,
                    token=hf_token,
                )

            logger.info("Model download finished! Now loading into memory...")
            self.set_status("loading")
            self._load_model_worker()

        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            self.set_status("error", str(e))

    def ensure_model_loading(self):
        """Start exactly one background model loader and keep the phrase worker alive."""
        with self._lifecycle_lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._worker_thread = threading.Thread(target=self._process_queue_worker, daemon=True)
                self._worker_thread.start()

            if not self.is_model_downloaded():
                self.set_status("not_downloaded")
                return

            if self.model is not None:
                if self.status != "transcribing":
                    self.set_status("ready")
                return

            if self._loader_thread is not None and self._loader_thread.is_alive():
                return
            if self._download_thread is not None and self._download_thread.is_alive():
                return

            self.set_status("loading")
            self._loader_thread = threading.Thread(target=self._load_model_worker, daemon=True)
            self._loader_thread.start()

    def _load_model_worker(self):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = self.get_cached_model_path() or COHERE_MODEL_ID
            logger.info(f"Loading Cohere model from '{model_path}' on device: {device}...")
            from nano_cohere_transcribe import from_pretrained
            self.model = from_pretrained(model_path, device=device)
            logger.info("Nano-cohere model loaded successfully!")
            self.set_status("ready")
        except Exception as e:
            logger.error(f"Failed to load nano-cohere model: {e}")
            self.error_message = str(e)
            self.set_status("error", str(e))

    def unload_model(self):
        """Unload Cohere model from memory and release GPU VRAM."""
        with self._lifecycle_lock:
            if self.model is not None:
                del self.model
                self.model = None
            try:
                import nano_cohere_transcribe.api as cohere_api
                cohere_api._MODEL_CACHE.clear()
            except Exception:
                pass
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass
            logger.info("Cohere model unloaded from memory and VRAM freed.")
            self.set_status("not_loaded" if self.is_model_downloaded() else "not_downloaded")


    def queue_phrase(
        self,
        session_id: str,
        phrase_id: str,
        speaker: str,
        start_time: float,
        end_time: float,
        audio_np: np.ndarray,
        language: str = DEFAULT_LANGUAGE,
    ):
        """Enqueue phrase for live transcription."""
        self.ensure_model_loading()
        item = PhraseItem(
            session_id=session_id,
            phrase_id=phrase_id,
            speaker=speaker,
            start_time=start_time,
            end_time=end_time,
            audio_np=audio_np,
            language=language,
        )
        self._queue.put(item)

    def _process_queue_worker(self):
        logger.info("STT phrase processing worker started.")
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            # Wait until model is ready or errored
            while self.status == "loading" and not self._stop_event.is_set():
                time.sleep(0.2)

            self._transcribe_phrase_item(item)
            self._queue.task_done()

    def _transcribe_phrase_item(self, item: PhraseItem):
        text = ""
        transcribe_error: Optional[str] = None
        transcribe_start = time.perf_counter()

        if self.model is not None:
            try:
                import torch
                prev_status = self.status
                self.set_status("transcribing")
                # Convert numpy float32 to tensor
                tensor = torch.from_numpy(item.audio_np.astype(np.float32)).float()
                # Run greedy transcription with production settings
                with self._inference_lock:
                    text = self.model.transcribe(
                        tensor,
                        language=item.language,
                        punctuation=PUNCTUATION,
                        max_new_tokens=256,
                        batch_size=BATCH_SIZE,
                        long_form_threshold_s=MAX_AUDIO_CLIP_S,
                    )
                self.set_status(prev_status if prev_status != "transcribing" else "ready")
            except Exception as e:
                logger.error(f"Inference error on phrase {item.phrase_id}: {e}")
                transcribe_error = str(e)
                self.set_status("ready")
        else:
            # Never persist fabricated transcript text when the model disappears.
            # This is reachable after a load/inference failure and must remain
            # visibly recoverable instead of looking like a successful transcript.
            transcribe_error = self.error_message or "Transcription model is not loaded"
            logger.error("Phrase %s could not be transcribed: %s", item.phrase_id, transcribe_error)

        latency = time.perf_counter() - transcribe_start
        phrase_dur = item.end_time - item.start_time
        if text:
            cleaned = text.strip().lower().rstrip(".!?,")
            if cleaned in ("thank you", "thanks", "thank you very much", "thank you for watching", "thanks for watching", "bye", "bye bye") and phrase_dur < 2.5:
                logger.info("Suppressing hallucinated '%s' on %s (dur=%.2fs)", text.strip(), item.phrase_id, phrase_dur)
                text = ""

        logger.info(f"Phrase {item.phrase_id} transcribed in {latency:.2f}s: '{text}'")

        if self.on_phrase_transcribed:
            result = {
                "session_id": item.session_id,
                "phrase_id": item.phrase_id,
                "speaker": "Me" if item.speaker.lower() in ("you", "me") else "Them",
                "start_time": round(item.start_time, 2),
                "end_time": round(item.end_time, 2),
                "duration": round(item.end_time - item.start_time, 2),
                "text": text.strip(),
                "latency_s": round(latency, 2),
            }
            if transcribe_error:
                result["error"] = transcribe_error
            self.on_phrase_transcribed(result)


    def reserve_hq_job(self, session_id: str) -> Optional[str]:
        """Reserve one HQ job per session so repeated actions cannot race the model."""
        with self._hq_jobs_lock:
            if session_id in self._active_hq_jobs:
                return None
            job_id = f"hq_{time.time_ns()}"
            self._active_hq_jobs[session_id] = job_id
            return job_id

    def has_active_hq_job(self, session_id: str) -> bool:
        with self._hq_jobs_lock:
            return session_id in self._active_hq_jobs

    def release_hq_job(self, session_id: str, job_id: str) -> None:
        """Release a reservation when persistence or task dispatch fails before inference starts."""
        with self._hq_jobs_lock:
            if self._active_hq_jobs.get(session_id) == job_id:
                self._active_hq_jobs.pop(session_id, None)

    def run_high_quality_pass(
        self,
        session_id: str,
        full_audio_np: np.ndarray,
        language: str = DEFAULT_LANGUAGE,
        job_id: Optional[str] = None,
        existing_phrases: Optional[List[Dict[str, Any]]] = None,
        audio_mic_np: Optional[np.ndarray] = None,
        audio_sys_np: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run one serialized HQ pass with progress updates and report explicit success or failure."""
        if job_id is None:
            job_id = self.reserve_hq_job(session_id)
            if job_id is None:
                return {"session_id": session_id, "status": "error", "error": "Transcription is already in progress"}

        logger.info("Starting High-Quality 2nd-pass transcription for session %s", session_id)
        t_start = time.perf_counter()
        hq_phrases: List[Dict[str, Any]] = []
        try:
            import torch
            if self.model is None:
                if self.is_model_downloaded():
                    self.ensure_model_loading()
                    deadline = time.time() + 60.0
                    while (self.model is None or self.status == "loading") and time.time() < deadline:
                        time.sleep(0.1)
                if self.model is None:
                    raise RuntimeError("Transcription model could not be loaded")

            HALLUCINATIONS = {
                "thank you", "thanks", "thank you very much",
                "thank you for watching", "thanks for watching", "bye", "bye bye"
            }

            if existing_phrases and len(existing_phrases) > 0:
                # 1. Normalize speaker labels and sort chronologically by start_time
                normalized_phrases = []
                for p in existing_phrases:
                    spk = p.get("speaker", "Me")
                    if spk.lower() in ("you", "me"):
                        spk = "Me"
                    elif spk.lower() in ("them", "system", "remote"):
                        spk = "Them"
                    st = float(p.get("start_time", 0.0))
                    et = float(p.get("end_time", st))
                    if et > st:
                        normalized_phrases.append({
                            "speaker": spk,
                            "start_time": st,
                            "end_time": et,
                            "text": p.get("text", ""),
                        })
                normalized_phrases.sort(key=lambda p: p["start_time"])

                # 2. Resolve cross-speaker conversational overlaps:
                # If speaker A spans [stA, etA] and speaker B speaks at [stB, etB] inside A's interval,
                # split speaker A around speaker B so speaker A's before and after turns become distinct chunks!
                resolved_segments: List[Dict[str, Any]] = []
                for p in normalized_phrases:
                    if not resolved_segments:
                        resolved_segments.append(dict(p))
                        continue

                    prev = resolved_segments[-1]
                    if prev["speaker"] != p["speaker"] and prev["end_time"] > p["start_time"]:
                        orig_prev_end = prev["end_time"]
                        prev["end_time"] = p["start_time"]
                        resolved_segments.append(dict(p))
                        if orig_prev_end > p["end_time"] + 0.3:
                            resolved_segments.append({
                                "speaker": prev["speaker"],
                                "start_time": p["end_time"],
                                "end_time": orig_prev_end,
                                "text": "",
                            })
                    else:
                        resolved_segments.append(dict(p))

                resolved_segments = [s for s in resolved_segments if (s["end_time"] - s["start_time"]) >= 0.2]
                resolved_segments.sort(key=lambda s: s["start_time"])

                # 3. Merge consecutive short phrases of the SAME speaker into larger chunks (up to 25s)
                merged_segments = []
                curr = None
                for seg in resolved_segments:
                    spk = seg["speaker"]
                    st = seg["start_time"]
                    et = seg["end_time"]
                    if curr is None:
                        curr = {"speaker": spk, "start_time": st, "end_time": et}
                    elif curr["speaker"] == spk and (st - curr["end_time"] <= 2.5) and (et - curr["start_time"] <= MAX_AUDIO_CLIP_S):
                        curr["end_time"] = et
                    else:
                        merged_segments.append(curr)
                        curr = {"speaker": spk, "start_time": st, "end_time": et}
                if curr:
                    merged_segments.append(curr)

                total_chunks = max(1, len(merged_segments))
                for idx, seg in enumerate(merged_segments):
                    pct = int((idx / total_chunks) * 100)
                    if self.on_hq_progress:
                        self.on_hq_progress({
                            "session_id": session_id,
                            "job_id": job_id,
                            "percent": pct,
                            "chunk": idx + 1,
                            "total_chunks": total_chunks,
                        })

                    st = seg["start_time"]
                    et = seg["end_time"]
                    start_samp = max(0, int(round(st * SAMPLE_RATE)))
                    dur = et - st

                    # Pick clean channel audio if available, else fall back to mixed full_audio_np
                    if seg["speaker"] == "Them" and audio_sys_np is not None and len(audio_sys_np) > 0:
                        source_stream = audio_sys_np
                    elif seg["speaker"] == "Me" and audio_mic_np is not None and len(audio_mic_np) > 0:
                        source_stream = audio_mic_np
                    else:
                        source_stream = full_audio_np

                    end_samp = min(len(source_stream), int(round(et * SAMPLE_RATE)))
                    seg_audio = source_stream[start_samp:end_samp]
                    if len(seg_audio) == 0:
                        continue

                    rms = float(np.sqrt(np.mean(seg_audio ** 2) + 1e-9))
                    if rms < 0.005 and dur < 2.5:
                        continue

                    piece_tensor = torch.from_numpy(seg_audio.astype(np.float32)).float()
                    with self._inference_lock:
                        try:
                            piece_text = self.model.transcribe(
                                piece_tensor,
                                language=language,
                                punctuation=True,
                                batch_size=BATCH_SIZE,
                                max_new_tokens=512,
                                long_form_threshold_s=MAX_AUDIO_CLIP_S,
                            )
                        except TypeError:
                            piece_text = self.model.transcribe(piece_tensor)

                    text_str = piece_text.strip() if isinstance(piece_text, str) else ""
                    cleaned = text_str.lower().rstrip(".!?,")
                    if cleaned in HALLUCINATIONS and dur < 2.5:
                        continue
                    if not text_str:
                        continue

                    hq_phrases.append({
                        "session_id": session_id,
                        "phrase_id": f"p_hq_{idx}_{int(st * 100)}",
                        "speaker": seg["speaker"],
                        "start_time": round(st, 2),
                        "end_time": round(et, 2),
                        "duration": round(dur, 2),
                        "text": text_str,
                    })

                    pct_done = int(((idx + 1) / total_chunks) * 100)
                    if self.on_hq_progress:
                        self.on_hq_progress({
                            "session_id": session_id,
                            "job_id": job_id,
                            "percent": pct_done,
                            "chunk": idx + 1,
                            "total_chunks": total_chunks,
                        })

                if hq_phrases:
                    full_text = "\n\n".join(f"{p['speaker']}: {p['text']}" for p in hq_phrases)
                else:
                    full_text = ""
            else:
                # Fallback to continuous audio chunking when phrase metadata is not available
                pieces = [full_audio_np]
                try:
                    from nano_cohere_transcribe.chunk import split_audio_chunks_energy
                    pieces = split_audio_chunks_energy(
                        waveform=full_audio_np,
                        sample_rate=SAMPLE_RATE,
                        max_audio_clip_s=MAX_AUDIO_CLIP_S,
                        overlap_chunk_second=OVERLAP_CHUNK_SECOND,
                        min_energy_window_samples=1600,
                    )
                except Exception as e:
                    logger.warning("Could not split audio into energy chunks: %s", e)
                    pieces = [full_audio_np]

                total_chunks = max(1, len(pieces))
                chunk_texts: List[str] = []
                current_start_s = 0.0

                for idx, piece in enumerate(pieces):
                    pct = int((idx / total_chunks) * 100)
                    if self.on_hq_progress:
                        self.on_hq_progress({
                            "session_id": session_id,
                            "job_id": job_id,
                            "percent": pct,
                            "chunk": idx + 1,
                            "total_chunks": total_chunks,
                        })

                    piece_tensor = torch.from_numpy(piece.astype(np.float32)).float()
                    with self._inference_lock:
                        try:
                            piece_text = self.model.transcribe(
                                piece_tensor,
                                language=language,
                                punctuation=True,
                                batch_size=BATCH_SIZE,
                                max_new_tokens=512,
                                long_form_threshold_s=MAX_AUDIO_CLIP_S,
                            )
                        except TypeError:
                            # Compatibility fallback for FakeModel in unit tests
                            piece_text = self.model.transcribe(piece_tensor)
                    
                    text_clean = piece_text.strip() if isinstance(piece_text, str) else ""
                    chunk_texts.append(text_clean)

                    piece_dur = len(piece) / SAMPLE_RATE
                    st = current_start_s
                    et = current_start_s + piece_dur
                    current_start_s = et

                    if text_clean:
                        hq_phrases.append({
                            "session_id": session_id,
                            "phrase_id": f"p_chunk_{idx}_{int(st * 100)}",
                            "speaker": "Me",
                            "start_time": round(st, 2),
                            "end_time": round(et, 2),
                            "duration": round(piece_dur, 2),
                            "text": text_clean,
                        })

                    pct_done = int(((idx + 1) / total_chunks) * 100)
                    if self.on_hq_progress:
                        self.on_hq_progress({
                            "session_id": session_id,
                            "job_id": job_id,
                            "percent": pct_done,
                            "chunk": idx + 1,
                            "total_chunks": total_chunks,
                        })

                full_text = "\n\n".join([t for t in chunk_texts if t])

            duration_s = len(full_audio_np) / SAMPLE_RATE
            elapsed_s = time.perf_counter() - t_start
            result = {
                "session_id": session_id,
                "job_id": job_id,
                "status": "completed",
                "high_quality_text": full_text.strip(),
                "phrases": hq_phrases if hq_phrases else None,
                "audio_duration_s": round(duration_s, 2),
                "inference_time_s": round(elapsed_s, 2),
                "realtime_factor": round(elapsed_s / duration_s, 3) if duration_s > 0 else 0.0,
            }
        except Exception as exc:
            logger.error("High quality pass error for session %s: %s", session_id, exc)
            result = {"session_id": session_id, "job_id": job_id, "status": "error", "error": str(exc)}
        finally:
            with self._hq_jobs_lock:
                if self._active_hq_jobs.get(session_id) == job_id:
                    self._active_hq_jobs.pop(session_id, None)
            # Unload model after post-recording HQ pass to free VRAM
            self.unload_model()

        if self.on_hq_pass_completed:
            self.on_hq_pass_completed(result)
        return result


    def shutdown(self):
        self._stop_event.set()
        self._queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
