import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import soundfile as sf

from .config import RECORDINGS_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)
_SESSION_ID_RE = re.compile(r"^sess_[A-Za-z0-9_-]+$")


def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


class StorageManager:
    """Thread-safe local session storage with validated paths and atomic metadata writes."""

    def __init__(self, base_dir: Path = RECORDINGS_DIR):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._session_locks: Dict[str, threading.RLock] = {}

    def _validate_session_id(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("Invalid session ID")
        return session_id

    def _get_session_dir(self, session_id: str) -> Path:
        session_id = self._validate_session_id(session_id)
        candidate = (self.base_dir / session_id).resolve()
        if candidate.parent != self.base_dir:
            raise ValueError("Invalid session ID")
        return candidate

    def _lock_for(self, session_id: str) -> threading.RLock:
        self._validate_session_id(session_id)
        with self._locks_guard:
            return self._session_locks.setdefault(session_id, threading.RLock())

    def create_session(self, session_id: str, mode: str = "mic_only", language: str = "en", title: Optional[str] = None) -> Dict[str, Any]:
        with self._lock_for(session_id):
            sdir = self._get_session_dir(session_id)
            sdir.mkdir(parents=True, exist_ok=False)
            if not title:
                title = f"Recording ({datetime.now().strftime('%b %d, %Y %I:%M %p')})"
            meta: Dict[str, Any] = {
                "id": session_id,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "duration": 0.0,
                "mode": mode,
                "language": language,
                "status": "recording",
                "status_error": None,
                "hq_job_id": None,
                "hq_base_revision": 0,
                "transcript_revision": 0,
                "phrases": [],
                "final_transcript": "",
                "audio_file": "audio.wav",
            }
            self._save_metadata(session_id, meta)
            return meta

    def save_audio(self, session_id: str, audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        with self._lock_for(session_id):
            sdir = self._get_session_dir(session_id)
            sdir.mkdir(parents=True, exist_ok=True)
            wav_path = sdir / "audio.wav"
            sf.write(str(wav_path), np.clip(audio_np, -1.0, 1.0), sample_rate, subtype="PCM_16")
            return str(wav_path)

    def append_phrase(self, session_id: str, phrase: Dict[str, Any]):
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta:
                return
            meta.setdefault("phrases", []).append(phrase)
            if meta.get("status") == "recording":
                texts = [p.get("text", "") for p in meta["phrases"] if p.get("text")]
                meta["final_transcript"] = "\n".join(texts)
            if phrase.get("end_time") is not None:
                meta["duration"] = max(meta.get("duration", 0.0), phrase["end_time"])
            self._save_and_export(session_id, meta)

    def mark_hq_processing(self, session_id: str, job_id: str, duration_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta:
                return None
            meta["status"] = "processing_hq"
            meta["status_error"] = None
            meta["hq_job_id"] = job_id
            meta["hq_base_revision"] = meta.get("transcript_revision", 0)
            if duration_s is not None:
                meta["duration"] = duration_s
            self._save_and_export(session_id, meta)
            return meta

    def complete_hq(self, session_id: str, job_id: str, full_text: str, duration_s: Optional[float] = None) -> bool:
        """Commit only the current job and never overwrite a transcript edited after it started."""
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta or meta.get("hq_job_id") != job_id:
                return False
            if meta.get("transcript_revision", 0) == meta.get("hq_base_revision", 0) and full_text:
                meta["final_transcript"] = full_text
            meta["status"] = "completed"
            meta["status_error"] = None
            meta["hq_job_id"] = None
            if duration_s is not None:
                meta["duration"] = duration_s
            self._save_and_export(session_id, meta)
            return True

    def fail_hq(self, session_id: str, job_id: str, error: str) -> bool:
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta or meta.get("hq_job_id") != job_id:
                return False
            meta["status"] = "hq_error"
            meta["status_error"] = error
            meta["hq_job_id"] = None
            self._save_metadata(session_id, meta)
            return True

    def update_final_transcript(self, session_id: str, full_text: str, duration_s: Optional[float] = None):
        """Compatibility helper for callers that perform an immediate successful finalization."""
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta:
                return
            meta["status"] = "completed"
            if full_text:
                meta["final_transcript"] = full_text
            if duration_s is not None:
                meta["duration"] = duration_s
            self._save_and_export(session_id, meta)

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta:
                return None
            for key, value in updates.items():
                if key in ("title", "final_transcript", "phrases", "language"):
                    meta[key] = value
            if "final_transcript" in updates:
                meta["transcript_revision"] = meta.get("transcript_revision", 0) + 1
            self._save_and_export(session_id, meta)
            return meta

    def _read_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        json_path = self._get_session_dir(session_id) / "session.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Error reading session %s: %s", session_id, exc)
            return None

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock_for(session_id):
            return self._read_metadata(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for sdir in self.base_dir.iterdir():
            if not sdir.is_dir() or not _SESSION_ID_RE.fullmatch(sdir.name):
                continue
            session = self.get_session(sdir.name)
            if session:
                sessions.append(session)
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    def recover_interrupted_sessions(self) -> int:
        """Make sessions left busy by a previous process crash retryable again."""
        recovered = 0
        for session in self.list_sessions():
            if session.get("status") not in ("recording", "processing_hq"):
                continue
            session_id = session["id"]
            with self._lock_for(session_id):
                current = self._read_metadata(session_id)
                if not current or current.get("status") not in ("recording", "processing_hq"):
                    continue
                previous = current["status"]
                current["status"] = "interrupted"
                current["status_error"] = (
                    "Recording was closed abruptly. Preserved audio and partial transcript."
                    if previous == "recording"
                    else "Transcription was interrupted when Sayso closed. Select Re-transcribe to retry."
                )
                current["hq_job_id"] = None
                if not current.get("final_transcript") and current.get("phrases"):
                    current["final_transcript"] = "\n".join(
                        p.get("text", "") for p in current["phrases"] if p.get("text")
                    )
                self._save_and_export(session_id, current)
                recovered += 1
        return recovered

    def delete_session(self, session_id: str) -> bool:
        with self._lock_for(session_id):
            sdir = self._get_session_dir(session_id)
            if not (sdir / "session.json").exists():
                return False
            for _ in range(3):
                try:
                    shutil.rmtree(sdir)
                    return True
                except Exception:
                    time.sleep(0.05)
            return False

    def get_audio_path(self, session_id: str) -> Optional[Path]:
        with self._lock_for(session_id):
            wav_path = self._get_session_dir(session_id) / "audio.wav"
            return wav_path if wav_path.exists() else None

    def load_audio(self, session_id: str) -> Optional[np.ndarray]:
        wav_path = self.get_audio_path(session_id)
        if not wav_path:
            return None
        try:
            data, sr = sf.read(str(wav_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE and len(data) > 0:
                from .recorder import resample_to_16k
                data = resample_to_16k(data, sr)
            return data
        except Exception as exc:
            logger.error("Failed to load audio for session %s: %s", session_id, exc)
            return None

    def _save_metadata(self, session_id: str, data: Dict[str, Any]):
        sdir = self._get_session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="session-", suffix=".tmp", dir=str(sdir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, sdir / "session.json")
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _save_and_export(self, session_id: str, meta: Dict[str, Any]):
        self._save_metadata(session_id, meta)
        self._export_text_files(session_id, meta)

    def _atomic_write_text(self, path: Path, content: str):
        fd, temp_name = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _export_text_files(self, session_id: str, meta: Dict[str, Any]):
        sdir = self._get_session_dir(session_id)
        lines = [
            f"Title: {meta.get('title', '')}",
            f"Date: {meta.get('created_at', '')}",
            f"Duration: {meta.get('duration', 0):.1f}s",
            "",
            "--- Phrases ---",
        ]
        for phrase in meta.get("phrases", []):
            lines.append(f"[{phrase.get('start_time', 0):.1f}s - {phrase.get('end_time', 0):.1f}s] {phrase.get('speaker', 'Speaker')}: {phrase.get('text', '')}")
        if meta.get("final_transcript"):
            lines.extend(["", "--- Full Transcript ---", meta["final_transcript"]])
        self._atomic_write_text(sdir / "transcript.txt", "\n".join(lines) + "\n")

        srt = "".join(
            f"{idx}\n{format_srt_time(p.get('start_time', 0))} --> {format_srt_time(p.get('end_time', 0))}\n{p.get('text', '')}\n\n"
            for idx, p in enumerate(meta.get("phrases", []), 1)
        )
        self._atomic_write_text(sdir / "transcript.srt", srt)
