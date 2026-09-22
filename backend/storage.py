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


def normalize_speaker(spk: Optional[str]) -> str:
    s = (spk or "").strip().lower()
    return "Them" if s in ("them", "system", "remote", "system audio", "system / remote") else "Me"


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

    def save_session_audios(
        self,
        session_id: str,
        audio_mixed: Optional[np.ndarray] = None,
        audio_mic: Optional[np.ndarray] = None,
        audio_system: Optional[np.ndarray] = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, str]:
        with self._lock_for(session_id):
            sdir = self._get_session_dir(session_id)
            sdir.mkdir(parents=True, exist_ok=True)
            saved_paths: Dict[str, str] = {}
            if audio_mixed is not None and len(audio_mixed) > 0:
                p = sdir / "audio.wav"
                sf.write(str(p), np.clip(audio_mixed, -1.0, 1.0), sample_rate, subtype="PCM_16")
                saved_paths["audio"] = str(p)
            if audio_mic is not None and len(audio_mic) > 0:
                p = sdir / "mic.wav"
                sf.write(str(p), np.clip(audio_mic, -1.0, 1.0), sample_rate, subtype="PCM_16")
                saved_paths["mic"] = str(p)
            if audio_system is not None and len(audio_system) > 0:
                p = sdir / "system.wav"
                sf.write(str(p), np.clip(audio_system, -1.0, 1.0), sample_rate, subtype="PCM_16")
                saved_paths["system"] = str(p)

            meta = self._read_metadata(session_id)
            if meta:
                if "audio" in saved_paths:
                    meta["audio_file"] = "audio.wav"
                if "mic" in saved_paths:
                    meta["audio_mic_file"] = "mic.wav"
                if "system" in saved_paths:
                    meta["audio_system_file"] = "system.wav"
                self._save_metadata(session_id, meta)
            return saved_paths

    def save_audio(self, session_id: str, audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        res = self.save_session_audios(session_id, audio_mixed=audio_np, sample_rate=sample_rate)
        return res.get("audio", "")

    def append_phrase(self, session_id: str, phrase: Dict[str, Any]):
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta:
                return
            text = (phrase.get("text") or "").strip()
            pid = phrase.get("phrase_id")
            phrases = meta.setdefault("phrases", [])
            if pid:
                phrases = [p for p in phrases if p.get("phrase_id") != pid]

            # Only append if phrase has valid non-empty transcribed text and no fatal error
            if text and not phrase.get("error"):
                if phrase.get("speaker"):
                    phrase["speaker"] = normalize_speaker(phrase["speaker"])
                phrase["text"] = text
                phrases.append(phrase)
                phrases.sort(key=lambda p: float(p.get("start_time", 0.0)))

            meta["phrases"] = phrases
            if meta.get("status") == "recording":
                lines = [f"{normalize_speaker(p.get('speaker', 'Me'))}: {p.get('text', '')}" for p in meta["phrases"] if p.get("text")]
                meta["final_transcript"] = "\n\n".join(lines)
            if phrase.get("end_time") is not None and text:
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

    def complete_hq(self, session_id: str, job_id: str, full_text: str, duration_s: Optional[float] = None, phrases: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Commit only the current job and never overwrite a transcript edited after it started."""
        with self._lock_for(session_id):
            meta = self._read_metadata(session_id)
            if not meta or meta.get("hq_job_id") != job_id:
                return False
            if meta.get("transcript_revision", 0) == meta.get("hq_base_revision", 0):
                if full_text:
                    meta["final_transcript"] = full_text
                if phrases is not None and len(phrases) > 0:
                    for p in phrases:
                        if p.get("speaker"):
                            p["speaker"] = normalize_speaker(p["speaker"])
                    meta["phrases"] = phrases
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
            if "phrases" in updates and "final_transcript" not in updates:
                lines = [
                    f"{normalize_speaker(p.get('speaker', 'Me'))}: {p.get('text', '')}"
                    for p in meta.get("phrases", [])
                    if p.get("text")
                ]
                meta["final_transcript"] = "\n\n".join(lines)
            if "final_transcript" in updates or "phrases" in updates:
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
                    current["final_transcript"] = "\n\n".join(
                        f"{normalize_speaker(p.get('speaker', 'Me'))}: {p.get('text', '')}"
                        for p in current["phrases"]
                        if p.get("text")
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

    def get_session_dir(self, session_id: str) -> Path:
        return self._get_session_dir(session_id)

    def get_audio_path(self, session_id: str, channel: str = "mixed") -> Optional[Path]:
        with self._lock_for(session_id):
            sdir = self._get_session_dir(session_id)
            if channel == "mic":
                p = sdir / "mic.wav"
                if p.exists():
                    return p
            elif channel == "system":
                p = sdir / "system.wav"
                if p.exists():
                    return p
            wav_path = sdir / "audio.wav"
            return wav_path if wav_path.exists() else None

    def load_audio(self, session_id: str, channel: str = "mixed") -> Optional[np.ndarray]:
        wav_path = self.get_audio_path(session_id, channel=channel)
        if not wav_path and channel != "mixed":
            wav_path = self.get_audio_path(session_id, channel="mixed")
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
            logger.error("Failed to load audio (%s) for session %s: %s", channel, session_id, exc)
            return None

    def _save_metadata(self, session_id: str, data: Dict[str, Any]):
        sdir = self._get_session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        self._atomic_write_text(sdir / "session.json", content)


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
            spk = normalize_speaker(phrase.get("speaker", "Me"))
            lines.append(f"[{phrase.get('start_time', 0):.1f}s - {phrase.get('end_time', 0):.1f}s] {spk}: {phrase.get('text', '')}")
        if meta.get("final_transcript"):
            lines.extend(["", "--- Full Transcript ---", meta["final_transcript"]])
        self._atomic_write_text(sdir / "transcript.txt", "\n".join(lines) + "\n")

        srt = "".join(
            f"{idx}\n{format_srt_time(p.get('start_time', 0))} --> {format_srt_time(p.get('end_time', 0))}\n{normalize_speaker(p.get('speaker', 'Me'))}: {p.get('text', '')}\n\n"
            for idx, p in enumerate(meta.get("phrases", []), 1)
        )
        self._atomic_write_text(sdir / "transcript.srt", srt)
