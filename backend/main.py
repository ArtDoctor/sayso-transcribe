import asyncio
import time
import threading
import uuid
import logging
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import soundfile as sf

import sys
import os
import subprocess
from .config import HOST, PORT, DEFAULT_LANGUAGE, LOGS_FILE
from .audio_devices import get_audio_devices
from .storage import StorageManager
from .stt_engine import STTEngine
from .recorder import AudioRecorder
from .ffmpeg_utils import is_ffmpeg_installed, get_ffmpeg_version, convert_audio_to_wav_16k

_device_info = {"device": "cpu", "cuda_device_name": None}
_torch_detection_lock = threading.Lock()
_torch_detected = False

def _detect_torch_device():
    global _torch_detected
    with _torch_detection_lock:
        if _torch_detected:
            return _device_info
        try:
            import torch
            if torch.cuda.is_available():
                _device_info["device"] = "cuda"
                try:
                    _device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                except Exception:
                    _device_info["cuda_device_name"] = None
            else:
                _device_info["device"] = "cpu"
        except Exception as e:
            logger.warning("Error detecting torch device: %s", e)
        finally:
            _torch_detected = True
        return _device_info

def get_torch_device_info():
    return _device_info["device"], _device_info["cuda_device_name"]

# Pre-warm torch and detect CUDA device in background without blocking server startup
threading.Thread(target=_detect_torch_device, daemon=True).start()

def init_logging():
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOGS_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Attach file handler if not already present
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOGS_FILE) for h in root.handlers):
        root.addHandler(file_handler)

    # Attach stream handler for console runs if stdout is available
    if sys.stdout is not None and hasattr(sys.stdout, "write"):
        try:
            if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(formatter)
                stream_handler.setLevel(logging.INFO)
                root.addHandler(stream_handler)
        except Exception:
            pass

    # Ensure uvicorn loggers route to file handler without duplicate propagation
    for u_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        u_logger = logging.getLogger(u_name)
        u_logger.handlers = [file_handler]
        u_logger.propagate = False

    # Guard against None stdout/stderr in pythonw background mode
    if sys.stdout is None:
        sys.stdout = file_handler.stream
    if sys.stderr is None:
        sys.stderr = file_handler.stream

init_logging()
logger = logging.getLogger("sayso.backend")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    ws_manager.set_loop(asyncio.get_running_loop())
    recovered = storage.recover_interrupted_sessions()
    if recovered:
        logger.warning("Recovered %s interrupted session(s).", recovered)
    logger.info("Sayso Backend started.")
    yield
    if recorder.is_recording:
        recorder.stop()
    stt_engine.shutdown()

app = FastAPI(title="Sayso Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers
storage = StorageManager()
stt_engine = STTEngine(auto_load=os.getenv("SAYSO_FORCE_MODEL_AUTOLOAD") == "1")
recorder = AudioRecorder()

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Remaining: {len(self.active_connections)}")

    def broadcast_sync(self, message: Dict[str, Any]):
        """Thread-safe non-blocking broadcast to all connected WebSocket clients."""
        if not self.active_connections or not self._loop:
            return

        async def _send():
            dead_sockets = []
            for ws in list(self.active_connections):
                try:
                    await ws.send_json(message)
                except Exception:
                    dead_sockets.append(ws)
            for ds in dead_sockets:
                if ds in self.active_connections:
                    self.active_connections.remove(ds)

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

ws_manager = ConnectionManager()

# Hook up callbacks
def on_stt_status(status: str, err: Optional[str]):
    ws_manager.broadcast_sync({
        "type": "model_status",
        "status": status,
        "error": err,
    })

def on_phrase_transcribed(phrase_data: Dict[str, Any]):
    # Phrase ownership is immutable; late results must never follow recorder.session_id.
    session_id = phrase_data.get("session_id")
    if session_id:
        storage.append_phrase(session_id, phrase_data)
        try:
            curr_audios = recorder.get_current_audios()
            if len(curr_audios.get("audio", [])) > 0:
                storage.save_session_audios(
                    session_id,
                    audio_mixed=curr_audios.get("audio"),
                    audio_mic=curr_audios.get("mic"),
                    audio_system=curr_audios.get("system"),
                )
        except Exception as e:
            logger.debug("Incremental audio save failed: %s", e)
    # Broadcast to UI
    ws_manager.broadcast_sync({
        "type": "phrase_transcribed",
        "phrase": phrase_data,
    })

def on_hq_pass_completed(res: Dict[str, Any]):
    session_id = res.get("session_id")
    job_id = res.get("job_id")
    if session_id and job_id:
        if res.get("status") == "completed":
            storage.complete_hq(
                session_id,
                job_id,
                res.get("high_quality_text", ""),
                res.get("audio_duration_s"),
                phrases=res.get("phrases"),
            )
        else:
            storage.fail_hq(session_id, job_id, res.get("error", "Transcription failed"))
    ws_manager.broadcast_sync({
        "type": "hq_pass_completed" if res.get("status") == "completed" else "hq_pass_failed",
        "result": res,
    })

def on_hq_progress(progress_data: Dict[str, Any]):
    ws_manager.broadcast_sync({
        "type": "hq_pass_progress",
        "progress": progress_data,
    })

def on_vad_level(data: Dict[str, Any]):
    ws_manager.broadcast_sync(data)

def on_phrase_ready(session_id: str, speaker: str, start_t: float, end_t: float, audio_np):
    # Tell the UI about the completed audio segment before handing it to STT.
    # The same ID is used for the eventual result so the skeleton can be filled
    # in-place rather than creating a second transcript row.
    phrase_id = f"p_{int(start_t * 1000)}_{uuid.uuid4().hex[:4]}"
    pending_phrase = {
        "session_id": session_id,
        "phrase_id": phrase_id,
        "speaker": speaker,
        "start_time": round(start_t, 2),
        "end_time": round(end_t, 2),
        "duration": round(end_t - start_t, 2),
        "text": "",
        "status": "pending",
    }
    ws_manager.broadcast_sync({
        "type": "phrase_pending",
        "phrase": pending_phrase,
    })

    # Enqueue into STT engine for fast live transcription.
    stt_engine.queue_phrase(
        session_id=session_id,
        phrase_id=phrase_id,
        speaker=speaker,
        start_time=start_t,
        end_time=end_t,
        audio_np=audio_np,
        language=current_recording_params.get("language", DEFAULT_LANGUAGE),
    )

def on_download_progress(progress_data: Dict[str, Any]):
    ws_manager.broadcast_sync({
        "type": "model_download_progress",
        "progress": progress_data,
    })


def on_recording_error(error_data: Dict[str, Any]):
    ws_manager.broadcast_sync(error_data)

stt_engine.on_status_change = on_stt_status
stt_engine.on_phrase_transcribed = on_phrase_transcribed
stt_engine.on_hq_pass_completed = on_hq_pass_completed
stt_engine.on_download_progress = on_download_progress
stt_engine.on_hq_progress = on_hq_progress
recorder.on_vad_level = on_vad_level
recorder.on_phrase_ready = on_phrase_ready
recorder.on_error = on_recording_error

current_recording_params: Dict[str, Any] = {
    "language": DEFAULT_LANGUAGE,
    "mode": "mic_only",
}
_recording_transition_lock = threading.Lock()


from .config import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    HOST,
    PORT,
    LOGS_FILE,
)


# REST Schemas
class StartRecordRequest(BaseModel):
    mode: Literal["mic_only", "mic_and_system"] = "mic_only"
    language: str = DEFAULT_LANGUAGE
    title: Optional[str] = None
    mic_device_index: Optional[int] = None
    system_device_index: Optional[int] = None
    mic_device_name: Optional[str] = None
    system_device_name: Optional[str] = None
    preload_model: bool = True

class ChangeLanguageRequest(BaseModel):
    language: str

class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    final_transcript: Optional[str] = None
    phrases: Optional[List[Dict[str, Any]]] = None



def _validate_session_id(session_id: str):
    try:
        storage._validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@app.get("/api/status")
def get_status():
    devices_summary = get_audio_devices()
    device_val, cuda_name = get_torch_device_info()
    return {
        "status": "online",
        "is_recording": recorder.is_recording,
        "active_session_id": recorder.session_id if recorder.is_recording else None,
        "active_recording_mode": recorder.recording_mode if recorder.is_recording else None,
        "recording_duration": max(0.0, time.time() - recorder.start_time) if recorder.is_recording else 0.0,
        "recording_error": recorder.last_error if recorder.is_recording else None,
        "model_status": stt_engine.status,
        "is_model_downloaded": stt_engine.is_model_downloaded(),
        "is_model_loaded": stt_engine.model is not None,
        "download_progress": stt_engine.download_progress if stt_engine.status == "downloading" else None,
        "model_error": stt_engine.error_message,
        "default_mic": devices_summary.get("default_mic"),
        "default_system": devices_summary.get("default_system"),
        "device": device_val,
        "cuda_device_name": cuda_name,
        "ffmpeg_installed": is_ffmpeg_installed(),
    }


@app.get("/api/system/ffmpeg")
def get_ffmpeg_status():
    installed = is_ffmpeg_installed()
    return {
        "installed": installed,
        "version": get_ffmpeg_version() if installed else None,
    }


@app.get("/api/devices")
def get_devices(force_refresh: bool = False):
    """Returns dynamic Windows audio devices (cached with 3s TTL to protect WASAPI)."""
    return get_audio_devices(force_refresh=force_refresh)


@app.post("/api/model/download")
def download_model():
    """Initiate downloading Cohere model weights with real-time progress."""
    if stt_engine.status == "downloading":
        return {"status": "downloading", "detail": "Download already in progress"}
    if stt_engine.status == "ready":
        return {"status": "ready", "detail": "Model is already downloaded and loaded"}
    stt_engine.start_model_download()
    return {"status": stt_engine.status, "is_downloaded": stt_engine.is_model_downloaded()}


@app.post("/api/model/preload")
def preload_model():
    """Explicitly preload model if user clicks Preload / Load Model."""
    if not stt_engine.is_model_downloaded():
        raise HTTPException(
            status_code=400,
            detail="Model is not downloaded. Please download it first before loading."
        )
    stt_engine.ensure_model_loading()
    return {"status": stt_engine.status}


@app.post("/api/record/start")
def start_recording(req: StartRecordRequest):
    # Serialize admission, session creation, and recorder transition as one transaction.
    with _recording_transition_lock:
        if recorder.is_recording:
            raise HTTPException(status_code=409, detail="Recording is already in progress")
        if stt_engine.status == "not_downloaded" or not stt_engine.is_model_downloaded():
            raise HTTPException(status_code=409, detail="Model is not ready. You must download and load the model before starting a meeting.")

        # On-demand loading: ensure model begins loading as soon as recording starts
        if stt_engine.model is None and stt_engine.status != "loading":
            stt_engine.ensure_model_loading()

        session_id = f"sess_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        current_recording_params["language"] = req.language
        current_recording_params["mode"] = req.mode
        meta = storage.create_session(session_id=session_id, mode=req.mode, language=req.language, title=req.title)
        success = recorder.start(
            session_id=session_id,
            mode=req.mode,
            mic_device_index=req.mic_device_index,
            system_device_index=req.system_device_index,
            mic_device_name=req.mic_device_name,
            system_device_name=req.system_device_name,
        )
        if not success:
            storage.delete_session(session_id)
            raise HTTPException(status_code=500, detail="Failed to initialize audio capture stream")

        def _periodic_flush():
            while recorder.is_recording:
                time.sleep(5)
                sid = recorder.session_id
                if sid and recorder.is_recording:
                    try:
                        curr_audios = recorder.get_current_audios()
                        if len(curr_audios.get("audio", [])) > 0:
                            storage.save_session_audios(
                                sid,
                                audio_mixed=curr_audios.get("audio"),
                                audio_mic=curr_audios.get("mic"),
                                audio_system=curr_audios.get("system"),
                            )
                    except Exception:
                        pass

        threading.Thread(target=_periodic_flush, daemon=True).start()

    # Broadcast session started
    ws_manager.broadcast_sync({
        "type": "recording_started",
        "session": meta,
    })

    return {"session_id": session_id, "status": "recording", "session": meta}


@app.post("/api/record/stop")
def stop_recording():
    # Start and stop share one transition lock so duplicate clicks/clients cannot
    # stop the same recorder twice or start while audio is still being finalized.
    with _recording_transition_lock:
        if not recorder.is_recording:
            raise HTTPException(status_code=409, detail="Recording is not active or is already stopping")
        session_id = recorder.session_id
        rec_result = recorder.stop()

    duration = rec_result.get("duration", 0.0)
    audio_np = rec_result.get("audio")
    audio_mic = rec_result.get("audio_mic")
    audio_sys = rec_result.get("audio_system")

    final_status = "processing_hq"
    final_error: Optional[str] = None
    job_id: Optional[str] = None
    try:
        if not session_id:
            raise RuntimeError("Recorder returned no session ID")

        # Mark the durable state before dispatch so the UI can always recover
        # after a refresh, even if the WebSocket event is missed.
        job_id = stt_engine.reserve_hq_job(session_id)
        if not job_id:
            raise RuntimeError("Transcription is already in progress")
        storage.mark_hq_processing(session_id, job_id, duration_s=duration)
        if audio_np is None or len(audio_np) == 0:
            raise RuntimeError("No audio was captured")
        storage.save_session_audios(
            session_id,
            audio_mixed=audio_np,
            audio_mic=audio_mic,
            audio_system=audio_sys,
        )
        sess = storage.get_session(session_id)
        existing_phrases = sess.get("phrases") if sess else None
        args = (
            session_id,
            audio_np,
            current_recording_params.get("language", DEFAULT_LANGUAGE),
            job_id,
            existing_phrases,
            audio_mic,
            audio_sys,
        )
        if ws_manager._loop and ws_manager._loop.is_running():
            ws_manager._loop.run_in_executor(None, stt_engine.run_high_quality_pass, *args)
        else:
            threading.Thread(target=stt_engine.run_high_quality_pass, args=args, daemon=True).start()
    except Exception as exc:
        final_status = "hq_error"
        final_error = str(exc)
        logger.error("Could not finalize recording %s: %s", session_id, exc, exc_info=True)
        if session_id and job_id:
            stt_engine.release_hq_job(session_id, job_id)
            try:
                storage.fail_hq(session_id, job_id, final_error)
            except Exception:
                logger.exception("Could not persist finalization failure for %s", session_id)
        stt_engine.unload_model()
        ws_manager.broadcast_sync({
            "type": "hq_pass_failed",
            "result": {"session_id": session_id, "status": "error", "error": final_error},
        })

    ws_manager.broadcast_sync({
        "type": "recording_stopped",
        "session_id": session_id,
        "duration": duration,
    })

    return {"session_id": session_id, "duration": duration, "status": final_status, "error": final_error}


@app.get("/api/recordings")
def list_recordings():
    return storage.list_sessions()


@app.get("/api/recordings/{session_id}")
def get_recording(session_id: str):
    _validate_session_id(session_id)
    sess = storage.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.get("/api/recordings/{session_id}/audio")
def get_recording_audio(session_id: str, channel: str = "mixed"):
    _validate_session_id(session_id)
    audio_path = storage.get_audio_path(session_id, channel=channel)
    if not audio_path or not audio_path.exists():
        audio_path = storage.get_audio_path(session_id, channel="mixed")
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    filename = f"{session_id}_{channel}.wav" if channel != "mixed" else f"{session_id}.wav"
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=filename,
    )


@app.post("/api/recordings/{session_id}/open_folder")
def open_recording_folder(session_id: str):
    """Open the local folder containing the recording audio and transcripts in system file explorer."""
    _validate_session_id(session_id)
    sdir = storage.get_session_dir(session_id)
    if not sdir.exists():
        raise HTTPException(status_code=404, detail="Recording directory not found")
    try:
        folder_str = str(sdir.resolve())
        if sys.platform == "win32":
            os.startfile(folder_str)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder_str])
        else:
            subprocess.Popen(["xdg-open", folder_str])
        return {"opened": True, "path": folder_str}
    except Exception as exc:
        logger.error("Could not open folder for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Could not open directory: {exc}")


@app.patch("/api/recordings/{session_id}")
def update_recording(session_id: str, req: UpdateSessionRequest):
    _validate_session_id(session_id)
    updates = req.model_dump(exclude_unset=True)
    sess = storage.update_session(session_id, updates)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.delete("/api/recordings/{session_id}")
def delete_recording(session_id: str):
    _validate_session_id(session_id)
    if (recorder.is_recording and recorder.session_id == session_id) or stt_engine.has_active_hq_job(session_id):
        raise HTTPException(status_code=409, detail="Recording or transcription is still active")
    success = storage.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or cannot be deleted")
    return {"deleted": True, "session_id": session_id}


@app.post("/api/recordings/{session_id}/retranscribe")
def retranscribe_recording(session_id: str):
    """Trigger one on-demand high-quality retranscription for a completed recording."""
    _validate_session_id(session_id)
    sess = storage.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    if recorder.is_recording:
        raise HTTPException(status_code=409, detail="Stop the active recording before starting a retranscription")

    audio_np = storage.load_audio(session_id, channel="mixed")
    if audio_np is None or len(audio_np) == 0:
        raise HTTPException(status_code=404, detail="Audio file not found or empty")

    audio_mic = storage.load_audio(session_id, channel="mic")
    audio_sys = storage.load_audio(session_id, channel="system")

    if not stt_engine.is_model_downloaded():
        raise HTTPException(
            status_code=400,
            detail="Model is not downloaded. Please download the Cohere model first."
        )

    if stt_engine.model is None and stt_engine.status != "loading":
        stt_engine.ensure_model_loading()

    lang = sess.get("language", DEFAULT_LANGUAGE)
    job_id = stt_engine.reserve_hq_job(session_id)
    if not job_id:
        raise HTTPException(status_code=409, detail="Transcription is already in progress")
    storage.mark_hq_processing(session_id, job_id)
    existing_phrases = sess.get("phrases")
    args = (session_id, audio_np, lang, job_id, existing_phrases, audio_mic, audio_sys)
    if ws_manager._loop and ws_manager._loop.is_running():
        ws_manager._loop.run_in_executor(None, stt_engine.run_high_quality_pass, *args)
    else:
        threading.Thread(target=stt_engine.run_high_quality_pass, args=args, daemon=True).start()

    return {"session_id": session_id, "status": "processing_hq", "job_id": job_id}


@app.post("/api/upload/transcribe")
async def upload_and_transcribe(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    title: Optional[str] = Form(None),
):
    """
    Upload an audio or video file of any format, convert it via FFmpeg, and run high-quality transcription.
    """
    if not is_ffmpeg_installed():
        raise HTTPException(
            status_code=400,
            detail="ffmpeg not installed, please install."
        )

    if recorder.is_recording:
        raise HTTPException(
            status_code=409,
            detail="Stop the active recording before transcribing a file"
        )

    if not stt_engine.is_model_downloaded():
        raise HTTPException(
            status_code=400,
            detail="Model is not downloaded. Please download the Cohere model first."
        )

    if stt_engine.model is None and stt_engine.status != "loading":
        stt_engine.ensure_model_loading()

    lang = language.strip().lower() if language else DEFAULT_LANGUAGE
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE

    orig_filename = file.filename or "uploaded_audio"
    clean_title = title.strip() if title and title.strip() else Path(orig_filename).stem

    session_id = f"sess_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    meta = storage.create_session(
        session_id=session_id,
        mode="uploaded",
        language=lang,
        title=clean_title,
    )

    sdir = storage._get_session_dir(session_id)
    suffix = Path(orig_filename).suffix or ".bin"
    temp_input = sdir / f"upload_{uuid.uuid4().hex[:6]}{suffix}"
    target_wav = sdir / "audio.wav"

    try:
        with open(temp_input, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):
                f_out.write(chunk)

        try:
            convert_audio_to_wav_16k(temp_input, target_wav)
        except Exception as e:
            storage.delete_session(session_id)
            raise HTTPException(status_code=400, detail=f"Failed to decode audio file with ffmpeg: {e}")

        try:
            audio_np, sr = sf.read(str(target_wav), dtype="float32")
        except Exception as e:
            storage.delete_session(session_id)
            raise HTTPException(status_code=400, detail=f"Could not read converted audio file: {e}")

        duration = len(audio_np) / sr if sr > 0 else 0.0
        if duration <= 0.05 or len(audio_np) == 0:
            storage.delete_session(session_id)
            raise HTTPException(status_code=400, detail="Audio file is empty or contains no readable audio")

        storage.save_session_audios(session_id, audio_mixed=audio_np)

        job_id = stt_engine.reserve_hq_job(session_id)
        if not job_id:
            storage.delete_session(session_id)
            raise HTTPException(status_code=409, detail="Transcription is already in progress")

        storage.mark_hq_processing(session_id, job_id, duration_s=duration)

        ws_manager.broadcast_sync({
            "type": "recording_started",
            "session": meta,
        })

        args = (session_id, audio_np, lang, job_id, None, None, None)
        if ws_manager._loop and ws_manager._loop.is_running():
            ws_manager._loop.run_in_executor(None, stt_engine.run_high_quality_pass, *args)
        else:
            threading.Thread(target=stt_engine.run_high_quality_pass, args=args, daemon=True).start()

        return {
            "session_id": session_id,
            "status": "processing_hq",
            "job_id": job_id,
            "session": meta,
        }
    finally:
        try:
            if temp_input.exists():
                temp_input.unlink()
        except Exception:
            pass


@app.post("/api/recording/language")
@app.patch("/api/recording/active")
def change_recording_language(req: ChangeLanguageRequest):
    """Dynamically switch transcription language even during an active recording session."""
    lang = req.language.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'. Supported: {SUPPORTED_LANGUAGES}")
    current_recording_params["language"] = lang
    if recorder.is_recording and recorder.session_id:
        storage.update_session(recorder.session_id, {"language": lang})
    ws_manager.broadcast_sync({
        "type": "recording_language_changed",
        "language": lang,
        "session_id": recorder.session_id if recorder.is_recording else None,
    })
    logger.info("Transcription language switched to %s (recording=%s)", lang, recorder.is_recording)
    return {"language": lang, "is_recording": recorder.is_recording}


@app.post("/api/model/unload")
def unload_model_endpoint():
    """Explicitly release model from GPU VRAM."""
    stt_engine.unload_model()
    return {
        "status": stt_engine.status,
        "is_model_loaded": stt_engine.model is not None,
        "is_model_downloaded": stt_engine.is_model_downloaded(),
    }



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep receiving or handle ping
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


def run_server():
    init_logging()
    with open(LOGS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n=======================================================\n")
        f.write(f"=== Meetily Backend Started: {time.strftime('%Y-%m-%d %H:%M:%S')} (PID: {os.getpid()}) ===\n")
        f.write(f"=======================================================\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", log_config=None)


if __name__ == "__main__":
    run_server()
