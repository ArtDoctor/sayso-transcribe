import time
import uuid

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app, storage, recorder, stt_engine


def test_full_recording_e2e_pipeline(monkeypatch):
    """Exercise the full lifecycle without audio hardware or a real ML model."""
    import backend.recorder as recorder_module

    class FakeModel:
        def transcribe(self, *_args, **kwargs):
            return "Live phrase." if kwargs.get("max_new_tokens") == 256 else "High quality final transcript."

    monkeypatch.setattr(recorder_module, "HAS_PYAUDIO", False)
    monkeypatch.setattr(recorder_module, "pyaudio", None)
    original_status, original_model = stt_engine.status, stt_engine.model
    client = TestClient(app)
    session_id = f"sess_e2e_{uuid.uuid4().hex[:8]}"

    try:
        # Model gate and status contract.
        stt_engine.status, stt_engine.model = "not_downloaded", None
        blocked = client.post("/api/record/start", json={"mode": "mic_only", "language": "en"})
        assert blocked.status_code == 409
        assert "Model is not ready" in blocked.json()["detail"]

        stt_engine.status, stt_engine.model = "ready", FakeModel()
        started = client.post("/api/record/start", json={
            "mode": "mic_only",
            "language": "en",
            "title": "E2E Meeting Verification",
        })
        assert started.status_code == 200
        # The API owns IDs, so use the returned ID and clean the unused fixture name.
        session_id = started.json()["session_id"]
        status = client.get("/api/status").json()
        assert status["is_recording"] is True
        assert status["active_session_id"] == session_id
        assert status["active_recording_mode"] == "mic_only"
        assert status["recording_duration"] >= 0

        duplicate = client.post("/api/record/start", json={"mode": "mic_only"})
        assert duplicate.status_code == 409

        # Live transcription.
        speech = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, endpoint=False))).astype(np.float32)
        recorder._handle_phrase_completed("You", 0.5, 1.5, speech)
        deadline = time.time() + 2
        while time.time() < deadline:
            session = storage.get_session(session_id)
            if session and session.get("phrases"):
                break
            time.sleep(0.01)
        assert session["phrases"][0]["text"] == "Live phrase."

        # Stop, persist and launch the HQ pass.
        recorder._mic_audio_chunks.append(speech)
        stopped = client.post("/api/record/stop")
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "processing_hq"
        assert client.post("/api/record/stop").status_code == 409

        deadline = time.time() + 2
        while time.time() < deadline:
            session = storage.get_session(session_id)
            if session and session["status"] == "completed":
                break
            time.sleep(0.01)
        assert session["final_transcript"] == "High quality final transcript."

        # Playback/export files and editing.
        audio_path = storage.get_audio_path(session_id)
        assert audio_path and audio_path.stat().st_size > 1000
        data, sample_rate = sf.read(str(audio_path))
        assert sample_rate == 16000 and len(data) > 0
        assert (audio_path.parent / "transcript.txt").exists()
        assert (audio_path.parent / "transcript.srt").exists()
        assert client.get(f"/api/recordings/{session_id}/audio").status_code == 200

        edited = client.patch(f"/api/recordings/{session_id}", json={
            "title": "Updated title",
            "final_transcript": "Corrected by the user.",
        })
        assert edited.status_code == 200
        assert edited.json()["final_transcript"] == "Corrected by the user."

        deleted = client.delete(f"/api/recordings/{session_id}")
        assert deleted.status_code == 200
        assert client.get(f"/api/recordings/{session_id}").status_code == 404
    finally:
        if recorder.is_recording:
            recorder.stop()
        deadline = time.time() + 2
        while stt_engine.has_active_hq_job(session_id) and time.time() < deadline:
            time.sleep(0.01)
        storage.delete_session(session_id)
        stt_engine.status, stt_engine.model = original_status, original_model
