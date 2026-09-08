import os
os.environ.setdefault("SAYSO_DISABLE_MODEL_AUTOLOAD", "1")

import pytest
import numpy as np
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from backend.audio_devices import get_audio_devices
from backend.vad_engine import SileroVADWrapper, PhraseSegmenter
from backend.storage import StorageManager
from backend.main import app


def test_device_enumeration():
    """Verify device enumeration returns required structure and defaults."""
    devices = get_audio_devices()
    assert isinstance(devices, dict)
    assert "default_mic" in devices
    assert "default_system" in devices
    assert "microphones" in devices
    assert "system_devices" in devices
    assert isinstance(devices["microphones"], list)
    assert isinstance(devices["system_devices"], list)


def test_vad_frame_prediction():
    """Verify Silero VAD frame prediction runs without error on 512 float32 samples."""
    vad = SileroVADWrapper()
    silence = np.zeros(512, dtype=np.float32)
    prob_silence = vad.predict_chunk(silence)
    assert 0.0 <= prob_silence <= 1.0

    # Synthetic tone frame
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    prob_tone = vad.predict_chunk(tone)
    assert 0.0 <= prob_tone <= 1.0


def test_phrase_segmenter():
    """Verify PhraseSegmenter processes frames and accumulates state."""
    completed = []

    def on_phrase(name, start, end, audio):
        completed.append((name, start, end, len(audio)))

    seg = PhraseSegmenter(
        stream_name="test_mic",
        speech_threshold=0.3,
        silence_duration_s=0.2,
        min_phrase_duration_s=0.1,
        on_phrase_completed=on_phrase,
    )

    # Feed silence
    silence = np.zeros(512, dtype=np.float32)
    seg.process_frame(silence, 0.0)
    assert not seg.is_speaking

    # Flush
    seg.flush(1.0)
    assert not seg.is_speaking


def test_phrase_segmenter_pre_and_post_roll_overlap():
    """Verify PhraseSegmenter includes pre-roll and post-roll overlap around speech."""
    completed = []

    def on_phrase(name, start, end, audio):
        completed.append({
            "name": name,
            "start": start,
            "end": end,
            "samples": len(audio),
            "audio": audio,
        })

    # Use small custom durations to test deterministic framing
    seg = PhraseSegmenter(
        stream_name="test_mic",
        speech_threshold=0.2,
        silence_duration_s=0.25,
        min_phrase_duration_s=0.1,
        pre_roll_s=0.3,
        post_roll_s=0.2,
        on_phrase_completed=on_phrase,
    )

    # Frame duration is 512 / 16000 = 0.032s
    # Feed 15 silence frames (0.48s) to build up pre-buffer
    silence = np.zeros(512, dtype=np.float32)
    t = 0.0
    for _ in range(15):
        seg.process_frame(silence, t)
        t += 0.032

    # Verify pre-buffer has accumulated
    assert len(seg.pre_buffer) == seg.pre_roll_frames

    # Mock VAD prediction: return 0.0 for silence, 0.9 for speech
    seg.vad.predict_chunk = lambda chunk: 0.9 if np.max(np.abs(chunk)) > 0.1 else 0.0

    tone_t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    speech_frame = (0.7 * np.sin(2 * np.pi * 440 * tone_t)).astype(np.float32)
    speech_onset_t = t
    for _ in range(10):
        seg.process_frame(speech_frame, t)
        t += 0.032

    assert seg.is_speaking is True
    # Start time must be before speech onset time due to pre-roll
    assert seg.speech_start_time < speech_onset_t

    # Feed silence until phrase completes (silence_duration_s = 0.25s -> ~8 frames)
    for _ in range(12):
        seg.process_frame(silence, t)
        t += 0.032

    # Phrase should now be completed
    assert len(completed) == 1
    phrase = completed[0]
    # Total samples should include pre-roll frames + 10 speech frames + post-roll frames
    expected_min_samples = (seg.pre_roll_frames + 10 + seg.post_roll_frames) * 512
    assert phrase["samples"] >= expected_min_samples
    assert phrase["start"] < speech_onset_t
    assert phrase["end"] > speech_onset_t + (10 * 0.032)


def test_phrase_segmenter_continuous_speech_overlap():
    """Verify continuous speech split at max_phrase_duration_s retains overlap in next phrase."""
    completed = []

    def on_phrase(name, start, end, audio):
        completed.append({
            "name": name,
            "start": start,
            "end": end,
            "samples": len(audio),
        })

    seg = PhraseSegmenter(
        stream_name="test_mic",
        speech_threshold=0.2,
        silence_duration_s=0.5,
        max_phrase_duration_s=0.32,  # Short max duration (10 frames)
        min_phrase_duration_s=0.05,
        pre_roll_s=0.1,  # ~3 frames overlap
        post_roll_s=0.1,
        on_phrase_completed=on_phrase,
    )
    seg.vad.predict_chunk = lambda chunk: 0.9

    speech_frame = np.ones(512, dtype=np.float32) * 0.5
    t = 0.0
    # Feed 15 continuous speech frames without pause
    for _ in range(15):
        seg.process_frame(speech_frame, t)
        t += 0.032

    # First chunk should have been triggered when reaching max_phrase_duration_s
    assert len(completed) >= 1
    first_chunk = completed[0]

    # Flush remaining continuous speech
    seg.flush(t)
    assert len(completed) >= 2
    second_chunk = completed[1]

    # Second chunk must start before the first chunk ended (overlap!)
    assert second_chunk["start"] < first_chunk["end"]


def test_storage_lifecycle():
    """Verify creating, writing audio, updating phrases, reading, and deleting a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageManager(base_dir=Path(tmpdir))

        # Create session
        sess = storage.create_session("sess_test_1", mode="mic_only", title="Unit Test Meeting")
        assert sess["id"] == "sess_test_1"
        assert sess["title"] == "Unit Test Meeting"

        # Save audio
        audio = np.zeros(16000, dtype=np.float32)
        wav_path = storage.save_audio("sess_test_1", audio)
        assert Path(wav_path).exists()

        # Append phrase
        storage.append_phrase("sess_test_1", {
            "phrase_id": "p1",
            "speaker": "You",
            "start_time": 0.0,
            "end_time": 1.0,
            "text": "Hello world.",
        })

        # Get session
        retrieved = storage.get_session("sess_test_1")
        assert retrieved is not None
        assert len(retrieved["phrases"]) == 1
        assert retrieved["phrases"][0]["text"] == "Hello world."

        # Update final transcript
        storage.update_final_transcript("sess_test_1", "Hello world. Full summary.", duration_s=1.0)
        updated = storage.get_session("sess_test_1")
        assert updated["final_transcript"] == "Hello world. Full summary."

        # List
        sessions = storage.list_sessions()
        assert len(sessions) == 1

        # Delete
        deleted = storage.delete_session("sess_test_1")
        assert deleted is True
        assert storage.get_session("sess_test_1") is None


def test_api_status_and_devices():
    """Verify FastAPI endpoints respond with valid JSON."""
    client = TestClient(app)

    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert data_status["status"] == "online"
    assert "model_status" in data_status

    res_devices = client.get("/api/devices")
    assert res_devices.status_code == 200
    data_devices = res_devices.json()
    assert "default_mic" in data_devices
    assert "microphones" in data_devices


def test_model_status_and_download_api(monkeypatch):
    """Verify model status, download endpoint and guards."""
    client = TestClient(app)
    from backend.main import stt_engine

    # Check status endpoint contains is_model_downloaded
    status_res = client.get("/api/status").json()
    assert "is_model_downloaded" in status_res
    assert "model_status" in status_res

    # Preload guard when not downloaded
    original_status = stt_engine.status
    stt_engine.status = "not_downloaded"
    monkeypatch.setattr(stt_engine, "is_model_downloaded", lambda: False)
    preload_res = client.post("/api/model/preload")
    assert preload_res.status_code == 400
    assert "not downloaded" in preload_res.json()["detail"].lower()

    # Reset
    stt_engine.status = original_status


def test_lock_free_model_cache_check():
    """Verify is_model_downloaded runs lock-free and quickly."""
    from backend.stt_engine import STTEngine
    engine = STTEngine(auto_load=False)
    assert isinstance(engine.is_model_downloaded(), bool)


def test_recorder_default_devices(monkeypatch):
    """Verify default device handling without touching real audio hardware."""
    import backend.recorder as recorder_module
    monkeypatch.setattr(recorder_module, "HAS_PYAUDIO", False)
    monkeypatch.setattr(recorder_module, "pyaudio", None)
    rec = recorder_module.AudioRecorder()
    started = rec.start(session_id="test_default_devs", mode="mic_only", mic_device_index=-1)
    assert started is True
    res = rec.stop()
    assert isinstance(res, dict)
    assert "audio" in res
    assert "duration" in res


def test_storage_load_audio():
    """Verify load_audio properly reads saved WAV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageManager(base_dir=Path(tmpdir))
        storage.create_session("sess_audio_test")
        sample_audio = (0.25 * np.sin(np.linspace(0, 10, 16000))).astype(np.float32)
        storage.save_audio("sess_audio_test", sample_audio)

        loaded = storage.load_audio("sess_audio_test")
        assert loaded is not None
        assert len(loaded) == 16000
        assert isinstance(loaded, np.ndarray)


def test_retranscribe_endpoint():
    """Verify retranscription completes with a deterministic in-memory model."""
    client = TestClient(app)
    from backend.main import storage, stt_engine

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return "Reliable replacement transcript."

    storage.create_session("sess_retranscribe_test")
    storage.save_audio("sess_retranscribe_test", np.zeros(16000, dtype=np.float32))
    original_status, original_model = stt_engine.status, stt_engine.model
    stt_engine.status, stt_engine.model = "ready", FakeModel()

    try:
        res = client.post("/api/recordings/sess_retranscribe_test/retranscribe")
        assert res.status_code == 200
        assert res.json()["status"] == "processing_hq"

        deadline = __import__("time").time() + 2
        while __import__("time").time() < deadline:
            session = storage.get_session("sess_retranscribe_test")
            if session and session["status"] == "completed":
                break
            __import__("time").sleep(0.01)
        assert session["status"] == "completed"
        assert session["final_transcript"] == "Reliable replacement transcript."
    finally:
        stt_engine.status, stt_engine.model = original_status, original_model
        storage.delete_session("sess_retranscribe_test")


def test_hq_completion_preserves_newer_user_edit():
    """A late HQ result must not overwrite text edited while it was running."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageManager(base_dir=Path(tmpdir))
        storage.create_session("sess_revision_test")
        storage.mark_hq_processing("sess_revision_test", "job_1")
        storage.update_session("sess_revision_test", {"final_transcript": "My correction"})

        assert storage.complete_hq("sess_revision_test", "job_1", "Late model output") is True
        session = storage.get_session("sess_revision_test")
        assert session["status"] == "completed"
        assert session["final_transcript"] == "My correction"


def test_storage_rejects_path_traversal():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageManager(base_dir=Path(tmpdir))
        with pytest.raises(ValueError):
            storage.get_session("../outside")


def test_hq_job_reservation_is_exclusive():
    from backend.stt_engine import STTEngine
    engine = STTEngine(auto_load=False)
    try:
        first = engine.reserve_hq_job("sess_one")
        assert first
        assert engine.reserve_hq_job("sess_one") is None
        assert engine.has_active_hq_job("sess_one") is True
    finally:
        engine.shutdown()


def test_ensure_model_loading_spawns_loader(monkeypatch):
    """Verify loading starts without touching the real model cache or GPU."""
    from backend.stt_engine import STTEngine
    engine = STTEngine(auto_load=False)
    loaded = __import__("threading").Event()

    def fake_load():
        engine.model = object()
        engine.set_status("ready")
        loaded.set()

    try:
        monkeypatch.setattr(engine, "is_model_downloaded", lambda: True)
        monkeypatch.setattr(engine, "_load_model_worker", fake_load)
        engine.model = None
        engine._download_thread = None
        engine._loader_thread = None
        engine.ensure_model_loading()
        assert loaded.wait(1)
        assert engine.status == "ready"
        assert engine.model is not None

        # Test unloading
        engine.unload_model()
        assert engine.model is None
        assert engine.status == "not_loaded"
    finally:
        engine.shutdown()


def test_recover_interrupted_sessions():
    """Verify interrupted sessions are recovered with 'interrupted' status and preserved transcript."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageManager(base_dir=Path(tmpdir))
        sess = storage.create_session("sess_abrupt_1", mode="mic_only", title="Abrupt Session")
        assert sess["status"] == "recording"

        storage.append_phrase("sess_abrupt_1", {
            "phrase_id": "p_live_1",
            "speaker": "You",
            "start_time": 0.0,
            "end_time": 2.5,
            "text": "Live transcribed before crash.",
        })

        audio = np.ones(16000, dtype=np.float32) * 0.1
        storage.save_audio("sess_abrupt_1", audio)

        recovered = storage.recover_interrupted_sessions()
        assert recovered == 1

        rec_sess = storage.get_session("sess_abrupt_1")
        assert rec_sess["status"] == "interrupted"
        assert "closed abruptly" in rec_sess["status_error"].lower()
        assert "Live transcribed before crash." in rec_sess["final_transcript"]
        assert storage.load_audio("sess_abrupt_1") is not None


def test_recorder_get_current_audio():
    """Verify get_current_audio mixes and returns audio accumulated so far."""
    from backend.recorder import AudioRecorder
    rec = AudioRecorder()
    assert len(rec.get_current_audio()) == 0

    chunk1 = np.ones(1600, dtype=np.float32) * 0.2
    chunk2 = np.ones(1600, dtype=np.float32) * 0.3
    rec._mic_audio_chunks.append(chunk1)
    rec._mic_audio_chunks.append(chunk2)

    current = rec.get_current_audio()
    assert len(current) == 3200
    assert np.allclose(current[:1600], 0.2)





