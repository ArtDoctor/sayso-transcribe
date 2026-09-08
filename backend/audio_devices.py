import time
import threading
from typing import Dict, Any, List, Optional
import logging

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

# Global lock to prevent concurrent PortAudio / WASAPI initialization crashes
PORTAUDIO_LOCK = threading.Lock()

# Cache to avoid constantly re-enumerating audio hardware on rapid HTTP requests
_cached_devices: Optional[Dict[str, Any]] = None
_last_query_time: float = 0.0
CACHE_TTL_SECONDS = 3.0


def get_audio_devices(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Enumerate all available audio devices on Windows safely with lock and TTL cache.
    Categorizes them into:
      1. Microphones / Input devices
      2. Windows WASAPI Loopback devices (System / PC Audio)
    And identifies current Windows default devices.
    """
    global _cached_devices, _last_query_time

    with PORTAUDIO_LOCK:
        now = time.time()
        if not force_refresh and _cached_devices is not None and (now - _last_query_time) < CACHE_TTL_SECONDS:
            return _cached_devices

        if not HAS_PYAUDIO or pyaudio is None:
            logger.warning("PyAudioWPatch is not installed. Returning fallback mock devices.")
            _cached_devices = {
                "default_mic": {"index": 0, "name": "Default Microphone (Simulated)"},
                "default_system": {"index": 1, "name": "Default System Audio [Loopback] (Simulated)"},
                "microphones": [{"index": 0, "name": "Default Microphone (Simulated)", "channels": 1, "sample_rate": 16000}],
                "system_devices": [{"index": 1, "name": "Default System Audio [Loopback] (Simulated)", "channels": 2, "sample_rate": 48000}],
            }
            _last_query_time = now
            return _cached_devices

        p = None
        microphones: List[Dict[str, Any]] = []
        system_devices: List[Dict[str, Any]] = []
        default_mic_info: Optional[Dict[str, Any]] = None
        default_system_info: Optional[Dict[str, Any]] = None

        try:
            p = pyaudio.PyAudio()

            # Detect current default input device
            try:
                default_in = p.get_default_input_device_info()
                default_mic_info = {
                    "index": default_in["index"],
                    "name": default_in["name"].strip(),
                }
            except Exception as e:
                logger.warning(f"Could not get default input device: {e}")

            # Detect current default WASAPI loopback device
            try:
                if hasattr(p, "get_default_wasapi_loopback"):
                    default_loop = p.get_default_wasapi_loopback()
                    if default_loop:
                        default_system_info = {
                            "index": default_loop["index"],
                            "name": default_loop["name"].strip(),
                        }
            except Exception as e:
                logger.warning(f"Could not get default wasapi loopback: {e}")

            # Enumerate all devices
            device_count = p.get_device_count()
            for i in range(device_count):
                try:
                    info = p.get_device_info_by_index(i)
                    name = info.get("name", "").strip()
                    max_inputs = info.get("maxInputChannels", 0)
                    is_loopback = bool(info.get("isLoopbackDevice", False)) or ("[loopback]" in name.lower())
                    sample_rate = int(info.get("defaultSampleRate", 16000))

                    if is_loopback:
                        system_devices.append({
                            "index": i,
                            "name": name,
                            "channels": max_inputs,
                            "sample_rate": sample_rate,
                        })
                    elif max_inputs > 0:
                        microphones.append({
                            "index": i,
                            "name": name,
                            "channels": max_inputs,
                            "sample_rate": sample_rate,
                        })
                except Exception as e:
                    logger.debug(f"Skipping audio device index {i}: {e}")

            # Fallbacks if default was not directly detected
            if not default_system_info and system_devices:
                default_system_info = {
                    "index": system_devices[0]["index"],
                    "name": system_devices[0]["name"],
                }

            if not default_mic_info and microphones:
                default_mic_info = {
                    "index": microphones[0]["index"],
                    "name": microphones[0]["name"],
                }

            _cached_devices = {
                "default_mic": default_mic_info or {"index": -1, "name": "No Microphone Detected"},
                "default_system": default_system_info or {"index": -1, "name": "No System Audio Loopback Detected"},
                "microphones": microphones,
                "system_devices": system_devices,
            }
            _last_query_time = now

        except Exception as e:
            logger.error(f"Error enumerating PortAudio devices: {e}")
            if _cached_devices is None:
                _cached_devices = {
                    "default_mic": {"index": -1, "name": "Audio Error"},
                    "default_system": {"index": -1, "name": "Audio Error"},
                    "microphones": [],
                    "system_devices": [],
                }
        finally:
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass

        return _cached_devices
