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
            seen_mic_names = set()
            seen_sys_names = set()

            for i in range(device_count):
                try:
                    info = p.get_device_info_by_index(i)
                    name = info.get("name", "").strip()
                    max_inputs = info.get("maxInputChannels", 0)
                    is_loopback = bool(info.get("isLoopbackDevice", False)) or ("[loopback]" in name.lower())
                    sample_rate = int(info.get("defaultSampleRate", 16000))

                    if is_loopback:
                        if name not in seen_sys_names:
                            seen_sys_names.add(name)
                            system_devices.append({
                                "index": i,
                                "name": name,
                                "channels": max_inputs,
                                "sample_rate": sample_rate,
                            })
                    elif max_inputs > 0:
                        if name not in seen_mic_names:
                            seen_mic_names.add(name)
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


def resolve_device_index(
    p: Any,
    target_name: Optional[str] = None,
    target_index: Optional[int] = None,
    is_loopback: bool = False,
) -> Optional[int]:
    """
    Resolve live PortAudio device index for a requested device name or index.
    Prioritizes matching by name so that device unplugging/reconnecting or PC restarts
    do not misdirect capture to the wrong audio hardware (e.g. monitors vs headphones).
    """
    if p is None:
        return None

    try:
        device_count = p.get_device_count()
    except Exception as e:
        logger.warning("Could not get device count: %s", e)
        return None

    # 1. Match by exact or partial device name if provided
    if target_name and target_name.strip():
        name_clean = target_name.strip().lower()
        candidates: List[tuple] = []
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                dev_name = info.get("name", "").strip()
                dev_lower = dev_name.lower()
                dev_is_loopback = bool(info.get("isLoopbackDevice", False)) or ("[loopback]" in dev_lower)
                max_inputs = info.get("maxInputChannels", 0)

                if is_loopback:
                    if not dev_is_loopback:
                        continue
                else:
                    if dev_is_loopback or max_inputs <= 0:
                        continue

                # Exact match
                if dev_lower == name_clean:
                    logger.info("Found exact match for device '%s' at live index %d", target_name, i)
                    return i

                # Fuzzy / partial match
                if name_clean in dev_lower or dev_lower in name_clean:
                    candidates.append((i, dev_name))
            except Exception:
                continue

        if candidates:
            best_idx, best_name = candidates[0]
            logger.info("Found partial match for device '%s' -> '%s' at live index %d", target_name, best_name, best_idx)
            return best_idx

        logger.warning(
            "Device named '%s' was requested but is not currently available (disconnected?). "
            "Falling back to default device instead of an arbitrary index.",
            target_name
        )
        target_index = None

    # 2. Check target_index if valid and compatible
    if target_index is not None and 0 <= target_index < device_count:
        try:
            info = p.get_device_info_by_index(target_index)
            dev_name = info.get("name", "").strip()
            dev_lower = dev_name.lower()
            dev_is_loopback = bool(info.get("isLoopbackDevice", False)) or ("[loopback]" in dev_lower)
            max_inputs = info.get("maxInputChannels", 0)
            if is_loopback and dev_is_loopback:
                return target_index
            elif not is_loopback and not dev_is_loopback and max_inputs > 0:
                return target_index
        except Exception:
            pass

    # 3. Fallback to system default
    try:
        if is_loopback:
            if hasattr(p, "get_default_wasapi_loopback"):
                def_loop = p.get_default_wasapi_loopback()
                if def_loop:
                    return def_loop.get("index")
        else:
            def_in = p.get_default_input_device_info()
            if def_in:
                return def_in.get("index")
    except Exception as e:
        logger.warning("Could not get default device info: %s", e)

    return None

