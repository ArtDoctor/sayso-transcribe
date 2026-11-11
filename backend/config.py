import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base Paths
# In a portable build BASE_DIR is the folder containing Sayso.exe, backend/, and
# models/.  Keeping these paths beside the executable prevents user data and
# Hugging Face downloads from leaking into the user's profile.
BASE_DIR = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_FILE = BASE_DIR / "logs.txt"

def load_env_file(env_path: Optional[Path] = None) -> bool:
    """Load key-value pairs from .env into os.environ if not already set."""
    candidates = [env_path] if env_path else [BASE_DIR / ".env", Path.cwd() / ".env"]
    for path in candidates:
        if path and path.is_file():
            try:
                from dotenv import load_dotenv
                load_dotenv(path, override=False)
                return True
            except ImportError:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                if k and k not in os.environ:
                                    os.environ[k] = v
                    return True
                except Exception:
                    pass
    return False

# Load environment variables on startup
load_env_file()

def get_hf_token() -> Optional[str]:
    """Return configured Hugging Face token from environment, or None."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    ).strip()
    return token if token else None

def mask_hf_token(token: Optional[str]) -> Optional[str]:
    """Return a masked representation of the token (e.g. hf_...1234) for safe display."""
    if not token:
        return None
    tok = token.strip()
    if len(tok) <= 8:
        return "****"
    return f"{tok[:4]}...{tok[-4:]}"

def save_hf_token_to_env(token: str) -> None:
    """Save or update HF_TOKEN in BASE_DIR / .env and update os.environ."""
    token = token.strip()
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        os.environ.pop("HUGGINGFACE_TOKEN", None)

    env_path = BASE_DIR / ".env"
    lines = []
    found = False
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            lines = []

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("HF_TOKEN=") or stripped.startswith("HF_TOKEN "):
            if token:
                new_lines.append(f"HF_TOKEN={token}\n")
            found = True
        else:
            new_lines.append(line)

    if not found and token:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"HF_TOKEN={token}\n")

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception as e:
        logger.error(f"Failed to write token to {env_path}: {e}")

MODEL_DIR = BASE_DIR / "models"
HF_HOME = MODEL_DIR / "huggingface"
TORCH_HOME = MODEL_DIR / "torch"
# The packaged executable sets these variables before importing the backend.
# Only force local caches when this is actually a portable folder; preserving
# the normal user cache keeps source checkouts and tests compatible with an
# already-downloaded developer model.
IS_PORTABLE = (BASE_DIR / "python").is_dir() or os.environ.get("SAYSO_PORTABLE_ROOT") == "1"
if IS_PORTABLE:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["TORCH_HOME"] = str(TORCH_HOME)

# Server Config
HOST = "127.0.0.1"
PORT = 48653

# Audio Configuration
SAMPLE_RATE = 16000  # 16 kHz mono required by Silero VAD and Cohere
CHANNELS = 1         # Mono
VAD_FRAME_SIZE = 512 # 512 samples = 32ms at 16kHz
FORMAT_WIDTH = 2     # 16-bit PCM (2 bytes)

# VAD & Segmentation Settings
VAD_SPEECH_THRESHOLD = 0.40          # Speech probability threshold
VAD_SILENCE_DURATION_S = 0.85        # Pause duration to trigger phrase completion
VAD_SHORT_PHRASE_SILENCE_S = 1.8     # Extended pause before finalizing short utterances (< 1.5s)
MAX_PHRASE_DURATION_S = 25.0         # Matches stt.md max_audio_clip_s recommendation
MIN_PHRASE_DURATION_S = 0.4          # Minimum phrase clip duration (seconds)
MIN_SPEECH_DURATION_S = 0.15         # Minimum actual speech frames duration to avoid transient noise/clicks
MIN_ENERGY_RMS = 0.0012              # Minimum RMS energy to qualify as audible speech (accommodates lower-gain mics)
VAD_PRE_ROLL_S = 0.50                # Audio padding/overlap before speech detection (seconds)
VAD_POST_ROLL_S = 0.50               # Audio padding/overlap after speech detection (seconds)

# STT Cohere Settings (From docs/stt.md)
COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = (
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"
)
BATCH_SIZE = 8
MAX_AUDIO_CLIP_S = 25.0
OVERLAP_CHUNK_SECOND = 3.0
PUNCTUATION = True

