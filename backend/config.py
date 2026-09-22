import os
from pathlib import Path

# Base Paths
# In a portable build BASE_DIR is the folder containing Sayso.exe, backend/, and
# models/.  Keeping these paths beside the executable prevents user data and
# Hugging Face downloads from leaking into the user's profile.
BASE_DIR = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_FILE = BASE_DIR / "logs.txt"

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

