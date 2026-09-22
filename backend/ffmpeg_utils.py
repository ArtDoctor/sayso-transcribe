import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_ffmpeg_installed() -> bool:
    """Return True if ffmpeg executable is available on the system PATH."""
    return shutil.which("ffmpeg") is not None


def get_ffmpeg_version() -> Optional[str]:
    """Return the first line of ffmpeg -version, or None if ffmpeg is not available."""
    if not is_ffmpeg_installed():
        return None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.splitlines()[0].strip()
    except Exception as e:
        logger.warning("Error running ffmpeg -version: %s", e)
    return None


def convert_audio_to_wav_16k(input_path: Path, output_path: Path) -> None:
    """
    Convert any media/audio file to 16kHz mono 16-bit PCM WAV using ffmpeg.
    Raises RuntimeError if ffmpeg is not installed or if conversion fails.
    """
    if not is_ffmpeg_installed():
        raise RuntimeError("ffmpeg not installed, please install.")

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    logger.info("Executing ffmpeg conversion: %s -> %s", input_path.name, output_path.name)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err_snippet = proc.stderr[-400:].strip() if proc.stderr else "Unknown ffmpeg error"
        logger.error("FFmpeg conversion failed: %s", err_snippet)
        raise RuntimeError(f"FFmpeg conversion failed: {err_snippet}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg generated an empty or missing audio file.")
