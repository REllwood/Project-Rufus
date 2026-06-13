"""Video assembly: combines rendered frames and an audio track into an MP4.

Uses ffmpeg directly (system binary, or the one bundled with
``imageio-ffmpeg``) so frames are streamed from disk rather than held
in memory, and so the output can optionally be motion-interpolated to
a higher frame rate (``config.smooth_fps``).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import GenerationConfig

logger = logging.getLogger(__name__)

_DIGITS_RE = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")


def find_ffmpeg() -> str:
    """Locate an ffmpeg executable, preferring the system install."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install it (e.g. `brew install ffmpeg`) or "
            "`pip install imageio-ffmpeg` for a bundled binary."
        )


def _sequence_pattern(first_frame: Path) -> tuple[str, int]:
    """Derive ffmpeg's printf-style input pattern from the first frame.

    ``frame_000123.png`` -> (``frame_%06d.png``, 123)
    """
    m = _DIGITS_RE.match(first_frame.name)
    if not m:
        raise ValueError(
            f"Frame filename {first_frame.name!r} has no numeric sequence."
        )
    prefix, digits, ext = m.groups()
    pattern = f"{prefix}%0{len(digits)}d{ext}"
    return str(first_frame.parent / pattern), int(digits)


def assemble(
    frame_paths: List[Path],
    audio_path: str,
    output_path: str,
    config: GenerationConfig,
    *,
    duration: Optional[float] = None,
) -> Path:
    """Stitch *frame_paths* into a video, mux with *audio_path*, and
    write to *output_path*.

    Parameters
    ----------
    frame_paths:
        Ordered list of image file paths (one per frame).  Filenames
        must contain a zero-padded frame number (``frame_000042.png``).
    audio_path:
        Path to the original audio file to mux into the video.
    output_path:
        Destination path for the finished ``.mp4``.
    config:
        Generation configuration (FPS, codec, bitrate, smooth_fps).
    duration:
        If provided, trim the output to this length in seconds.

    Returns
    -------
    Path
        The resolved output path.
    """
    if not frame_paths:
        raise ValueError("No frames to assemble.")

    ffmpeg = find_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pattern, start_number = _sequence_pattern(Path(frame_paths[0]))

    logger.info(
        "Assembling %d frames at %d FPS into %s",
        len(frame_paths), config.fps, output_path,
    )

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(config.fps),
        "-start_number", str(start_number),
        "-i", pattern,
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
    ]

    filters = []
    if config.smooth_fps and config.smooth_fps > config.fps:
        logger.info("Motion-interpolating to %d FPS (this is slow)", config.smooth_fps)
        filters.append(
            f"minterpolate=fps={config.smooth_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
        )
    # yuv420p needs even dimensions; diffusion sizes are multiples of 8,
    # but scale defensively for arbitrary inputs.
    filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    cmd += ["-vf", ",".join(filters)]

    cmd += [
        "-c:v", config.video_codec,
        "-b:v", config.video_bitrate,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
    ]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{tail}")

    logger.info("Video written to %s", output_path)
    return output_path


def probe_duration(video_path: str) -> float:
    """Return a video's duration in seconds via ffprobe (for tests)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        ffmpeg = find_ffmpeg()
        sibling = Path(ffmpeg).parent / "ffprobe"
        if sibling.exists():
            ffprobe = str(sibling)
        else:
            raise RuntimeError("ffprobe not found.")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())
