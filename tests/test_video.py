"""Tests for video assembly using synthetic frames and audio.

Generates solid-colour PNG frames and a short WAV, then assembles
them into an MP4 with ffmpeg and verifies the result with ffprobe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from rufus.config import GenerationConfig
from rufus.video import assemble, find_ffmpeg, probe_duration

try:
    find_ffmpeg()
    HAS_FFMPEG = True
except RuntimeError:
    HAS_FFMPEG = False


def _make_gradient_frames(output_dir: Path, count: int, size: int = 128):
    """Create *count* PNG frames with a colour gradient for visual
    verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        t = i / max(count - 1, 1)
        r = int(255 * t)
        g = int(128 * (1 - t))
        b = int(255 * (1 - t))
        img = Image.new("RGB", (size, size), (r, g, b))
        p = output_dir / f"frame_{i:04d}.png"
        img.save(p)
        paths.append(p)
    return paths


def _make_silent_wav(path: Path, duration: float = 2.0, sr: int = 22050):
    """Generate a short silent WAV for audio muxing tests."""
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(path), samples, sr)
    return path


@pytest.fixture
def test_assets(tmp_path):
    if not HAS_SOUNDFILE:
        pytest.skip("soundfile not installed")
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")

    fps = 8
    duration = 2.0
    n_frames = int(fps * duration)

    frame_dir = tmp_path / "frames"
    frame_paths = _make_gradient_frames(frame_dir, n_frames)
    wav_path = _make_silent_wav(tmp_path / "audio.wav", duration)

    return frame_paths, wav_path, fps, duration


class TestVideoAssembly:
    def test_output_created(self, test_assets, tmp_path):
        frame_paths, wav_path, fps, duration = test_assets
        out = tmp_path / "output.mp4"
        config = GenerationConfig(fps=fps, device="cpu")
        result = assemble(frame_paths, str(wav_path), str(out), config)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_output_duration(self, test_assets, tmp_path):
        frame_paths, wav_path, fps, duration = test_assets
        out = tmp_path / "output.mp4"
        config = GenerationConfig(fps=fps, device="cpu")
        assemble(frame_paths, str(wav_path), str(out), config)
        assert abs(probe_duration(out) - duration) < 0.5

    def test_explicit_duration_trim(self, test_assets, tmp_path):
        frame_paths, wav_path, fps, duration = test_assets
        out = tmp_path / "output.mp4"
        config = GenerationConfig(fps=fps, device="cpu")
        trim = 1.0
        assemble(frame_paths, str(wav_path), str(out), config, duration=trim)
        assert probe_duration(out) <= trim + 0.5

    def test_smooth_fps_upsampling(self, test_assets, tmp_path):
        frame_paths, wav_path, fps, duration = test_assets
        out = tmp_path / "smooth.mp4"
        config = GenerationConfig(fps=fps, smooth_fps=24, device="cpu")
        result = assemble(frame_paths, str(wav_path), str(out), config)
        assert Path(result).exists()
        assert abs(probe_duration(out) - duration) < 0.5

    def test_empty_frames_raises(self, tmp_path):
        if not HAS_FFMPEG:
            pytest.skip("ffmpeg not available")
        config = GenerationConfig(device="cpu")
        with pytest.raises(ValueError):
            assemble([], "audio.wav", str(tmp_path / "o.mp4"), config)
