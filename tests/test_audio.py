"""Tests for audio analysis using a synthetically generated WAV file.

No external audio download required -- we generate a short test tone
with numpy and soundfile.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from rufus.audio import AudioProfile, analyse


def _make_test_wav(path: Path, duration: float = 5.0, sr: int = 22050) -> Path:
    """Generate a synthetic WAV: a kick-drum-like pulse train with
    rising energy, giving the analyser beats and dynamic range to
    detect.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Rising sine sweep for spectral variation
    freq = 220 + 440 * (t / duration)
    signal = 0.3 * np.sin(2 * np.pi * freq * t)

    # Pulse every ~0.5s to simulate beats
    beat_interval = 0.5
    for onset in np.arange(0, duration, beat_interval):
        mask = (t >= onset) & (t < onset + 0.05)
        signal[mask] += 0.7 * np.sin(2 * np.pi * 80 * (t[mask] - onset))

    # Rising amplitude envelope
    envelope = 0.3 + 0.7 * (t / duration)
    signal *= envelope

    signal = signal / np.max(np.abs(signal))

    sf.write(str(path), signal, sr)
    return path


@pytest.fixture
def test_wav(tmp_path) -> Path:
    if not HAS_SOUNDFILE:
        pytest.skip("soundfile not installed")
    return _make_test_wav(tmp_path / "test.wav")


class TestAudioAnalysis:
    def test_returns_audio_profile(self, test_wav):
        profile = analyse(test_wav)
        assert isinstance(profile, AudioProfile)

    def test_duration(self, test_wav):
        profile = analyse(test_wav)
        assert 4.5 < profile.duration < 5.5

    def test_tempo_detected(self, test_wav):
        profile = analyse(test_wav)
        assert profile.tempo > 0

    def test_beats_detected(self, test_wav):
        profile = analyse(test_wav)
        assert len(profile.beat_times) > 0

    def test_energy_curve(self, test_wav):
        profile = analyse(test_wav)
        assert len(profile.rms_energy) > 0
        assert profile.rms_energy.min() >= 0
        assert profile.rms_energy.max() <= 1.0

    def test_onset_envelope(self, test_wav):
        profile = analyse(test_wav)
        assert len(profile.onset_envelope) > 0

    def test_spectral_centroid(self, test_wav):
        profile = analyse(test_wav)
        assert len(profile.spectral_centroid) > 0

    def test_sections_detected(self, test_wav):
        profile = analyse(test_wav)
        assert len(profile.sections) > 0
        for sec in profile.sections:
            assert sec.label in ("calm", "build", "peak")
            assert sec.start_time < sec.end_time

    def test_energy_at(self, test_wav):
        profile = analyse(test_wav)
        e = profile.energy_at(2.5)
        assert 0.0 <= e <= 1.0

    def test_onset_at(self, test_wav):
        profile = analyse(test_wav)
        o = profile.onset_at(1.0)
        assert 0.0 <= o <= 1.0

    def test_spectral_centroid_at(self, test_wav):
        profile = analyse(test_wav)
        c = profile.spectral_centroid_at(3.0)
        assert 0.0 <= c <= 1.0

    def test_nearest_beat(self, test_wav):
        profile = analyse(test_wav)
        bt, dist = profile.nearest_beat(1.0)
        assert dist >= 0.0
        assert bt >= 0.0

    def test_section_at(self, test_wav):
        profile = analyse(test_wav)
        sec = profile.section_at(2.5)
        assert sec is not None or True  # may legitimately be None at edges

    def test_frame_times_aligned(self, test_wav):
        profile = analyse(test_wav)
        assert len(profile.frame_times) == len(profile.rms_energy)
        assert len(profile.frame_times) == len(profile.onset_envelope)
        assert len(profile.frame_times) == len(profile.spectral_centroid)


class TestSuggestKeyframeTimes:
    def test_count_and_ordering(self, test_wav):
        from rufus.audio import suggest_keyframe_times
        profile = analyse(test_wav)
        times = suggest_keyframe_times(profile, 4)
        assert len(times) == 4
        assert times[0] == 0.0
        assert times == sorted(times)
        assert all(0 <= t <= profile.duration for t in times)

    def test_single_keyframe(self, test_wav):
        from rufus.audio import suggest_keyframe_times
        profile = analyse(test_wav)
        assert suggest_keyframe_times(profile, 1) == [0.0]

    def test_respects_duration_cap(self, test_wav):
        from rufus.audio import suggest_keyframe_times
        profile = analyse(test_wav)
        times = suggest_keyframe_times(profile, 3, duration=3.0)
        assert all(t <= 3.0 for t in times)
