from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np


@dataclass
class Section:
    """A detected structural section of the audio track."""

    label: str
    start_time: float
    end_time: float
    mean_energy: float


@dataclass
class AudioProfile:
    """Complete audio analysis result consumed by the generation pipeline.

    All time-series arrays share the same hop-aligned time axis available
    via ``frame_times``.
    """

    path: str
    duration: float
    sample_rate: int
    tempo: float
    beat_times: np.ndarray
    frame_times: np.ndarray
    onset_envelope: np.ndarray
    rms_energy: np.ndarray
    spectral_centroid: np.ndarray
    sections: List[Section]

    # ------------------------------------------------------------------
    # Convenience look-ups
    # ------------------------------------------------------------------

    def energy_at(self, t: float) -> float:
        """Return normalised RMS energy (0-1) at time *t* seconds."""
        idx = np.searchsorted(self.frame_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.rms_energy) - 1)
        return float(self.rms_energy[idx])

    def onset_at(self, t: float) -> float:
        """Return normalised onset strength (0-1) at time *t* seconds."""
        idx = np.searchsorted(self.frame_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.onset_envelope) - 1)
        return float(self.onset_envelope[idx])

    def spectral_centroid_at(self, t: float) -> float:
        """Return normalised spectral centroid (0-1) at time *t* seconds."""
        idx = np.searchsorted(self.frame_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.spectral_centroid) - 1)
        return float(self.spectral_centroid[idx])

    def nearest_beat(self, t: float) -> Tuple[float, float]:
        """Return ``(beat_time, distance)`` for the beat closest to *t*."""
        if len(self.beat_times) == 0:
            return (0.0, t)
        idx = np.argmin(np.abs(self.beat_times - t))
        bt = float(self.beat_times[idx])
        return (bt, abs(bt - t))

    def section_at(self, t: float) -> Optional[Section]:
        """Return the section that contains time *t*, or ``None``."""
        for sec in self.sections:
            if sec.start_time <= t < sec.end_time:
                return sec
        return None


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if constant."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving-average smoothing."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _detect_sections(
    rms: np.ndarray,
    frame_times: np.ndarray,
    duration: float,
) -> List[Section]:
    """Segment the track into broad structural sections based on energy.

    Uses a three-tier energy threshold (low / mid / high) to label
    contiguous regions as *calm*, *build*, or *peak*.  Adjacent regions
    with the same label are merged.
    """
    low_thresh = 0.33
    high_thresh = 0.66

    labels: List[str] = []
    for val in rms:
        if val < low_thresh:
            labels.append("calm")
        elif val < high_thresh:
            labels.append("build")
        else:
            labels.append("peak")

    sections: List[Section] = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            start_t = float(frame_times[start_idx])
            end_t = float(frame_times[i])
            mean_e = float(rms[start_idx:i].mean())
            sections.append(Section(current_label, start_t, end_t, mean_e))
            current_label = labels[i]
            start_idx = i

    start_t = float(frame_times[start_idx])
    end_t = duration
    mean_e = float(rms[start_idx:].mean())
    sections.append(Section(current_label, start_t, end_t, mean_e))

    return sections


def suggest_keyframe_times(
    profile: AudioProfile,
    n: int,
    duration: Optional[float] = None,
) -> List[float]:
    """Place *n* keyframes at musically sensible times.

    Targets are spread evenly across the track, then snapped to nearby
    section boundaries (where the music's character changes) and then
    to the nearest beat, so scene changes land on musical events.
    """
    d = min(duration or profile.duration, profile.duration)
    if n <= 1:
        return [0.0]

    bounds = np.array(
        [s.start_time for s in profile.sections if 0.0 < s.start_time < d]
    )
    spacing = d / (n - 1)
    tolerance = spacing / 2

    times = [0.0]
    for k in range(1, n):
        t = spacing * k
        if len(bounds):
            nearest = float(bounds[np.argmin(np.abs(bounds - t))])
            if abs(nearest - t) <= tolerance:
                t = nearest
        if len(profile.beat_times):
            beat = float(
                profile.beat_times[np.argmin(np.abs(profile.beat_times - t))]
            )
            if abs(beat - t) <= 1.0:
                t = beat
        # Keep strictly ordered with a minimum gap.
        t = min(max(t, times[-1] + 0.5), d)
        times.append(float(t))
    return times


def analyse(
    audio_path: str | Path,
    *,
    sr: int = 22050,
    hop_length: int = 512,
    energy_smoothing: int = 5,
) -> AudioProfile:
    """Analyse an audio file and return a complete :class:`AudioProfile`.

    Parameters
    ----------
    audio_path:
        Path to the audio file (wav, mp3, flac, ogg, etc.).
    sr:
        Target sample rate for analysis.
    hop_length:
        Hop length in samples for frame-level features.
    energy_smoothing:
        Moving-average window size applied to the RMS energy curve.
    """
    audio_path = str(audio_path)
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr_actual)

    tempo_arr, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr_actual, hop_length=hop_length
    )
    tempo = float(np.atleast_1d(tempo_arr)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr_actual, hop_length=hop_length)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr_actual, hop_length=hop_length)
    onset_env = _normalise(_smooth(onset_env, energy_smoothing))

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = _normalise(_smooth(rms, energy_smoothing))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr_actual, hop_length=hop_length)[0]
    cent = _normalise(_smooth(cent, energy_smoothing))

    n_frames = min(len(onset_env), len(rms), len(cent))
    onset_env = onset_env[:n_frames]
    rms = rms[:n_frames]
    cent = cent[:n_frames]

    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr_actual, hop_length=hop_length
    )

    sections = _detect_sections(rms, frame_times, duration)

    return AudioProfile(
        path=audio_path,
        duration=duration,
        sample_rate=sr_actual,
        tempo=tempo,
        beat_times=beat_times,
        frame_times=frame_times,
        onset_envelope=onset_env,
        rms_energy=rms,
        spectral_centroid=cent,
        sections=sections,
    )
