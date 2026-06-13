"""Audio-reactive frame planning.

Precomputes every per-frame control signal for the whole video as a
pure function of the audio analysis, the prompt timeline, and the
config.  Doing this up-front (rather than per-frame inside the render
loop) guarantees two properties the old implementation lacked:

* **Monotonic progress**: audio energy modulates the *rate* at which
  a transition advances, so progress never moves backwards (no morph
  jitter), yet transitions still accelerate during builds and drops.
* **Deterministic resume**: the plan depends only on inputs, so an
  interrupted render can recompute it exactly and continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .audio import AudioProfile
from .config import GenerationConfig
from .prompts import PromptTimeline

# A beat influences transition speed within this window (seconds).
_BEAT_WINDOW = 0.1


@dataclass
class FramePlan:
    """Per-frame control signals for the full video.

    All arrays have length ``total_frames``.
    """

    times: np.ndarray          # frame timestamp in seconds
    pair_index: np.ndarray     # index of keyframe A in the active transition
    progress: np.ndarray       # 0-1 blend within the active transition
    velocity: np.ndarray       # transition speed in progress-per-second
    strength: np.ndarray       # img2img denoise strength (flow mode)
    color_anchor: np.ndarray   # 0-1 colour-stabiliser strength per frame
    zoom: np.ndarray           # per-frame multiplicative zoom factor
    pan_x: np.ndarray          # per-frame horizontal drift (pixels)
    pan_y: np.ndarray          # per-frame vertical drift (pixels)
    energy: np.ndarray         # audio features sampled per frame
    onset: np.ndarray
    centroid: np.ndarray
    section_labels: List[str]

    def __len__(self) -> int:
        return len(self.times)


def _sample_curve(
    frame_times: np.ndarray,
    curve_times: np.ndarray,
    curve: np.ndarray,
) -> np.ndarray:
    """Sample an audio feature curve at each frame timestamp."""
    idx = np.searchsorted(curve_times, frame_times, side="right") - 1
    idx = np.clip(idx, 0, len(curve) - 1)
    return curve[idx]


def _beat_proximity(frame_times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Per-frame proximity to the nearest beat, 1 on the beat -> 0 at
    ``_BEAT_WINDOW`` seconds away."""
    if len(beat_times) == 0:
        return np.zeros_like(frame_times)
    idx = np.searchsorted(beat_times, frame_times)
    idx_lo = np.clip(idx - 1, 0, len(beat_times) - 1)
    idx_hi = np.clip(idx, 0, len(beat_times) - 1)
    dist = np.minimum(
        np.abs(frame_times - beat_times[idx_lo]),
        np.abs(frame_times - beat_times[idx_hi]),
    )
    return np.clip(1.0 - dist / _BEAT_WINDOW, 0.0, 1.0)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < 2:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def build_plan(
    timeline: PromptTimeline,
    audio: AudioProfile,
    config: GenerationConfig,
    total_frames: int,
) -> FramePlan:
    """Compute the full per-frame control plan."""
    fps = config.fps
    times = np.arange(total_frames) / fps

    energy = _sample_curve(times, audio.frame_times, audio.rms_energy)
    onset = _sample_curve(times, audio.frame_times, audio.onset_envelope)
    centroid = _sample_curve(times, audio.frame_times, audio.spectral_centroid)
    beat_prox = _beat_proximity(times, audio.beat_times)

    section_labels = []
    for t in times:
        sec = audio.section_at(float(t))
        section_labels.append(sec.label if sec else "---")

    # ------------------------------------------------------------------
    # Transition progress: audio modulates the *rate* of advance.
    # Within each keyframe span the per-frame weights are accumulated
    # and normalised, so progress starts at 0, ends at 1, and is
    # strictly non-decreasing.
    # ------------------------------------------------------------------
    keyframes = timeline.keyframes
    kf_times = [k.time for k in keyframes]
    n_pairs = max(len(keyframes) - 1, 1)

    pair_index = np.zeros(total_frames, dtype=np.int64)
    progress = np.zeros(total_frames, dtype=np.float64)

    rate_weights = (
        1.0
        + config.energy_rate_influence * energy
        + config.onset_sensitivity * 2.0 * np.clip(onset - 0.5, 0.0, None)
        + config.beat_transition_boost * beat_prox
    )

    if len(keyframes) == 1:
        pass  # single keyframe: pair 0, progress 0 everywhere
    else:
        for pair in range(n_pairs):
            t_a, t_b = kf_times[pair], kf_times[pair + 1]
            # The first pair also owns the pre-roll before the first
            # keyframe; the last pair also owns the tail after the last
            # one.  (A 2-keyframe timeline has a single pair owning both.)
            lo = np.ones_like(times, dtype=bool) if pair == 0 else times >= t_a
            hi = np.ones_like(times, dtype=bool) if pair == n_pairs - 1 else times < t_b
            in_span = lo & hi

            idx = np.nonzero(in_span)[0]
            if len(idx) == 0:
                continue
            pair_index[idx] = pair

            # Frames after the last keyframe stay fully blended.
            active = idx[(times[idx] >= t_a) & (times[idx] < t_b)]
            done = idx[times[idx] >= t_b]
            progress[done] = 1.0
            if len(active) == 0:
                continue

            w = rate_weights[active]
            total_w = w.sum()
            if total_w <= 0:
                continue
            # Cumulative weight *before* each frame: first frame is 0,
            # the last approaches (but does not reach) 1, and the next
            # transition picks up seamlessly at 0.
            progress[active] = (np.cumsum(w) - w) / total_w

    # Keyframe hold: let each scene rest before its morph begins.  The
    # first `hold` fraction of accumulated progress is clamped to 0 and
    # the remainder rescaled, so the morph still completes on time.
    hold = float(np.clip(config.keyframe_hold, 0.0, 0.9))
    if hold > 0:
        progress = np.clip((progress - hold) / (1.0 - hold), 0.0, 1.0)

    # Smoothstep easing: transitions start and end gently instead of
    # snapping between constant speeds.  Monotonicity is preserved.
    ease = float(np.clip(config.transition_ease, 0.0, 1.0))
    if ease > 0:
        smooth = progress * progress * (3.0 - 2.0 * progress)
        progress = (1.0 - ease) * progress + ease * smooth

    # ------------------------------------------------------------------
    # Flow-mode schedules
    # ------------------------------------------------------------------
    # Transition velocity in progress-per-second.  A uniform 10-second
    # transition moves at 0.1/s; short or audio-accelerated transitions
    # move much faster and the image needs more denoise to keep up.
    dprog = np.diff(progress, prepend=progress[0])
    velocity = _smooth(np.clip(dprog, 0.0, None) * fps, max(int(fps // 2), 1))
    # 0.5 progress/s (a ~2-second full scene change) counts as "fast".
    velocity_norm = np.clip(velocity / 0.5, 0.0, 1.0)

    # Denoise strength: energy drives morph speed, strong onsets add a
    # kick, and prompt velocity stops scenes lagging behind the prompt.
    s_min, s_max = config.flow_strength_min, config.flow_strength_max
    drive = np.clip(
        0.8 * energy
        + 0.4 * np.clip(onset - 0.5, 0.0, None) * 2.0
        + config.flow_velocity_influence * velocity_norm,
        0.0, 1.0,
    )
    strength = s_min + (s_max - s_min) * drive

    # Frame-rate compensation: strengths are defined at the reference
    # fps.  At higher frame rates each frame changes proportionally
    # less, so the morph speed per second (and the smoothness gain of
    # extra frames) is real rather than just a faster boil.
    ref = max(config.morph_rate_reference_fps, 1)
    fps_factor = float(np.clip(ref / fps, 0.4, 1.5))
    strength = np.clip(strength * fps_factor, 0.10, 0.75)

    # Colour anchoring: hold the palette steady within a scene, but
    # relax while the prompt is moving fast so the look can follow the
    # journey (snow is allowed to become savanna).
    color_anchor = 1.0 - 0.7 * velocity_norm

    # Camera: energy drives zoom rate, spectral brightness steers the
    # pan, and each beat lands a momentary zoom kick.
    z_min, z_max = config.flow_zoom_min, config.flow_zoom_max
    zoom = (z_min + (z_max - z_min) * energy) * (
        1.0 + config.beat_zoom_pulse * beat_prox
    )

    pan_window = max(int(fps), 1)
    pan_x = _smooth((centroid - 0.5) * 2.0 * config.flow_pan_max, pan_window)
    pan_y = _smooth((onset - 0.5) * 2.0 * config.flow_pan_max * 0.5, pan_window)

    return FramePlan(
        times=times,
        pair_index=pair_index,
        progress=progress,
        velocity=velocity,
        strength=strength,
        color_anchor=color_anchor,
        zoom=zoom,
        pan_x=pan_x,
        pan_y=pan_y,
        energy=energy,
        onset=onset,
        centroid=centroid,
        section_labels=section_labels,
    )
