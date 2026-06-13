"""Tests for the precomputed audio-reactive frame plan."""

from __future__ import annotations

import numpy as np
import pytest

from rufus.audio import AudioProfile, Section
from rufus.config import GenerationConfig
from rufus.prompts import Keyframe, PromptTimeline
from rufus.reactivity import build_plan


def _make_profile(duration: float = 30.0, sr: int = 22050) -> AudioProfile:
    """Synthetic audio profile: energy ramps 0 -> 1, beats every 0.5s."""
    n = int(duration * 43)  # ~43 analysis frames/sec at hop 512
    frame_times = np.linspace(0, duration, n, endpoint=False)
    rising = np.linspace(0, 1, n)
    onset = (np.sin(frame_times * 4.0) * 0.5 + 0.5)
    return AudioProfile(
        path="synthetic",
        duration=duration,
        sample_rate=sr,
        tempo=120.0,
        beat_times=np.arange(0, duration, 0.5),
        frame_times=frame_times,
        onset_envelope=onset,
        rms_energy=rising,
        spectral_centroid=rising[::-1].copy(),
        sections=[Section("calm", 0, duration / 2, 0.2),
                  Section("peak", duration / 2, duration, 0.8)],
    )


def _make_timeline() -> PromptTimeline:
    return PromptTimeline([
        Keyframe(time=0, prompt="desert"),
        Keyframe(time=10, prompt="mountains"),
        Keyframe(time=20, prompt="forest"),
    ])


@pytest.fixture
def plan():
    audio = _make_profile()
    cfg = GenerationConfig(device="cpu", fps=12)
    total = int(audio.duration * cfg.fps)
    return build_plan(_make_timeline(), audio, cfg, total)


class TestFramePlan:
    def test_shapes(self, plan):
        n = len(plan)
        for arr in (plan.times, plan.pair_index, plan.progress, plan.strength,
                    plan.zoom, plan.pan_x, plan.pan_y, plan.energy,
                    plan.onset, plan.centroid):
            assert len(arr) == n
        assert len(plan.section_labels) == n

    def test_progress_in_range(self, plan):
        assert plan.progress.min() >= 0.0
        assert plan.progress.max() <= 1.0

    def test_progress_monotonic_within_transitions(self, plan):
        """The old implementation could move progress backwards; the
        plan must be non-decreasing inside every transition span."""
        for pair in np.unique(plan.pair_index):
            mask = plan.pair_index == pair
            p = plan.progress[mask]
            assert (np.diff(p) >= -1e-12).all(), f"progress regressed in pair {pair}"

    def test_progress_starts_at_zero_per_transition(self, plan):
        for pair in np.unique(plan.pair_index):
            first = np.nonzero(plan.pair_index == pair)[0][0]
            assert plan.progress[first] < 0.05

    def test_tail_fully_blended(self, plan):
        """Frames after the last keyframe hold progress 1."""
        assert plan.progress[-1] == pytest.approx(1.0)
        assert plan.pair_index[-1] == 1  # last transition pair

    def test_energy_accelerates_progress(self):
        """With rising energy, the second half of a transition should
        cover more progress than the first half."""
        audio = _make_profile()
        cfg = GenerationConfig(device="cpu", fps=12, energy_rate_influence=3.0)
        total = int(audio.duration * cfg.fps)
        plan = build_plan(_make_timeline(), audio, cfg, total)

        span = np.nonzero(plan.pair_index == 0)[0]
        half = len(span) // 2
        first_half = plan.progress[span[half]] - plan.progress[span[0]]
        second_half = plan.progress[span[-1]] - plan.progress[span[half]]
        assert second_half > first_half

    def test_strength_within_bounds(self, plan):
        cfg = GenerationConfig(device="cpu")
        assert plan.strength.min() >= cfg.flow_strength_min - 1e-9
        assert plan.strength.max() <= cfg.flow_strength_max + 1e-9

    def test_strength_tracks_energy(self, plan):
        """Rising energy must produce higher strength at the end."""
        assert plan.strength[-1] > plan.strength[0]

    def test_zoom_within_bounds(self, plan):
        cfg = GenerationConfig(device="cpu")
        assert plan.zoom.min() >= cfg.flow_zoom_min - 1e-9
        # Beats may add a momentary pulse on top of the energy zoom.
        ceiling = cfg.flow_zoom_max * (1.0 + cfg.beat_zoom_pulse)
        assert plan.zoom.max() <= ceiling + 1e-9

    def test_beat_zoom_pulse(self):
        """Frames on a beat must zoom harder than identical frames with
        the pulse disabled."""
        audio = _make_profile()
        total = int(audio.duration * 12)
        on = build_plan(_make_timeline(), audio,
                        GenerationConfig(device="cpu", fps=12, beat_zoom_pulse=0.02),
                        total)
        off = build_plan(_make_timeline(), audio,
                         GenerationConfig(device="cpu", fps=12, beat_zoom_pulse=0.0),
                         total)
        assert on.zoom.max() > off.zoom.max()
        # Away from beats the schedules agree.
        far = np.argmin(np.abs(on.zoom - off.zoom))
        assert abs(on.zoom[far] - off.zoom[far]) < 1e-9

    def test_velocity_reflects_transition_speed(self):
        """A 2-second transition must register much higher velocity than
        a 10-second one (same audio)."""
        audio = _make_profile(duration=20.0)
        cfg = GenerationConfig(device="cpu", fps=12)
        total = int(audio.duration * cfg.fps)
        tl = PromptTimeline([
            Keyframe(time=0, prompt="a"),
            Keyframe(time=2, prompt="b"),    # fast transition
            Keyframe(time=12, prompt="c"),   # slow transition
        ])
        plan = build_plan(tl, audio, cfg, total)
        fast = plan.velocity[plan.pair_index == 0].mean()
        slow = plan.velocity[plan.pair_index == 1].mean()
        assert fast > slow * 2

    def test_velocity_raises_strength(self):
        """Fast transitions need extra denoise even in quiet audio, so
        the image keeps up with the moving prompt target."""
        audio = _make_profile(duration=20.0)
        cfg = GenerationConfig(device="cpu", fps=12)
        total = int(audio.duration * cfg.fps)
        fast_tl = PromptTimeline([Keyframe(time=0, prompt="a"),
                                  Keyframe(time=2, prompt="b")])
        slow_tl = PromptTimeline([Keyframe(time=0, prompt="a"),
                                  Keyframe(time=18, prompt="b")])
        fast_plan = build_plan(fast_tl, audio, cfg, total)
        slow_plan = build_plan(slow_tl, audio, cfg, total)
        # Compare within the first 2 seconds (identical audio there).
        window = slice(0, 2 * cfg.fps)
        assert fast_plan.strength[window].mean() > slow_plan.strength[window].mean()

    def test_color_anchor_relaxes_during_fast_transitions(self):
        audio = _make_profile(duration=20.0)
        cfg = GenerationConfig(device="cpu", fps=12)
        total = int(audio.duration * cfg.fps)
        tl = PromptTimeline([Keyframe(time=0, prompt="a"),
                             Keyframe(time=2, prompt="b")])
        plan = build_plan(tl, audio, cfg, total)
        assert (plan.color_anchor >= 0.0).all()
        assert (plan.color_anchor <= 1.0).all()
        during = plan.color_anchor[: 2 * cfg.fps].mean()
        after = plan.color_anchor[10 * cfg.fps :].mean()
        assert during < after  # anchor relaxed while transitioning

    def test_two_keyframes_with_tail(self):
        """Regression: with exactly two keyframes the single pair is both
        first and last; tail frames after the last keyframe must hold
        progress 1 rather than regress to 0."""
        audio = _make_profile(duration=8.0)
        cfg = GenerationConfig(device="cpu", fps=8)
        total = int(audio.duration * cfg.fps)
        tl = PromptTimeline([Keyframe(time=0, prompt="a"),
                             Keyframe(time=4, prompt="b")])
        plan = build_plan(tl, audio, cfg, total)
        assert (np.diff(plan.progress) >= -1e-12).all()
        assert plan.progress[-1] == pytest.approx(1.0)

    def test_single_keyframe(self):
        audio = _make_profile()
        cfg = GenerationConfig(device="cpu", fps=12)
        total = int(audio.duration * cfg.fps)
        tl = PromptTimeline([Keyframe(time=0, prompt="only")])
        plan = build_plan(tl, audio, cfg, total)
        assert (plan.pair_index == 0).all()
        assert (plan.progress == 0.0).all()

    def test_transition_ease(self):
        """Eased progress must start more gently than linear while
        staying monotonic with identical endpoints."""
        audio = _make_profile()
        total = int(audio.duration * 12)
        eased = build_plan(_make_timeline(), audio,
                           GenerationConfig(device="cpu", fps=12,
                                            transition_ease=1.0, keyframe_hold=0.0),
                           total)
        linear = build_plan(_make_timeline(), audio,
                            GenerationConfig(device="cpu", fps=12,
                                             transition_ease=0.0, keyframe_hold=0.0),
                            total)
        span = np.nonzero(eased.pair_index == 0)[0]
        quarter = span[len(span) // 4]
        # Early in the transition, eased progress lags linear.
        assert eased.progress[quarter] < linear.progress[quarter]
        # Monotonic and endpoint-preserving.
        assert (np.diff(eased.progress[span]) >= -1e-12).all()
        assert eased.progress[span[0]] == pytest.approx(0.0, abs=1e-6)
        assert eased.progress[-1] == pytest.approx(1.0)

    def test_strength_scales_down_with_fps(self):
        """Doubling the frame rate must roughly halve per-frame strength,
        so extra frames mean smaller steps rather than a faster morph."""
        audio = _make_profile()
        low = build_plan(_make_timeline(), audio,
                         GenerationConfig(device="cpu", fps=12),
                         int(audio.duration * 12))
        high = build_plan(_make_timeline(), audio,
                          GenerationConfig(device="cpu", fps=24),
                          int(audio.duration * 24))
        ratio = high.strength.mean() / low.strength.mean()
        assert 0.4 < ratio < 0.7
        assert high.strength.min() >= 0.10  # absolute floor holds

    def test_keyframe_hold(self):
        """With a hold, each scene rests (progress 0) for the early part
        of its span before morphing; without one it morphs immediately."""
        audio = _make_profile()
        total = int(audio.duration * 12)
        held = build_plan(_make_timeline(), audio,
                          GenerationConfig(device="cpu", fps=12,
                                           keyframe_hold=0.4, transition_ease=0.0),
                          total)
        unheld = build_plan(_make_timeline(), audio,
                            GenerationConfig(device="cpu", fps=12,
                                             keyframe_hold=0.0, transition_ease=0.0),
                            total)
        span = np.nonzero(held.pair_index == 0)[0]
        early = span[: len(span) // 5]
        # Early frames rest at 0 with the hold, but have moved without it.
        assert (held.progress[early] == 0.0).all()
        assert unheld.progress[early[-1]] > 0.0
        # The morph still completes.
        assert held.progress[-1] == pytest.approx(1.0)

    def test_deterministic(self):
        """Same inputs -> identical plan (resume relies on this)."""
        audio = _make_profile()
        cfg = GenerationConfig(device="cpu", fps=12)
        total = int(audio.duration * cfg.fps)
        p1 = build_plan(_make_timeline(), audio, cfg, total)
        p2 = build_plan(_make_timeline(), audio, cfg, total)
        assert np.array_equal(p1.progress, p2.progress)
        assert np.array_equal(p1.strength, p2.strength)
