from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Keyframe:
    """A single visual keyframe tied to a point in the audio timeline.

    A keyframe needs a *prompt*, an *image*, or both:

    * prompt only: the scene is generated from text (classic mode).
    * image only: the video is pulled toward this photo; frames near
      this keyframe resemble it.
    * both: the photo anchors the content while the prompt steers the
      style and guides the diffusion.

    Parameters
    ----------
    time:
        Position in seconds where this keyframe should be fully realised.
    prompt:
        The positive text prompt describing the scene.
    image:
        Optional path to an image (photo, artwork) for this keyframe.
    negative_prompt:
        Optional per-keyframe negative prompt.  Falls back to the global
        negative prompt from :class:`~rufus.config.GenerationConfig` when
        ``None``.
    style:
        Optional style suffix appended to the prompt (e.g.
        ``"cinematic, 8k, dramatic lighting"``).
    """

    time: float
    prompt: str = ""
    image: Optional[str] = None
    negative_prompt: Optional[str] = None
    style: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.prompt and not self.image:
            raise ValueError(
                f"Keyframe at t={self.time} needs a prompt, an image, or both."
            )

    @property
    def full_prompt(self) -> str:
        if self.style and self.prompt:
            return f"{self.prompt}, {self.style}"
        return self.prompt

    @property
    def label(self) -> str:
        """Display name: the prompt, or the image filename."""
        if self.full_prompt:
            return self.full_prompt
        from pathlib import Path as _P

        return f"[{_P(self.image).name}]"


class PromptTimeline:
    """Maps :class:`Keyframe` objects onto a continuous timeline and
    resolves which pair of keyframes is active at any moment, together
    with the interpolation progress between them.

    Keyframes are kept sorted by time.  Queries before the first keyframe
    return the first keyframe with progress 0; queries after the last
    return the last keyframe with progress 1.
    """

    def __init__(self, keyframes: List[Keyframe]) -> None:
        if not keyframes:
            raise ValueError("At least one keyframe is required.")
        self._keyframes = sorted(keyframes, key=lambda k: k.time)
        self._times = [k.time for k in self._keyframes]

    @classmethod
    def from_dicts(cls, items: List[Dict]) -> "PromptTimeline":
        """Build a timeline from a list of plain dictionaries.

        Each dict must contain ``time`` plus ``prompt`` and/or
        ``image``.  Optional keys: ``negative_prompt``, ``style``.
        """
        keyframes = [Keyframe(**item) for item in items]
        return cls(keyframes)

    @property
    def keyframes(self) -> List[Keyframe]:
        return list(self._keyframes)

    @property
    def start_time(self) -> float:
        return self._keyframes[0].time

    @property
    def end_time(self) -> float:
        return self._keyframes[-1].time

    def __len__(self) -> int:
        return len(self._keyframes)

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def at(self, t: float) -> Tuple[Keyframe, Keyframe, float]:
        """Return ``(kf_a, kf_b, progress)`` for time *t*.

        *progress* is in ``[0, 1]`` where 0 means fully *kf_a* and 1
        means fully *kf_b*.

        If there is only one keyframe, both *kf_a* and *kf_b* point to it
        and progress is 0.
        """
        if len(self._keyframes) == 1:
            kf = self._keyframes[0]
            return (kf, kf, 0.0)

        # Before or at first keyframe
        if t <= self._times[0]:
            return (self._keyframes[0], self._keyframes[1], 0.0)

        # After or at last keyframe
        if t >= self._times[-1]:
            return (self._keyframes[-2], self._keyframes[-1], 1.0)

        idx = bisect.bisect_right(self._times, t) - 1
        idx = min(idx, len(self._keyframes) - 2)

        kf_a = self._keyframes[idx]
        kf_b = self._keyframes[idx + 1]

        span = kf_b.time - kf_a.time
        if span < 1e-9:
            progress = 1.0
        else:
            progress = (t - kf_a.time) / span

        return (kf_a, kf_b, float(progress))

    def transition_pairs(self) -> List[Tuple[Keyframe, Keyframe]]:
        """Return all consecutive ``(kf_a, kf_b)`` transition pairs."""
        return [
            (self._keyframes[i], self._keyframes[i + 1])
            for i in range(len(self._keyframes) - 1)
        ]
