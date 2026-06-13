"""High-level pipeline that orchestrates the full Rufus workflow.

Usage::

    from rufus import RufusPipeline

    pipeline = RufusPipeline(model="stabilityai/stable-diffusion-xl-base-1.0")
    pipeline.generate(
        audio_path="song.mp3",
        keyframes=[
            {"time": 0, "prompt": "vast desert dunes under purple twilight sky"},
            {"time": 30, "prompt": "towering mountain range with cascading waterfalls"},
        ],
        output_path="output.mp4",
    )
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Union

from .audio import analyse as analyse_audio, suggest_keyframe_times
from .config import GenerationConfig
from .display import Dashboard, is_available as _dashboard_available
from .generator import FrameGenerator
from .prompts import Keyframe, PromptTimeline
from .video import assemble as assemble_video

logger = logging.getLogger(__name__)


class RufusPipeline:
    """End-to-end pipeline: audio analysis -> frame generation -> video assembly.

    Parameters
    ----------
    model:
        HuggingFace model ID or local path to a diffusers-compatible
        checkpoint.  Defaults to SDXL base.
    device:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  When left empty the
        best available device is auto-detected (CUDA > MPS > CPU).
    config:
        An explicit :class:`GenerationConfig`.  If provided, *model*
        and *device* are taken from the config and the constructor
        arguments are ignored.
    """

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "",
        config: Optional[GenerationConfig] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = GenerationConfig(model=model, device=device)

        self._generator: Optional[FrameGenerator] = None

    # ------------------------------------------------------------------
    # Lazy model access
    # ------------------------------------------------------------------

    @property
    def generator(self) -> FrameGenerator:
        if self._generator is None:
            self._generator = FrameGenerator(self.config)
        return self._generator

    def preload_model(self) -> None:
        """Eagerly load the diffusion model into memory.

        Call this if you want to pay the load cost up-front rather than
        on the first ``generate()`` call.
        """
        self.generator.load_model()

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------

    def generate(
        self,
        audio_path: str,
        keyframes: Union[List[Dict], List[Keyframe]],
        output_path: str = "output.mp4",
        *,
        fps: Optional[int] = None,
        resolution: Optional[tuple] = None,
        seed: Optional[int] = None,
        duration: Optional[float] = None,
        init_image: Optional[str] = None,
        frame_dir: Optional[str] = None,
        keep_frames: bool = False,
        use_dashboard: bool = False,
    ) -> Path:
        """Run the full generation pipeline.

        Parameters
        ----------
        audio_path:
            Path to the input audio file.
        keyframes:
            Landscape keyframes as :class:`Keyframe` objects or plain
            dicts with ``time`` and ``prompt`` keys (plus optional
            ``negative_prompt`` and ``style``).
        output_path:
            Where to write the final ``.mp4``.
        fps:
            Override the configured frames-per-second.
        resolution:
            Override the configured ``(width, height)``.
        seed:
            Random seed for reproducible generation.
        duration:
            Only render the first *duration* seconds of the track.
        init_image:
            Optional path to an image (e.g. a photo) used as the
            starting frame in flow mode.  The video begins from this
            image stylised toward the first prompt, then morphs through
            the keyframe journey.
        frame_dir:
            Directory to store intermediate frame images.  When
            ``None`` a temporary directory is used.
        keep_frames:
            If ``True`` the frame directory is preserved after video
            assembly.  Ignored when *frame_dir* is explicitly provided
            (frames are always kept in that case).
        use_dashboard:
            If ``True`` (and the ``rich`` package is installed), show a
            live terminal dashboard with progress, audio meters, and
            transition details instead of a plain tqdm bar.

        Returns
        -------
        Path
            The path to the finished video file.
        """
        # Apply per-call overrides
        cfg = self.config
        if fps is not None:
            cfg = replace(cfg, fps=fps)
        if resolution is not None:
            cfg = replace(cfg, width=resolution[0], height=resolution[1])
        if seed is not None:
            cfg = replace(cfg, seed=seed)
        self.generator.config = cfg

        # 1. Analyse audio
        logger.info("Analysing audio: %s", audio_path)
        audio_profile = analyse_audio(
            audio_path,
            energy_smoothing=cfg.energy_smoothing_window,
        )
        logger.info(
            "Audio: %.1fs, %.0f BPM, %d beats detected",
            audio_profile.duration,
            audio_profile.tempo,
            len(audio_profile.beat_times),
        )

        effective_duration = min(
            duration or audio_profile.duration, audio_profile.duration
        )

        # 2. Build prompt timeline
        if keyframes and isinstance(keyframes[0], dict):
            untimed = [kf for kf in keyframes if "time" not in kf]
            if untimed and len(untimed) != len(keyframes):
                raise ValueError(
                    "Either give every keyframe a 'time' or none "
                    "(auto-placement assigns all of them)."
                )
            if untimed:
                # With looping, reserve the final span for the return
                # to the first keyframe.
                placement_window = effective_duration
                if cfg.loop:
                    n = len(keyframes)
                    placement_window = effective_duration * n / (n + 1)
                times = suggest_keyframe_times(
                    audio_profile, len(keyframes), placement_window
                )
                keyframes = [
                    {**kf, "time": t} for kf, t in zip(keyframes, times)
                ]
                logger.info(
                    "Auto-placed keyframes at: %s",
                    ", ".join(f"{t:.1f}s" for t in times),
                )
            timeline = PromptTimeline.from_dicts(keyframes)
        else:
            timeline = PromptTimeline(keyframes)

        # Looping: the journey returns to the first keyframe at the end.
        if cfg.loop and timeline.end_time < effective_duration:
            from dataclasses import replace as _replace

            return_kf = _replace(timeline.keyframes[0], time=effective_duration)
            timeline = PromptTimeline(timeline.keyframes + [return_kf])
            logger.info("Loop: returning to the first keyframe at %.1fs",
                        effective_duration)
        logger.info(
            "Timeline: %d keyframes spanning %.1fs - %.1fs",
            len(timeline),
            timeline.start_time,
            timeline.end_time,
        )

        # Fail fast on missing keyframe images, before the model loads.
        for kf in timeline.keyframes:
            if kf.image and not Path(kf.image).is_file():
                raise FileNotFoundError(
                    f"Keyframe image not found: {kf.image} (t={kf.time})"
                )

        # 3. Render frames
        use_temp = frame_dir is None
        if use_temp:
            frame_dir_path = Path(
                cfg.frame_cache_dir
                or tempfile.mkdtemp(prefix="rufus_frames_")
            )
        else:
            frame_dir_path = Path(frame_dir)

        dashboard = None
        if use_dashboard and _dashboard_available():
            dashboard = Dashboard()
        elif use_dashboard:
            logger.warning(
                "Dashboard requested but 'rich' is not installed. "
                "Falling back to tqdm. Install with: pip install rich"
            )

        init_img = None
        if init_image is not None:
            from PIL import Image as _PILImage

            init_img = _PILImage.open(init_image).convert("RGB")
            logger.info("Starting from init image: %s", init_image)

        frame_paths = self.generator.render_sequence(
            timeline,
            audio_profile,
            frame_dir_path,
            seed=cfg.seed,
            dashboard=dashboard,
            duration=duration,
            init_image=init_img,
        )

        # 4. Assemble video
        result = assemble_video(
            frame_paths,
            audio_path,
            output_path,
            cfg,
            duration=effective_duration,
        )

        # 5. Clean up temp frames
        if use_temp and not keep_frames:
            shutil.rmtree(frame_dir_path, ignore_errors=True)
            logger.info("Cleaned up temporary frame directory.")

        logger.info("Generation complete: %s", result)
        return result

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release GPU memory held by the diffusion model."""
        if self._generator is not None:
            self._generator.unload_model()
            self._generator = None
