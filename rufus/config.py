from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def _detect_device() -> str:
    """Return the best available torch device string.

    Detection order: NVIDIA CUDA, Apple Metal (MPS), Intel XPU, then
    CPU.  Anything else falls back to CPU.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def describe_device(device: str) -> str:
    """Human-readable description of what a device string maps to."""
    try:
        import torch
    except ImportError:
        return "CPU (torch not installed)"

    if device == "cuda" and torch.cuda.is_available():
        return f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "Apple Silicon (Metal/MPS)"
    if device == "xpu":
        return "Intel XPU"
    return "CPU"


def _default_dtype_for_device(device: str) -> str:
    """Choose a safe default dtype based on the device.

    CUDA (NVIDIA/ROCm) and Intel XPU use float16 for speed.  MPS
    supports float16 for most operations but some SDXL components hit
    dtype mismatches there, so we default to float32 for reliability.
    CPU stays float32 (fp16 on CPU is slower, not faster).
    """
    if device in ("cuda", "xpu"):
        return "float16"
    return "float32"


# Per-frame inference defaults applied when a fast mode is selected and
# the user has not overridden steps/guidance themselves.
_FAST_MODE_DEFAULTS = {
    "lcm": {"num_inference_steps": 6, "guidance_scale": 1.5},
    "lightning": {"num_inference_steps": 4, "guidance_scale": 0.0},
}

_DEFAULT_STEPS = 20
_DEFAULT_GUIDANCE = 7.5


@dataclass
class GenerationConfig:
    """Centralised configuration for the Rufus video generation pipeline."""

    # --- Model -----------------------------------------------------------
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = ""
    torch_dtype: str = ""
    # Optional distilled-inference mode: "lcm" (LCM-LoRA) or "lightning"
    # (SDXL-Lightning LoRA).  Cuts steps-per-frame from ~20 to 4-8.
    fast_mode: Optional[str] = None

    # --- Rendering mode ----------------------------------------------------
    # "flow"  is an img2img feedback loop: every frame is derived from
    #         the previous one (temporally coherent, Deforum-style).
    # "morph" renders each frame independently from interpolated
    #         embeddings/latents (legacy; flickers, kept for comparison).
    mode: str = "flow"

    # --- Resolution & frames ---------------------------------------------
    width: int = 768
    height: int = 768
    fps: int = 12

    # --- Diffusion -------------------------------------------------------
    num_inference_steps: int = _DEFAULT_STEPS
    guidance_scale: float = _DEFAULT_GUIDANCE
    negative_prompt: str = (
        "blurry, low quality, distorted, watermark, text, "
        "poorly drawn, cartoon, disfigured, bad art, deformed"
    )

    # --- Flow mode (feedback loop) -----------------------------------------
    # img2img denoise strength range; audio energy maps calm -> peak.
    # These are defined at `morph_rate_reference_fps` and scaled for the
    # actual frame rate, so higher fps = less change per frame (smoother)
    # while the morph speed per *second* stays the same.
    flow_strength_min: float = 0.30
    flow_strength_max: float = 0.55
    morph_rate_reference_fps: int = 12
    # Noise used for re-diffusion: "fixed" reuses the same noise every
    # frame (stable textures, far less shimmer); "varying" draws fresh
    # noise per frame (livelier but visibly boiling).
    temporal_noise: str = "fixed"
    # How much fast prompt transitions raise denoise strength so the
    # image keeps up with the moving embedding target (0 disables).
    flow_velocity_influence: float = 0.6
    # Subtle unsharp mask applied each frame before re-diffusion to
    # counteract the feedback loop's progressive softening (0 disables).
    sharpen_amount: float = 0.3
    # Per-frame multiplicative zoom factor range (compounds over time).
    flow_zoom_min: float = 1.002
    flow_zoom_max: float = 1.012
    # Maximum per-frame pan drift in pixels.
    flow_pan_max: float = 4.0
    # Momentary zoom kick on each beat (0 disables).
    beat_zoom_pulse: float = 0.006
    # Depth-aware parallax warp instead of flat zoom (downloads a small
    # depth-estimation model on first use).
    enable_depth_warp: bool = False
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    # 0 = flat zoom, 1 = motion fully scaled by depth.
    depth_parallax: float = 0.6
    # Match each frame's colour statistics to a rolling reference to
    # prevent the feedback loop drifting (magenta shift / blow-out).
    color_coherence: bool = True
    color_coherence_decay: float = 0.98
    # When an init image is supplied, how strongly frame 0 is re-diffused
    # toward the first prompt (0 = keep the photo, 1 = ignore it).
    init_image_strength: float = 0.5
    # Image keyframes: per-frame blend toward the cross-dissolved target
    # photos before re-diffusion.  Higher = tighter tracking of the
    # photos but more ghosting for the diffusion to clean up.
    image_pull: float = 0.22
    # IP-Adapter: feed keyframe photos' CLIP embeddings directly into
    # the diffusion cross-attention (much stronger image guidance than
    # captions alone).  Loaded only when image keyframes are present.
    # Keep the scale moderate: in a feedback loop strong image guidance
    # overrides transition progress and amplifies texture artifacts.
    enable_ip_adapter: bool = True
    ip_adapter_scale: float = 0.35
    # Auto-caption image keyframes that have no text prompt (BLIP).
    # Without a content-bearing prompt the diffusion has no idea what
    # the photo blend depicts and the morph wanders off-subject.
    auto_caption: bool = True
    caption_model: str = "Salesforce/blip-image-captioning-base"
    # Fallback guidance prompt when auto-captioning is disabled.
    image_keyframe_prompt: str = (
        "a photograph, high quality, detailed, natural light"
    )

    # --- Audio reactivity ----------------------------------------------------
    energy_smoothing_window: int = 5
    onset_sensitivity: float = 1.0
    beat_transition_boost: float = 1.5
    # How strongly audio energy accelerates transition progress.
    energy_rate_influence: float = 1.5
    # Smoothstep easing on transition progress: 0 = linear, 1 = fully
    # eased (transitions start and end gently).
    transition_ease: float = 1.0
    # Fraction of each keyframe span spent *holding* on the arrived
    # scene before the morph to the next one begins.  0 = always
    # morphing (no rest), 0.35 = scene rests for a third of its span.
    keyframe_hold: float = 0.35

    # --- Morph mode 2D motion (legacy) -------------------------------------
    enable_motion: bool = True
    max_zoom: float = 1.04
    max_pan_x: float = 10.0
    max_pan_y: float = 6.0
    motion_intensity: float = 0.3
    slerp_dot_threshold: float = 0.9995

    # --- Memory optimisations --------------------------------------------
    enable_cpu_offload: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_xformers: bool = False

    # --- Looping -----------------------------------------------------------
    # Seamless loop: the prompt journey returns to the first keyframe
    # and the closing frames converge back onto frame 0.
    loop: bool = False
    # How many final seconds blend back toward the opening frame.
    loop_blend_seconds: float = 1.5

    # --- Output & checkpointing ------------------------------------------
    frame_cache_dir: Optional[str] = None
    resume: bool = True
    video_codec: str = "libx264"
    video_bitrate: str = "8M"
    # If set (and higher than fps), the final video is motion-interpolated
    # up to this frame rate with ffmpeg's minterpolate filter.
    smooth_fps: Optional[int] = None

    # --- Seed ------------------------------------------------------------
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.device:
            self.device = _detect_device()
        if not self.torch_dtype:
            self.torch_dtype = _default_dtype_for_device(self.device)

        if self.mode not in ("flow", "morph"):
            raise ValueError(f"Unknown mode: {self.mode!r} (use 'flow' or 'morph')")
        if self.temporal_noise not in ("fixed", "varying"):
            raise ValueError(
                f"Unknown temporal_noise: {self.temporal_noise!r} "
                "(use 'fixed' or 'varying')"
            )
        if self.fast_mode is not None and self.fast_mode not in _FAST_MODE_DEFAULTS:
            raise ValueError(
                f"Unknown fast_mode: {self.fast_mode!r} (use 'lcm' or 'lightning')"
            )

        # Fast modes need far fewer steps and little/no CFG.  Only apply
        # when the user left the stock defaults in place.
        if self.fast_mode:
            fast = _FAST_MODE_DEFAULTS[self.fast_mode]
            if self.num_inference_steps == _DEFAULT_STEPS:
                self.num_inference_steps = fast["num_inference_steps"]
            if self.guidance_scale == _DEFAULT_GUIDANCE:
                self.guidance_scale = fast["guidance_scale"]

    # --- Convenience presets ---------------------------------------------

    @classmethod
    def preview(cls, **overrides) -> "GenerationConfig":
        """Fast, low-resolution preset for quick iteration."""
        defaults = dict(
            width=512,
            height=512,
            fps=8,
            num_inference_steps=12,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def production(cls, **overrides) -> "GenerationConfig":
        """High-quality preset for final renders."""
        defaults = dict(
            width=1024,
            height=1024,
            fps=24,
            num_inference_steps=30,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)
