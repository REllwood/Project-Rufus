"""Core frame generation engine.

Two rendering modes:

* **flow** (default) is a Deforum-style img2img feedback loop.  Every
  frame is derived from the previous one: the last frame is warped by a
  small audio-driven camera move, partially re-noised, and re-diffused
  with the current interpolated prompt embeddings.  Audio energy drives
  the denoise strength (morph speed), giving temporally coherent
  terrain that *flows* between scenes.

* **morph** is the legacy approach: each frame is rendered
  independently from SLERP-interpolated embeddings and latents.  Kept
  for comparison; expect flicker.

Models are loaded through diffusers' ``AutoPipeline``, so both
SDXL-class checkpoints (two text encoders, pooled embeddings) and
SD 1.5-class checkpoints (single text encoder, much lighter, the
practical choice for CPU or low-VRAM GPUs) work.  Prompt encoding uses
the pipeline's own ``encode_prompt`` and adapts to whichever embedding
set the model family returns.

Hardware: NVIDIA (CUDA), AMD (ROCm, exposed as ``cuda``), Apple
Silicon (MPS), Intel (XPU), and plain CPU.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from .audio import AudioProfile
from .config import GenerationConfig
from .display import Dashboard, FrameStatus
from .interpolation import lerp, slerp
from .motion import ColorStabiliser, DepthWarper, sharpen, zoom_pan
from .prompts import Keyframe, PromptTimeline
from .reactivity import FramePlan, build_plan

logger = logging.getLogger(__name__)

_FRAME_RE = re.compile(r"frame_(\d+)\.png$")

# (prompt_embeds, negative_embeds, pooled_embeds, negative_pooled_embeds)
# The pooled slots are None for single-encoder (SD 1.5-class) models.
EmbeddingSet = Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
]


class _EmbeddingCache:
    """Caches prompt embeddings so each unique prompt is encoded once."""

    def __init__(self):
        self._cache: Dict[str, EmbeddingSet] = {}

    def get_or_encode(self, prompt: str, negative_prompt: str, encode_fn) -> EmbeddingSet:
        key = f"{prompt}||{negative_prompt}"
        if key not in self._cache:
            self._cache[key] = encode_fn(prompt, negative_prompt)
        return self._cache[key]


class FrameGenerator:
    """Renders video frames from the diffusion model.

    Owns the model lifecycle (loading, memory optimisation, fast-mode
    LoRAs) and exposes :meth:`render_sequence`, which the pipeline calls
    with a prompt timeline and an audio profile.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._pipe = None
        self._img2img = None
        self._embedding_cache = _EmbeddingCache()
        self._depth_warper: Optional[DepthWarper] = None
        self._device = config.device
        self._dtype = getattr(torch, config.torch_dtype, torch.float32)
        # Pluggable image captioner (PIL.Image -> str); lazily set to a
        # BLIP pipeline on first use, replaceable in tests.
        self._captioner = None
        self._ip_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @property
    def _is_sdxl(self) -> bool:
        """True for two-encoder (SDXL-class) pipelines."""
        return getattr(self._pipe, "text_encoder_2", None) is not None

    def load_model(self) -> None:
        """Load the diffusion model and apply memory optimisations.

        Uses ``AutoPipeline`` so any diffusers text-to-image checkpoint
        (SDXL, SD 1.5/2.x, ...) resolves to the right pipeline class.
        """
        from diffusers import (
            AutoPipelineForImage2Image,
            AutoPipelineForText2Image,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
            LCMScheduler,
        )

        from .config import describe_device

        cfg = self.config
        logger.info("Loading model: %s", cfg.model)
        logger.info(
            "Device: %s [%s]  dtype: %s  mode: %s",
            self._device, describe_device(self._device), self._dtype, cfg.mode,
        )

        # Prefer the fp16 weight variant when running half precision,
        # but fall back gracefully for checkpoints that don't ship one.
        try:
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                cfg.model,
                torch_dtype=self._dtype,
                variant="fp16" if self._dtype == torch.float16 else None,
            )
        except (ValueError, OSError):
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                cfg.model, torch_dtype=self._dtype
            )

        # SD1.5-class pipelines ship a safety checker that replaces
        # flagged frames with black images. In a feedback loop one
        # false positive corrupts every subsequent frame, so disable it.
        if getattr(self._pipe, "safety_checker", None) is not None:
            self._pipe.safety_checker = None
            logger.info("Safety checker disabled (incompatible with the feedback loop).")

        if self._device == "cpu" and self._is_sdxl:
            logger.warning(
                "SDXL on CPU is extremely slow (minutes per frame). Consider "
                "a lighter model, e.g. GenerationConfig(model="
                "'stable-diffusion-v1-5/stable-diffusion-v1-5', fast_mode='lcm')."
            )

        # Fast modes swap in a distilled LoRA + matching scheduler.
        fast_mode = cfg.fast_mode
        if fast_mode == "lightning" and not self._is_sdxl:
            logger.warning(
                "SDXL-Lightning only exists for SDXL models; "
                "falling back to LCM-LoRA for this checkpoint."
            )
            fast_mode = "lcm"

        if fast_mode == "lcm":
            lora = (
                "latent-consistency/lcm-lora-sdxl"
                if self._is_sdxl
                else "latent-consistency/lcm-lora-sdv1-5"
            )
            logger.info("Applying LCM-LoRA fast mode (%s)", lora)
            self._pipe.load_lora_weights(lora)
            self._pipe.fuse_lora()
            self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)
        elif fast_mode == "lightning":
            logger.info("Applying SDXL-Lightning fast mode")
            self._pipe.load_lora_weights(
                "ByteDance/SDXL-Lightning",
                weight_name="sdxl_lightning_4step_lora.safetensors",
            )
            self._pipe.fuse_lora()
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(
                self._pipe.scheduler.config, timestep_spacing="trailing"
            )
        else:
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipe.scheduler.config
            )

        is_mps = self._device == "mps"

        # Model CPU offload needs an accelerator that accelerate can
        # page against (CUDA/ROCm/XPU); on MPS/CPU move the whole model.
        if cfg.enable_cpu_offload and self._device in ("cuda", "xpu"):
            try:
                self._pipe.enable_model_cpu_offload()
            except Exception:
                logger.warning("CPU offload unavailable; loading fully on device.")
                self._pipe.to(self._device)
        else:
            self._pipe.to(self._device)

        if cfg.enable_attention_slicing or is_mps:
            self._pipe.enable_attention_slicing()
        if cfg.enable_vae_slicing:
            self._pipe.enable_vae_slicing()
        if cfg.enable_vae_tiling:
            self._pipe.enable_vae_tiling()
        # xformers needs CUDA (NVIDIA or ROCm builds).
        if cfg.enable_xformers and self._device == "cuda":
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.warning("xformers not available; skipping.")

        self._pipe.set_progress_bar_config(disable=True)

        # The img2img pipeline shares all weights with the base pipeline.
        if cfg.mode == "flow":
            self._img2img = AutoPipelineForImage2Image.from_pipe(self._pipe)
            self._img2img.set_progress_bar_config(disable=True)

        logger.info("Model loaded successfully.")

    def unload_model(self) -> None:
        """Release model from memory."""
        self._img2img = None
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def _execution_device(self):
        return getattr(self._pipe, "_execution_device", None) or self._pipe.device

    def _encode_prompt(self, prompt: str, negative_prompt: str) -> EmbeddingSet:
        """Encode a prompt pair via the pipeline's own encoder.

        SDXL-class pipelines return four tensors (sequence + pooled,
        positive + negative); SD 1.5-class pipelines return two.  The
        result is normalised to a 4-slot set with ``None`` pooled slots
        for single-encoder models.
        """
        with torch.inference_mode():
            result = self._pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=self._execution_device(),
                do_classifier_free_guidance=True,
                num_images_per_prompt=1,
            )

        if len(result) == 4:
            pos, neg, pool_pos, pool_neg = result
            return (pos.cpu(), neg.cpu(), pool_pos.cpu(), pool_neg.cpu())
        pos, neg = result
        return (pos.cpu(), neg.cpu(), None, None)

    def _caption_image(self, image: Image.Image) -> str:
        """Describe an image with the captioning model (lazy-loaded)."""
        if self._captioner is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            logger.info("Loading caption model: %s", self.config.caption_model)
            processor = BlipProcessor.from_pretrained(self.config.caption_model)
            model = BlipForConditionalGeneration.from_pretrained(
                self.config.caption_model
            )
            try:
                model = model.to(self._device)
                device = self._device
            except Exception:
                logger.warning("Caption model failed on %s; using CPU.", self._device)
                model, device = model.to("cpu"), "cpu"
            model.eval()

            def caption(img: Image.Image) -> str:
                with torch.inference_mode():
                    inputs = processor(img, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_new_tokens=30)
                return processor.decode(out[0], skip_special_tokens=True).strip()

            self._captioner = caption
        return self._captioner(image)

    # ------------------------------------------------------------------
    # IP-Adapter (image conditioning)
    # ------------------------------------------------------------------

    def _load_ip_adapter(self) -> None:
        """Load IP-Adapter weights so keyframe photos can condition the
        diffusion directly through CLIP image embeddings."""
        if self._ip_loaded:
            return
        repo = "h94/IP-Adapter"
        if self._is_sdxl:
            sub, weight = "sdxl_models", "ip-adapter_sdxl.safetensors"
        else:
            sub, weight = "models", "ip-adapter_sd15.safetensors"

        logger.info("Loading IP-Adapter (%s/%s)", sub, weight)
        # Attention slicing's processors can't host the IP-Adapter ones;
        # reset to defaults first and stay unsliced (slicing afterwards
        # would clobber the adapter).
        if hasattr(self._pipe, "unet") and hasattr(self._pipe.unet, "set_default_attn_processor"):
            self._pipe.unet.set_default_attn_processor()
            if self.config.enable_attention_slicing:
                logger.info("Attention slicing disabled for IP-Adapter compatibility.")
        try:
            self._pipe.load_ip_adapter(repo, subfolder=sub, weight_name=weight)
        except Exception:
            self._pipe.load_ip_adapter(
                repo, subfolder=sub, weight_name=weight.replace(".safetensors", ".bin")
            )
        self._pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)

        # Rebuild the img2img view so it shares the image encoder.
        try:
            from diffusers import AutoPipelineForImage2Image

            self._img2img = AutoPipelineForImage2Image.from_pipe(self._pipe)
            self._img2img.set_progress_bar_config(disable=True)
        except Exception:
            pass  # test doubles aren't from_pipe-able; they share state anyway
        self._ip_loaded = True

    def _ip_image_embed(self, image: Image.Image) -> torch.Tensor:
        """CLIP-encode a keyframe photo for IP-Adapter conditioning."""
        with torch.inference_mode():
            embeds = self._pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[image],
                ip_adapter_image_embeds=None,
                device=self._execution_device(),
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
        return embeds[0].cpu()

    def keyframe_prompt(self, keyframe: Keyframe) -> str:
        """Resolve the guidance prompt for a keyframe.

        Image-only keyframes are auto-captioned (when enabled) so the
        diffusion knows what the photo depicts. Without a
        content-bearing prompt the morph wanders off-subject.
        """
        if keyframe.full_prompt:
            return keyframe.full_prompt
        if keyframe.image and self.config.auto_caption:
            image = Image.open(keyframe.image).convert("RGB")
            caption = self._caption_image(image)
            logger.info("Auto-caption %s: %r", Path(keyframe.image).name, caption)
            return caption
        return self.config.image_keyframe_prompt

    def encode_keyframe(self, keyframe: Keyframe, fallback_negative: str) -> EmbeddingSet:
        """Encode a keyframe's prompts, using the cache."""
        prompt = self.keyframe_prompt(keyframe)
        neg = keyframe.negative_prompt or fallback_negative
        return self._embedding_cache.get_or_encode(prompt, neg, self._encode_prompt)

    @staticmethod
    def _interpolate_embeddings(a: EmbeddingSet, b: EmbeddingSet, t: float,
                                dot_threshold: float) -> EmbeddingSet:
        """SLERP the sequence embeddings, lerp the pooled ones (if any)."""
        if t <= 0.0:
            return a
        if t >= 1.0:
            return b
        seq_pos = slerp(a[0].squeeze(0), b[0].squeeze(0), t, dot_threshold).unsqueeze(0)
        seq_neg = slerp(a[1].squeeze(0), b[1].squeeze(0), t, dot_threshold).unsqueeze(0)
        pool_pos = lerp(a[2], b[2], t) if a[2] is not None else None
        pool_neg = lerp(a[3], b[3], t) if a[3] is not None else None
        return (seq_pos, seq_neg, pool_pos, pool_neg)

    def _embed_kwargs(self, embeds: EmbeddingSet, device) -> dict:
        """Build the pipeline kwargs for an embedding set, including the
        pooled tensors only for model families that produce them."""
        kwargs = {
            "prompt_embeds": embeds[0].to(device),
            "negative_prompt_embeds": embeds[1].to(device),
        }
        if embeds[2] is not None:
            kwargs["pooled_prompt_embeds"] = embeds[2].to(device)
            kwargs["negative_pooled_prompt_embeds"] = embeds[3].to(device)
        return kwargs

    # ------------------------------------------------------------------
    # Latents & single-frame rendering
    # ------------------------------------------------------------------

    def make_latents(self, seed: Optional[int] = None) -> torch.Tensor:
        """Create a random noise latent tensor for the current config."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        latent_channels = self._pipe.unet.config.in_channels
        h = self.config.height // 8
        w = self.config.width // 8
        return torch.randn(
            (1, latent_channels, h, w), generator=generator, dtype=self._dtype
        )

    @torch.inference_mode()
    def render_txt2img(
        self,
        embeds: EmbeddingSet,
        latents: torch.Tensor,
        ip_embeds: Optional[torch.Tensor] = None,
    ) -> Image.Image:
        """Generate a frame from scratch (used for frame 0 and morph mode)."""
        cfg = self.config
        device = self._pipe.device
        kwargs = self._embed_kwargs(embeds, device)
        if ip_embeds is not None:
            kwargs["ip_adapter_image_embeds"] = [ip_embeds.to(device)]
        return self._pipe(
            **kwargs,
            latents=latents.to(device),
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            output_type="pil",
        ).images[0]

    @torch.inference_mode()
    def render_img2img(
        self,
        embeds: EmbeddingSet,
        image: Image.Image,
        strength: float,
        seed: Optional[int] = None,
        ip_embeds: Optional[torch.Tensor] = None,
    ) -> Image.Image:
        """Re-diffuse *image* at the given denoise strength (flow mode)."""
        cfg = self.config
        device = self._img2img.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # img2img only runs int(steps * strength) steps; make sure that
        # is at least one full step.
        steps = max(cfg.num_inference_steps, math.ceil(1.0 / max(strength, 1e-3)) + 1)

        kwargs = self._embed_kwargs(embeds, device)
        if ip_embeds is not None:
            kwargs["ip_adapter_image_embeds"] = [ip_embeds.to(device)]

        return self._img2img(
            image=image,
            strength=float(strength),
            **kwargs,
            num_inference_steps=steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            output_type="pil",
        ).images[0]

    # ------------------------------------------------------------------
    # Sequence rendering
    # ------------------------------------------------------------------

    def render_sequence(
        self,
        timeline: PromptTimeline,
        audio: AudioProfile,
        output_dir: Path,
        *,
        seed: Optional[int] = None,
        dashboard: Optional[Dashboard] = None,
        duration: Optional[float] = None,
        init_image: Optional[Image.Image] = None,
    ) -> List[Path]:
        """Render every frame for the video and save to *output_dir*.

        Returns a sorted list of frame file paths.
        """
        if self._pipe is None:
            self.load_model()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        effective_duration = min(duration or audio.duration, audio.duration)
        total_frames = int(math.ceil(effective_duration * cfg.fps))

        plan = build_plan(timeline, audio, cfg, total_frames)

        keyframes = timeline.keyframes
        embeddings = [
            self.encode_keyframe(kf, cfg.negative_prompt) for kf in keyframes
        ]

        logger.info(
            "Rendering %d frames at %d FPS (%.1fs, mode=%s)",
            total_frames, cfg.fps, effective_duration, cfg.mode,
        )

        if dashboard is not None:
            dashboard.start(
                total_frames=total_frames,
                fps=cfg.fps,
                device=cfg.device,
                resolution=cfg.resolution,
            )

        if cfg.mode == "flow":
            frame_paths = self._render_flow(
                plan, keyframes, embeddings, output_dir, seed, dashboard,
                init_image=init_image,
            )
        else:
            if init_image is not None:
                logger.warning("init_image is only used in flow mode; ignoring.")
            frame_paths = self._render_morph(
                plan, keyframes, embeddings, output_dir, seed, dashboard
            )

        if dashboard is not None:
            dashboard.finish()

        logger.info("Rendered %d frames to %s", len(frame_paths), output_dir)
        return frame_paths

    # ------------------------------------------------------------------
    # Flow mode: img2img feedback loop
    # ------------------------------------------------------------------

    def _frame_embeds(self, plan: FramePlan, embeddings: List[EmbeddingSet],
                      idx: int) -> EmbeddingSet:
        pair = int(plan.pair_index[idx])
        pair_b = min(pair + 1, len(embeddings) - 1)
        return self._interpolate_embeddings(
            embeddings[pair],
            embeddings[pair_b],
            float(plan.progress[idx]),
            self.config.slerp_dot_threshold,
        )

    def _find_resume_frame(self, output_dir: Path, total_frames: int) -> int:
        """Return the index of the first frame that still needs rendering.

        Flow mode can only resume from a contiguous prefix, since every
        frame depends on the previous one.
        """
        existing = set()
        for p in output_dir.glob("frame_*.png"):
            m = _FRAME_RE.search(p.name)
            if m:
                existing.add(int(m.group(1)))
        next_idx = 0
        while next_idx in existing and next_idx < total_frames:
            next_idx += 1
        return next_idx

    def _render_flow(
        self,
        plan: FramePlan,
        keyframes: List[Keyframe],
        embeddings: List[EmbeddingSet],
        output_dir: Path,
        seed: Optional[int],
        dashboard: Optional[Dashboard],
        init_image: Optional[Image.Image] = None,
    ) -> List[Path]:
        cfg = self.config
        total_frames = len(plan)

        warper = None
        if cfg.enable_depth_warp:
            warper = DepthWarper(cfg.depth_model, cfg.device, cfg.depth_parallax)

        stabiliser = ColorStabiliser(cfg.color_coherence_decay) if cfg.color_coherence else None

        # Pre-load keyframe images at render resolution.
        kf_images: List[Optional[Image.Image]] = []
        for kf in keyframes:
            if kf.image:
                kf_images.append(
                    Image.open(kf.image).convert("RGB").resize(
                        (cfg.width, cfg.height), Image.LANCZOS
                    )
                )
            else:
                kf_images.append(None)

        # With no explicit init image, a photo on the first keyframe
        # seeds frame 0 so the video opens on (a stylised version of) it.
        if init_image is None and kf_images[0] is not None:
            init_image = kf_images[0]

        # IP-Adapter: encode each keyframe photo so it can condition the
        # diffusion directly (in addition to the pixel pull + caption).
        ip_embeds: List[Optional[torch.Tensor]] = [None] * len(kf_images)
        use_ip = cfg.enable_ip_adapter and any(im is not None for im in kf_images)
        if use_ip:
            try:
                self._load_ip_adapter()
                for k, im in enumerate(kf_images):
                    if im is not None:
                        ip_embeds[k] = self._ip_image_embed(im)
            except Exception as exc:
                logger.warning(
                    "IP-Adapter unavailable for this model (%s); "
                    "continuing with pixel pull + captions only.", exc,
                )
                use_ip = False

        def frame_ip(idx: int):
            """Interpolated image embedding + adapter scale for a frame."""
            if not use_ip:
                return None, None
            pair = int(plan.pair_index[idx])
            pair_b = min(pair + 1, len(ip_embeds) - 1)
            a_e, b_e = ip_embeds[pair], ip_embeds[pair_b]
            p = float(plan.progress[idx])
            base = cfg.ip_adapter_scale
            if a_e is not None and b_e is not None:
                return slerp(a_e, b_e, p), base
            if b_e is not None:
                return b_e, base * p
            if a_e is not None:
                return a_e, base * (1.0 - p)
            # Text-only span: keep the adapter quiet but fed.
            any_embed = next(e for e in ip_embeds if e is not None)
            return any_embed, 0.0

        start_idx = self._find_resume_frame(output_dir, total_frames) if cfg.resume else 0
        prev_image: Optional[Image.Image] = None
        first_frame: Optional[Image.Image] = None
        if start_idx > 0:
            prev_image = Image.open(output_dir / f"frame_{start_idx - 1:06d}.png").convert("RGB")
            first_frame = Image.open(output_dir / "frame_000000.png").convert("RGB")
            if stabiliser is not None:
                stabiliser.reset(prev_image)
            logger.info("Resuming from frame %d", start_idx)

        iterator = range(start_idx, total_frames)
        if dashboard is None:
            iterator = tqdm(iterator, desc="Rendering frames",
                            initial=start_idx, total=total_frames)

        frame_paths = [output_dir / f"frame_{i:06d}.png" for i in range(start_idx)]

        for i in iterator:
            frame_path = output_dir / f"frame_{i:06d}.png"
            embeds = self._frame_embeds(plan, embeddings, i)

            ip_emb, ip_scale = frame_ip(i)
            if ip_emb is not None:
                self._img2img.set_ip_adapter_scale(ip_scale)
                self._pipe.set_ip_adapter_scale(ip_scale)

            if prev_image is None:
                if init_image is not None:
                    # Frame 0 from a user photo: stylise it toward the
                    # first prompt while keeping the subject.
                    source = init_image.convert("RGB").resize(
                        (cfg.width, cfg.height), Image.LANCZOS
                    )
                    image = self.render_img2img(
                        embeds, source, cfg.init_image_strength, seed=seed,
                        ip_embeds=ip_emb,
                    )
                else:
                    # Frame 0: render from scratch.
                    latents = self.make_latents(seed=seed)
                    image = self.render_txt2img(embeds, latents, ip_embeds=ip_emb)
                if stabiliser is not None:
                    stabiliser.reset(image)
            else:
                # Camera move on the previous frame, then re-diffuse.
                zoom = float(plan.zoom[i])
                pan_x = float(plan.pan_x[i])
                pan_y = float(plan.pan_y[i])
                if warper is not None:
                    moved = warper.warp(prev_image, zoom, pan_x, pan_y)
                else:
                    moved = zoom_pan(prev_image, zoom, pan_x, pan_y)

                # Image keyframes: pull the frame toward the
                # cross-dissolve of the surrounding photos.  The
                # re-diffusion below re-coheres the blend into a clean
                # image, so the morph reads as transformation rather
                # than double exposure.
                pair = int(plan.pair_index[i])
                pair_b = min(pair + 1, len(kf_images) - 1)
                img_a, img_b = kf_images[pair], kf_images[pair_b]
                if img_a is not None or img_b is not None:
                    p = float(plan.progress[i])
                    if img_a is not None and img_b is not None:
                        target = Image.blend(img_a, img_b, p)
                        pull = cfg.image_pull
                    elif img_b is not None:
                        target, pull = img_b, cfg.image_pull * p
                    else:
                        target, pull = img_a, cfg.image_pull * (1.0 - p)
                    if pull > 0:
                        moved = Image.blend(moved, target, pull)

                # Seamless looping: in the final seconds, ramp a pull
                # back toward the opening frame so the video ends where
                # it began.
                if cfg.loop and first_frame is not None:
                    blend_frames = max(int(cfg.loop_blend_seconds * cfg.fps), 1)
                    blend_start = total_frames - blend_frames
                    if i >= blend_start:
                        ramp = (i - blend_start + 1) / blend_frames
                        moved = Image.blend(moved, first_frame, 0.45 * ramp)

                # Counteract the loop's progressive softening.
                moved = sharpen(moved, cfg.sharpen_amount)

                # Fixed temporal noise re-uses one noise pattern across
                # all frames, which suppresses texture shimmer/boiling.
                if cfg.temporal_noise == "fixed":
                    frame_seed = seed if seed is not None else 0
                else:
                    frame_seed = (seed + i) if seed is not None else None
                image = self.render_img2img(
                    embeds, moved, float(plan.strength[i]), seed=frame_seed,
                    ip_embeds=ip_emb,
                )
                if stabiliser is not None:
                    # Anchor the palette within scenes, relax it while
                    # the prompt is transitioning fast.
                    image = stabiliser.apply(
                        image, amount=float(plan.color_anchor[i])
                    )

            image.save(frame_path)
            frame_paths.append(frame_path)
            prev_image = image
            if first_frame is None:
                first_frame = image

            if dashboard is not None:
                dashboard.update(self._status(plan, keyframes, i, frame_path,
                                              strength=float(plan.strength[i])))

        return frame_paths

    # ------------------------------------------------------------------
    # Morph mode: independent frames (legacy)
    # ------------------------------------------------------------------

    def _render_morph(
        self,
        plan: FramePlan,
        keyframes: List[Keyframe],
        embeddings: List[EmbeddingSet],
        output_dir: Path,
        seed: Optional[int],
        dashboard: Optional[Dashboard],
    ) -> List[Path]:
        cfg = self.config
        total_frames = len(plan)

        # One latent per keyframe, chained so each transition ends on the
        # exact latent the next one starts from (no snap-back at keyframes).
        base_seed = seed if seed is not None else torch.seed() % (2**31)
        latents = [self.make_latents(seed=base_seed + k) for k in range(len(keyframes))]

        iterator = range(total_frames)
        if dashboard is None:
            iterator = tqdm(iterator, desc="Rendering frames")

        frame_paths: List[Path] = []
        for i in iterator:
            frame_path = output_dir / f"frame_{i:06d}.png"

            if cfg.resume and frame_path.exists():
                frame_paths.append(frame_path)
                if dashboard is not None:
                    dashboard.update(self._status(plan, keyframes, i, frame_path,
                                                  skipped=True))
                continue

            embeds = self._frame_embeds(plan, embeddings, i)

            pair = int(plan.pair_index[i])
            pair_b = min(pair + 1, len(latents) - 1)
            latent = slerp(
                latents[pair].squeeze(0),
                latents[pair_b].squeeze(0),
                float(plan.progress[i]),
                cfg.slerp_dot_threshold,
            ).unsqueeze(0)

            image = self.render_txt2img(embeds, latent)

            if cfg.enable_motion:
                zoom = 1.0 + float(plan.energy[i]) * (cfg.max_zoom - 1.0)
                pan_x = (float(plan.centroid[i]) - 0.5) * cfg.max_pan_x * cfg.motion_intensity
                pan_y = (float(plan.onset[i]) - 0.5) * cfg.max_pan_y * cfg.motion_intensity
                image = zoom_pan(image, zoom, pan_x, pan_y)

            image.save(frame_path)
            frame_paths.append(frame_path)

            if dashboard is not None:
                dashboard.update(self._status(plan, keyframes, i, frame_path))

        return frame_paths

    # ------------------------------------------------------------------
    # Dashboard plumbing
    # ------------------------------------------------------------------

    def _status(self, plan: FramePlan, keyframes: List[Keyframe], idx: int,
                frame_path: Path, *, strength: float = 0.0,
                skipped: bool = False) -> FrameStatus:
        pair = int(plan.pair_index[idx])
        pair_b = min(pair + 1, len(keyframes) - 1)
        return FrameStatus(
            frame_idx=idx,
            total_frames=len(plan),
            timestamp=float(plan.times[idx]),
            duration=float(plan.times[-1]) if len(plan) else 0.0,
            prompt_a=keyframes[pair].label,
            prompt_b=keyframes[pair_b].label,
            progress=float(plan.progress[idx]),
            energy=float(plan.energy[idx]),
            onset=float(plan.onset[idx]),
            centroid=float(plan.centroid[idx]),
            section_label=plan.section_labels[idx],
            frame_path=str(frame_path),
            strength=strength,
            skipped=skipped,
        )
