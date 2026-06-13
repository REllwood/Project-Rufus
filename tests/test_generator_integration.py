"""Integration tests for the render loop using a fake diffusion pipeline.

A lightweight stand-in for SDXL exercises the full generator plumbing
(prompt encoding, embedding interpolation, the flow feedback loop,
morph mode, resume) without a GPU or model download.  This is the test
layer that would have caught the original SDXL embedding shape bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from rufus.config import GenerationConfig
from rufus.generator import FrameGenerator
from rufus.motion import ColorStabiliser, zoom_pan
from rufus.prompts import Keyframe, PromptTimeline

from tests.test_reactivity import _make_profile


# ---------------------------------------------------------------------
# Fake SDXL pipelines
# ---------------------------------------------------------------------

class _FakeUnetConfig:
    in_channels = 4


class _FakeUnet:
    config = _FakeUnetConfig()


class _FakeResult:
    def __init__(self, images):
        self.images = images


def _prompt_color(embeds: torch.Tensor) -> tuple:
    """Deterministic colour derived from the embedding contents, so
    tests can verify which prompt drove a frame."""
    v = float(embeds.float().abs().mean())
    rng = np.random.default_rng(int(v * 1e6) % (2**32))
    return tuple(int(c) for c in rng.integers(0, 255, 3))


class FakeSDXLPipe:
    """Mimics the parts of StableDiffusionXLPipeline the generator uses."""

    def __init__(self, size=(64, 64)):
        self.size = size
        self.device = torch.device("cpu")
        self.unet = _FakeUnet()
        self.calls = []
        self.encoded = []
        self.ip_loaded = False
        self.ip_scales = []

    text_encoder_2 = object()  # marks this as an SDXL-class pipeline

    def load_ip_adapter(self, *args, **kwargs):
        self.ip_loaded = True

    def set_ip_adapter_scale(self, scale):
        self.ip_scales.append(float(scale))

    def prepare_ip_adapter_image_embeds(self, ip_adapter_image=None,
                                        ip_adapter_image_embeds=None,
                                        device=None, num_images_per_prompt=1,
                                        do_classifier_free_guidance=True):
        arr = np.asarray(ip_adapter_image[0], dtype=np.float32)
        gen = torch.Generator().manual_seed(int(arr.mean() * 1000) % (2**31))
        return [torch.randn((2, 1, 32), generator=gen)]

    def encode_prompt(self, prompt, negative_prompt=None, device=None,
                      do_classifier_free_guidance=True, num_images_per_prompt=1):
        """Deterministic embeddings seeded by the prompt text, with the
        real SDXL shapes: (1, 77, 2048) sequence + (1, 1280) pooled."""
        self.encoded.append(prompt)

        def embed(text, dim_seq=(1, 77, 2048), dim_pool=(1, 1280)):
            gen = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
            return (torch.randn(dim_seq, generator=gen),
                    torch.randn(dim_pool, generator=gen))

        pos_seq, pos_pool = embed(prompt)
        neg_seq, neg_pool = embed(negative_prompt or "")
        return pos_seq, neg_seq, pos_pool, neg_pool

    def __call__(self, prompt_embeds=None, negative_prompt_embeds=None,
                 pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None,
                 latents=None, ip_adapter_image_embeds=None, **kwargs):
        assert prompt_embeds is not None and pooled_prompt_embeds is not None
        assert prompt_embeds.shape[-1] == 2048
        assert pooled_prompt_embeds.shape[-1] == 1280
        self.calls.append({"kind": "txt2img",
                           "has_ip": ip_adapter_image_embeds is not None, **kwargs})
        img = Image.new("RGB", self.size, _prompt_color(prompt_embeds))
        return _FakeResult([img])


class FakeImg2ImgPipe:
    """Mimics StableDiffusionXLImg2ImgPipeline: blends the input image
    toward the prompt colour proportionally to strength."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = []
        self.ip_scales = []

    def set_ip_adapter_scale(self, scale):
        self.ip_scales.append(float(scale))

    def __call__(self, image=None, strength=0.5, prompt_embeds=None,
                 negative_prompt_embeds=None, pooled_prompt_embeds=None,
                 negative_pooled_prompt_embeds=None, num_inference_steps=20,
                 guidance_scale=7.5, generator=None,
                 ip_adapter_image_embeds=None, output_type="pil"):
        assert image is not None
        assert 0.0 < strength <= 1.0
        assert int(num_inference_steps * strength) >= 1
        self.calls.append({"strength": strength, "steps": num_inference_steps,
                           "has_ip": ip_adapter_image_embeds is not None,
                           "seed": generator.initial_seed() if generator else None})

        target = Image.new("RGB", image.size, _prompt_color(prompt_embeds))
        blended = Image.blend(image, target, alpha=float(strength) * 0.5)
        return _FakeResult([blended])


class FakeSD15Pipe:
    """Mimics a single-encoder (SD 1.5-class) pipeline: encode_prompt
    returns only two tensors, and the call signature has no pooled
    embedding parameters at all -- passing them must fail loudly."""

    text_encoder_2 = None

    def __init__(self, size=(64, 64)):
        self.size = size
        self.device = torch.device("cpu")
        self.unet = _FakeUnet()
        self.calls = []

    def encode_prompt(self, prompt, negative_prompt=None, device=None,
                      do_classifier_free_guidance=True, num_images_per_prompt=1):
        def embed(text):
            gen = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
            return torch.randn((1, 77, 768), generator=gen)

        return embed(prompt), embed(negative_prompt or "")

    def __call__(self, prompt_embeds=None, negative_prompt_embeds=None,
                 latents=None, height=None, width=None,
                 num_inference_steps=20, guidance_scale=7.5, output_type="pil"):
        assert prompt_embeds is not None
        assert prompt_embeds.shape[-1] == 768
        self.calls.append({"kind": "txt2img"})
        img = Image.new("RGB", self.size, _prompt_color(prompt_embeds))
        return _FakeResult([img])


class FakeSD15Img2ImgPipe:
    """SD 1.5-class img2img: no pooled embedding parameters."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = []

    def __call__(self, image=None, strength=0.5, prompt_embeds=None,
                 negative_prompt_embeds=None, num_inference_steps=20,
                 guidance_scale=7.5, generator=None, output_type="pil"):
        assert image is not None
        assert int(num_inference_steps * strength) >= 1
        self.calls.append({"strength": strength})
        target = Image.new("RGB", image.size, _prompt_color(prompt_embeds))
        return _FakeResult([Image.blend(image, target, alpha=float(strength) * 0.5)])


def _make_generator(cfg: GenerationConfig) -> FrameGenerator:
    gen = FrameGenerator(cfg)
    gen._pipe = FakeSDXLPipe(size=(cfg.width, cfg.height))
    gen._img2img = FakeImg2ImgPipe()
    return gen


def _make_sd15_generator(cfg: GenerationConfig) -> FrameGenerator:
    gen = FrameGenerator(cfg)
    gen._pipe = FakeSD15Pipe(size=(cfg.width, cfg.height))
    gen._img2img = FakeSD15Img2ImgPipe()
    return gen


def _timeline():
    return PromptTimeline([
        Keyframe(time=0, prompt="desert"),
        Keyframe(time=5, prompt="forest"),
    ])


@pytest.fixture
def audio():
    return _make_profile(duration=10.0)


# ---------------------------------------------------------------------
# Flow mode
# ---------------------------------------------------------------------

class TestFlowMode:
    def _config(self, **kw):
        defaults = dict(device="cpu", mode="flow", fps=4, width=64, height=64,
                        enable_depth_warp=False)
        defaults.update(kw)
        return GenerationConfig(**defaults)

    def test_renders_all_frames(self, audio, tmp_path):
        cfg = self._config()
        gen = _make_generator(cfg)
        paths = gen.render_sequence(_timeline(), audio, tmp_path, seed=1)
        assert len(paths) == int(audio.duration * cfg.fps)
        assert all(p.exists() for p in paths)

    def test_frame_zero_txt2img_rest_img2img(self, audio, tmp_path):
        cfg = self._config()
        gen = _make_generator(cfg)
        gen.render_sequence(_timeline(), audio, tmp_path, seed=1)
        assert len(gen._pipe.calls) == 1  # only frame 0
        assert len(gen._img2img.calls) == int(audio.duration * cfg.fps) - 1

    def test_duration_cap(self, audio, tmp_path):
        cfg = self._config()
        gen = _make_generator(cfg)
        paths = gen.render_sequence(_timeline(), audio, tmp_path, seed=1,
                                    duration=2.0)
        assert len(paths) == int(2.0 * cfg.fps)

    def test_temporal_coherence(self, audio, tmp_path):
        """Consecutive flow frames must be visually close -- the whole
        point of the feedback loop."""
        cfg = self._config(color_coherence=False)
        gen = _make_generator(cfg)
        paths = gen.render_sequence(_timeline(), audio, tmp_path, seed=1)
        a = np.asarray(Image.open(paths[5]), dtype=np.float32)
        b = np.asarray(Image.open(paths[6]), dtype=np.float32)
        assert np.abs(a - b).mean() < 60  # small per-frame change

    def test_resume_continues_from_last_frame(self, audio, tmp_path):
        cfg = self._config()
        gen = _make_generator(cfg)
        total = int(audio.duration * cfg.fps)

        # First run: render only the first 2 seconds.
        gen.render_sequence(_timeline(), audio, tmp_path, seed=1, duration=2.0)
        rendered_first = len(gen._img2img.calls) + len(gen._pipe.calls)

        # Second run: full duration resumes from the existing frames.
        gen2 = _make_generator(cfg)
        paths = gen2.render_sequence(_timeline(), audio, tmp_path, seed=1)
        assert len(paths) == total
        # No txt2img on resume, and only the missing frames re-rendered.
        assert len(gen2._pipe.calls) == 0
        assert len(gen2._img2img.calls) == total - rendered_first

    def test_fixed_temporal_noise(self, audio, tmp_path):
        """Default noise mode re-uses one seed across all frames."""
        cfg = self._config(temporal_noise="fixed")
        gen = _make_generator(cfg)
        gen.render_sequence(_timeline(), audio, tmp_path, seed=7, duration=2.0)
        seeds = {c["seed"] for c in gen._img2img.calls}
        assert seeds == {7}

    def test_varying_temporal_noise(self, audio, tmp_path):
        cfg = self._config(temporal_noise="varying")
        gen = _make_generator(cfg)
        gen.render_sequence(_timeline(), audio, tmp_path, seed=7, duration=2.0)
        seeds = [c["seed"] for c in gen._img2img.calls]
        assert len(set(seeds)) == len(seeds)  # all distinct

    def test_init_image_skips_txt2img(self, audio, tmp_path):
        """With an init image, frame 0 is img2img from the photo and
        txt2img is never called."""
        cfg = self._config()
        gen = _make_generator(cfg)
        photo = Image.new("RGB", (640, 480), (90, 120, 60))
        paths = gen.render_sequence(_timeline(), audio, tmp_path, seed=1,
                                    duration=2.0, init_image=photo)
        assert len(gen._pipe.calls) == 0
        assert len(gen._img2img.calls) == len(paths)
        # First frame derives from the (resized) photo.
        first = Image.open(paths[0])
        assert first.size == (cfg.width, cfg.height)

    def test_image_keyframes_pull_toward_photos(self, audio, tmp_path):
        """With photo keyframes, early frames must resemble photo A and
        late frames photo B -- the morph tracks the image series."""
        red = tmp_path / "red.png"
        blue = tmp_path / "blue.png"
        Image.new("RGB", (64, 64), (220, 30, 30)).save(red)
        Image.new("RGB", (64, 64), (30, 30, 220)).save(blue)

        cfg = self._config(color_coherence=False)
        gen = _make_generator(cfg)
        gen._captioner = lambda img: "a solid colour square"
        tl = PromptTimeline([
            Keyframe(time=0, image=str(red)),
            Keyframe(time=10, image=str(blue)),
        ])
        paths = gen.render_sequence(tl, audio, tmp_path / "out", seed=1)

        early = np.asarray(Image.open(paths[2]), dtype=np.float32).mean(axis=(0, 1))
        late = np.asarray(Image.open(paths[-1]), dtype=np.float32).mean(axis=(0, 1))
        # The morph must track the photo series: red falls, blue rises.
        # (Trend-based, since the fake pipe's prompt-derived colour adds
        # an arbitrary constant offset to every frame.)
        assert early[0] > late[0] + 20
        assert late[2] > early[2] + 20
        # Frame 0 seeds from photo A automatically (img2img, no txt2img).
        assert len(gen._pipe.calls) == 0

    def test_image_keyframes_are_auto_captioned(self, audio, tmp_path):
        """Image-only keyframes must encode a content-bearing caption,
        not the generic fallback prompt -- without it the morph loses
        the subject (verified empirically: owls become milk jars)."""
        photo = tmp_path / "owl.png"
        Image.new("RGB", (64, 64), (200, 180, 150)).save(photo)

        cfg = self._config(auto_caption=True)
        gen = _make_generator(cfg)
        gen._captioner = lambda img: "a barn owl perched on a branch"

        tl = PromptTimeline([
            Keyframe(time=0, prompt="a red fox"),
            Keyframe(time=5, image=str(photo)),
        ])
        gen.render_sequence(tl, audio, tmp_path / "out", seed=1, duration=1.0)
        assert "a barn owl perched on a branch" in gen._pipe.encoded
        assert cfg.image_keyframe_prompt not in gen._pipe.encoded

    def test_ip_adapter_conditions_photo_frames(self, audio, tmp_path):
        """With photo keyframes, the IP-Adapter loads, every img2img call
        carries image embeddings, and the adapter scale ramps with
        progress on mixed text->photo transitions."""
        photo = tmp_path / "p.png"
        Image.new("RGB", (64, 64), (120, 80, 40)).save(photo)
        cfg = self._config()
        gen = _make_generator(cfg)
        gen._captioner = lambda img: "an animal"
        tl = PromptTimeline([
            Keyframe(time=0, prompt="a desert"),
            Keyframe(time=10, image=str(photo)),
        ])
        gen.render_sequence(tl, audio, tmp_path / "out", seed=1)

        assert gen._pipe.ip_loaded
        assert all(c["has_ip"] for c in gen._img2img.calls)
        scales = gen._img2img.ip_scales
        # Text->photo transition: scale ramps from ~0 toward full.
        assert min(scales) < 0.1
        assert max(scales) > cfg.ip_adapter_scale * 0.8
        assert scales[-1] > scales[0]

    def test_ip_adapter_disabled(self, audio, tmp_path):
        photo = tmp_path / "p.png"
        Image.new("RGB", (64, 64), (120, 80, 40)).save(photo)
        cfg = self._config(enable_ip_adapter=False)
        gen = _make_generator(cfg)
        gen._captioner = lambda img: "an animal"
        tl = PromptTimeline([
            Keyframe(time=0, image=str(photo)),
            Keyframe(time=5, prompt="a forest"),
        ])
        gen.render_sequence(tl, audio, tmp_path / "out", seed=1, duration=2.0)
        assert not gen._pipe.ip_loaded
        assert not any(c["has_ip"] for c in gen._img2img.calls)

    def test_caption_disabled_uses_fallback(self, audio, tmp_path):
        photo = tmp_path / "p.png"
        Image.new("RGB", (64, 64), (1, 2, 3)).save(photo)
        cfg = self._config(auto_caption=False)
        gen = _make_generator(cfg)
        tl = PromptTimeline([Keyframe(time=0, image=str(photo)),
                             Keyframe(time=5, prompt="a forest")])
        gen.render_sequence(tl, audio, tmp_path / "out", seed=1, duration=1.0)
        assert cfg.image_keyframe_prompt in gen._pipe.encoded

    def test_mixed_text_and_image_keyframes(self, audio, tmp_path):
        photo = tmp_path / "photo.png"
        Image.new("RGB", (64, 64), (10, 200, 10)).save(photo)
        cfg = self._config()
        gen = _make_generator(cfg)
        gen._captioner = lambda img: "a green field"
        tl = PromptTimeline([
            Keyframe(time=0, prompt="a desert"),
            Keyframe(time=5, image=str(photo)),
        ])
        paths = gen.render_sequence(tl, audio, tmp_path / "out", seed=1)
        assert len(paths) == int(audio.duration * cfg.fps)

    def test_strength_respects_step_floor(self, audio, tmp_path):
        """int(steps * strength) >= 1 even with tiny strength + few steps."""
        cfg = self._config(num_inference_steps=4, flow_strength_min=0.1,
                           flow_strength_max=0.2)
        gen = _make_generator(cfg)
        gen.render_sequence(_timeline(), audio, tmp_path, seed=1, duration=2.0)
        # FakeImg2ImgPipe asserts the invariant on every call.


class TestLooping:
    def _config(self, **kw):
        defaults = dict(device="cpu", mode="flow", fps=4, width=64, height=64,
                        color_coherence=False)
        defaults.update(kw)
        return GenerationConfig(**defaults)

    def _loop_timeline(self, duration: float):
        # What the pipeline builds with loop=True: journey returns home.
        return PromptTimeline([
            Keyframe(time=0, prompt="desert"),
            Keyframe(time=duration / 2, prompt="forest"),
            Keyframe(time=duration, prompt="desert"),
        ])

    def test_loop_converges_to_first_frame(self, audio, tmp_path):
        tl = self._loop_timeline(audio.duration)

        gen_loop = _make_generator(self._config(loop=True))
        looped = gen_loop.render_sequence(tl, audio, tmp_path / "loop", seed=1)

        gen_open = _make_generator(self._config(loop=False))
        opened = gen_open.render_sequence(tl, audio, tmp_path / "open", seed=1)

        def gap(paths):
            first = np.asarray(Image.open(paths[0]), dtype=np.float32)
            last = np.asarray(Image.open(paths[-1]), dtype=np.float32)
            return np.abs(first - last).mean()

        # The loop pull must bring the final frame markedly closer to
        # frame 0 than the un-looped render manages.
        assert gap(looped) < gap(opened) * 0.7


class TestFullPipeline:
    """End-to-end: audio file -> analysis -> auto-timed keyframes ->
    flow render (fake model) -> ffmpeg assembly."""

    def test_auto_timed_loop_video(self, tmp_path):
        sf = pytest.importorskip("soundfile")
        from tests.test_video import HAS_FFMPEG
        if not HAS_FFMPEG:
            pytest.skip("ffmpeg not available")

        sr, dur = 22050, 4.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sig = np.sin(2 * np.pi * 220 * t) * (0.3 + 0.7 * t / dur)
        wav = tmp_path / "song.wav"
        sf.write(str(wav), sig.astype(np.float32), sr)

        from rufus.pipeline import RufusPipeline

        cfg = GenerationConfig(device="cpu", mode="flow", fps=4,
                               width=64, height=64, loop=True)
        pipeline = RufusPipeline(config=cfg)
        pipeline._generator = _make_generator(cfg)

        out = pipeline.generate(
            audio_path=str(wav),
            keyframes=[{"prompt": "desert"}, {"prompt": "forest"}],  # no times
            output_path=str(tmp_path / "out.mp4"),
            frame_dir=str(tmp_path / "frames"),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_partial_times_rejected(self, tmp_path):
        sf = pytest.importorskip("soundfile")
        sr = 22050
        wav = tmp_path / "s.wav"
        sf.write(str(wav), np.zeros(sr, dtype=np.float32), sr)

        from rufus.pipeline import RufusPipeline

        cfg = GenerationConfig(device="cpu", fps=4, width=64, height=64)
        pipeline = RufusPipeline(config=cfg)
        pipeline._generator = _make_generator(cfg)
        with pytest.raises(ValueError, match="every keyframe"):
            pipeline.generate(
                audio_path=str(wav),
                keyframes=[{"prompt": "a"}, {"time": 1, "prompt": "b"}],
                output_path=str(tmp_path / "o.mp4"),
            )


# ---------------------------------------------------------------------
# SD 1.5-class models (single text encoder, no pooled embeddings)
# ---------------------------------------------------------------------

class TestSD15Family:
    """The generator must adapt to single-encoder models -- the
    practical choice for CPU and low-VRAM GPUs."""

    def _config(self, **kw):
        defaults = dict(device="cpu", fps=4, width=64, height=64)
        defaults.update(kw)
        return GenerationConfig(**defaults)

    def test_flow_mode(self, audio, tmp_path):
        cfg = self._config(mode="flow")
        gen = _make_sd15_generator(cfg)
        paths = gen.render_sequence(_timeline(), audio, tmp_path, seed=1)
        assert len(paths) == int(audio.duration * cfg.fps)
        assert all(p.exists() for p in paths)

    def test_morph_mode(self, audio, tmp_path):
        cfg = self._config(mode="morph")
        gen = _make_sd15_generator(cfg)
        paths = gen.render_sequence(_timeline(), audio, tmp_path, seed=1)
        assert len(paths) == int(audio.duration * cfg.fps)

    def test_embeddings_have_no_pooled_slots(self):
        cfg = self._config()
        gen = _make_sd15_generator(cfg)
        embeds = gen._encode_prompt("desert", "blurry")
        assert embeds[0].shape == (1, 77, 768)
        assert embeds[2] is None and embeds[3] is None


# ---------------------------------------------------------------------
# Morph mode
# ---------------------------------------------------------------------

class TestMorphMode:
    def _config(self, **kw):
        defaults = dict(device="cpu", mode="morph", fps=4, width=64, height=64)
        defaults.update(kw)
        return GenerationConfig(**defaults)

    def test_renders_all_frames(self, audio, tmp_path):
        cfg = self._config()
        gen = _make_generator(cfg)
        paths = gen.render_sequence(_timeline(), audio, tmp_path, seed=1)
        assert len(paths) == int(audio.duration * cfg.fps)
        assert all(p.exists() for p in paths)
        assert len(gen._img2img.calls) == 0  # morph never uses img2img

    def test_resume_skips_existing(self, audio, tmp_path):
        cfg = self._config()
        gen = _make_generator(cfg)
        gen.render_sequence(_timeline(), audio, tmp_path, seed=1)
        first_calls = len(gen._pipe.calls)

        gen2 = _make_generator(cfg)
        gen2.render_sequence(_timeline(), audio, tmp_path, seed=1)
        assert first_calls > 0
        assert len(gen2._pipe.calls) == 0  # everything skipped


# ---------------------------------------------------------------------
# Motion & colour utilities
# ---------------------------------------------------------------------

class TestMotion:
    def test_zoom_pan_preserves_size(self):
        img = Image.new("RGB", (64, 48), (10, 20, 30))
        out = zoom_pan(img, 1.05, 2.0, -1.0)
        assert out.size == (64, 48)

    def test_identity_is_noop(self):
        img = Image.new("RGB", (32, 32), (1, 2, 3))
        assert zoom_pan(img, 1.0, 0.0, 0.0) is img

    def test_color_stabiliser_corrects_drift(self):
        ref = Image.new("RGB", (32, 32), (100, 100, 100))
        stab = ColorStabiliser(decay=1.0)
        stab.reset(ref)
        drifted = Image.new("RGB", (32, 32), (180, 90, 140))
        out = np.asarray(stab.apply(drifted), dtype=np.float32)
        # A solid frame has zero std, so only the mean can be matched.
        assert abs(out.mean() - 100) < 5

    def test_color_stabiliser_partial_amount(self):
        """amount=0 leaves the frame untouched; 0.5 lands in between."""
        ref = Image.new("RGB", (32, 32), (100, 100, 100))
        drifted = Image.new("RGB", (32, 32), (200, 200, 200))

        stab = ColorStabiliser(decay=1.0)
        stab.reset(ref)
        untouched = np.asarray(stab.apply(drifted, amount=0.0), dtype=np.float32)
        assert abs(untouched.mean() - 200) < 1

        stab = ColorStabiliser(decay=1.0)
        stab.reset(ref)
        halfway = np.asarray(stab.apply(drifted, amount=0.5), dtype=np.float32)
        assert 130 < halfway.mean() < 170

    def test_sharpen(self):
        from rufus.motion import sharpen
        img = Image.effect_noise((64, 64), 40).convert("RGB")
        blurred = img.filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(1))
        sharpened = sharpen(blurred, 0.5)
        assert sharpened.size == blurred.size
        # Sharpening must increase local contrast (std of laplacian-ish proxy).
        b = np.asarray(blurred, dtype=np.float32)
        s = np.asarray(sharpened, dtype=np.float32)
        assert s.std() > b.std()
        # amount=0 is a no-op
        assert sharpen(blurred, 0.0) is blurred
