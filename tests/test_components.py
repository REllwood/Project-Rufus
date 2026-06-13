"""Unit tests for Rufus components that can run without a GPU.

Validates audio analysis, prompt timeline, interpolation maths, and
video assembly using synthetic test data.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rufus.config import GenerationConfig
from rufus.interpolation import circular_walk, lerp, slerp, slerp_batch
from rufus.prompts import Keyframe, PromptTimeline


# =====================================================================
# Interpolation
# =====================================================================


class TestSlerp:
    def test_endpoints(self):
        v0 = torch.randn(64)
        v1 = torch.randn(64)
        at_zero = slerp(v0, v1, 0.0)
        at_one = slerp(v0, v1, 1.0)
        assert torch.allclose(at_zero, v0, atol=1e-4)
        assert torch.allclose(at_one, v1, atol=1e-4)

    def test_midpoint_differs(self):
        v0 = torch.randn(64)
        v1 = torch.randn(64)
        mid = slerp(v0, v1, 0.5)
        assert not torch.allclose(mid, v0, atol=1e-2)
        assert not torch.allclose(mid, v1, atol=1e-2)

    def test_batch(self):
        v0 = torch.randn(32)
        v1 = torch.randn(32)
        t = torch.linspace(0, 1, 10)
        result = slerp(v0, v1, t)
        assert result.shape == (10, 32)

    def test_parallel_vectors_fallback(self):
        v0 = torch.randn(64)
        v1 = v0 + 1e-7 * torch.randn(64)
        mid = slerp(v0, v1, 0.5)
        assert mid.shape == v0.shape
        assert torch.isfinite(mid).all()

    def test_shape_preserved(self):
        v0 = torch.randn(4, 8, 8)
        v1 = torch.randn(4, 8, 8)
        result = slerp(v0, v1, 0.3)
        assert result.shape == v0.shape


class TestLerp:
    def test_endpoints(self):
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])
        assert torch.allclose(lerp(v0, v1, 0.0), v0)
        assert torch.allclose(lerp(v0, v1, 1.0), v1)

    def test_midpoint(self):
        v0 = torch.tensor([0.0, 0.0])
        v1 = torch.tensor([2.0, 4.0])
        mid = lerp(v0, v1, 0.5)
        assert torch.allclose(mid, torch.tensor([1.0, 2.0]))

    def test_batch(self):
        v0 = torch.randn(16)
        v1 = torch.randn(16)
        t = torch.linspace(0, 1, 5)
        result = lerp(v0, v1, t)
        assert result.shape == (5, 16)


class TestSlerpBatch:
    def test_shape(self):
        v0 = torch.randn(32)
        v1 = torch.randn(32)
        result = slerp_batch(v0, v1, 20)
        assert result.shape == (20, 32)

    def test_endpoints_match(self):
        v0 = torch.randn(32)
        v1 = torch.randn(32)
        result = slerp_batch(v0, v1, 10)
        assert torch.allclose(result[0], v0, atol=1e-4)
        assert torch.allclose(result[-1], v1, atol=1e-4)


class TestCircularWalk:
    def test_shape(self):
        lx = torch.randn(4, 8, 8)
        ly = torch.randn(4, 8, 8)
        result = circular_walk(lx, ly, 30)
        assert result.shape == (30, 4, 8, 8)

    def test_loop_closure(self):
        lx = torch.randn(64)
        ly = torch.randn(64)
        result = circular_walk(lx, ly, 100)
        assert torch.allclose(result[0], result[-1], atol=1e-4)


# =====================================================================
# Prompt timeline
# =====================================================================


class TestKeyframe:
    def test_full_prompt_without_style(self):
        kf = Keyframe(time=0, prompt="a beautiful mountain")
        assert kf.full_prompt == "a beautiful mountain"

    def test_full_prompt_with_style(self):
        kf = Keyframe(time=0, prompt="a mountain", style="cinematic, 8k")
        assert kf.full_prompt == "a mountain, cinematic, 8k"

    def test_image_only_keyframe(self):
        kf = Keyframe(time=0, image="photo.jpg")
        assert kf.full_prompt == ""
        assert kf.label == "[photo.jpg]"

    def test_image_and_prompt_keyframe(self):
        kf = Keyframe(time=0, prompt="a dog", image="dog.jpg")
        assert kf.label == "a dog"

    def test_empty_keyframe_raises(self):
        with pytest.raises(ValueError):
            Keyframe(time=0)

    def test_from_dicts_with_images(self):
        tl = PromptTimeline.from_dicts([
            {"time": 0, "image": "a.jpg"},
            {"time": 10, "prompt": "a forest"},
        ])
        assert tl.keyframes[0].image == "a.jpg"
        assert tl.keyframes[1].image is None


class TestPromptTimeline:
    def _make_timeline(self):
        return PromptTimeline([
            Keyframe(time=0, prompt="desert"),
            Keyframe(time=30, prompt="mountains"),
            Keyframe(time=60, prompt="forest"),
        ])

    def test_at_start(self):
        tl = self._make_timeline()
        a, b, p = tl.at(0)
        assert a.prompt == "desert"
        assert b.prompt == "mountains"
        assert p == 0.0

    def test_at_midpoint(self):
        tl = self._make_timeline()
        a, b, p = tl.at(15)
        assert a.prompt == "desert"
        assert b.prompt == "mountains"
        assert abs(p - 0.5) < 1e-6

    def test_at_end(self):
        tl = self._make_timeline()
        a, b, p = tl.at(60)
        assert b.prompt == "forest"
        assert p == 1.0

    def test_beyond_end(self):
        tl = self._make_timeline()
        a, b, p = tl.at(100)
        assert p == 1.0
        assert b.prompt == "forest"

    def test_before_start(self):
        tl = self._make_timeline()
        a, b, p = tl.at(-5)
        assert p == 0.0
        assert a.prompt == "desert"

    def test_single_keyframe(self):
        tl = PromptTimeline([Keyframe(time=0, prompt="only")])
        a, b, p = tl.at(10)
        assert a.prompt == "only"
        assert b.prompt == "only"
        assert p == 0.0

    def test_from_dicts(self):
        tl = PromptTimeline.from_dicts([
            {"time": 0, "prompt": "alpha"},
            {"time": 10, "prompt": "beta"},
        ])
        assert len(tl) == 2
        assert tl.keyframes[0].prompt == "alpha"

    def test_transition_pairs(self):
        tl = self._make_timeline()
        pairs = tl.transition_pairs()
        assert len(pairs) == 2
        assert pairs[0][0].prompt == "desert"
        assert pairs[0][1].prompt == "mountains"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            PromptTimeline([])


# =====================================================================
# Config
# =====================================================================


class TestConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.width == 768
        assert cfg.fps == 12
        assert cfg.device in ("cuda", "mps", "cpu")
        assert cfg.torch_dtype in ("float16", "float32")

    def test_auto_detect_device(self):
        cfg = GenerationConfig()
        # device should be populated by __post_init__
        assert cfg.device != ""

    def test_explicit_device(self):
        cfg = GenerationConfig(device="cpu")
        assert cfg.device == "cpu"
        assert cfg.torch_dtype == "float32"

    def test_cuda_default_dtype(self):
        cfg = GenerationConfig(device="cuda")
        assert cfg.torch_dtype == "float16"

    def test_mps_default_dtype(self):
        cfg = GenerationConfig(device="mps")
        assert cfg.torch_dtype == "float32"

    def test_xpu_default_dtype(self):
        cfg = GenerationConfig(device="xpu")
        assert cfg.torch_dtype == "float16"

    def test_describe_device(self):
        from rufus.config import describe_device
        assert describe_device("cpu") == "CPU"
        assert "Metal" in describe_device("mps")
        assert "Intel" in describe_device("xpu")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            GenerationConfig(device="cpu", mode="bogus")

    def test_invalid_fast_mode_raises(self):
        with pytest.raises(ValueError):
            GenerationConfig(device="cpu", fast_mode="turbo")

    def test_preview_preset(self):
        cfg = GenerationConfig.preview()
        assert cfg.width == 512
        assert cfg.fps == 8

    def test_production_preset(self):
        cfg = GenerationConfig.production()
        assert cfg.width == 1024
        assert cfg.fps == 24

    def test_preset_override(self):
        cfg = GenerationConfig.preview(fps=10)
        assert cfg.fps == 10
        assert cfg.width == 512

    def test_resolution_property(self):
        cfg = GenerationConfig(width=640, height=480)
        assert cfg.resolution == (640, 480)
