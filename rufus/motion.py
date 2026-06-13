"""Camera motion and colour coherence for the feedback-loop renderer.

Provides the per-frame image transforms applied to the previous frame
before it is re-diffused:

* :func:`zoom_pan`: flat 2D zoom + pan (cheap, always available).
* :class:`DepthWarper`: depth-aware parallax warp using a monocular
  depth estimator, for a genuine "flying through the scene" feel.
* :class:`ColorStabiliser`: matches each frame's colour statistics to
  a rolling reference so the img2img feedback loop doesn't drift
  toward magenta or blow out over hundreds of frames.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Flat 2D zoom + pan
# ------------------------------------------------------------------

def zoom_pan(image: Image.Image, zoom: float, pan_x: float, pan_y: float) -> Image.Image:
    """Zoom into *image* by *zoom* (>1) and drift by (*pan_x*, *pan_y*) px."""
    w, h = image.size
    if abs(zoom - 1.0) < 1e-6 and abs(pan_x) < 0.25 and abs(pan_y) < 0.25:
        return image

    new_w = max(int(round(w * zoom)), w)
    new_h = max(int(round(h * zoom)), h)
    resized = image.resize((new_w, new_h), Image.BICUBIC)

    cx = (new_w - w) / 2 + pan_x
    cy = (new_h - h) / 2 + pan_y
    left = int(max(0, min(cx, new_w - w)))
    top = int(max(0, min(cy, new_h - h)))
    return resized.crop((left, top, left + w, top + h))


def sharpen(image: Image.Image, amount: float) -> Image.Image:
    """Subtle unsharp mask to counteract the feedback loop's softening.

    Each img2img round trip (resample -> encode -> denoise -> decode)
    loses a little high-frequency detail; without correction a long
    render drifts toward mush.  ``amount`` of ~0.3 restores roughly what
    one round trip loses.
    """
    if amount <= 0:
        return image
    from PIL import ImageFilter

    percent = int(80 * float(np.clip(amount, 0.0, 1.0)))
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=2))


# ------------------------------------------------------------------
# Depth-aware parallax warp
# ------------------------------------------------------------------

class DepthWarper:
    """Warps frames with depth-scaled motion for a parallax effect.

    Near pixels move more than far pixels, which turns a flat zoom into
    an apparent camera translation through the scene.  The depth model
    is loaded lazily on first use.
    """

    def __init__(self, model_name: str, device: str, parallax: float = 0.6) -> None:
        self._model_name = model_name
        self._device = device
        self.parallax = float(np.clip(parallax, 0.0, 1.0))
        self._estimator = None

    def _load(self) -> None:
        if self._estimator is not None:
            return
        from transformers import pipeline as hf_pipeline

        logger.info("Loading depth model: %s", self._model_name)
        # MPS works for these small ViT depth models; fall back to CPU
        # on any failure rather than aborting the render.
        try:
            self._estimator = hf_pipeline(
                "depth-estimation", model=self._model_name, device=self._device
            )
        except Exception:
            logger.warning("Depth model failed on %s; using CPU.", self._device)
            self._estimator = hf_pipeline(
                "depth-estimation", model=self._model_name, device="cpu"
            )

    def estimate(self, image: Image.Image) -> np.ndarray:
        """Return relative depth in [0, 1] (1 = nearest) at image size."""
        self._load()
        result = self._estimator(image)
        depth = np.asarray(result["depth"], dtype=np.float32)
        if depth.shape != (image.height, image.width):
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize(image.size, Image.BILINEAR)
            depth = np.asarray(depth_img, dtype=np.float32)
        mn, mx = depth.min(), depth.max()
        if mx - mn < 1e-9:
            return np.ones_like(depth)
        # Depth-Anything outputs higher = nearer already; normalise.
        return (depth - mn) / (mx - mn)

    def warp(
        self,
        image: Image.Image,
        zoom: float,
        pan_x: float,
        pan_y: float,
        depth: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """Apply zoom + pan with per-pixel magnitude scaled by depth."""
        from scipy.ndimage import map_coordinates

        if depth is None:
            depth = self.estimate(image)

        w, h = image.size
        arr = np.asarray(image, dtype=np.float32)

        # Per-pixel motion scale: far pixels (depth 0) keep
        # (1 - parallax) of the motion, near pixels (depth 1) get all of it.
        scale = (1.0 - self.parallax) + self.parallax * depth

        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing="ij")
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        # Inverse mapping: where in the source does each output pixel come
        # from?  Zoom contracts coordinates toward the centre; pan shifts.
        zoom_amount = (zoom - 1.0) * scale
        src_x = xx - (xx - cx) * zoom_amount - pan_x * scale
        src_y = yy - (yy - cy) * zoom_amount - pan_y * scale

        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            out[:, :, c] = map_coordinates(
                arr[:, :, c], [src_y, src_x], order=1, mode="nearest"
            )
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------
# Colour coherence
# ------------------------------------------------------------------

class ColorStabiliser:
    """Keeps the feedback loop's colours anchored to a rolling reference.

    Maintains an exponential moving average of per-channel mean/std and
    remaps each new frame to match.  ``decay`` close to 1 anchors hard
    to the opening look; lower values let the palette evolve with the
    prompt journey while still suppressing frame-to-frame drift.
    """

    def __init__(self, decay: float = 0.98) -> None:
        self.decay = decay
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_std: Optional[np.ndarray] = None

    def reset(self, image: Image.Image) -> None:
        arr = np.asarray(image, dtype=np.float32)
        self._ref_mean = arr.reshape(-1, 3).mean(axis=0)
        self._ref_std = arr.reshape(-1, 3).std(axis=0) + 1e-6

    def apply(self, image: Image.Image, amount: float = 1.0) -> Image.Image:
        """Match *image* toward the rolling reference.

        ``amount`` scales the correction: 1 = full anchoring, 0 = leave
        the frame untouched.  The render plan lowers it during fast
        prompt transitions so the palette can follow the scene journey.
        """
        arr = np.asarray(image, dtype=np.float32)
        flat = arr.reshape(-1, 3)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-6

        if self._ref_mean is None:
            self._ref_mean, self._ref_std = mean, std
            return image

        matched = (arr - mean) / std * self._ref_std + self._ref_mean
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount < 1.0:
            matched = arr * (1.0 - amount) + matched * amount
        result = Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))

        # Let the reference drift slowly toward the corrected frame so the
        # palette can follow the prompt journey.
        d = self.decay
        corrected = np.asarray(result, dtype=np.float32).reshape(-1, 3)
        self._ref_mean = d * self._ref_mean + (1 - d) * corrected.mean(axis=0)
        self._ref_std = d * self._ref_std + (1 - d) * (corrected.std(axis=0) + 1e-6)
        return result
