"""Latent-space interpolation utilities.

Provides SLERP (spherical linear interpolation), standard linear
interpolation, and circular latent walks for the Rufus pipeline.

The SLERP implementation is adapted from Andrej Karpathy's gist:
https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


# ------------------------------------------------------------------
# Spherical linear interpolation
# ------------------------------------------------------------------

def slerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: Union[float, torch.Tensor],
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    """Spherical linear interpolation between tensors *v0* and *v1*.

    Parameters
    ----------
    v0, v1:
        Tensors of identical shape.
    t:
        Interpolation factor(s) in ``[0, 1]``.  A scalar produces a
        single interpolated tensor; a 1-D tensor produces a batch.
    dot_threshold:
        When the cosine similarity exceeds this value the vectors are
        nearly parallel and we fall back to linear interpolation to
        avoid numerical instability.

    Returns
    -------
    torch.Tensor
        Interpolated tensor(s).  If *t* is scalar the shape matches
        *v0*; if *t* is 1-D the result has an extra leading dimension.
    """
    device = v0.device
    dtype = v0.dtype

    v0_flat = v0.detach().float().cpu().flatten()
    v1_flat = v1.detach().float().cpu().flatten()

    v0_norm = v0_flat / torch.linalg.norm(v0_flat).clamp(min=1e-8)
    v1_norm = v1_flat / torch.linalg.norm(v1_flat).clamp(min=1e-8)
    dot = torch.dot(v0_norm, v1_norm).clamp(-1.0, 1.0)

    scalar_t = not isinstance(t, torch.Tensor)
    if scalar_t:
        t = torch.tensor([t], dtype=torch.float32)
    else:
        t = t.float()

    if torch.abs(dot) > dot_threshold:
        result = (1.0 - t[:, None]) * v0_flat[None] + t[:, None] * v1_flat[None]
    else:
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        result = s0[:, None] * v0_flat[None] + s1[:, None] * v1_flat[None]

    result = result.reshape(-1, *v0.shape).to(dtype=dtype, device=device)

    if scalar_t:
        return result.squeeze(0)
    return result


# ------------------------------------------------------------------
# Linear interpolation (lerp)
# ------------------------------------------------------------------

def lerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Standard linear interpolation between *v0* and *v1*.

    Parameters match :func:`slerp`.
    """
    if isinstance(t, (int, float)):
        return v0 + (v1 - v0) * t

    t_shape = [-1] + [1] * v0.dim()
    t_broadcast = t.to(device=v0.device, dtype=v0.dtype).reshape(t_shape)
    return v0.unsqueeze(0) + (v1 - v0).unsqueeze(0) * t_broadcast


# ------------------------------------------------------------------
# Batch SLERP: interpolate between two tensors for N steps
# ------------------------------------------------------------------

def slerp_batch(
    v0: torch.Tensor,
    v1: torch.Tensor,
    steps: int,
    *,
    t_start: float = 0.0,
    t_end: float = 1.0,
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    """Return *steps* evenly-spaced SLERP interpolations.

    Returns a tensor of shape ``(steps, *v0.shape)``.
    """
    t = torch.linspace(t_start, t_end, steps)
    return slerp(v0, v1, t, dot_threshold=dot_threshold)


# ------------------------------------------------------------------
# Circular latent walk (for loopable sequences)
# ------------------------------------------------------------------

def circular_walk(
    latent_x: torch.Tensor,
    latent_y: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Walk a circle through two orthogonal noise bases.

    The walk starts and ends at the same point, making it ideal for
    seamlessly looping video segments.

    Parameters
    ----------
    latent_x, latent_y:
        Two random latent tensors of identical shape acting as the
        x and y axes of the circle.
    steps:
        Number of frames in the loop.

    Returns
    -------
    torch.Tensor
        Shape ``(steps, *latent_x.shape)``.
    """
    device = latent_x.device
    dtype = latent_x.dtype

    angles = torch.linspace(0, 2 * np.pi, steps, device=device, dtype=dtype)
    cos_w = torch.cos(angles)
    sin_w = torch.sin(angles)

    x_flat = latent_x.flatten()
    y_flat = latent_y.flatten()

    frames = cos_w[:, None] * x_flat[None, :] + sin_w[:, None] * y_flat[None, :]
    return frames.reshape(steps, *latent_x.shape)
