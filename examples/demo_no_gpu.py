"""CPU-only Rufus demo -- no GPU or diffusion model required.

Downloads a royalty-free audio clip, analyses it, generates synthetic
gradient frames that react to the music's energy, and assembles a
video.  This demonstrates the full pipeline flow (audio analysis,
timeline, frame creation, video assembly) without needing a GPU.

Usage:
    python examples/demo_no_gpu.py

The output video won't have AI-generated landscapes, but the colour
transitions will visually follow the audio energy -- proving the
music-reactive pipeline works end to end.

Audio source:
    "Winter" by Zoe Blade -- CC BY-NC-ND, via Internet Archive
    https://archive.org/details/zoeb_-_winter
"""

from __future__ import annotations

import colorsys
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rufus.audio import analyse
from rufus.config import GenerationConfig
from rufus.prompts import Keyframe, PromptTimeline
from rufus.reactivity import build_plan
from rufus.video import assemble

AUDIO_URL = "https://archive.org/download/zoeb_-_winter/winter_64kb.mp3"
AUDIO_FILENAME = "demo_audio.mp3"

# Palette: each keyframe maps to an HSV hue (0-1)
SCENE_HUES = [0.75, 0.1, 0.45, 0.0]  # purple, gold, teal, red
SCENE_NAMES = ["twilight desert", "golden mountains", "forest night", "sunset city"]


def download_audio(dest: Path) -> Path:
    if dest.exists():
        print(f"Audio cached: {dest}")
        return dest
    print("Downloading demo audio from Internet Archive...")
    try:
        # requests bundles certifi CA certs; system urllib may lack them.
        import requests

        resp = requests.get(AUDIO_URL, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except ImportError:
        urllib.request.urlretrieve(AUDIO_URL, str(dest))
    print(f"Saved to: {dest}")
    return dest


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def render_reactive_frame(
    width: int,
    height: int,
    hue_a: float,
    hue_b: float,
    progress: float,
    energy: float,
    onset: float,
    centroid: float,
) -> Image.Image:
    """Generate a synthetic frame whose colour and structure respond
    to the audio features, simulating what the AI generator does."""
    hue = _lerp(hue_a, hue_b, progress) % 1.0
    saturation = 0.5 + 0.5 * energy
    value = 0.3 + 0.5 * energy

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    base_colour = (int(r * 255), int(g * 255), int(b * 255))

    img = Image.new("RGB", (width, height), base_colour)
    draw = ImageDraw.Draw(img)

    # Layered radial gradient driven by spectral centroid
    cx, cy = width // 2, height // 2
    max_radius = int(min(width, height) * 0.45)
    n_rings = 12
    for i in range(n_rings, 0, -1):
        ring_t = i / n_rings
        radius = int(max_radius * ring_t)
        shift = centroid * 0.15
        ring_hue = (hue + shift * ring_t) % 1.0
        ring_val = value * (0.5 + 0.5 * ring_t)
        rr, rg, rb = colorsys.hsv_to_rgb(ring_hue, saturation, ring_val)
        ring_col = (int(rr * 255), int(rg * 255), int(rb * 255))
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=ring_col,
        )

    # Beat pulse: bright flash ring on high onset
    if onset > 0.6:
        pulse_radius = int(max_radius * onset)
        pulse_alpha = int(200 * (onset - 0.6) / 0.4)
        pr, pg, pb = colorsys.hsv_to_rgb(hue, 0.3, 1.0)
        pulse_col = (int(pr * 255), int(pg * 255), int(pb * 255))
        ring_w = max(2, int(8 * onset))
        draw.ellipse(
            (cx - pulse_radius, cy - pulse_radius,
             cx + pulse_radius, cy + pulse_radius),
            outline=pulse_col,
            width=ring_w,
        )

    # Horizontal scan lines for texture (energy-driven density)
    line_spacing = max(4, int(20 * (1 - energy)))
    for y in range(0, height, line_spacing):
        line_alpha = int(30 + 40 * energy)
        draw.line(
            [(0, y), (width, y)],
            fill=(255, 255, 255, line_alpha) if img.mode == "RGBA" else (
                min(255, base_colour[0] + line_alpha),
                min(255, base_colour[1] + line_alpha),
                min(255, base_colour[2] + line_alpha),
            ),
            width=1,
        )

    # Soft blur for dreamlike quality
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    return img


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    audio_path = project_root / AUDIO_FILENAME
    output_path = project_root / "demo_no_gpu_output.mp4"
    frame_dir = project_root / "demo_no_gpu_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download audio
    download_audio(audio_path)

    # 2. Analyse
    print("\nAnalysing audio...")
    profile = analyse(audio_path)
    print(f"  Duration: {profile.duration:.1f}s")
    print(f"  Tempo:    {profile.tempo:.0f} BPM")
    print(f"  Beats:    {len(profile.beat_times)} detected")
    print(f"  Sections: {len(profile.sections)} detected")
    for sec in profile.sections[:8]:
        print(f"    {sec.label:6s} {sec.start_time:6.1f}s - {sec.end_time:6.1f}s  "
              f"(energy {sec.mean_energy:.2f})")

    # 3. Build timeline
    duration = min(profile.duration, 60.0)
    interval = duration / len(SCENE_HUES)
    keyframes = [
        Keyframe(time=i * interval, prompt=SCENE_NAMES[i])
        for i in range(len(SCENE_HUES))
    ]
    timeline = PromptTimeline(keyframes)

    # 4. Render synthetic frames using the precomputed audio-reactive
    #    plan (the same monotonic progress/strength schedules the AI
    #    renderer uses).
    fps = 12
    width, height = 512, 512
    total_frames = int(duration * fps)

    config = GenerationConfig(fps=fps, width=width, height=height, device="cpu")
    plan = build_plan(timeline, profile, config, total_frames)

    print(f"\nRendering {total_frames} synthetic frames at {fps} FPS...")
    frame_paths = []
    for frame_idx in range(total_frames):
        t = plan.times[frame_idx]
        idx_a = int(plan.pair_index[frame_idx])
        idx_b = min(idx_a + 1, len(keyframes) - 1)
        progress = float(plan.progress[frame_idx])

        energy = float(plan.energy[frame_idx])
        onset = float(plan.onset[frame_idx])
        centroid = float(plan.centroid[frame_idx])

        img = render_reactive_frame(
            width, height,
            SCENE_HUES[idx_a], SCENE_HUES[idx_b],
            progress, energy, onset, centroid,
        )

        path = frame_dir / f"frame_{frame_idx:06d}.png"
        img.save(path)
        frame_paths.append(path)

        if frame_idx % (fps * 5) == 0:
            print(f"  Frame {frame_idx}/{total_frames} "
                  f"({t:.1f}s, energy={energy:.2f})")

    # 5. Assemble video
    print(f"\nAssembling video ({len(frame_paths)} frames)...")
    result = assemble(frame_paths, str(audio_path), str(output_path), config,
                      duration=duration)

    print(f"\nDemo complete! Video saved to: {result}")
    print(f"Frames preserved in: {frame_dir}")
    print(
        "\nThis video uses synthetic colour gradients reacting to the "
        "music's energy.\nWith a GPU (CUDA or Apple Silicon MPS), run "
        "examples/demo.py for AI-generated landscapes."
    )


if __name__ == "__main__":
    main()
