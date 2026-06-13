"""Self-contained Rufus demo.

Downloads a royalty-free audio clip from the Internet Archive,
then runs the full generation pipeline to produce a music-reactive
landscape video.

Usage:
    python examples/demo.py

Requirements:
    - GPU: CUDA (NVIDIA) or MPS (Apple Silicon M1/M2/M3)
    - All dependencies from requirements.txt installed
    - Internet connection (for audio download + model weights on first run)

The demo uses the preview preset (512x512, 8 FPS) to keep generation
time reasonable.  Approximate times for a 30-second clip:
    - NVIDIA RTX 3080 class:  ~30 minutes
    - Apple M1 Max (32 GB):   ~45-60 minutes (float32, slower per-step)

Audio source:
    "Winter" by Zoe Blade -- CC BY-NC-ND, via Internet Archive
    https://archive.org/details/zoeb_-_winter
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rufus import GenerationConfig, RufusPipeline

AUDIO_URL = "https://archive.org/download/zoeb_-_winter/winter_64kb.mp3"
AUDIO_FILENAME = "demo_audio.mp3"

# Only use the first 30 seconds to keep the demo fast
DEMO_DURATION_SECONDS = 30


def download_audio(dest: Path) -> Path:
    """Download the demo audio clip if not already cached."""
    if dest.exists():
        print(f"Audio already downloaded: {dest}")
        return dest

    print(f"Downloading demo audio from Internet Archive...")
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    audio_path = project_root / AUDIO_FILENAME
    output_path = project_root / "demo_output.mp4"
    frame_dir = project_root / "demo_frames"

    # 1. Download audio
    download_audio(audio_path)

    # 2. Configure -- preview preset + flow mode (img2img feedback loop)
    #    Add fast_mode="lightning" for 4-step inference (extra ~1 GB
    #    LoRA download, ~4x faster per frame).
    config = GenerationConfig.preview(seed=42, mode="flow")
    pipeline = RufusPipeline(config=config)

    # 3. Define the landscape journey
    #    These keyframes are spaced across the first 30 seconds.
    #    The library will smoothly morph between each scene, with
    #    the rate of change driven by the music's energy.
    keyframes = [
        {
            "time": 0,
            "prompt": (
                "vast desert dunes stretching to the horizon under a "
                "purple and magenta twilight sky, cinematic wide angle, "
                "volumetric god rays, photorealistic"
            ),
        },
        {
            "time": 10,
            "prompt": (
                "towering snow-capped mountain peaks with waterfalls "
                "cascading into a misty valley, golden hour sunlight, "
                "epic landscape photography"
            ),
        },
        {
            "time": 20,
            "prompt": (
                "dense bioluminescent forest at night, glowing blue and "
                "green mushrooms, ethereal mist, moonlight filtering "
                "through ancient trees, dreamlike atmosphere"
            ),
        },
        {
            "time": 30,
            "prompt": (
                "aerial view of a glowing coastal city at twilight, "
                "neon reflections on calm ocean water, dramatic cumulus "
                "clouds lit from below, cinematic drone shot"
            ),
        },
    ]

    # 4. Generate
    print(f"\nStarting generation (preview preset: 512x512 @ 8 FPS)...")
    print(f"Device: {config.device} (dtype: {config.torch_dtype})")
    print(f"Frames will be saved to: {frame_dir}")
    print(f"This allows resuming if interrupted.\n")

    result = pipeline.generate(
        audio_path=str(audio_path),
        keyframes=keyframes,
        output_path=str(output_path),
        duration=DEMO_DURATION_SECONDS,
        frame_dir=str(frame_dir),
        keep_frames=True,
        use_dashboard=True,
    )

    print(f"\nDemo complete! Video saved to: {result}")
    print(f"Frame images preserved in: {frame_dir}")


if __name__ == "__main__":
    main()
