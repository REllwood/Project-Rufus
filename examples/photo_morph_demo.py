"""Photo-series morph demo -- the reference "smooth" configuration.

Morphs through a series of photos in sync with a music track, with
seamless looping, depth-parallax camera motion, scene holds, and
fixed temporal noise for shimmer-free frames.

Usage:
    python examples/photo_morph_demo.py --audio song.mp3 \
        --photos a.jpg b.jpg c.jpg -o out.mp4

The settings here are the smoothness recipe:
- base fps 24 (per-frame morph strength auto-scales down, so high fps
  means small steps per frame, not a faster morph)
- temporal_noise="fixed" (no texture shimmer)
- keyframes auto-placed at section boundaries with loop=True
- scene holds via the keyframe_hold default
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rufus import GenerationConfig, RufusPipeline


def main():
    parser = argparse.ArgumentParser(description="Photo-series morph demo.")
    parser.add_argument("--audio", required=True, help="Path to the music track.")
    parser.add_argument("--photos", nargs="+", required=True,
                        help="Photo paths, in journey order.")
    parser.add_argument("-o", "--output", default="photo_morph.mp4")
    parser.add_argument("--duration", type=float, default=None,
                        help="Only use the first N seconds of the track.")
    parser.add_argument("--model",
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = GenerationConfig(
        model=args.model,
        mode="flow",
        width=512, height=512,
        fps=args.fps,
        num_inference_steps=12,
        loop=not args.no_loop,
        enable_depth_warp=not args.no_depth,
        seed=args.seed,
    )
    pipeline = RufusPipeline(config=config)

    result = pipeline.generate(
        audio_path=args.audio,
        keyframes=[{"image": p} for p in args.photos],  # auto-placed
        output_path=args.output,
        duration=args.duration,
        frame_dir=str(Path(args.output).with_suffix("")) + "_frames",
        use_dashboard=True,
    )
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
