"""Basic Rufus usage example.

Generates a music-reactive landscape video from an audio file and a
set of keyframed landscape prompts.

Usage:
    python examples/basic_usage.py --audio path/to/song.mp3

Adjust the keyframes, model, and configuration to taste.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rufus import GenerationConfig, RufusPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate a Rufus landscape video.")
    parser.add_argument("--audio", required=True, help="Path to the audio file.")
    parser.add_argument("--output", default="output.mp4", help="Output video path.")
    parser.add_argument(
        "--preset",
        choices=["preview", "default", "production"],
        default="default",
        help="Quality preset.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--mode",
        choices=["flow", "morph"],
        default="flow",
        help="flow = coherent img2img feedback loop (recommended); "
             "morph = legacy independent frames.",
    )
    parser.add_argument(
        "--fast",
        choices=["lcm", "lightning"],
        default=None,
        help="Distilled fast mode (4-8 inference steps per frame).",
    )
    parser.add_argument(
        "--frame-dir",
        default=None,
        help="Directory for intermediate frames (enables resume).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show a rich terminal dashboard (requires: pip install rich).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- Configuration --
    overrides = dict(mode=args.mode, fast_mode=args.fast)
    if args.preset == "preview":
        config = GenerationConfig.preview(**overrides)
    elif args.preset == "production":
        config = GenerationConfig.production(**overrides)
    else:
        config = GenerationConfig(**overrides)

    pipeline = RufusPipeline(config=config)
    print(f"Device: {config.device} (dtype: {config.torch_dtype})")

    # -- Keyframes --
    # Define your landscape journey here.  Each keyframe specifies a
    # timestamp (seconds) and a prompt describing the scene at that
    # moment.  The library smoothly morphs between consecutive prompts.
    keyframes = [
        {
            "time": 0,
            "prompt": "vast desert dunes under a purple and orange twilight sky, "
                      "cinematic wide shot, volumetric lighting",
        },
        {
            "time": 25,
            "prompt": "towering snow-capped mountain range with cascading waterfalls, "
                      "golden hour, mist rising from valleys",
        },
        {
            "time": 50,
            "prompt": "dense bioluminescent forest at night, glowing mushrooms, "
                      "ethereal blue and green light filtering through canopy",
        },
        {
            "time": 75,
            "prompt": "aerial view of a neon-lit coastal city at sunset, "
                      "reflections on calm ocean water, dramatic clouds",
        },
        {
            "time": 100,
            "prompt": "abstract crystalline landscape under a star-filled sky, "
                      "aurora borealis, otherworldly rock formations",
        },
    ]

    # -- Generate --
    result = pipeline.generate(
        audio_path=args.audio,
        keyframes=keyframes,
        output_path=args.output,
        seed=args.seed,
        frame_dir=args.frame_dir,
        keep_frames=args.frame_dir is not None,
        use_dashboard=args.dashboard,
    )

    print(f"\nVideo saved to: {result}")


if __name__ == "__main__":
    main()
