"""Command-line interface for Rufus.

Installed as the ``rufus`` console script::

    rufus song.mp3 --keyframes keyframes.json -o output.mp4

The keyframes file is a JSON list of ``{"time": ..., "prompt": ...}``
objects (optional keys: ``negative_prompt``, ``style``).  Alternatively
pass inline keyframes with repeated ``--kf "TIME:PROMPT"`` flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List


def _parse_inline_keyframe(spec: str) -> Dict:
    time_str, _, prompt = spec.partition(":")
    if not prompt:
        raise argparse.ArgumentTypeError(
            f"--kf expects 'TIME:PROMPT', got {spec!r}"
        )
    try:
        return {"time": float(time_str), "prompt": prompt.strip()}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--kf time must be a number, got {time_str!r}"
        )


def _parse_inline_image_keyframe(spec: str) -> Dict:
    time_str, _, path = spec.partition(":")
    if not path:
        raise argparse.ArgumentTypeError(
            f"--kf-image expects 'TIME:PATH', got {spec!r}"
        )
    try:
        return {"time": float(time_str), "image": path.strip()}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--kf-image time must be a number, got {time_str!r}"
        )


def _load_keyframes(args: argparse.Namespace) -> List[Dict]:
    if args.keyframes:
        data = json.loads(Path(args.keyframes).read_text())
        if not isinstance(data, list):
            raise SystemExit("Keyframes file must contain a JSON list.")
        return data
    inline = (args.kf or []) + (args.kf_image or [])
    if inline:
        return sorted(inline, key=lambda k: k["time"])
    raise SystemExit(
        "No keyframes given. Use --keyframes FILE.json, or repeated "
        "--kf 'TIME:PROMPT' / --kf-image 'TIME:PATH' flags."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rufus",
        description="Generate a music-reactive AI landscape video.",
    )
    parser.add_argument("audio", help="Path to the input audio file.")
    parser.add_argument("-o", "--output", default="output.mp4",
                        help="Output video path (default: output.mp4).")
    parser.add_argument("--keyframes", help="JSON file with keyframe list.")
    parser.add_argument("--kf", action="append", type=_parse_inline_keyframe,
                        metavar="TIME:PROMPT",
                        help="Inline text keyframe; repeat for each scene.")
    parser.add_argument("--kf-image", action="append",
                        type=_parse_inline_image_keyframe, metavar="TIME:PATH",
                        help="Inline photo keyframe; the video morphs "
                             "through these images. Mixes with --kf.")
    parser.add_argument("--preset", choices=["preview", "default", "production"],
                        default="default", help="Quality preset.")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID or local checkpoint path. "
                             "SDXL-class and SD1.5-class models both work; "
                             "use an SD1.5 model for CPU or low-VRAM GPUs.")
    parser.add_argument("--device", default=None,
                        choices=["cuda", "mps", "xpu", "cpu"],
                        help="Override device (default: auto-detect).")
    parser.add_argument("--mode", choices=["flow", "morph"], default="flow",
                        help="flow = coherent feedback loop (recommended); "
                             "morph = legacy independent frames.")
    parser.add_argument("--fast", choices=["lcm", "lightning"], default=None,
                        help="Distilled fast mode (4-8 steps/frame).")
    parser.add_argument("--depth-warp", action="store_true",
                        help="Depth-aware parallax camera motion (flow mode).")
    parser.add_argument("--smooth-fps", type=int, default=None,
                        help="Motion-interpolate the final video to this FPS.")
    parser.add_argument("--loop", action="store_true",
                        help="Seamless loop: the video returns to its "
                             "opening scene and frame.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Only render the first N seconds.")
    parser.add_argument("--init-image", default=None,
                        help="Start the video from this image (photo or "
                             "artwork), stylised toward the first prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--frame-dir", default=None,
                        help="Directory for intermediate frames (enables resume).")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show the rich terminal dashboard.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    keyframes = _load_keyframes(args)

    from .config import GenerationConfig, describe_device
    from .pipeline import RufusPipeline

    overrides = dict(
        mode=args.mode,
        fast_mode=args.fast,
        enable_depth_warp=args.depth_warp,
        smooth_fps=args.smooth_fps,
        loop=args.loop,
        seed=args.seed,
    )
    if args.model:
        overrides["model"] = args.model
    if args.device:
        overrides["device"] = args.device
    if args.preset == "preview":
        config = GenerationConfig.preview(**overrides)
    elif args.preset == "production":
        config = GenerationConfig.production(**overrides)
    else:
        config = GenerationConfig(**overrides)

    pipeline = RufusPipeline(config=config)
    print(f"Device: {config.device} [{describe_device(config.device)}] "
          f"(dtype: {config.torch_dtype}, mode: {config.mode}, "
          f"fast: {config.fast_mode or 'off'})")

    result = pipeline.generate(
        audio_path=args.audio,
        keyframes=keyframes,
        output_path=args.output,
        duration=args.duration,
        init_image=args.init_image,
        frame_dir=args.frame_dir,
        keep_frames=args.frame_dir is not None,
        use_dashboard=args.dashboard,
    )
    print(f"\nVideo saved to: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
