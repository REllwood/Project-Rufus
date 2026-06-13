# Project Rufus

Project Rufus turns a song and a handful of scenes into a video where the scenery morphs in time with the music. Give it an audio track plus some text prompts or your own photos, and it renders one continuous journey that speeds up and slows down with the song.

Inspired by the [RUFUS DU SOL "Next To Me" music video](https://www.youtube.com/watch?v=GPXiL6ynVG8) by Osk Studio.





Three photos (a fox, an owl and a lion) flowing into one another in time with a track and looping back to the start. On the left, a journey rendered from text prompts; on the right, a seamless loop. These are quick low step preview renders kept small on purpose, so full quality output looks sharper than what you see here.

## Background

This started with a music video. I came across the clip for RUFUS DU SOL's "Next To Me" and could not stop watching it: AI-generated landscapes melting from one scene into the next, perfectly in time with the music. That sent me down a rabbit hole. Who made this, and how did they do it?

It turned out to be the work of Osk Studio, made in 2021. They trained AI on huge collections of landscape and cityscape imagery, then rendered a continuous journey through that learned space, so a desert becomes mountains becomes a forest becomes a glowing city, with the rate of change rising and falling with the song. It was GAN era work, a smooth walk through latent space rather than a sequence of cuts, and the transformation was tied to the structure of the track.

I wanted to see whether I could replicate that effect myself: on my own machine, with my own music, and with my own images. This is my attempt at that.

## How it works

1. **Audio analysis.** librosa pulls out beat positions, onset strength, RMS energy and spectral centroid, and works out the broad sections of the track (calm, build, peak).
2. **Prompt timeline.** You lay out keyframes, each one a scene pinned to a moment in the track, and Rufus maps them onto the audio.
3. **Frame plan.** Before any rendering happens, every per-frame control signal (how far through a transition you are, how hard to morph, how the camera moves) is worked out as a function of the audio. Louder, busier passages push transitions along faster, and the progress never runs backwards.
4. **Flow rendering** (the default). Each frame grows out of the one before it. Rufus takes the previous frame, nudges it with a small camera move (a little zoom and pan, optionally a depth-aware parallax), sharpens it slightly, adds some noise, then re-diffuses it with prompt embeddings interpolated between the surrounding keyframes. Because every frame inherits the last, the scenery flows instead of flickering. How hard each frame morphs follows the music and how fast the prompt is changing, a colour stabiliser keeps the palette from drifting, and the sharpening offsets the softening a feedback loop would otherwise build up.
5. **Video assembly.** The frames are stitched into an MP4 with ffmpeg, the original audio is muxed back in, and the result can be motion-interpolated up to a higher frame rate.

There is also an older **morph** mode, where each frame is rendered on its own from interpolated embeddings. It is kept for comparison via `mode="morph"`, and it flickers.

## Requirements

- Python 3.9 or newer
- ffmpeg (`brew install ffmpeg`, or `pip install imageio-ffmpeg` for a bundled binary)

### Hardware

The device is detected automatically (override with `device=` or `--device`):


| Hardware      | Backend       | Notes                                                                                               |
| ------------- | ------------- | --------------------------------------------------------------------------------------------------- |
| NVIDIA GPU    | `cuda`        | fp16, CPU offloading; about 6 GB VRAM for SDXL                                                      |
| Apple Silicon | `mps` (Metal) | fp32 plus attention slicing for reliability; 32 GB unified memory for SDXL                          |
| Intel GPU     | `xpu`         | fp16                                                                                                |
| CPU           | `cpu`         | the fallback for everything else; use an SD 1.5-class model, since SDXL on CPU is minutes per frame |


Both SDXL-class and SD 1.5-class checkpoints work (loaded through the diffusers `AutoPipeline`). For CPU or a low-VRAM GPU, pick a lighter model:

```python
config = GenerationConfig(
    model="stable-diffusion-v1-5/stable-diffusion-v1-5",
    fast_mode="lcm",       # LCM-LoRA: about 6 steps per frame
    width=512, height=512,
)
```

## Installation

```bash
pip install -e .                # library plus the `rufus` CLI
pip install -e ".[dashboard]"   # also installs the rich terminal dashboard
```

## Quick start

### CLI

```bash
rufus song.mp3 \
  --kf "0:vast desert dunes under purple twilight sky" \
  --kf "30:towering mountain range with cascading waterfalls" \
  --kf "60:dense bioluminescent forest at night" \
  --kf "90:aerial view of glowing coastal city at sunset" \
  -o output.mp4 --preset preview --fast lightning --dashboard
```

Or point it at a JSON keyframes file: `rufus song.mp3 --keyframes scenes.json -o output.mp4`

### Python API

```python
from rufus import RufusPipeline

pipeline = RufusPipeline(
    model="stabilityai/stable-diffusion-xl-base-1.0",
    # device is auto-detected: cuda > mps > cpu
)

pipeline.generate(
    audio_path="song.mp3",
    keyframes=[
        {"time": 0,  "prompt": "vast desert dunes under purple twilight sky"},
        {"time": 30, "prompt": "towering mountain range with cascading waterfalls"},
        {"time": 60, "prompt": "dense bioluminescent forest at night"},
        {"time": 90, "prompt": "aerial view of glowing coastal city at sunset"},
    ],
    output_path="output.mp4",
    fps=12,
    resolution=(768, 768),
    seed=42,
)
```

## Configuration

### Presets

```python
from rufus import GenerationConfig

# Fast preview (512x512, 8 FPS, fewer inference steps)
config = GenerationConfig.preview()

# High quality (1024x1024, 24 FPS, more inference steps)
config = GenerationConfig.production()

# 4-step distilled inference (much faster per frame)
config = GenerationConfig.preview(fast_mode="lightning")
```

### Key parameters


| Parameter                   | Default                                    | Description                                                                                                                                                                                                             |
| --------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                     | `stabilityai/stable-diffusion-xl-base-1.0` | Hugging Face model ID                                                                                                                                                                                                   |
| `device`                    | auto-detect                                | `cuda`, `mps`, `xpu` or `cpu`                                                                                                                                                                                           |
| `mode`                      | `flow`                                     | `flow` (coherent feedback loop) or `morph` (legacy)                                                                                                                                                                     |
| `fast_mode`                 | `None`                                     | `"lightning"` or `"lcm"`, for 4 to 8 step distilled inference                                                                                                                                                           |
| `width` / `height`          | 768                                        | Output resolution                                                                                                                                                                                                       |
| `fps`                       | 12                                         | Frames per second                                                                                                                                                                                                       |
| `flow_strength_min` / `max` | 0.30 / 0.55                                | img2img denoise range; audio energy maps it from calm to peak. Set at `morph_rate_reference_fps` (12) and scaled to the frame rate, so a higher fps gives a smaller change per frame at the same morph speed per second |
| `temporal_noise`            | `fixed`                                    | `fixed` reuses one noise pattern across frames (stable, no shimmer); `varying` redraws it per frame (livelier, but it boils)                                                                                            |
| `keyframe_hold`             | 0.35                                       | Fraction of each span the scene rests before it starts morphing                                                                                                                                                         |
| `flow_velocity_influence`   | 0.6                                        | Extra denoise during fast prompt transitions so scenes do not lag the prompt                                                                                                                                            |
| `sharpen_amount`            | 0.3                                        | Per-frame unsharp mask that counters feedback-loop softening                                                                                                                                                            |
| `flow_zoom_min` / `max`     | 1.002 / 1.012                              | Per-frame zoom factor range (compounds into motion)                                                                                                                                                                     |
| `enable_depth_warp`         | False                                      | Depth-aware parallax camera (downloads a small depth model)                                                                                                                                                             |
| `color_coherence`           | True                                       | Stops the feedback loop drifting in colour over time                                                                                                                                                                    |
| `smooth_fps`                | None                                       | Motion-interpolate the final video to this frame rate with ffmpeg                                                                                                                                                       |
| `onset_sensitivity`         | 1.0                                        | How much audio onsets speed up transitions                                                                                                                                                                              |
| `beat_transition_boost`     | 1.5                                        | Extra transition speed near beats                                                                                                                                                                                       |
| `beat_zoom_pulse`           | 0.006                                      | A small zoom kick on each beat                                                                                                                                                                                          |
| `resume`                    | True                                       | Continue an interrupted render from the last saved frame                                                                                                                                                                |


## Beyond landscapes

Prompts are free text, so the same audio-reactive morphing works for any subject: animals, architecture, abstract art, portraits. You can also start the video from your own photo instead of a generated frame:

```python
pipeline.generate(
    audio_path="song.mp3",
    init_image="my_dog.jpg",   # the video starts from this photo,
    keyframes=[                # stylised toward the first prompt
        {"time": 0,  "prompt": "a golden retriever in a sunlit meadow"},
        {"time": 30, "prompt": "a wolf on a moonlit mountain ridge"},
        {"time": 60, "prompt": "a white tiger in a misty bamboo forest"},
    ],
    output_path="output.mp4",
)
```

On the CLI that is `--init-image photo.jpg`. The `init_image_strength` setting controls how strongly frame 0 is pushed toward the first prompt: 0 keeps the photo, 1 ignores it.

### Morphing through a photo series

Keyframes can be images instead of prompts, or as well as them, and the video morphs through the photo series in time with the music:

```python
pipeline.generate(
    audio_path="song.mp3",
    keyframes=[
        {"time": 0,  "image": "holiday_beach.jpg"},
        {"time": 30, "image": "holiday_mountains.jpg"},
        {"time": 60, "image": "holiday_city.jpg",
         "prompt": "a vibrant city at night"},   # an optional style steer
    ],
    output_path="memories.mp4",
)
```

On the CLI: `rufus song.mp3 --kf-image 0:beach.jpg --kf-image 30:mountains.jpg`, which mixes freely with `--kf` text keyframes.

Photo keyframes guide the render in three ways at once:

1. **Pixel pull.** Each frame is gently blended toward the cross-dissolve of the surrounding photos before it is re-diffused (`image_pull`).
2. **Auto-captioning.** Photos without a prompt are described by BLIP, and those captions drive the text-embedding journey (`auto_caption`).
3. **IP-Adapter.** Each photo's CLIP image embedding conditions the diffusion directly and is interpolated per frame, so the model sees the actual photo rather than a one-line description of it (`enable_ip_adapter`, `ip_adapter_scale`; the weights download on first use).

## Looping, auto-timing and easing

```python
pipeline.generate(
    audio_path="song.mp3",
    keyframes=[                       # no timestamps, so Rufus places
        {"prompt": "desert dunes"},   # them at detected section
        {"prompt": "mountain peaks"}, # boundaries, snapped to beats
        {"image": "city.jpg"},
    ],
    output_path="loop.mp4",
)
```

- **Auto-timing.** Leave `time` off every keyframe and they are placed at musically sensible moments (section changes, snapped to beats).
- **Seamless loop.** With `loop=True` (or `--loop`), the journey returns to the first keyframe and the closing frames converge back onto frame 0, so the video loops cleanly.
- **Eased transitions.** Scene changes start and end gently (`transition_ease`, where 0 is linear and 1 is full smoothstep).
- **Scene holds.** Each scene rests before its morph begins (`keyframe_hold`, default 0.35, roughly the first third of each span), so the scenes read as stable images rather than a constant mid-morph.

For smooth results, give transitions room to breathe, with scene changes every 15 to 30 seconds rather than every 2 or 3. For per-frame smoothness, raise the base `fps`: the per-frame morph strength scales down to match, so 24 FPS gives small steps every frame rather than a faster boil. Rendering at 8 to 12 FPS with `smooth_fps=24` is the cheaper alternative.

## Terminal dashboard

Pass `use_dashboard=True` to `generate()` (or `--dashboard` on the CLI) for a live terminal display showing progress with an ETA, render speed, the current transition and its blend progress, the audio meters (energy, onset, brightness) and the current morph speed.

## Architecture

```
Audio File ──> Audio Analysis ──┐
                                ├──> Frame Plan (progress/strength/camera) ──┐
Landscape Prompts ──> Timeline ─┘                                            │
                                                                             v
              ┌── frame N-1 ── camera move ── img2img re-diffuse ── frame N ──┐
              └───────────────────────── feedback loop ──────────────────────┘
                                                                             │
                                              ffmpeg assembly + audio mux <──┘
```

### Modules

- `rufus.audio`: audio analysis via librosa
- `rufus.prompts`: keyframe and timeline management
- `rufus.reactivity`: the precomputed per-frame plan (progress, strength, camera)
- `rufus.interpolation`: SLERP, lerp and circular latent walks
- `rufus.motion`: camera transforms, depth parallax warp and the colour stabiliser
- `rufus.generator`: the diffusion rendering engine (flow and morph modes)
- `rufus.video`: MP4 assembly via ffmpeg
- `rufus.pipeline`: the high-level orchestrator
- `rufus.cli`: the `rufus` command-line entry point
- `rufus.config`: central configuration

## Running the demos

### No GPU required: synthetic visualiser

```bash
python examples/demo_no_gpu.py
```

Downloads a royalty-free Creative Commons audio clip ("Winter" by Zoe Blade, via the Internet Archive), analyses it, generates synthetic colour-gradient frames driven by the same audio-reactive frame plan the AI renderer uses, and assembles an MP4.

### Full AI generation (needs a GPU)

```bash
python examples/demo.py
```

Downloads the same audio clip and generates AI landscapes with SDXL in preview mode (512x512 at 8 FPS), trimmed to the first 30 seconds. The model weights (about 6 GB) download on first run, and frames are saved as they go so you can interrupt and resume.

### Running the tests

```bash
python -m pytest tests/ -v
```

The suite covers the interpolation maths, the prompt timeline, audio analysis (against a synthetic WAV), the audio-reactive frame plan (monotonicity, determinism, response to energy), ffmpeg video assembly, and, through a fake diffusion pipeline, the full render loop in both modes, including resume and temporal coherence.

## Performance notes

Per-frame diffusion is what takes the time. A few ways to keep it down, roughly in order of how much they help:

- **Fast mode.** `fast_mode="lightning"` (or `"lcm"`) cuts inference from about 20 steps to 4 to 8 per frame.
- **Render low, smooth up.** Render at `fps=10` and set `smooth_fps=24`; ffmpeg's motion interpolation fills in the rest, and it suits the morphing style well.
- **Preview preset.** 512x512 for fast iteration.
- **Resume.** Interrupt and restart without losing progress (pass `frame_dir=` to keep the frames).
- **CPU offloading.** On by default on CUDA to reduce VRAM use.
- **Apple Silicon.** The MPS backend is detected automatically, with attention slicing and float32 for reliability.

## Licence and models

The Project Rufus source code in this repository is released under the MIT licence (see `LICENSE`).

Project Rufus does not bundle any model weights. It downloads them at runtime from Hugging Face. Each model has its own licence, and it is your responsibility to check and follow the licence for every model you use, particularly for any commercial use. The models Rufus can download are:

- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)
- [LCM-LoRA for SDXL](https://huggingface.co/latent-consistency/lcm-lora-sdxl)
- [LCM-LoRA for SD 1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [BLIP image captioning](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)

This is general information, not legal advice.

## Responsible use

Project Rufus will cheerfully render whatever you point it at and ask no questions, so a few sensible boundaries:

- **Safety checker is disabled.** The iterative generation loop breaks if the safety checker rejects even a single frame — downstream frames inherit the black output and the entire sequence collapses. Disabling it was a practical necessity, but it means there is no content filter on the output. Use responsibly.
- **Respect image rights.** Only use source images you have the rights to. Do not use this tool to manipulate real people's likenesses or to replicate a living artist's style and pass it off as their work.
- **License your audio.** The demo uses "Winter" by Zoe Blade under Creative Commons BY-NC-ND. The "no derivatives" clause makes any derivative music video unsuitable for public distribution. For anything you intend to share, use your own audio or a track with an appropriate licence.
- **First run downloads models.** Several gigabytes of model weights are fetched on the initial run. A stable internet connection is required.

