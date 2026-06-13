"""Project Rufus: music-reactive AI landscape video generator."""

from .audio import AudioProfile, Section, analyse, suggest_keyframe_times
from .config import GenerationConfig, describe_device
from .display import Dashboard, FrameStatus
from .generator import FrameGenerator
from .interpolation import circular_walk, lerp, slerp, slerp_batch
from .motion import ColorStabiliser, DepthWarper, zoom_pan
from .pipeline import RufusPipeline
from .prompts import Keyframe, PromptTimeline
from .reactivity import FramePlan, build_plan
from .video import assemble

__version__ = "0.2.0"

__all__ = [
    "AudioProfile",
    "Section",
    "analyse",
    "suggest_keyframe_times",
    "GenerationConfig",
    "describe_device",
    "Dashboard",
    "FrameStatus",
    "FrameGenerator",
    "circular_walk",
    "lerp",
    "slerp",
    "slerp_batch",
    "ColorStabiliser",
    "DepthWarper",
    "zoom_pan",
    "RufusPipeline",
    "Keyframe",
    "PromptTimeline",
    "FramePlan",
    "build_plan",
    "assemble",
]
