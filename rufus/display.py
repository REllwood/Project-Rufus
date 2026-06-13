"""Rich terminal dashboard for monitoring generation progress.

Provides a live-updating terminal display showing:
- Overall progress bar with ETA
- Current frame rate (frames/second)
- Active transition (prompt A -> prompt B) and interpolation progress
- Audio section label and reactive feature meters
- Last rendered frame path

The dashboard is entirely optional. The generator works without it
by falling back to a plain tqdm bar.  Enable it by passing
``use_dashboard=True`` to :meth:`RufusPipeline.generate`.

Requires the ``rich`` package (``pip install rich``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def is_available() -> bool:
    """Return ``True`` if the rich library is installed."""
    return HAS_RICH


# ------------------------------------------------------------------
# Per-frame snapshot pushed from the generator
# ------------------------------------------------------------------

@dataclass
class FrameStatus:
    """Snapshot of the current generation state for one frame."""

    frame_idx: int
    total_frames: int
    timestamp: float
    duration: float
    prompt_a: str
    prompt_b: str
    progress: float
    energy: float
    onset: float
    centroid: float
    section_label: str
    frame_path: str
    strength: float = 0.0
    skipped: bool = False


# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

class Dashboard:
    """Rich live-terminal dashboard for Rufus generation.

    Usage::

        dash = Dashboard()
        dash.start(total_frames=240, fps=12, device="mps", resolution=(512, 512))
        for frame in frames:
            ...
            dash.update(status)
        dash.finish()
    """

    def __init__(self) -> None:
        if not HAS_RICH:
            raise ImportError(
                "The 'rich' package is required for the dashboard. "
                "Install it with: pip install rich"
            )
        self._console = Console()
        self._live: Optional[Live] = None
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        self._start_time: float = 0.0
        self._frames_rendered: int = 0
        self._last_status: Optional[FrameStatus] = None
        self._fps_window: list[float] = []

        self._total_frames: int = 0
        self._target_fps: int = 0
        self._device: str = ""
        self._resolution: tuple = (0, 0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        total_frames: int,
        fps: int,
        device: str,
        resolution: tuple,
    ) -> None:
        """Begin the live display."""
        self._total_frames = total_frames
        self._target_fps = fps
        self._device = device
        self._resolution = resolution
        self._start_time = time.monotonic()
        self._frames_rendered = 0
        self._fps_window = []

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Rendering"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[dim]|"),
            TimeElapsedColumn(),
            TextColumn("[dim]remaining"),
            TimeRemainingColumn(),
        )
        self._task_id = self._progress.add_task(
            "render", total=total_frames
        )

        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.start()

    def update(self, status: FrameStatus) -> None:
        """Push a frame status update to the dashboard."""
        self._last_status = status

        if not status.skipped:
            now = time.monotonic()
            self._fps_window.append(now)
            # Keep a 20-sample sliding window for FPS calculation
            if len(self._fps_window) > 20:
                self._fps_window = self._fps_window[-20:]
            self._frames_rendered += 1

        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=status.frame_idx + 1)

        if self._live:
            self._live.update(self._build_layout())

    def finish(self, output_path: str = "") -> None:
        """Stop the live display and print the completion summary."""
        if self._live:
            self._live.stop()
            self._live = None

        elapsed = time.monotonic() - self._start_time
        avg_fps = self._frames_rendered / elapsed if elapsed > 0 else 0

        self._console.print()
        self._console.rule("[bold green]Generation complete")
        summary = Table(show_header=False, box=None, padding=(0, 2))
        summary.add_column(style="dim")
        summary.add_column()
        summary.add_row("Frames rendered", str(self._frames_rendered))
        summary.add_row("Total time", _format_duration(elapsed))
        summary.add_row("Average speed", f"{avg_fps:.2f} frames/sec")
        if output_path:
            summary.add_row("Output", output_path)
        self._console.print(summary)
        self._console.print()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> Group:
        """Assemble the full dashboard layout."""
        parts = []

        # Header
        header = Table(show_header=False, box=None, padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row(
            "Device", f"{self._device}  |  "
                      f"{self._resolution[0]}x{self._resolution[1]}  |  "
                      f"{self._target_fps} FPS target"
        )
        parts.append(header)

        # Progress bar
        if self._progress:
            parts.append(self._progress)

        # Frame details
        if self._last_status:
            parts.append(self._build_detail_panel(self._last_status))

        return Group(*parts)

    def _build_detail_panel(self, s: FrameStatus) -> Panel:
        """Build the detail panel showing per-frame information."""
        table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        table.add_column("Key", style="dim", width=18)
        table.add_column("Value")

        # Speed
        speed = self._current_fps()
        speed_str = f"{speed:.2f} frames/sec" if speed > 0 else "calculating..."
        table.add_row("Speed", speed_str)

        # Time position
        table.add_row(
            "Position",
            f"{s.timestamp:.1f}s / {s.duration:.1f}s"
        )

        # Section
        table.add_row("Section", s.section_label)

        # Active transition
        prompt_a_short = _truncate(s.prompt_a, 55)
        prompt_b_short = _truncate(s.prompt_b, 55)
        blend_bar = _blend_bar(s.progress, width=30)
        table.add_row("From", prompt_a_short)
        table.add_row("To", prompt_b_short)
        table.add_row("Blend", blend_bar)

        # Audio meters
        table.add_row("Energy", _meter(s.energy, width=30))
        table.add_row("Onset", _meter(s.onset, width=30))
        table.add_row("Brightness", _meter(s.centroid, width=30))
        if s.strength > 0:
            table.add_row("Morph speed", _meter(s.strength, width=30))

        # Last frame
        table.add_row("Last frame", s.frame_path)

        return Panel(table, title="[bold]Frame details", border_style="blue")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        if span < 0.01:
            return 0.0
        return (len(self._fps_window) - 1) / span


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _meter(value: float, width: int = 20) -> str:
    """Render a 0-1 value as a coloured bar."""
    filled = int(value * width)
    empty = width - filled

    if value < 0.33:
        colour = "green"
    elif value < 0.66:
        colour = "yellow"
    else:
        colour = "red"

    bar = f"[{colour}]{'|' * filled}[/{colour}][dim]{'.' * empty}[/dim]"
    return f"{bar} {value:.0%}"


def _blend_bar(progress: float, width: int = 20) -> str:
    """Render the A->B blend progress as a two-tone bar."""
    pos = int(progress * width)
    left = width - pos
    bar = f"[cyan]{'=' * left}[/cyan][magenta]{'=' * pos}[/magenta]"
    return f"A {bar} B  ({progress:.0%})"


def _format_duration(seconds: float) -> str:
    """Format seconds as ``Xh Ym Zs``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
