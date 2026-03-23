"""Rich console helpers for readable pipeline logs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from models import PipelineConfig, PipelinePaths

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Basic message helpers
# ---------------------------------------------------------------------------

def print_section(step_number: int, title: str, goal: str) -> None:
    """Print a rich panel header for a pipeline step."""
    console.print()
    console.print(Panel(
        f"[dim]{goal}[/dim]",
        title=f"[bold cyan]Step {step_number}[/bold cyan]  [bold white]{title}[/bold white]",
        border_style="cyan",
        padding=(0, 1),
    ))


def print_info(message: str) -> None:
    console.print(f"  [blue]→[/blue]  {message}")


def print_warning(message: str) -> None:
    console.print(f"  [yellow]⚠[/yellow]  [yellow]{message}[/yellow]")


def print_success(message: str) -> None:
    console.print(f"  [green]✓[/green]  {message}")


def print_error(message: str) -> None:
    console.print(f"  [red]✗[/red]  [red bold]{message}[/red bold]")


def print_pipeline_error(exc: Exception) -> None:
    """Show a formatted error panel when a pipeline step fails."""
    console.print()
    console.print(Panel(
        str(exc),
        title="[bold red]Pipeline stopped[/bold red]",
        border_style="red",
        padding=(1, 2),
    ))
    console.print(
        "  [dim]Run [bold]python preprocess.py doctor[/bold] to check your environment, "
        "or add [bold]--help[/bold] to see available options.[/dim]"
    )


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def create_progress(description_width: int = 32) -> Progress:
    """Return a standard rich Progress bar for pipeline loops."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[progress.description]{{task.description:<{description_width}}}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Context manager that shows an animated spinner for indeterminate work."""
    with console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
        yield


# ---------------------------------------------------------------------------
# Run-plan panel
# ---------------------------------------------------------------------------

def _yes_no(flag: bool) -> str:
    return "[green]yes[/green]" if flag else "[dim]no[/dim]"


def _on_off(flag: bool) -> str:
    return "[green]on[/green]" if flag else "[dim]off[/dim]"


def print_run_plan(config: PipelineConfig) -> None:
    """Show the resolved runtime config in a rich table before the run starts."""
    console.print()
    console.print(Panel.fit(
        "[bold green]Video to Dataset[/bold green]  [dim]— preprocessing pipeline[/dim]",
        border_style="green",
    ))

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
    table.add_column("Setting", style="dim", no_wrap=True)
    table.add_column("Value")

    table.add_row("Preset", f"[bold]{config.preset}[/bold]")
    table.add_row("Input", config.video_path)
    table.add_row("Output", config.output_root)
    table.add_row("Extraction FPS", f"{config.extraction_fps:.2f}")
    table.add_row("Blur threshold", f"{config.blur_threshold:.2f}")
    table.add_row(
        "Overlap range",
        f"{config.min_overlap:.2f} [dim]→[/dim] {config.max_overlap:.2f}  "
        f"[dim](target {config.target_overlap:.2f})[/dim]",
    )
    table.add_row("Auto tune", _on_off(config.auto_tune))
    max_dim = str(config.max_image_dim) if config.max_image_dim > 0 else "[dim]original[/dim]"
    table.add_row("Max image dim", max_dim)
    table.add_row("Color mode", config.color_mode)
    table.add_row("Deblur strength", f"{config.deblur_strength:.2f}")
    cleanup = (
        f"white-balance={_on_off(config.enable_white_balance)}  "
        f"CLAHE={_on_off(config.enable_clahe)}  "
        f"contrast={_on_off(config.enable_local_contrast)}"
    )
    table.add_row("Image cleanup", cleanup)
    table.add_row("Semantic masking", _yes_no(config.run_semantic_masking))
    if config.run_semantic_masking:
        table.add_row("  Mask backend", config.mask_backend)
        table.add_row("  Mask model", config.mask_model)
        table.add_row("  Mask classes", ", ".join(config.mask_classes))
        table.add_row("  Mask device", config.mask_device)
        table.add_row("  Confidence", f"{config.mask_confidence:.2f}")
        table.add_row("  Strict masking", _on_off(config.strict_static_masking))
    max_images = str(config.output_max_images) if config.output_max_images > 0 else "[dim]unbounded[/dim]"
    table.add_row(
        "Angle coverage",
        f"{_on_off(config.validate_angle_coverage)}  [dim]({config.angle_bins} bins)[/dim]",
    )
    table.add_row("Output image range", f"{config.output_min_images} [dim]→[/dim] {max_images}")
    table.add_row("Run COLMAP", _yes_no(config.run_sfm))
    if config.run_sfm:
        table.add_row("  Matcher", config.matcher)
        table.add_row("  Device", config.colmap_device)
        table.add_row("  Parallel", _on_off(config.colmap_parallel))
        threads = str(config.colmap_num_threads) if config.colmap_num_threads > 0 else "[dim]auto[/dim]"
        table.add_row("  Threads", threads)
    if config.quality_gate_min_overlap > 0:
        table.add_row(
            "Quality gate",
            f"mean overlap [bold]≥ {config.quality_gate_min_overlap:.2f}[/bold]  "
            f"[dim](fail={'yes' if config.quality_gate_fail else 'warn'})[/dim]",
        )

    console.print(table)
    console.rule(style="dim")


# ---------------------------------------------------------------------------
# Final summary panel
# ---------------------------------------------------------------------------

def print_final_summary(paths: PipelinePaths, run_sfm: bool) -> None:
    """Show a rich summary panel when the pipeline finishes."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
    table.add_column("Item", style="dim", no_wrap=True)
    table.add_column("Path", style="bold green")

    table.add_row("Dataset folder", paths.root)
    table.add_row("Images", paths.images)
    table.add_row("Semantic masks", paths.masks)
    if run_sfm:
        table.add_row("Undistorted COLMAP", paths.undistorted)
    table.add_row("Run report", paths.report)

    console.print()
    console.print(Panel(
        table,
        title="[bold green]Pipeline complete[/bold green]",
        border_style="green",
    ))

    # Next-step hints
    console.print()
    if run_sfm:
        console.print("  [dim]What to do next:[/dim]")
        console.print("  [dim]  • Load the undistorted folder into NeRFStudio, Instant-NGP, or 3D Gaussian Splatting.[/dim]")
        console.print("  [dim]  • Re-run with [bold]--export-format nerfstudio[/bold] to get a transforms.json automatically.[/dim]")
        console.print("  [dim]  • Inspect quality metrics: [bold]python preprocess.py inspect " + paths.root + "[/bold][/dim]")
    else:
        console.print("  [dim]What to do next:[/dim]")
        console.print("  [dim]  • Run COLMAP on the images folder, or re-run without [bold]--no-colmap[/bold].[/dim]")
        console.print("  [dim]  • Inspect quality metrics: [bold]python preprocess.py inspect " + paths.root + "[/bold][/dim]")
