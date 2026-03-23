"""Entry point for the video preprocessing framework.

This file stays intentionally small.
It forwards the user command to the framework CLI and runner.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

from rich import box
from rich.panel import Panel
from rich.table import Table

from cli import build_config_from_args, handle_special_command, parse_args, run_interactive_shell
from console import console, print_error, print_info, print_pipeline_error, print_success, print_warning
from runner import PipelineRunner


def run_pipeline_from_args(args: argparse.Namespace, video_path: str | None = None) -> None:
    """Build the config for one video and run the pipeline."""

    config = build_config_from_args(args, video_path=video_path)
    PipelineRunner(config).run()


def run_batch(args: argparse.Namespace) -> int:
    """Process multiple videos sequentially and show a rich summary table."""

    videos: List[str] = args.resolved_videos
    if len(videos) == 1:
        try:
            run_pipeline_from_args(args, video_path=videos[0])
        except RuntimeError as exc:
            print_pipeline_error(exc)
            return 1
        return 0

    console.print()
    console.print(Panel.fit(
        f"[bold green]Batch run[/bold green]  [dim]— {len(videos)} videos[/dim]",
        border_style="green",
    ))

    results: List[Tuple[str, str, float, str]] = []  # (name, status, elapsed, note)

    for idx, video_path in enumerate(videos, start=1):
        name = os.path.basename(video_path)
        print_info(f"[{idx}/{len(videos)}] Starting: {name}")
        t0 = time.monotonic()
        try:
            run_pipeline_from_args(args, video_path=video_path)
            elapsed = time.monotonic() - t0
            results.append((name, "done", elapsed, ""))
            print_success(f"Finished: {name}  ({elapsed:.1f}s)")
        except RuntimeError as exc:
            elapsed = time.monotonic() - t0
            results.append((name, "failed", elapsed, str(exc).splitlines()[0]))
            print_pipeline_error(exc)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            results.append((name, "failed", elapsed, str(exc)))
            print_error(f"Failed: {name} — {exc}")

    # Summary table
    table = Table(title="Batch Summary", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Video", style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Elapsed", justify="right", style="dim")
    table.add_column("Note", style="dim")

    for i, (name, status, elapsed, note) in enumerate(results, start=1):
        status_str = "[green]done[/green]" if status == "done" else "[red]failed[/red]"
        table.add_row(str(i), name, status_str, f"{elapsed:.1f}s", note)

    console.print()
    console.print(table)

    failed = sum(1 for _, s, _, _ in results if s == "failed")
    if failed:
        print_warning(f"{failed}/{len(videos)} video(s) failed.")
        return 1
    print_success(f"All {len(videos)} video(s) processed successfully.")
    return 0


def main() -> None:
    """Run the framework command requested by the user."""

    args = parse_args()
    if args.command == "shell":
        raise SystemExit(run_interactive_shell(run_pipeline_from_args))

    special_command_exit_code = handle_special_command(args)
    if special_command_exit_code is not None:
        raise SystemExit(special_command_exit_code)

    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()
