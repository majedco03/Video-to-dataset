"""Entry point for the video preprocessing framework.

This file stays intentionally small.
It forwards the user command to the framework CLI and runner.
"""

from __future__ import annotations

import argparse

from cli import build_config_from_args, handle_special_command, parse_args, run_interactive_shell
from runner import PipelineRunner


def run_pipeline_from_args(args: argparse.Namespace) -> None:
    """Build the config and run the pipeline once."""

    config = build_config_from_args(args)
    PipelineRunner(config).run()


def main() -> None:
    """Run the framework command requested by the user."""

    args = parse_args()
    if args.command == "shell":
        raise SystemExit(run_interactive_shell(run_pipeline_from_args))

    special_command_exit_code = handle_special_command(args)
    if special_command_exit_code is not None:
        raise SystemExit(special_command_exit_code)

    run_pipeline_from_args(args)


if __name__ == "__main__":
    main()
