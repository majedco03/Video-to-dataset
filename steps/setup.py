"""Initial setup step."""

from __future__ import annotations

from console import print_info, print_run_plan
from filesystem import resume_directories, setup_directories
from models import PipelineContext
from .base import PipelineStep


class DirectorySetupStep(PipelineStep):
    """Prepare folders before the heavy work starts."""

    def run(self, context: PipelineContext) -> PipelineContext:
        print_run_plan(self.config)
        if context.resuming:
            print_info("[dim]Resume mode — preserving existing output files.[/dim]")
            context.paths = resume_directories(self.config.output_root)
        else:
            context.paths = setup_directories(self.config.output_root)
        return context
