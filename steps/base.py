"""Base class for all pipeline steps."""

from __future__ import annotations

from console import print_section
from models import PipelineConfig, PipelineContext


class PipelineStep:
    """Small base class used by all pipeline steps."""

    step_number: int | None = None
    title = ""
    goal = ""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def announce(self) -> None:
        if self.step_number is None:
            return
        print_section(self.step_number, self.title, self.goal)

    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError
