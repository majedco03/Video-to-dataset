"""Pipeline runner that wires all steps together."""

from __future__ import annotations

from models import PipelineConfig, PipelineContext
from steps import (
    BlurDetectionStep,
    DirectorySetupStep,
    FinalizationStep,
    OverlapSelectionStep,
    RadiometricNormalizationStep,
    SemanticMaskingStep,
    StructureFromMotionStep,
)


class PipelineRunner:
    """Small orchestrator that runs all pipeline steps in order."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.steps = [
            DirectorySetupStep(config),
            BlurDetectionStep(config),
            OverlapSelectionStep(config),
            RadiometricNormalizationStep(config),
            SemanticMaskingStep(config),
            StructureFromMotionStep(config),
            FinalizationStep(config),
        ]

    def run(self) -> PipelineContext:
        context = PipelineContext()
        for step in self.steps:
            context = step.run(context)
        return context
