"""Pipeline runner that wires all steps together."""

from __future__ import annotations

import os
from typing import Set

from checkpoint import clear_checkpoint, completed_step_names, mark_step_complete
from console import print_info
from filesystem import resume_directories
from models import PipelineConfig, PipelineContext
from steps import (
    BlurDetectionStep,
    DirectorySetupStep,
    ExportFormatsStep,
    FinalizationStep,
    OverlapSelectionStep,
    RadiometricNormalizationStep,
    SemanticMaskingStep,
    StructureFromMotionStep,
)


class PipelineRunner:
    """Small orchestrator that runs all pipeline steps in order."""

    # These steps always run — setup wires the paths; finalize writes the report.
    _ALWAYS_RUN: Set[str] = {"DirectorySetupStep", "FinalizationStep"}

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.steps = [
            DirectorySetupStep(config),
            BlurDetectionStep(config),
            OverlapSelectionStep(config),
            RadiometricNormalizationStep(config),
            SemanticMaskingStep(config),
            StructureFromMotionStep(config),
            ExportFormatsStep(config),
            FinalizationStep(config),
        ]

    def _restore_context(self, context: PipelineContext, already_done: Set[str]) -> PipelineContext:
        """Re-populate context fields from disk for steps that were skipped."""

        if context.paths is None:
            return context

        # Re-load saved_images from the images directory when radiometric was already done.
        if "RadiometricNormalizationStep" in already_done and not context.saved_images:
            images_dir = context.paths.images
            if os.path.isdir(images_dir):
                exts = {".png", ".jpg", ".jpeg"}
                context.saved_images = sorted(
                    os.path.join(images_dir, f)
                    for f in os.listdir(images_dir)
                    if os.path.splitext(f)[1].lower() in exts
                )

        return context

    def run(self) -> PipelineContext:
        output_root = self.config.output_root

        if not self.config.resume:
            clear_checkpoint(output_root)

        already_done = completed_step_names(output_root)
        is_resuming = bool(already_done) and self.config.resume

        if is_resuming:
            resumable = [
                s.step_name for s in self.steps
                if s.step_name not in self._ALWAYS_RUN and s.step_name in already_done
            ]
            if resumable:
                print_info(f"Resuming — skipping completed steps: {', '.join(resumable)}")

        context = PipelineContext(resuming=is_resuming)

        for step in self.steps:
            name = step.step_name

            skip = (
                is_resuming
                and name not in self._ALWAYS_RUN
                and name in already_done
            )

            if skip:
                print_info(f"  [dim]↩  Skipping (checkpoint):[/dim] {name}")
                # After setup runs, restore filesystem-based context for skipped steps.
                if context.paths is not None:
                    context = self._restore_context(context, already_done)
                continue

            context = step.run(context)

            if name not in self._ALWAYS_RUN:
                mark_step_complete(output_root, name)

        # Clean up checkpoint on successful completion.
        clear_checkpoint(output_root)
        return context
