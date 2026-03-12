"""Finalization step."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

import numpy as np

from console import print_final_summary
from filesystem import cleanup_temporary_candidates
from models import PipelineContext
from .base import PipelineStep


class FinalizationStep(PipelineStep):
    """Write the report and clean temporary files."""

    def build_report(self, context: PipelineContext) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "video_path": self.config.video_path,
            "output_root": self.config.output_root,
            "parameters": {
                key: value
                for key, value in asdict(self.config).items()
                if key not in {"video_path", "output_root"}
            },
            "candidate_stats": context.candidate_stats,
            "selection_stats": {
                "selected_frames": len(context.selected_frames),
                "overlap_violations": context.overlap_violations,
                "mean_overlap": float(np.mean(context.overlaps)) if context.overlaps else 0.0,
                "min_overlap_observed": float(min(context.overlaps)) if context.overlaps else 0.0,
                "max_overlap_observed": float(max(context.overlaps)) if context.overlaps else 0.0,
            },
            "output_images": len(context.saved_images),
            "semantic_masking": context.semantic_masking_result,
            "colmap": context.colmap_result,
        }

    @staticmethod
    def write_report(report_path: str, report: Dict[str, Any]) -> None:
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.paths is None:
            raise RuntimeError("Output folders were not prepared before finalization.")

        report = self.build_report(context)
        self.write_report(context.paths.report, report)
        cleanup_temporary_candidates(context.paths.candidates)
        print_final_summary(context.paths, run_sfm=self.config.run_sfm)
        return context
