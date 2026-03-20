"""Finalization step."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

import numpy as np

from console import print_final_summary, print_warning
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
            "selection_coverage": context.selection_coverage,
            "output_images": len(context.saved_images),
            "semantic_masking": context.semantic_masking_result,
            "colmap": context.colmap_result,
        }

    def evaluate_quality_gates(self, context: PipelineContext) -> Dict[str, Any]:
        mean_overlap = float(np.mean(context.overlaps)) if context.overlaps else 0.0
        checks = {
            "min_overlap": {
                "enabled": self.config.quality_gate_min_overlap > 0.0,
                "required": self.config.quality_gate_min_overlap,
                "observed": mean_overlap,
                "passed": (
                    True
                    if self.config.quality_gate_min_overlap <= 0.0
                    else mean_overlap >= self.config.quality_gate_min_overlap
                ),
            }
        }

        failed_labels = [name for name, result in checks.items() if result["enabled"] and not result["passed"]]
        summary = "All enabled quality gates passed." if not failed_labels else f"Failed gates: {', '.join(failed_labels)}"

        return {
            "enabled": any(bool(result["enabled"]) for result in checks.values()),
            "fail_on_error": self.config.quality_gate_fail,
            "summary": summary,
            "checks": checks,
            "passed": len(failed_labels) == 0,
        }

    @staticmethod
    def write_report(report_path: str, report: Dict[str, Any]) -> None:
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.paths is None:
            raise RuntimeError("Output folders were not prepared before finalization.")

        quality_gates = self.evaluate_quality_gates(context)
        if quality_gates["enabled"] and not quality_gates["passed"]:
            print_warning(f"Quality gate check failed. {quality_gates['summary']}")
            if self.config.quality_gate_fail:
                raise RuntimeError(f"Quality gate check failed. {quality_gates['summary']}")

        report = self.build_report(context)
        report["quality_gates"] = quality_gates
        self.write_report(context.paths.report, report)
        cleanup_temporary_candidates(context.paths.candidates)
        print_final_summary(context.paths, run_sfm=self.config.run_sfm)
        return context
