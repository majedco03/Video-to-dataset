from __future__ import annotations

import json

import pytest

from steps.finalize import FinalizationStep


def test_quality_gate_passes_when_overlap_high(config_factory, context_factory):
    config = config_factory(quality_gate_min_overlap=0.75, quality_gate_fail=True)
    step = FinalizationStep(config)
    context = context_factory(overlaps=[0.8, 0.82, 0.9])

    result = step.evaluate_quality_gates(context)

    assert result["enabled"] is True
    assert result["passed"] is True


def test_quality_gate_fails_when_overlap_low(config_factory, context_factory):
    config = config_factory(quality_gate_min_overlap=0.85, quality_gate_fail=False)
    step = FinalizationStep(config)
    context = context_factory(overlaps=[0.6, 0.7])

    result = step.evaluate_quality_gates(context)

    assert result["enabled"] is True
    assert result["passed"] is False


def test_finalize_run_raises_on_quality_gate_fail(config_factory, context_factory, paths_factory):
    config = config_factory(quality_gate_min_overlap=0.9, quality_gate_fail=True)
    step = FinalizationStep(config)
    context = context_factory(
        paths=paths_factory(),
        overlaps=[0.6, 0.7],
        selected_frames=[],
        saved_images=[],
    )

    with pytest.raises(RuntimeError):
        step.run(context)


def test_finalize_run_writes_report_when_warn_only(config_factory, context_factory, paths_factory):
    config = config_factory(quality_gate_min_overlap=0.9, quality_gate_fail=False)
    step = FinalizationStep(config)
    paths = paths_factory()
    context = context_factory(
        paths=paths,
        overlaps=[0.6, 0.7],
        selected_frames=[],
        saved_images=[],
        selection_coverage={"bins_total": 4, "bins_covered_after": 2},
    )

    step.run(context)

    with open(paths.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    assert report["quality_gates"]["enabled"] is True
    assert report["quality_gates"]["passed"] is False
    assert report["selection_coverage"]["bins_total"] == 4
