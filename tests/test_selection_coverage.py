from __future__ import annotations

from steps.selection import OverlapSelectionStep


def test_selection_backfills_missing_angle_bins(config_factory, candidate_factory):
    config = config_factory(angle_bins=4, validate_angle_coverage=True, output_min_images=4, output_max_images=0)
    step = OverlapSelectionStep(config)

    all_candidates = [
        candidate_factory(0, sharpness=90),
        candidate_factory(25, sharpness=91),
        candidate_factory(50, sharpness=92),
        candidate_factory(75, sharpness=93),
    ]
    selected = [all_candidates[0]]

    updated, summary = step.enforce_coverage_and_output_range(all_candidates, selected)

    assert len(updated) >= 4
    assert summary["bins_covered_after"] == 4
    assert summary["added_for_coverage"] >= 1


def test_selection_fills_min_range_from_least_covered_bins(config_factory, candidate_factory):
    config = config_factory(angle_bins=5, validate_angle_coverage=False, output_min_images=5, output_max_images=0)
    step = OverlapSelectionStep(config)

    all_candidates = [candidate_factory(index * 20, sharpness=100 + index) for index in range(7)]
    selected = [all_candidates[0], all_candidates[1]]

    updated, summary = step.enforce_coverage_and_output_range(all_candidates, selected)

    assert len(updated) >= 5
    assert summary["added_for_range"] >= 3


def test_selection_trims_to_max_range(config_factory, candidate_factory):
    config = config_factory(angle_bins=4, validate_angle_coverage=True, output_min_images=0, output_max_images=3)
    step = OverlapSelectionStep(config)

    all_candidates = [candidate_factory(index * 10, sharpness=200 - index) for index in range(6)]
    selected = list(all_candidates)

    updated, summary = step.enforce_coverage_and_output_range(all_candidates, selected)

    assert len(updated) == 3
    assert summary["removed_for_max_range"] == 3
