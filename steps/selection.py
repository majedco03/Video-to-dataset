"""Frame selection step based on overlap grouping."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from console import print_info
from models import CandidateFrame, PipelineContext
from .base import PipelineStep


class OverlapSelectionStep(PipelineStep):
    """Pick one sharp representative frame from each overlap group."""

    step_number = 2
    title = "Frame spacing and overlap selection"
    goal = "Walk through the frames in order and keep the sharpest frame from each useful overlap group."

    def __init__(self, config) -> None:
        super().__init__(config)
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._feature_cache: Dict[int, Tuple[Any, Any]] = {}

    def get_features(self, frame: CandidateFrame) -> Tuple[Any, Any]:
        cached = self._feature_cache.get(frame.frame_idx)
        if cached is not None:
            return cached

        keypoints, descriptors = self.orb.detectAndCompute(frame.gray_small, None)
        cached = (keypoints, descriptors)
        self._feature_cache[frame.frame_idx] = cached
        return cached

    @staticmethod
    def fallback_overlap_ratio(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
        a = gray_a.astype(np.float32)
        b = gray_b.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        height, width = gray_a.shape[:2]
        overlap_w = max(0.0, width - abs(dx))
        overlap_h = max(0.0, height - abs(dy))
        return float((overlap_w * overlap_h) / (width * height + 1e-9))

    def estimate_overlap_ratio(self, anchor: CandidateFrame, candidate: CandidateFrame) -> float:
        keypoints_a, descriptors_a = self.get_features(anchor)
        keypoints_b, descriptors_b = self.get_features(candidate)

        if descriptors_a is None or descriptors_b is None:
            return self.fallback_overlap_ratio(anchor.gray_small, candidate.gray_small)
        if len(keypoints_a) < 8 or len(keypoints_b) < 8:
            return self.fallback_overlap_ratio(anchor.gray_small, candidate.gray_small)

        raw_matches = self.matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            first_match, second_match = match_pair
            if first_match.distance < 0.75 * second_match.distance:
                good_matches.append(first_match)

        if len(good_matches) < 8:
            return self.fallback_overlap_ratio(anchor.gray_small, candidate.gray_small)

        source_points = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 3.0)
        if homography is None or inlier_mask is None:
            return self.fallback_overlap_ratio(anchor.gray_small, candidate.gray_small)

        if int(np.sum(inlier_mask)) < 8:
            return self.fallback_overlap_ratio(anchor.gray_small, candidate.gray_small)

        anchor_height, anchor_width = anchor.gray_small.shape[:2]
        candidate_height, candidate_width = candidate.gray_small.shape[:2]
        anchor_corners = np.float32(
            [[0, 0], [anchor_width - 1, 0], [anchor_width - 1, anchor_height - 1], [0, anchor_height - 1]]
        ).reshape(-1, 1, 2)
        warped_anchor = cv2.perspectiveTransform(anchor_corners, homography).reshape(-1, 2).astype(np.float32)
        candidate_corners = np.float32(
            [[0, 0], [candidate_width - 1, 0], [candidate_width - 1, candidate_height - 1], [0, candidate_height - 1]]
        )

        try:
            intersection_area, _ = cv2.intersectConvexConvex(candidate_corners, warped_anchor)
        except cv2.error:
            return self.fallback_overlap_ratio(anchor.gray_small, candidate.gray_small)

        if intersection_area <= 0:
            return 0.0

        candidate_area = float(candidate_width * candidate_height)
        overlap = intersection_area / max(candidate_area, 1.0)
        return float(max(0.0, min(1.0, overlap)))

    @staticmethod
    def choose_sharper_frame(current_best: CandidateFrame, new_frame: CandidateFrame) -> CandidateFrame:
        return new_frame if new_frame.sharpness > current_best.sharpness else current_best

    def compute_selected_overlaps(self, selected_frames: List[CandidateFrame]) -> List[float]:
        overlaps: List[float] = []
        for first_frame, second_frame in zip(selected_frames, selected_frames[1:]):
            overlaps.append(self.estimate_overlap_ratio(first_frame, second_frame))
        return overlaps

    @staticmethod
    def frame_bin_index(frame_idx: int, min_idx: int, max_idx: int, bin_count: int) -> int:
        if max_idx <= min_idx:
            return 0
        position = (frame_idx - min_idx) / float(max_idx - min_idx)
        return max(0, min(bin_count - 1, int(position * bin_count)))

    @staticmethod
    def compute_candidate_quality_scores(candidates: List[CandidateFrame]) -> Dict[int, float]:
        if not candidates:
            return {}

        sharpness = np.array([item.sharpness for item in candidates], dtype=np.float32)
        texture = np.array([item.texture for item in candidates], dtype=np.float32)
        exposure = np.array([item.exposure_score for item in candidates], dtype=np.float32)

        sharpness_norm = (sharpness - sharpness.min()) / (np.ptp(sharpness) + 1e-6)
        texture_norm = (texture - texture.min()) / (np.ptp(texture) + 1e-6)
        exposure_norm = np.clip(exposure, 0.0, 1.0)

        scores = 0.55 * sharpness_norm + 0.30 * texture_norm + 0.15 * exposure_norm
        return {
            candidate.frame_idx: float(score)
            for candidate, score in zip(candidates, scores)
        }

    def enforce_coverage_and_output_range(
        self,
        all_candidates: List[CandidateFrame],
        selected_frames: List[CandidateFrame],
    ) -> Tuple[List[CandidateFrame], Dict[str, float | int | List[int]]]:
        if not selected_frames:
            return selected_frames, {
                "bins_total": self.config.angle_bins,
                "bins_covered_before": 0,
                "bins_covered_after": 0,
                "added_for_coverage": 0,
                "added_for_range": 0,
                "removed_for_max_range": 0,
                "missing_bins_before": [],
                "missing_bins_after": [],
            }

        min_idx = min(item.frame_idx for item in all_candidates)
        max_idx = max(item.frame_idx for item in all_candidates)
        bin_count = max(3, self.config.angle_bins)
        quality_scores = self.compute_candidate_quality_scores(all_candidates)
        candidate_lookup = {item.frame_idx: item for item in all_candidates}

        selected = list(selected_frames)
        selected_ids = {item.frame_idx for item in selected}

        def selected_bin_counts() -> Dict[int, int]:
            counts = {index: 0 for index in range(bin_count)}
            for frame in selected:
                counts[self.frame_bin_index(frame.frame_idx, min_idx, max_idx, bin_count)] += 1
            return counts

        def get_remaining_in_bin(bin_index: int) -> List[CandidateFrame]:
            frames = [
                frame
                for frame in all_candidates
                if frame.frame_idx not in selected_ids
                and self.frame_bin_index(frame.frame_idx, min_idx, max_idx, bin_count) == bin_index
            ]
            return sorted(frames, key=lambda item: quality_scores.get(item.frame_idx, 0.0), reverse=True)

        bin_counts_before = selected_bin_counts()
        missing_bins_before = [index for index, count in bin_counts_before.items() if count == 0]

        added_for_coverage = 0
        if self.config.validate_angle_coverage:
            for missing_bin in missing_bins_before:
                candidates_in_bin = get_remaining_in_bin(missing_bin)
                if not candidates_in_bin:
                    continue
                chosen = candidates_in_bin[0]
                selected.append(chosen)
                selected_ids.add(chosen.frame_idx)
                added_for_coverage += 1

        added_for_range = 0
        while len(selected) < self.config.output_min_images:
            bin_counts = selected_bin_counts()
            ordered_bins = sorted(bin_counts.keys(), key=lambda index: (bin_counts[index], index))

            chosen_frame: CandidateFrame | None = None
            for bin_index in ordered_bins:
                candidates_in_bin = get_remaining_in_bin(bin_index)
                if not candidates_in_bin:
                    continue
                chosen_frame = candidates_in_bin[0]
                break

            if chosen_frame is None:
                break

            selected.append(chosen_frame)
            selected_ids.add(chosen_frame.frame_idx)
            added_for_range += 1

        removed_for_max_range = 0
        if self.config.output_max_images > 0 and len(selected) > self.config.output_max_images:
            while len(selected) > self.config.output_max_images:
                bin_counts = selected_bin_counts()

                removable = [
                    frame
                    for frame in selected
                    if bin_counts[self.frame_bin_index(frame.frame_idx, min_idx, max_idx, bin_count)] > 1
                ]
                if not removable:
                    removable = selected

                removable_sorted = sorted(
                    removable,
                    key=lambda frame: (
                        -bin_counts[self.frame_bin_index(frame.frame_idx, min_idx, max_idx, bin_count)],
                        quality_scores.get(frame.frame_idx, 0.0),
                        frame.frame_idx,
                    ),
                )
                to_remove = removable_sorted[0]
                selected = [frame for frame in selected if frame.frame_idx != to_remove.frame_idx]
                selected_ids.discard(to_remove.frame_idx)
                removed_for_max_range += 1

        selected.sort(key=lambda item: item.frame_idx)
        bin_counts_after = selected_bin_counts()
        missing_bins_after = [index for index, count in bin_counts_after.items() if count == 0]

        # Ensure each selected frame still points to the canonical candidate object.
        selected = [candidate_lookup[item.frame_idx] for item in selected]

        summary: Dict[str, float | int | List[int]] = {
            "bins_total": bin_count,
            "bins_covered_before": int(sum(1 for count in bin_counts_before.values() if count > 0)),
            "bins_covered_after": int(sum(1 for count in bin_counts_after.values() if count > 0)),
            "added_for_coverage": added_for_coverage,
            "added_for_range": added_for_range,
            "removed_for_max_range": removed_for_max_range,
            "missing_bins_before": missing_bins_before,
            "missing_bins_after": missing_bins_after,
        }
        return selected, summary

    def run(self, context: PipelineContext) -> PipelineContext:
        self.announce()
        if not context.candidates:
            raise RuntimeError("No candidates are available for overlap selection.")

        selected: List[CandidateFrame] = []
        overlap_violations = 0

        group_anchor = context.candidates[0]
        group_best = context.candidates[0]

        for candidate in context.candidates[1:]:
            overlap = self.estimate_overlap_ratio(group_anchor, candidate)

            if overlap >= self.config.min_overlap:
                group_best = self.choose_sharper_frame(group_best, candidate)
                continue

            selected.append(group_best)
            overlap_violations += 1
            group_anchor = candidate
            group_best = candidate

        selected.append(group_best)

        unique_selected: List[CandidateFrame] = []
        for frame in selected:
            if unique_selected and unique_selected[-1].frame_idx == frame.frame_idx:
                unique_selected[-1] = frame
                continue
            unique_selected.append(frame)

        overlaps = self.compute_selected_overlaps(unique_selected)
        unique_selected, coverage_summary = self.enforce_coverage_and_output_range(
            context.candidates,
            unique_selected,
        )
        overlaps = self.compute_selected_overlaps(unique_selected)
        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        print_info(
            f"Selected {len(unique_selected)} frames | mean overlap: {mean_overlap:.3f} | "
            f"group changes: {overlap_violations}"
        )
        print_info(
            "Coverage check -> "
            f"bins: {coverage_summary['bins_covered_after']}/{coverage_summary['bins_total']} | "
            f"added(coverage): {coverage_summary['added_for_coverage']} | "
            f"added(range): {coverage_summary['added_for_range']} | "
            f"removed(max): {coverage_summary['removed_for_max_range']}"
        )

        if not unique_selected:
            raise RuntimeError("Frame selection failed. No useful frame sequence could be built.")

        context.selected_frames = unique_selected
        context.overlap_violations = overlap_violations
        context.overlaps = overlaps
        context.selection_coverage = coverage_summary
        return context
