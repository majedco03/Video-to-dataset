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
        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        print_info(
            f"Selected {len(unique_selected)} frames | mean overlap: {mean_overlap:.3f} | "
            f"group changes: {overlap_violations}"
        )

        if not unique_selected:
            raise RuntimeError("Frame selection failed. No useful frame sequence could be built.")

        context.selected_frames = unique_selected
        context.overlap_violations = overlap_violations
        context.overlaps = overlaps
        return context
