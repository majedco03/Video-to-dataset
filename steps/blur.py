"""Frame sampling and blur filtering step."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from console import print_info
from constants import EXPOSURE_SCORE_FLOOR, JPEG_QUALITY, OVERLAP_PREVIEW_MAX_DIM
from models import CandidateFrame, PipelineContext
from .base import PipelineStep


class BlurDetectionStep(PipelineStep):
    """Sample the video and keep frames that look usable."""

    step_number = 1
    title = "Frame sampling and quick quality check"
    goal = "Pull frames from the video and drop the ones that are too blurry or badly exposed."

    @staticmethod
    def compute_sharpness(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_texture(gray: np.ndarray) -> float:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(cv2.magnitude(grad_x, grad_y)))

    @staticmethod
    def compute_exposure_score(gray: np.ndarray) -> float:
        low_clip = float(np.mean(gray <= 4))
        high_clip = float(np.mean(gray >= 251))
        mean_intensity = float(np.mean(gray)) / 255.0
        centered = 1.0 - min(1.0, abs(mean_intensity - 0.5) / 0.5)
        clip_penalty = max(0.0, 1.0 - 3.0 * (low_clip + high_clip))
        return max(0.0, centered * clip_penalty)

    @staticmethod
    def resize_for_overlap(gray: np.ndarray, max_dim: int = OVERLAP_PREVIEW_MAX_DIM) -> np.ndarray:
        height, width = gray.shape[:2]
        scale = min(1.0, max_dim / max(height, width))
        if scale == 1.0:
            return gray
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def open_video_capture(video_path: str) -> Tuple[cv2.VideoCapture, float, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            native_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return cap, float(native_fps), total_frames

    def sample_candidate_frames(self, candidates_dir: str) -> Tuple[List[CandidateFrame], float, int, int]:
        cap, native_fps, total_frames = self.open_video_capture(self.config.video_path)
        stride = max(1, int(round(native_fps / self.config.extraction_fps)))

        sampled_candidates: List[CandidateFrame] = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_path = os.path.join(candidates_dir, f"candidate_{frame_idx:07d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            sampled_candidates.append(
                CandidateFrame(
                    frame_idx=frame_idx,
                    gray_small=self.resize_for_overlap(gray),
                    frame_path=frame_path,
                    sharpness=self.compute_sharpness(gray),
                    texture=self.compute_texture(gray),
                    exposure_score=self.compute_exposure_score(gray),
                )
            )
            frame_idx += 1

        cap.release()
        return sampled_candidates, native_fps, stride, total_frames

    def build_candidate_stats(
        self,
        sampled_candidates: List[CandidateFrame],
        sharpness_threshold: float,
        texture_threshold: float,
    ) -> Dict[str, float]:
        if not sampled_candidates:
            return {
                "sharpness_threshold": float(sharpness_threshold),
                "texture_threshold": float(texture_threshold),
                "sampled_count": 0.0,
                "kept_count": 0.0,
            }

        sharpness_values = np.array([item.sharpness for item in sampled_candidates], dtype=np.float32)
        exposure_values = np.array([item.exposure_score for item in sampled_candidates], dtype=np.float32)

        return {
            "sharpness_threshold": float(sharpness_threshold),
            "texture_threshold": float(texture_threshold),
            "sampled_count": float(len(sampled_candidates)),
            "kept_count": 0.0,
            "sharpness_p50": float(np.percentile(sharpness_values, 50)),
            "sharpness_p75": float(np.percentile(sharpness_values, 75)),
            "exposure_score_p50": float(np.percentile(exposure_values, 50)),
        }

    def filter_clear_candidates(self, sampled_candidates: List[CandidateFrame]) -> Tuple[List[CandidateFrame], Dict[str, float]]:
        if not sampled_candidates:
            return [], self.build_candidate_stats([], self.config.blur_threshold, 0.0)

        sharpness_values = np.array([item.sharpness for item in sampled_candidates], dtype=np.float32)
        texture_values = np.array([item.texture for item in sampled_candidates], dtype=np.float32)

        adaptive_sharpness = float(np.percentile(sharpness_values, self.config.sharpness_percentile))
        adaptive_texture = float(np.percentile(texture_values, self.config.texture_percentile))

        sharpness_threshold = (
            max(self.config.blur_threshold, adaptive_sharpness)
            if self.config.auto_tune
            else self.config.blur_threshold
        )
        texture_threshold = adaptive_texture if self.config.auto_tune else 0.0

        candidates = [
            candidate
            for candidate in sampled_candidates
            if candidate.sharpness >= sharpness_threshold
            and candidate.texture >= texture_threshold
            and candidate.exposure_score >= EXPOSURE_SCORE_FLOOR
        ]

        stats = self.build_candidate_stats(sampled_candidates, sharpness_threshold, texture_threshold)
        stats["kept_count"] = float(len(candidates))
        return candidates, stats

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.paths is None:
            raise RuntimeError("Output folders were not prepared before frame sampling.")

        self.announce()
        sampled_candidates, native_fps, stride, total_frames = self.sample_candidate_frames(
            context.paths.candidates
        )
        candidates, stats = self.filter_clear_candidates(sampled_candidates)

        print_info(
            "Video FPS: "
            f"{native_fps:.2f} | stride: {stride} | total frames: {total_frames} | "
            f"sampled: {len(sampled_candidates)} | kept: {len(candidates)}"
        )
        print_info(
            "Thresholds used -> "
            f"sharpness: {stats['sharpness_threshold']:.2f}, "
            f"texture: {stats['texture_threshold']:.2f}, "
            f"exposure floor: {EXPOSURE_SCORE_FLOOR:.2f}"
        )

        if not candidates:
            raise RuntimeError(
                "No clear frames were found. Try lowering the blur threshold or raising the extraction FPS."
            )

        context.candidates = candidates
        context.native_fps = native_fps
        context.stride = stride
        context.candidate_stats = stats
        return context
