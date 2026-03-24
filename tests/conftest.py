from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models import CandidateFrame, PipelineConfig, PipelineContext, PipelinePaths


@pytest.fixture
def config_factory():
    def _build(**overrides):
        values = {
            "video_path": "input.mp4",
            "output_root": "processed_dataset",
            "extraction_fps": 3.0,
            "blur_threshold": 90.0,
            "min_overlap": 0.85,
            "max_overlap": 0.97,
            "target_overlap": 0.90,
            "auto_tune": True,
            "sharpness_percentile": 55.0,
            "texture_percentile": 35.0,
            "max_image_dim": 1920,
            "color_mode": "color",
            "deblur_strength": 1.0,
            "enable_white_balance": True,
            "enable_clahe": True,
            "enable_local_contrast": True,
            "run_semantic_masking": True,
            "strict_static_masking": False,
            "mask_backend": "yolo",
            "mask_model": "yolov8n-seg.pt",
            "mask_classes": ("all",),
            "mask_device": "cpu",
            "mask_image_size": 640,
            "mask_confidence": 0.35,
            "colmap_device": "cpu",
            "colmap_parallel": False,
            "colmap_num_threads": 1,
            "matcher": "sequential",
            "sequential_overlap": 12,
            "loop_detection": True,
            "vocab_tree_path": "",
            "use_gpu": False,
            "run_sfm": False,
            "angle_bins": 12,
            "validate_angle_coverage": True,
            "output_min_images": 0,
            "output_max_images": 0,
            "quality_gate_min_overlap": 0.0,
            "quality_gate_fail": False,
            "preset": "balanced",
        }
        values.update(overrides)
        return PipelineConfig(**values)

    return _build


@pytest.fixture
def candidate_factory(tmp_path: Path):
    def _build(frame_idx: int, sharpness: float = 100.0, texture: float = 10.0, exposure_score: float = 0.9):
        return CandidateFrame(
            frame_idx=frame_idx,
            gray_small=np.zeros((16, 16), dtype=np.uint8),
            frame_path=str(tmp_path / f"candidate_{frame_idx:05d}.jpg"),
            sharpness=sharpness,
            texture=texture,
            exposure_score=exposure_score,
        )

    return _build


@pytest.fixture
def paths_factory(tmp_path: Path):
    def _build():
        root = tmp_path / "out"
        images = root / "images"
        candidates = root / "_candidates"
        masks = root / "masks"
        colmap = root / "colmap"
        sparse = colmap / "sparse"
        dense = colmap / "dense"
        undistorted = colmap / "undistorted"
        sparse_txt = colmap / "sparse_txt"

        for directory in (images, candidates, masks, colmap, sparse, dense, undistorted, sparse_txt):
            directory.mkdir(parents=True, exist_ok=True)

        return PipelinePaths(
            root=str(root),
            images=str(images),
            candidates=str(candidates),
            masks=str(masks),
            colmap=str(colmap),
            database=str(colmap / "database.db"),
            sparse=str(sparse),
            dense=str(dense),
            undistorted=str(undistorted),
            sparse_txt=str(sparse_txt),
            report=str(root / "preprocessing_report.json"),
        )

    return _build


@pytest.fixture
def context_factory():
    def _build(**overrides):
        context = PipelineContext()
        for key, value in overrides.items():
            setattr(context, key, value)
        return context

    return _build
