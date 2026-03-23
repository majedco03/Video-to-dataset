"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class CandidateFrame:
    """One sampled frame and the scores used to decide if it should stay."""

    frame_idx: int
    gray_small: np.ndarray
    frame_path: str
    sharpness: float
    texture: float
    exposure_score: float
    quality_score: float = 0.0


@dataclass(frozen=True)
class PipelinePaths:
    """All folders and files created by the pipeline."""

    root: str
    images: str
    candidates: str
    masks: str
    colmap: str
    database: str
    sparse: str
    dense: str
    undistorted: str
    sparse_txt: str
    report: str


@dataclass(frozen=True)
class PipelineConfig:
    """All user-facing settings resolved into one runtime config."""

    video_path: str
    output_root: str
    extraction_fps: float
    blur_threshold: float
    min_overlap: float
    max_overlap: float
    target_overlap: float
    auto_tune: bool
    sharpness_percentile: float
    texture_percentile: float
    max_image_dim: int
    color_mode: str
    deblur_strength: float
    enable_white_balance: bool
    enable_clahe: bool
    enable_local_contrast: bool
    run_semantic_masking: bool
    strict_static_masking: bool
    mask_backend: str
    mask_model: str
    mask_classes: Tuple[str, ...]
    mask_device: str
    mask_image_size: int
    mask_confidence: float
    colmap_device: str
    colmap_parallel: bool
    colmap_num_threads: int
    matcher: str
    sequential_overlap: int
    loop_detection: bool
    vocab_tree_path: str
    use_gpu: bool
    run_sfm: bool
    angle_bins: int
    validate_angle_coverage: bool
    output_min_images: int
    output_max_images: int
    quality_gate_min_overlap: float
    quality_gate_fail: bool
    export_format: str = "colmap"
    resume: bool = True
    preset: str = "balanced"


@dataclass
class PipelineContext:
    """Shared data passed from one step to the next."""

    paths: PipelinePaths | None = None
    resuming: bool = False
    candidates: List[CandidateFrame] = field(default_factory=list)
    native_fps: float = 0.0
    stride: int = 0
    candidate_stats: Dict[str, float] = field(default_factory=dict)
    selected_frames: List[CandidateFrame] = field(default_factory=list)
    overlap_violations: int = 0
    overlaps: List[float] = field(default_factory=list)
    selection_coverage: Dict[str, Any] = field(default_factory=dict)
    saved_images: List[str] = field(default_factory=list)
    semantic_masking_result: Dict[str, Any] = field(
        default_factory=lambda: {"summary": "Semantic masking not executed."}
    )
    colmap_result: Dict[str, Any] = field(
        default_factory=lambda: {"summary": "COLMAP not executed."}
    )
