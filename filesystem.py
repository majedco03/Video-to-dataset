"""Filesystem helpers for pipeline outputs."""

from __future__ import annotations

import os
import shutil

from models import PipelinePaths


def safe_remove(path: str) -> None:
    """Delete a file or folder if it already exists."""

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def setup_directories(base_dir: str) -> PipelinePaths:
    """Create a clean output folder tree for a fresh run."""

    paths = PipelinePaths(
        root=base_dir,
        images=os.path.join(base_dir, "images"),
        candidates=os.path.join(base_dir, "_candidates"),
        masks=os.path.join(base_dir, "masks"),
        colmap=os.path.join(base_dir, "colmap"),
        database=os.path.join(base_dir, "colmap", "database.db"),
        sparse=os.path.join(base_dir, "colmap", "sparse"),
        dense=os.path.join(base_dir, "colmap", "dense"),
        undistorted=os.path.join(base_dir, "colmap", "undistorted"),
        sparse_txt=os.path.join(base_dir, "colmap", "sparse_txt"),
        report=os.path.join(base_dir, "preprocessing_report.json"),
    )

    safe_remove(paths.images)
    safe_remove(paths.candidates)
    safe_remove(paths.masks)
    safe_remove(paths.colmap)

    os.makedirs(paths.images, exist_ok=True)
    os.makedirs(paths.candidates, exist_ok=True)
    os.makedirs(paths.masks, exist_ok=True)
    os.makedirs(paths.colmap, exist_ok=True)
    return paths


def cleanup_temporary_candidates(candidates_dir: str) -> None:
    """Remove temporary frame files after the run."""

    safe_remove(candidates_dir)
