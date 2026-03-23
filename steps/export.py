"""Export step — writes transforms.json for NeRFStudio and Instant-NGP.

Reads the COLMAP sparse reconstruction (cameras.txt + images.txt) and
converts the camera poses and intrinsics into the JSON format expected by
NeRFStudio and Instant-NGP.

Only runs when COLMAP has produced a valid sparse model and
--export-format includes nerfstudio, instant-ngp, or all.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from console import print_info, print_success, print_warning
from models import PipelineContext
from .base import PipelineStep


# ---------------------------------------------------------------------------
# COLMAP text-format parsers
# ---------------------------------------------------------------------------

def _parse_cameras_txt(path: str) -> Dict[int, Dict[str, Any]]:
    """Parse COLMAP cameras.txt and return {camera_id: {params}}."""
    cameras: Dict[int, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return cameras
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {"model": model, "width": width, "height": height, "params": params}
    return cameras


def _parse_images_txt(path: str) -> List[Dict[str, Any]]:
    """Parse COLMAP images.txt and return a list of image records."""
    images: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return images
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    i = 0
    while i + 1 < len(lines):
        parts = lines[i].split()
        if len(parts) < 9:
            i += 2
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9] if len(parts) > 9 else f"image_{image_id}"
        images.append({
            "image_id": image_id,
            "qvec": (qw, qx, qy, qz),
            "tvec": (tx, ty, tz),
            "camera_id": cam_id,
            "name": name,
        })
        i += 2
    return images


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _qvec_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a unit quaternion to a 3×3 rotation matrix."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 0:
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _colmap_to_transform_matrix(qvec: Tuple, tvec: Tuple) -> List[List[float]]:
    """Convert COLMAP world-to-camera pose to a camera-to-world 4×4 matrix."""
    qw, qx, qy, qz = qvec
    tx, ty, tz = tvec
    R = _qvec_to_rotation_matrix(qw, qx, qy, qz)
    t = np.array([tx, ty, tz], dtype=np.float64)
    # COLMAP: X_camera = R * X_world + t  →  X_world = R^T * (X_camera - t)
    Rt = R.T
    c = -Rt @ t
    # Build 4×4 c2w matrix; flip y and z axes to convert to OpenGL convention.
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Rt
    c2w[:3, 3] = c
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1
    return c2w.tolist()


# ---------------------------------------------------------------------------
# Intrinsics helpers
# ---------------------------------------------------------------------------

def _extract_intrinsics(cam: Dict[str, Any]) -> Dict[str, float]:
    model = cam["model"]
    params = cam["params"]
    w, h = cam["width"], cam["height"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        return {"fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "w": w, "h": h}
    if model in ("PINHOLE", "RADIAL", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w, "h": h}
    # Fallback: treat first param as focal length, use image centre.
    f = params[0] if params else float(max(w, h))
    return {"fl_x": f, "fl_y": f, "cx": w / 2.0, "cy": h / 2.0, "w": w, "h": h}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_nerfstudio(
    cameras: Dict[int, Dict[str, Any]],
    images: List[Dict[str, Any]],
    images_dir: str,
    output_path: str,
) -> int:
    """Write a NeRFStudio-compatible transforms.json."""
    if not cameras or not images:
        return 0

    # Use the first camera's intrinsics for the global fields.
    first_cam = next(iter(cameras.values()))
    intr = _extract_intrinsics(first_cam)

    frames = []
    for img in sorted(images, key=lambda x: x["name"]):
        file_path = os.path.join(images_dir, img["name"])
        if not os.path.isfile(file_path):
            continue
        cam = cameras.get(img["camera_id"], first_cam)
        i = _extract_intrinsics(cam)
        transform = _colmap_to_transform_matrix(img["qvec"], img["tvec"])
        frame: Dict[str, Any] = {
            "file_path": os.path.relpath(file_path, os.path.dirname(output_path)),
            "transform_matrix": transform,
            "fl_x": i["fl_x"],
            "fl_y": i["fl_y"],
            "cx": i["cx"],
            "cy": i["cy"],
            "w": int(i["w"]),
            "h": int(i["h"]),
        }
        frames.append(frame)

    data: Dict[str, Any] = {
        "camera_model": "OPENCV",
        "fl_x": intr["fl_x"],
        "fl_y": intr["fl_y"],
        "cx": intr["cx"],
        "cy": intr["cy"],
        "w": int(intr["w"]),
        "h": int(intr["h"]),
        "frames": frames,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return len(frames)


def _write_instant_ngp(
    cameras: Dict[int, Dict[str, Any]],
    images: List[Dict[str, Any]],
    images_dir: str,
    output_path: str,
) -> int:
    """Write an Instant-NGP-compatible transforms.json."""
    if not cameras or not images:
        return 0

    first_cam = next(iter(cameras.values()))
    intr = _extract_intrinsics(first_cam)
    w, h = int(intr["w"]), int(intr["h"])
    fl_x = intr["fl_x"]
    fl_y = intr["fl_y"]

    camera_angle_x = 2.0 * math.atan(w / (2.0 * fl_x))
    camera_angle_y = 2.0 * math.atan(h / (2.0 * fl_y))

    frames = []
    for img in sorted(images, key=lambda x: x["name"]):
        file_path = os.path.join(images_dir, img["name"])
        if not os.path.isfile(file_path):
            continue
        transform = _colmap_to_transform_matrix(img["qvec"], img["tvec"])
        frames.append({
            "file_path": os.path.relpath(file_path, os.path.dirname(output_path)),
            "transform_matrix": transform,
        })

    data: Dict[str, Any] = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": intr["cx"],
        "cy": intr["cy"],
        "w": w,
        "h": h,
        "frames": frames,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return len(frames)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class ExportFormatsStep(PipelineStep):
    """Convert COLMAP output to NeRFStudio / Instant-NGP transforms.json."""

    step_number = 6
    title = "Export formats"
    goal = (
        "Parse the COLMAP sparse model (cameras.txt + images.txt) and convert each camera pose "
        "from COLMAP's world-to-camera quaternion format into the camera-to-world matrices "
        "expected by NeRFStudio and Instant-NGP."
    )

    def run(self, context: PipelineContext) -> PipelineContext:
        fmt = self.config.export_format
        if fmt == "colmap":
            return context  # Nothing extra to do.

        if not self.config.run_sfm:
            print_warning("Export formats require COLMAP output. Skipping (--no-colmap was set).")
            return context

        if context.paths is None:
            return context

        sparse_txt = context.paths.sparse_txt
        cameras_txt = os.path.join(sparse_txt, "cameras.txt")
        images_txt = os.path.join(sparse_txt, "images.txt")

        if not os.path.isfile(cameras_txt) or not os.path.isfile(images_txt):
            print_warning(
                "COLMAP sparse_txt output not found — skipping transforms.json export. "
                "Run COLMAP first or check that the reconstruction succeeded."
            )
            return context

        self.announce()
        cameras = _parse_cameras_txt(cameras_txt)
        images = _parse_images_txt(images_txt)
        images_dir = context.paths.images

        if not cameras or not images:
            print_warning("No camera/image data found in COLMAP sparse_txt — skipping export.")
            return context

        root = context.paths.root
        do_nerfstudio = fmt in ("nerfstudio", "all")
        do_instant_ngp = fmt in ("instant-ngp", "all")

        if do_nerfstudio:
            out_path = os.path.join(root, "transforms_nerfstudio.json")
            count = _write_nerfstudio(cameras, images, images_dir, out_path)
            print_success(f"NeRFStudio transforms.json written ({count} frames): {out_path}")

        if do_instant_ngp:
            out_path = os.path.join(root, "transforms_instant_ngp.json")
            count = _write_instant_ngp(cameras, images, images_dir, out_path)
            print_success(f"Instant-NGP transforms.json written ({count} frames): {out_path}")

        return context
