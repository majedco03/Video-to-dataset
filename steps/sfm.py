"""Structure-from-motion step powered by COLMAP."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, List, Tuple

from console import print_info, print_success, print_warning, spinner
from constants import COLMAP_MAX_IMAGE_SIZE
from filesystem import safe_remove
from models import PipelineContext
from .base import PipelineStep


class StructureFromMotionStep(PipelineStep):
    """Run the COLMAP stages on the cleaned image set."""

    step_number = 5
    title = "COLMAP structure-from-motion"
    goal = (
        "Run COLMAP feature extraction, feature matching, and sparse mapping to triangulate "
        "3D points and recover the camera pose for every selected frame. "
        "The best sparse model is then undistorted and exported."
    )

    @staticmethod
    def parse_analyzer_value(summary: str, label: str) -> float:
        match = re.search(rf"{re.escape(label)}\s*:\s*([-+0-9.eE]+)", summary)
        if not match:
            return 0.0
        return float(match.group(1))

    def select_best_sparse_model(self, sparse_dir: str) -> Tuple[str, str, Dict[str, float]]:
        best_model_dir = ""
        best_summary = "No sparse model produced."
        best_score = -1.0
        best_metrics: Dict[str, float] = {}

        if not os.path.isdir(sparse_dir):
            return best_model_dir, best_summary, best_metrics

        for child in sorted(os.listdir(sparse_dir)):
            model_dir = os.path.join(sparse_dir, child)
            if not os.path.isdir(model_dir):
                continue

            analyzer = subprocess.run(
                ["colmap", "model_analyzer", "--path", model_dir],
                check=False,
                capture_output=True,
                text=True,
            )
            if analyzer.returncode != 0:
                continue  # empty or corrupt sub-model; skip without aborting
            summary = (analyzer.stdout or "") + ("\n" if analyzer.stdout and analyzer.stderr else "") + (analyzer.stderr or "")
            images = self.parse_analyzer_value(summary, "Registered images")
            points = self.parse_analyzer_value(summary, "Points")
            mean_track = self.parse_analyzer_value(summary, "Mean track length")
            mean_obs = self.parse_analyzer_value(summary, "Mean observations per image")
            score = images * 1000.0 + points + mean_track * 100.0 + mean_obs * 10.0

            if score > best_score:
                best_score = score
                best_model_dir = model_dir
                best_summary = summary
                best_metrics = {
                    "registered_images": images,
                    "points": points,
                    "mean_track_length": mean_track,
                    "mean_observations_per_image": mean_obs,
                }

        return best_model_dir, best_summary, best_metrics

    @staticmethod
    def run_command(command: List[str], label: str) -> None:
        with spinner(f"{label}..."):
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_tail = "\n".join((result.stderr or "").splitlines()[-10:]).strip()
            raise subprocess.CalledProcessError(result.returncode, command, stderr=stderr_tail)
        print_success(f"Finished {label}.")

    def get_colmap_thread_count(self) -> int:
        if not self.config.colmap_parallel:
            return 1
        if self.config.colmap_num_threads > 0:
            return self.config.colmap_num_threads
        return max(1, os.cpu_count() or 1)

    def get_colmap_gpu_index(self) -> str:
        if not self.config.colmap_device.startswith("cuda"):
            return "0"
        _, _, maybe_index = self.config.colmap_device.partition(":")
        return maybe_index or "0"

    def build_feature_command(self, images_dir: str, database_path: str, mask_path: str = "") -> List[str]:
        num_threads = str(self.get_colmap_thread_count())
        # COLMAP 3.13+: thread count and GPU control moved to FeatureExtraction.* namespace.
        command = [
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", images_dir,
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.num_threads", num_threads,
            "--SiftExtraction.max_num_features", "16384",
        ]
        if self.config.use_gpu:
            # GPU-only flags: only present in CUDA-enabled COLMAP builds.
            command.extend([
                "--FeatureExtraction.gpu_index", self.get_colmap_gpu_index(),
                "--SiftExtraction.estimate_affine_shape", "1",
                "--SiftExtraction.domain_size_pooling", "1",
            ])
        if mask_path:
            command.extend(["--ImageReader.mask_path", mask_path])
        return command

    def build_match_command(self, database_path: str, matcher: str | None = None) -> List[str]:
        num_threads = str(self.get_colmap_thread_count())
        # COLMAP 3.13+: thread count, GPU control, and guided_matching moved to FeatureMatching.* namespace.
        shared_match_args = ["--FeatureMatching.num_threads", num_threads]
        if self.config.use_gpu:
            # GPU-only flag: only present in CUDA-enabled COLMAP builds.
            shared_match_args = ["--FeatureMatching.gpu_index", self.get_colmap_gpu_index()] + shared_match_args

        selected_matcher = matcher or self.config.matcher
        if selected_matcher == "sequential":
            command = [
                "colmap", "sequential_matcher",
                "--database_path", database_path,
                "--FeatureMatching.guided_matching", "1",
                "--SequentialMatching.overlap", str(self.config.sequential_overlap),
                "--SequentialMatching.quadratic_overlap", "1",
                *shared_match_args,
            ]
            if self.config.loop_detection and self.config.vocab_tree_path:
                command.extend([
                    "--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", self.config.vocab_tree_path,
                ])
            return command

        return [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.guided_matching", "1",
            *shared_match_args,
        ]

    def build_mapper_command(self, images_dir: str, database_path: str, sparse_dir: str) -> List[str]:
        return [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", images_dir,
            "--output_path", sparse_dir,
            "--Mapper.num_threads", str(self.get_colmap_thread_count()),
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "1",
            "--Mapper.min_num_matches", "20",
            "--Mapper.init_min_num_inliers", "60",
            "--Mapper.abs_pose_min_num_inliers", "30",
            "--Mapper.filter_max_reproj_error", "4",
        ]

    def export_best_colmap_model(self, images_dir: str, best_model_dir: str, undistorted_dir: str, sparse_txt_dir: str) -> None:
        self.run_command(
            [
                "colmap", "image_undistorter",
                "--image_path", images_dir,
                "--input_path", best_model_dir,
                "--output_path", undistorted_dir,
                "--output_type", "COLMAP",
                "--max_image_size", str(COLMAP_MAX_IMAGE_SIZE),
            ],
            "COLMAP image undistorter",
        )
        self.run_command(
            [
                "colmap", "model_converter",
                "--input_path", best_model_dir,
                "--output_path", sparse_txt_dir,
                "--output_type", "TXT",
            ],
            "COLMAP model converter",
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        if not self.config.run_sfm:
            return context
        if context.paths is None:
            raise RuntimeError("Output folders were not prepared before COLMAP.")

        self.announce()
        safe_remove(context.paths.database)
        safe_remove(context.paths.sparse)
        safe_remove(context.paths.undistorted)
        safe_remove(context.paths.sparse_txt)
        os.makedirs(context.paths.sparse, exist_ok=True)
        os.makedirs(context.paths.undistorted, exist_ok=True)
        os.makedirs(context.paths.sparse_txt, exist_ok=True)

        requested_matcher = self.config.matcher
        actual_matcher = self.config.matcher
        mask_path = context.paths.masks if context.semantic_masking_result.get("masks_written", 0) > 0 else ""
        thread_count = self.get_colmap_thread_count()

        print_info(
            f"COLMAP runtime -> device: {self.config.colmap_device}, parallel: {'on' if self.config.colmap_parallel else 'off'}, threads: {thread_count}"
        )

        try:
            self.run_command(
                self.build_feature_command(context.paths.images, context.paths.database, mask_path=mask_path),
                "COLMAP feature extractor",
            )
            try:
                self.run_command(
                    self.build_match_command(context.paths.database, matcher=requested_matcher),
                    f"COLMAP {requested_matcher} matcher",
                )
            except subprocess.CalledProcessError:
                if requested_matcher != "sequential":
                    raise
                print_warning("Sequential matcher failed, so the script will try the exhaustive matcher.")
                actual_matcher = "exhaustive"
                self.run_command(
                    self.build_match_command(context.paths.database, matcher=actual_matcher),
                    "COLMAP exhaustive matcher",
                )

            self.run_command(
                self.build_mapper_command(context.paths.images, context.paths.database, context.paths.sparse),
                "COLMAP mapper",
            )
            print_success(f"COLMAP sparse reconstruction completed in: {context.paths.colmap}")
            best_model_dir, best_summary, best_metrics = self.select_best_sparse_model(context.paths.sparse)
            if not best_model_dir:
                print_warning(
                    "COLMAP mapper ran but produced no reconstruction.\n"
                    "  This usually means there was not enough frame overlap for feature matching.\n\n"
                    "  Things to try:\n"
                    "    • Increase frame sampling:              --fps 5.0\n"
                    "    • Lower the minimum overlap threshold:  --min-overlap 0.70\n"
                    "    • Switch to exhaustive matcher:         --matcher exhaustive\n"
                    "    • Reduce image size for faster matching: --max-image-dim 1280\n"
                    "  The cleaned image dataset is still ready to use without reconstruction."
                )
                context.colmap_result = {
                    "summary": "No sparse model folder found in colmap/sparse",
                    "best_model_dir": "",
                    "metrics": {},
                    "undistorted_dir": "",
                    "matcher_requested": requested_matcher,
                    "matcher_used": actual_matcher,
                    "device": self.config.colmap_device,
                    "parallel": self.config.colmap_parallel,
                    "threads": thread_count,
                }
                return context

            self.export_best_colmap_model(context.paths.images, best_model_dir, context.paths.undistorted, context.paths.sparse_txt)
            context.colmap_result = {
                "summary": best_summary,
                "best_model_dir": best_model_dir,
                "metrics": best_metrics,
                "undistorted_dir": context.paths.undistorted,
                "matcher_requested": requested_matcher,
                "matcher_used": actual_matcher,
                "device": self.config.colmap_device,
                "parallel": self.config.colmap_parallel,
                "threads": thread_count,
            }
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            print_warning("COLMAP failed or is not installed.")
            stderr = getattr(error, "stderr", "") or ""
            if stderr:
                print_info(f"COLMAP output (last lines):\n{stderr}")
            print_info(
                "Things to try:\n"
                "  • Run 'python preprocess.py doctor' to check COLMAP is installed and on PATH.\n"
                "  • Try a CPU-only run:          --colmap-device cpu\n"
                "  • Skip reconstruction for now: --no-colmap\n"
                "  The cleaned image dataset is still saved and ready to use."
            )
            context.colmap_result = {
                "summary": f"COLMAP failed: {error}",
                "device": self.config.colmap_device,
                "parallel": self.config.colmap_parallel,
                "threads": thread_count,
            }

        return context
