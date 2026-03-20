"""Small console helpers for readable pipeline logs."""

from __future__ import annotations

from models import PipelineConfig, PipelinePaths


def print_section(step_number: int, title: str, goal: str) -> None:
    """Print a short header so each part of the run is easy to spot."""

    print(f"\n{'=' * 72}")
    print(f"STEP {step_number}: {title}")
    print(f"Goal: {goal}")
    print(f"{'=' * 72}")


def print_info(message: str) -> None:
    print(f"[info] {message}")


def print_warning(message: str) -> None:
    print(f"[warning] {message}")


def print_success(message: str) -> None:
    print(f"[done] {message}")


def print_run_plan(config: PipelineConfig) -> None:
    """Show the main runtime choices before the pipeline starts."""

    print("\nVideo preprocessing framework")
    print("-" * 72)
    print(f"Preset             : {config.preset}")
    print(f"Input video        : {config.video_path}")
    print(f"Output folder      : {config.output_root}")
    print(f"Frame extraction   : {config.extraction_fps:.2f} FPS")
    print(f"Blur threshold     : {config.blur_threshold:.2f}")
    print(
        f"Overlap range      : {config.min_overlap:.2f} to {config.max_overlap:.2f} "
        f"(target {config.target_overlap:.2f})"
    )
    print(f"Auto tune          : {'on' if config.auto_tune else 'off'}")
    print(f"Max image size     : {config.max_image_dim if config.max_image_dim > 0 else 'original'}")
    print(f"Output color mode  : {config.color_mode}")
    print(f"Deblur strength    : {config.deblur_strength:.2f}")
    print(
        "Image cleanup      : "
        f"white balance={'on' if config.enable_white_balance else 'off'}, "
        f"clahe={'on' if config.enable_clahe else 'off'}, "
        f"local contrast={'on' if config.enable_local_contrast else 'off'}"
    )
    print(f"Semantic masking   : {'yes' if config.run_semantic_masking else 'no'}")
    if config.run_semantic_masking:
        print(f"Mask backend       : {config.mask_backend}")
        print(f"Mask model         : {config.mask_model}")
        print(f"Mask classes       : {', '.join(config.mask_classes)}")
        print(f"Mask device        : {config.mask_device}")
        print(f"Mask confidence    : {config.mask_confidence:.2f}")
        print(f"Strict static mask : {'on' if config.strict_static_masking else 'off'}")
    max_images = config.output_max_images if config.output_max_images > 0 else 'unbounded'
    print(f"Angle validation   : {'on' if config.validate_angle_coverage else 'off'} ({config.angle_bins} bins)")
    print(f"Output image range : {config.output_min_images} to {max_images}")
    print(f"Run COLMAP         : {'yes' if config.run_sfm else 'no'}")
    if config.run_sfm:
        print(f"COLMAP matcher     : {config.matcher}")
        print(f"COLMAP device      : {config.colmap_device}")
        print(f"Use GPU in COLMAP  : {'yes' if config.use_gpu else 'no'}")
        print(f"COLMAP parallel    : {'yes' if config.colmap_parallel else 'no'}")
        print(
            f"COLMAP threads     : {config.colmap_num_threads if config.colmap_num_threads > 0 else 'auto'}"
        )
    if config.quality_gate_min_overlap > 0:
        print(f"Quality gate       : mean overlap >= {config.quality_gate_min_overlap:.2f}")
        print(f"Gate failure mode  : {'fail run' if config.quality_gate_fail else 'warn only'}")
    print("-" * 72)


def print_final_summary(paths: PipelinePaths, run_sfm: bool) -> None:
    """End the run with a short summary."""

    print("\nPipeline complete.")
    print(f"Processed dataset folder : {paths.root}")
    print(f"Images for training      : {paths.images}")
    print(f"Semantic masks           : {paths.masks}")
    if run_sfm:
        print(f"Undistorted COLMAP data  : {paths.undistorted}")
    print(f"Run report               : {paths.report}")
