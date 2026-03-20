"""User-facing CLI for the preprocessing framework."""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import shutil
import sys
from typing import Callable

from constants import (
    DEFAULT_ANGLE_BINS,
    DEFAULT_COLMAP_DEVICE,
    DEFAULT_COLOR_MODE,
    DEFAULT_MASK_BACKEND,
    DEFAULT_MASK_CLASSES,
    DEFAULT_MASK_CONFIDENCE,
    DEFAULT_MASK_IMAGE_SIZE,
    DEFAULT_MASK_MODEL,
    DEFAULT_OUTPUT_MAX_IMAGES,
    DEFAULT_OUTPUT_MIN_IMAGES,
    DEFAULT_QUALITY_GATE_FAIL,
    DEFAULT_QUALITY_GATE_MIN_OVERLAP,
    DEFAULT_RCNN_MODEL,
    DEFAULT_STRICT_STATIC_MASKING,
    DEFAULT_VALIDATE_ANGLE_COVERAGE,
    PRESET_DEFAULTS,
)
from console import print_info, print_success, print_warning
from models import PipelineConfig
from pipeline_profiles import (
    list_pipeline_profiles,
    load_pipeline_profile,
    save_pipeline_profile,
    validate_profile_name,
)


class FriendlyHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Help formatter that keeps examples readable and also shows defaults."""


PROFILE_EXCLUDED_FIELDS = {"video", "profile", "show_settings", "help", "command"}


def _format_option_strings(action: argparse.Action) -> str:
    """Build a compact label for one CLI option."""

    option_strings = ", ".join(action.option_strings)
    if action.metavar or action.type or action.nargs not in (0, None):
        value_name = action.metavar or action.dest.upper()
        if action.option_strings:
            return f"{option_strings} {value_name}"
    return option_strings or action.dest


def _iter_run_parser_groups(run_parser: argparse.ArgumentParser):
    """Yield visible action groups from the run command parser."""

    for group in run_parser._action_groups:  # noqa: SLF001 - argparse stores groups on the parser itself.
        actions = [action for action in group._group_actions if action.dest != "help"]  # noqa: SLF001
        if actions:
            yield group, actions


def _get_subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:  # noqa: SLF001
    """Return the argparse subparser action stored on the root parser."""

    return next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))  # noqa: SLF001


def get_run_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Return the `run` subparser so helper code can inspect its defaults."""

    return _get_subparsers_action(parser).choices["run"]


def get_run_parser_defaults(parser: argparse.ArgumentParser) -> dict[str, object]:
    """Build a default-value map for the `run` command."""

    defaults: dict[str, object] = {}
    run_parser = get_run_parser(parser)
    for action in run_parser._actions:  # noqa: SLF001
        if action.dest == "help" or action.default is argparse.SUPPRESS:
            continue
        defaults[action.dest] = copy.deepcopy(action.default)
    return defaults


def build_profile_options_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Keep only profile-worthy run options when saving a reusable pipeline."""

    return {
        key: value
        for key, value in vars(args).items()
        if key not in PROFILE_EXCLUDED_FIELDS
    }


def apply_profile_to_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Apply a saved local profile when the user requests one."""

    profile_name = getattr(args, "profile", "")
    if not profile_name:
        return args

    profile_options = load_pipeline_profile(profile_name)
    if profile_options is None:
        parser.error(f"Saved profile was not found: {profile_name}")

    defaults = get_run_parser_defaults(parser)
    for field_name, saved_value in profile_options.items():
        if field_name in PROFILE_EXCLUDED_FIELDS or not hasattr(args, field_name):
            continue
        current_value = getattr(args, field_name)
        if current_value == defaults.get(field_name):
            setattr(args, field_name, copy.deepcopy(saved_value))
    return args


def print_saved_profiles() -> None:
    """Show all saved local pipeline profiles."""

    profiles = list_pipeline_profiles()
    if not profiles:
        print("No saved pipeline profiles were found on this device.")
        print("Use 'python preprocess.py setup' to create one.")
        return

    print("Saved pipeline profiles\n")
    for name, entry in sorted(profiles.items()):
        options = entry.get("options", {}) if isinstance(entry, dict) else {}
        saved_at = entry.get("saved_at", "unknown time") if isinstance(entry, dict) else "unknown time"
        preset = options.get("preset", "balanced")
        mask_backend = options.get("mask_backend", DEFAULT_MASK_BACKEND)
        run_sfm = not options.get("no_colmap", False)
        print(
            f"- {name} | preset={preset} | masking={mask_backend} | colmap={'on' if run_sfm else 'off'} | saved={saved_at}"
        )
    print("\nUse a saved profile with:")
    print("  python preprocess.py run input.mp4 --profile PROFILE_NAME")


def prompt_choice(question: str, options: list[tuple[str, object]], default_index: int = 0) -> object:
    """Ask the user to choose one item from a numbered list."""

    print(f"\n{question}")
    for index, (label, _) in enumerate(options, start=1):
        default_marker = " [default]" if index - 1 == default_index else ""
        print(f"  {index}. {label}{default_marker}")

    while True:
        raw = input(f"Choose 1-{len(options)} [{default_index + 1}]: ").strip()
        if not raw:
            return options[default_index][1]
        if raw.isdigit():
            selected_index = int(raw) - 1
            if 0 <= selected_index < len(options):
                return options[selected_index][1]

        lowered = raw.lower()
        for label, value in options:
            if lowered == str(value).lower() or lowered == label.lower():
                return value
        print_warning("Please choose one of the listed options.")


def prompt_text(question: str, default: str = "") -> str:
    """Prompt for free-form text while still showing the default value."""

    suffix = f" [{default}]" if default else ""
    raw = input(f"{question}{suffix}: ").strip()
    return raw or default


def prompt_numeric(question: str, default: float | int, suggestions: list[float | int], cast: type) -> float | int:
    """Prompt for a numeric setting with common choices and a custom path."""

    options: list[tuple[str, object]] = []
    all_values: list[float | int] = []
    for value in [default, *suggestions]:
        if value not in all_values:
            all_values.append(value)
    for value in all_values:
        options.append((f"Use {value}", value))
    options.append(("Enter a custom value", "__custom__"))

    default_index = all_values.index(default)
    selected = prompt_choice(question, options, default_index=default_index)
    if selected != "__custom__":
        return cast(selected)

    while True:
        raw = input("Enter your custom value: ").strip()
        try:
            return cast(raw)
        except ValueError:
            print_warning("Please enter a valid number.")


def prompt_bool(question: str, default: bool) -> bool:
    """Prompt for a yes or no answer using numbered choices."""

    default_index = 0 if default else 1
    return bool(
        prompt_choice(
            question,
            [("Yes", True), ("No", False)],
            default_index=default_index,
        )
    )


def build_wizard_defaults(parser: argparse.ArgumentParser, preset_name: str) -> dict[str, object]:
    """Start the setup wizard from CLI defaults plus the chosen preset."""

    values = get_run_parser_defaults(parser)
    values["preset"] = preset_name
    values.update(PRESET_DEFAULTS.get(preset_name, {}))
    return values


def run_setup_wizard() -> int:
    """Walk the user through every major pipeline option and save a named profile."""

    parser = build_argument_parser()
    print("Pipeline setup wizard\n")
    print("This wizard will walk through the pipeline step by step and save a reusable local profile.")

    while True:
        profile_name = prompt_text("Choose a profile name", "my_pipeline")
        if not validate_profile_name(profile_name):
            print_warning("Use 2-64 characters with letters, numbers, dashes, or underscores.")
            continue

        existing_profile = load_pipeline_profile(profile_name)
        if existing_profile is None:
            break
        if prompt_bool(f"A profile named '{profile_name}' already exists. Overwrite it?", default=False):
            break

    preset_name = str(
        prompt_choice(
            "Choose a starting preset",
            [
                ("balanced - safe default for most cases", "balanced"),
                ("laptop-fast - lighter settings for easier runs", "laptop-fast"),
                ("quality - stronger settings when quality matters more", "quality"),
            ],
            default_index=0,
        )
    )
    values = build_wizard_defaults(parser, preset_name)

    values["output"] = prompt_text("Default output folder for this profile", str(values["output"]))

    values["fps"] = prompt_numeric("Frame sampling rate (FPS)", float(values["fps"]), [2.0, 3.0, 4.0, 6.0], float)
    values["blur_threshold"] = prompt_numeric(
        "Blur threshold",
        float(values["blur_threshold"]),
        [60.0, 90.0, 120.0],
        float,
    )
    values["min_overlap"] = prompt_numeric("Minimum overlap", float(values["min_overlap"]), [0.75, 0.85, 0.9], float)
    values["max_overlap"] = prompt_numeric("Maximum overlap", float(values["max_overlap"]), [0.95, 0.97, 0.99], float)
    values["target_overlap"] = prompt_numeric(
        "Target overlap",
        float(values["target_overlap"]),
        [0.85, 0.9, 0.93],
        float,
    )
    values["no_auto_tune"] = not prompt_bool("Enable automatic threshold tuning?", default=not bool(values["no_auto_tune"]))
    values["sharpness_percentile"] = prompt_numeric(
        "Sharpness percentile",
        float(values["sharpness_percentile"]),
        [45.0, 55.0, 65.0],
        float,
    )
    values["texture_percentile"] = prompt_numeric(
        "Texture percentile",
        float(values["texture_percentile"]),
        [25.0, 35.0, 50.0],
        float,
    )

    values["max_image_dim"] = prompt_numeric(
        "Maximum output image dimension",
        int(values["max_image_dim"]),
        [0, 1600, 1920, 2560],
        int,
    )
    values["color_mode"] = prompt_choice(
        "Choose the output color mode",
        [("Color images", "color"), ("Grayscale images", "grayscale")],
        default_index=0 if values["color_mode"] == "color" else 1,
    )
    values["deblur_strength"] = prompt_numeric(
        "Deblur strength",
        float(values["deblur_strength"]),
        [0.0, 0.75, 1.0, 1.25],
        float,
    )
    values["disable_white_balance"] = not prompt_bool(
        "Enable white balance?",
        default=not bool(values["disable_white_balance"]),
    )
    values["disable_clahe"] = not prompt_bool(
        "Enable CLAHE local equalization?",
        default=not bool(values["disable_clahe"]),
    )
    values["disable_local_contrast"] = not prompt_bool(
        "Enable local contrast boost?",
        default=not bool(values["disable_local_contrast"]),
    )

    values["no_semantic_mask"] = not prompt_bool(
        "Enable semantic masking?",
        default=not bool(values["no_semantic_mask"]),
    )
    if not values["no_semantic_mask"]:
        values["mask_backend"] = prompt_choice(
            "Choose the masking backend",
            [
                ("YOLO segmentation", "yolo"),
                ("Mask R-CNN", "rcnn"),
            ],
            default_index=0 if values["mask_backend"] == "yolo" else 1,
        )
        default_mask_model = resolve_mask_model(str(values["mask_backend"]), str(values["mask_model"]))
        values["mask_model"] = prompt_text("Mask model name or path", default_mask_model)
        mask_classes_default = values["mask_classes"]
        if isinstance(mask_classes_default, tuple):
            mask_classes_default = ",".join(str(item) for item in mask_classes_default)
        values["mask_classes"] = prompt_text("Mask classes (comma-separated or 'all')", str(mask_classes_default))
        values["mask_device"] = prompt_choice(
            "Choose the masking device",
            [
                ("Auto detect the best device", "auto"),
                ("CPU", "cpu"),
                ("Apple MPS", "mps"),
                ("CUDA GPU", "cuda"),
                ("Custom device string", "__custom__"),
            ],
            default_index=0,
        )
        if values["mask_device"] == "__custom__":
            values["mask_device"] = prompt_text("Enter the custom masking device", "cuda:0")
        values["mask_image_size"] = prompt_numeric(
            "Mask inference image size",
            int(values["mask_image_size"]),
            [512, 640, 768, 1024],
            int,
        )
        values["mask_confidence"] = prompt_numeric(
            "Mask confidence threshold",
            float(values["mask_confidence"]),
            [0.25, 0.35, 0.5, 0.7],
            float,
        )

    values["no_colmap"] = not prompt_bool(
        "Run COLMAP reconstruction?",
        default=not bool(values["no_colmap"]),
    )
    if not values["no_colmap"]:
        values["matcher"] = prompt_choice(
            "Choose the COLMAP matcher",
            [("Sequential matcher", "sequential"), ("Exhaustive matcher", "exhaustive")],
            default_index=0 if values["matcher"] == "sequential" else 1,
        )
        values["sequential_overlap"] = prompt_numeric(
            "Sequential matcher overlap",
            int(values["sequential_overlap"]),
            [8, 12, 16, 24],
            int,
        )
        values["disable_loop_detection"] = not prompt_bool(
            "Enable loop detection?",
            default=not bool(values["disable_loop_detection"]),
        )
        if not values["disable_loop_detection"]:
            values["vocab_tree_path"] = prompt_choice(
                "Vocabulary tree path",
                [
                    ("Do not use a vocabulary tree", ""),
                    ("Enter a custom path", "__custom__"),
                ],
                default_index=0 if not values["vocab_tree_path"] else 1,
            )
            if values["vocab_tree_path"] == "__custom__":
                values["vocab_tree_path"] = prompt_text("Enter the vocabulary tree path", str(values["vocab_tree_path"]))
        values["colmap_device"] = prompt_choice(
            "Choose the COLMAP device",
            [
                ("Auto detect the best device", "auto"),
                ("CPU", "cpu"),
                ("CUDA GPU", "cuda"),
                ("Custom device string", "__custom__"),
            ],
            default_index=0,
        )
        if values["colmap_device"] == "__custom__":
            values["colmap_device"] = prompt_text("Enter the custom COLMAP device", "cuda:0")
        values["colmap_parallel"] = prompt_choice(
            "Choose the COLMAP parallel mode",
            [
                ("Automatic", None),
                ("Force parallel on", True),
                ("Force parallel off", False),
            ],
            default_index=0 if values["colmap_parallel"] is None else (1 if values["colmap_parallel"] else 2),
        )
        values["colmap_threads"] = prompt_choice(
            "Choose the COLMAP thread limit",
            [
                ("Automatic thread count", 0),
                ("Use 4 threads", 4),
                ("Use 8 threads", 8),
                ("Use 16 threads", 16),
                ("Enter a custom number", "__custom__"),
            ],
            default_index=0 if int(values["colmap_threads"]) == 0 else 4,
        )
        if values["colmap_threads"] == "__custom__":
            values["colmap_threads"] = prompt_numeric("Enter the custom thread limit", 0, [4, 8, 16], int)
        values["cpu_only"] = False

    print("\nProfile summary")
    print(f"- name: {profile_name}")
    print(f"- preset: {values['preset']}")
    print(f"- output: {values['output']}")
    print(f"- masking: {'off' if values['no_semantic_mask'] else values['mask_backend']}")
    print(f"- colmap: {'off' if values['no_colmap'] else values['matcher']}")

    if not prompt_bool("Save this profile locally on this device?", default=True):
        print_info("Setup wizard canceled before saving.")
        return 0

    save_pipeline_profile(profile_name, values)
    print_success(f"Saved pipeline profile '{profile_name}'.")
    print_info(f"Use it later with: python preprocess.py run input.mp4 --profile {profile_name}")
    return 0


def show_cli_settings() -> None:
    """Print a user-friendly overview of every available run setting."""

    parser = build_argument_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)  # noqa: SLF001
    )
    run_parser = subparsers_action.choices["run"]

    print("Available pipeline settings\n")
    print("Use one of these forms:")
    print("- python preprocess.py run input.mp4 [options]")
    print("- python preprocess.py input.mp4 [options]\n")

    for group, actions in _iter_run_parser_groups(run_parser):
        print(f"{group.title}")
        if group.description:
            print(f"  {group.description}")
        for action in actions:
            label = _format_option_strings(action)
            details = action.help or ""
            default_value = None if action.default is argparse.SUPPRESS else action.default
            print(f"  {label}")
            if details:
                print(f"    {details}")
            if action.choices:
                print(f"    Choices: {', '.join(str(choice) for choice in action.choices)}")
            if default_value not in (None, False):
                print(f"    Default: {default_value}")
        print()

    print("Built-in presets")
    for preset_name, preset_values in PRESET_DEFAULTS.items():
        if preset_values:
            formatted_values = ", ".join(f"{key}={value}" for key, value in preset_values.items())
        else:
            formatted_values = "uses the standard defaults"
        print(f"  {preset_name}: {formatted_values}")


def show_effective_settings(args: argparse.Namespace) -> None:
    """Show the resolved settings for one specific run command without starting the pipeline."""

    config = build_config_from_args(args)
    print("Resolved settings for this run\n")
    print(f"Preset                : {config.preset}")
    if getattr(args, "profile", ""):
        print(f"Saved profile         : {args.profile}")
    print(f"Input video           : {config.video_path}")
    print(f"Output folder         : {config.output_root}")
    print(f"Frame extraction FPS  : {config.extraction_fps}")
    print(f"Blur threshold        : {config.blur_threshold}")
    print(f"Overlap window        : {config.min_overlap} -> {config.max_overlap} (target {config.target_overlap})")
    print(f"Auto tune             : {config.auto_tune}")
    print(f"Sharpness percentile  : {config.sharpness_percentile}")
    print(f"Texture percentile    : {config.texture_percentile}")
    print(f"Max image dimension   : {config.max_image_dim}")
    print(f"Color mode            : {config.color_mode}")
    print(f"Deblur strength       : {config.deblur_strength}")
    print(f"White balance         : {config.enable_white_balance}")
    print(f"CLAHE                 : {config.enable_clahe}")
    print(f"Local contrast        : {config.enable_local_contrast}")
    print(f"Semantic masking      : {config.run_semantic_masking}")
    print(f"Mask backend          : {config.mask_backend}")
    print(f"Mask model            : {config.mask_model}")
    print(f"Mask classes          : {', '.join(config.mask_classes)}")
    print(f"Mask device           : {config.mask_device}")
    print(f"Mask image size       : {config.mask_image_size}")
    print(f"Mask confidence       : {config.mask_confidence}")
    print(f"Strict static masking : {config.strict_static_masking}")
    print(f"Validate angle cover  : {config.validate_angle_coverage}")
    print(f"Angle bins            : {config.angle_bins}")
    max_images = config.output_max_images if config.output_max_images > 0 else "unbounded"
    print(f"Output image range    : {config.output_min_images} -> {max_images}")
    print(f"Run COLMAP            : {config.run_sfm}")
    print(f"COLMAP device         : {config.colmap_device}")
    print(f"COLMAP matcher        : {config.matcher}")
    print(f"Sequential overlap    : {config.sequential_overlap}")
    print(f"Loop detection        : {config.loop_detection}")
    print(f"Vocabulary tree       : {config.vocab_tree_path or 'not set'}")
    print(f"Parallel COLMAP       : {config.colmap_parallel}")
    print(f"COLMAP threads        : {config.colmap_num_threads if config.colmap_num_threads > 0 else 'auto'}")
    print(f"Use GPU in COLMAP     : {config.use_gpu}")
    print(f"Quality gate overlap  : {config.quality_gate_min_overlap}")
    print(f"Quality gate fail     : {config.quality_gate_fail}")


def cuda_is_available() -> bool:
    """Best-effort CUDA check used to pick sensible defaults."""

    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    return shutil.which("nvidia-smi") is not None


def normalize_colmap_device(device: str) -> str:
    """Accept friendly CUDA spellings and keep a compact normalized value."""

    normalized = device.strip().lower()
    if normalized == "cuda":
        return "cuda:0"
    return normalized


def resolve_colmap_device(requested_device: str, cpu_only: bool) -> str:
    """Turn the CLI request into the actual COLMAP runtime device."""

    if cpu_only:
        return "cpu"

    normalized = normalize_colmap_device(requested_device)
    if normalized == "auto":
        return "cuda:0" if cuda_is_available() else "cpu"
    if normalized.startswith("cuda") and not cuda_is_available():
        print_warning("CUDA was requested for COLMAP, but no CUDA runtime was detected. Falling back to CPU.")
        return "cpu"
    return normalized


def resolve_colmap_parallel(requested_parallel: bool | None, resolved_device: str) -> bool:
    """Enable parallel COLMAP work automatically when CUDA is active."""

    if requested_parallel is not None:
        return requested_parallel
    return resolved_device.startswith("cuda")


def resolve_mask_model(mask_backend: str, requested_model: str) -> str:
    """Pick a backend-appropriate default masking model when the generic default was left unchanged."""

    if mask_backend == "rcnn" and requested_model == DEFAULT_MASK_MODEL:
        return DEFAULT_RCNN_MODEL
    return requested_model


def parse_output_image_range(value: str) -> tuple[int, int]:
    """Parse the output image range string in the form min:max."""

    text = (value or "").strip()
    if not text:
        return DEFAULT_OUTPUT_MIN_IMAGES, DEFAULT_OUTPUT_MAX_IMAGES

    if ":" not in text:
        number = int(text)
        if number < 0:
            raise ValueError("output image range cannot be negative")
        return number, number

    min_text, max_text = text.split(":", maxsplit=1)
    min_value = int(min_text.strip() or "0")
    max_value = int(max_text.strip() or "0")
    if min_value < 0 or max_value < 0:
        raise ValueError("output image range cannot be negative")
    return min_value, max_value


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Run the full preprocessing pipeline from video input to a cleaned image dataset.\n"
        "Use --show-settings to preview the final resolved configuration before starting."
    )
    parser.epilog = (
        "Examples:\n"
        "  python preprocess.py run input.mp4\n"
        "  python preprocess.py run input.mp4 --preset laptop-fast\n"
        "  python preprocess.py run input.mp4 --color-mode grayscale --no-semantic-mask\n"
        "  python preprocess.py run input.mp4 --colmap-device cuda --colmap-parallel\n"
        "  python preprocess.py run input.mp4 --mask-backend rcnn --mask-model maskrcnn_resnet50_fpn_v2\n"
        "  python preprocess.py run input.mp4 --show-settings\n"
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_DEFAULTS.keys()),
        default="balanced",
        help="Ready-to-use setup. 'laptop-fast' is the easiest starting point.",
    )

    io_group = parser.add_argument_group("Input and output")
    io_group.add_argument(
        "--output",
        default="processed_dataset",
        help="Output folder. If you give a relative path, it is created in the current folder.",
    )
    io_group.add_argument(
        "--profile",
        default="",
        help="Load a saved local pipeline profile created by the setup wizard.",
    )
    io_group.add_argument(
        "--show-settings",
        action="store_true",
        help="Print the fully resolved settings for this run and stop before processing.",
    )

    frame_group = parser.add_argument_group(
        "Frame sampling and selection",
        "These options control how frames are sampled and which ones survive the quality filter.",
    )
    frame_group.add_argument("--fps", type=float, default=3.0, help="How many frames per second to sample.")
    frame_group.add_argument("--blur-threshold", type=float, default=90.0, help="Minimum sharpness score.")
    frame_group.add_argument("--min-overlap", type=float, default=0.85, help="Minimum overlap ratio between groups.")
    frame_group.add_argument("--max-overlap", type=float, default=0.97, help="Maximum overlap ratio for reporting and tuning.")
    frame_group.add_argument("--target-overlap", type=float, default=0.90, help="Preferred overlap ratio.")
    frame_group.add_argument("--no-auto-tune", action="store_true", help="Use fixed thresholds only.")
    frame_group.add_argument("--sharpness-percentile", type=float, default=55.0, help="Adaptive sharpness percentile.")
    frame_group.add_argument("--texture-percentile", type=float, default=35.0, help="Adaptive texture percentile.")
    frame_group.add_argument(
        "--validate-angle-coverage",
        dest="validate_angle_coverage",
        action="store_true",
        default=DEFAULT_VALIDATE_ANGLE_COVERAGE,
        help="Check selected frame coverage across angle bins and backfill missing bins.",
    )
    frame_group.add_argument(
        "--no-validate-angle-coverage",
        dest="validate_angle_coverage",
        action="store_false",
        help="Skip angle-coverage validation and backfilling.",
    )
    frame_group.add_argument(
        "--angle-bins",
        type=int,
        default=DEFAULT_ANGLE_BINS,
        help="How many angle bins to use while checking frame coverage.",
    )
    frame_group.add_argument(
        "--output-image-range",
        default=f"{DEFAULT_OUTPUT_MIN_IMAGES}:{DEFAULT_OUTPUT_MAX_IMAGES}",
        help="Desired output image range as min:max. Use 0 for an open upper bound, for example 80:0.",
    )

    image_group = parser.add_argument_group(
        "Image clean-up",
        "Fine-tune the exported images without changing the rest of the pipeline.",
    )
    image_group.add_argument(
        "--max-image-dim",
        type=int,
        default=1920,
        help="Resize exported images so the largest side is at most this value. Use 0 to keep the original size.",
    )
    image_group.add_argument(
        "--color-mode",
        choices=["color", "grayscale"],
        default=DEFAULT_COLOR_MODE,
        help="Save the final dataset in full color or grayscale.",
    )
    image_group.add_argument(
        "--deblur-strength",
        type=float,
        default=1.0,
        help="Global strength multiplier for the restoration pass. 0 disables sharpening, 1 keeps the default behavior.",
    )
    image_group.add_argument("--disable-white-balance", action="store_true", help="Skip gray-world white balancing.")
    image_group.add_argument("--disable-clahe", action="store_true", help="Skip local luminance equalization.")
    image_group.add_argument("--disable-local-contrast", action="store_true", help="Skip the final local contrast boost.")

    masking_group = parser.add_argument_group(
        "Semantic masking",
        "Use instance segmentation to hide moving objects before reconstruction.",
    )
    masking_group.add_argument("--no-semantic-mask", action="store_true", help="Skip semantic masking.")
    masking_group.add_argument(
        "--mask-backend",
        choices=["yolo", "rcnn"],
        default=DEFAULT_MASK_BACKEND,
        help="Segmentation backend. Use YOLO for Ultralytics models or R-CNN for Torchvision Mask R-CNN.",
    )
    masking_group.add_argument(
        "--mask-model",
        default=DEFAULT_MASK_MODEL,
        help=(
            "Segmentation model name or path. Examples: yolov8n-seg.pt for YOLO, "
            "maskrcnn_resnet50_fpn or maskrcnn_resnet50_fpn_v2 for R-CNN."
        ),
    )
    masking_group.add_argument(
        "--mask-classes",
        default=",".join(DEFAULT_MASK_CLASSES),
        help="Comma-separated classes to consider for masking. Use 'all' to use every class from the segmentation model.",
    )
    masking_group.add_argument(
        "--mask-device",
        default="auto",
        help="Masking device: auto, cpu, mps, cuda, or a specific CUDA device such as cuda:1.",
    )
    masking_group.add_argument("--mask-image-size", type=int, default=DEFAULT_MASK_IMAGE_SIZE, help="Inference image size for masking.")
    masking_group.add_argument(
        "--mask-confidence",
        type=float,
        default=DEFAULT_MASK_CONFIDENCE,
        help="Minimum confidence score for a detection to become part of the ignore mask.",
    )
    masking_group.add_argument(
        "--strict-static-masking",
        action="store_true",
        default=DEFAULT_STRICT_STATIC_MASKING,
        help="Mask all detected selected classes without keeping a focused subject instance.",
    )

    colmap_group = parser.add_argument_group(
        "COLMAP",
        "Reconstruction options for feature extraction, matching, and sparse mapping.",
    )
    colmap_group.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential", help="Matcher type.")
    colmap_group.add_argument("--sequential-overlap", type=int, default=12, help="How many nearby frames the sequential matcher compares.")
    colmap_group.add_argument("--disable-loop-detection", action="store_true", help="Turn off loop detection in the sequential matcher.")
    colmap_group.add_argument("--vocab-tree-path", default="", help="Optional path to a COLMAP vocabulary tree file.")
    colmap_group.add_argument(
        "--colmap-device",
        default=DEFAULT_COLMAP_DEVICE,
        help="COLMAP device: auto, cpu, cuda, or a specific CUDA device such as cuda:0.",
    )
    parallel_group = colmap_group.add_mutually_exclusive_group()
    parallel_group.add_argument(
        "--colmap-parallel",
        dest="colmap_parallel",
        action="store_true",
        default=None,
        help="Let COLMAP use multiple CPU threads where supported.",
    )
    parallel_group.add_argument(
        "--no-colmap-parallel",
        dest="colmap_parallel",
        action="store_false",
        help="Keep COLMAP in a more conservative single-threaded mode.",
    )
    colmap_group.add_argument(
        "--colmap-threads",
        type=int,
        default=0,
        help="Maximum CPU threads for COLMAP. Use 0 to let the script choose automatically.",
    )
    colmap_group.add_argument("--cpu-only", action="store_true", help="Use CPU only for COLMAP feature extraction and matching.")
    colmap_group.add_argument("--no-colmap", action="store_true", help="Skip COLMAP and only export the cleaned image dataset.")

    quality_group = parser.add_argument_group(
        "Quality gates",
        "Optional checks for minimum reconstruction readiness before finishing the run.",
    )
    quality_group.add_argument(
        "--quality-gate-min-overlap",
        type=float,
        default=DEFAULT_QUALITY_GATE_MIN_OVERLAP,
        help="Require at least this mean overlap in selected output frames. Use 0 to disable.",
    )
    quality_group.add_argument(
        "--quality-gate-fail",
        action="store_true",
        default=DEFAULT_QUALITY_GATE_FAIL,
        help="Fail the run when a quality gate check does not pass instead of only warning.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the main framework CLI."""

    parser = argparse.ArgumentParser(
        prog="preprocess.py",
        description=(
            "Turn a video into a cleaner image dataset for 3D reconstruction.\n\n"
            "Main commands:\n"
            "  run       run the full pipeline\n"
            "  setup     create a saved pipeline step by step\n"
            "  profiles  list saved local pipeline profiles\n"
            "  settings  list every available setting with defaults\n"
            "  presets   show the built-in presets\n"
            "  doctor    check the local environment\n"
            "  shell     open the interactive shell"
        ),
        formatter_class=FriendlyHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  python preprocess.py run input.mp4 --preset laptop-fast\n"
            "  python preprocess.py setup\n"
            "  python preprocess.py profiles\n"
            "  python preprocess.py run input.mp4 --output processed_dataset\n"
            "  python preprocess.py settings\n"
            "\nLegacy short form also works:\n"
            "  python preprocess.py input.mp4 --preset laptop-fast\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full preprocessing pipeline.",
        formatter_class=FriendlyHelpFormatter,
    )
    _add_run_arguments(run_parser)

    subparsers.add_parser("shell", help="Open the interactive framework shell.")
    subparsers.add_parser("setup", help="Interactive step-by-step pipeline builder that saves a local profile.")
    subparsers.add_parser("profiles", help="List locally saved pipeline profiles.")
    subparsers.add_parser("settings", help="List every available run setting with defaults.")
    subparsers.add_parser("presets", help="Show the built-in presets.")
    subparsers.add_parser("doctor", help="Check if common dependencies look available.")
    return parser


def normalize_argv(argv: list[str] | None = None) -> list[str]:
    """Keep the old one-line usage working by inserting the run command."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return ["shell"]

    known_commands = {"run", "shell", "setup", "profiles", "settings", "presets", "doctor", "-h", "--help"}
    if args[0] not in known_commands:
        return ["run", *args]
    return args


def show_shell_help() -> None:
    """Print the short command list used inside the interactive shell."""

    print("Interactive shell commands:\n")
    print("- run <video> [options]   : run the full pipeline")
    print("- <video> [options]       : same as run, short form")
    print("- setup                   : build a pipeline profile step by step")
    print("- profiles                : list saved local pipeline profiles")
    print("- settings                : list every run setting and its default")
    print("- presets                 : show built-in presets")
    print("- doctor                  : check the environment")
    print("- help                    : show this shell help")
    print("- exit                    : leave the shell")


def run_interactive_shell(execute_run_command: Callable[[argparse.Namespace], None]) -> int:
    """Keep the framework open so the user can run many commands in one session."""

    print("3D Preprocessing interactive shell")
    print("Type 'help' to see commands. Type 'exit' to close the shell.\n")

    while True:
        try:
            raw_command = input("preprocess> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nUse 'exit' to close the shell.")
            continue

        if not raw_command:
            continue
        if raw_command.lower() in {"exit", "quit"}:
            print_info("Leaving the interactive shell.")
            return 0
        if raw_command.lower() in {"help", "?"}:
            show_shell_help()
            continue

        try:
            args = parse_args(shlex.split(raw_command))
        except SystemExit:
            # argparse already printed the reason.
            continue

        if args.command == "shell":
            show_shell_help()
            continue

        special_command_exit_code = handle_special_command(args)
        if special_command_exit_code is not None:
            continue

        try:
            execute_run_command(args)
        except Exception as error:
            print_warning(f"Run failed: {error}")


def show_presets() -> None:
    """Print the built-in presets."""

    print("Available presets:\n")
    print("- balanced     : safe default for normal use")
    print("- laptop-fast  : lighter settings for easier runs on a laptop")
    print("- quality      : higher image settings when quality matters most")


def run_doctor() -> int:
    """Check a few common tools and packages."""

    print("Environment check\n")
    python_ok = True
    print_success("Python environment is active.")

    if shutil.which("colmap"):
        print_success("COLMAP command found.")
    else:
        print_warning("COLMAP command was not found in PATH.")
        python_ok = False

    try:
        import cv2  # noqa: F401

        print_success("OpenCV is installed.")
    except Exception:
        print_warning("OpenCV is missing.")
        python_ok = False

    try:
        import ultralytics  # noqa: F401

        print_success("Ultralytics is installed.")
    except Exception:
        print_warning("Ultralytics is missing. The YOLO masking backend will not work.")

    try:
        import torchvision  # noqa: F401

        print_success("Torchvision is installed.")
    except Exception:
        print_warning("Torchvision is missing. The R-CNN masking backend will not work.")

    if cuda_is_available():
        print_success("CUDA runtime detected. NVIDIA acceleration should be available.")
    else:
        print_info("CUDA runtime was not detected. COLMAP and masking will fall back to CPU or MPS.")

    return 0 if python_ok else 1


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply preset values only when the user did not override them."""

    preset_values = PRESET_DEFAULTS.get(args.preset, {})
    parser_defaults = {
        "fps": 3.0,
        "max_image_dim": 1920,
        "color_mode": DEFAULT_COLOR_MODE,
        "mask_image_size": DEFAULT_MASK_IMAGE_SIZE,
        "matcher": "sequential",
    }

    for field_name, preset_value in preset_values.items():
        if getattr(args, field_name) == parser_defaults.get(field_name):
            setattr(args, field_name, preset_value)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Keep argument checks in one place."""

    if args.command in {"shell", "setup", "profiles", "settings", "presets", "doctor"}:
        return
    if args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.max_overlap < args.min_overlap:
        parser.error("--max-overlap must be greater than or equal to --min-overlap")
    if not (args.min_overlap <= args.target_overlap <= args.max_overlap):
        parser.error("--target-overlap must stay between --min-overlap and --max-overlap")
    if not (0 <= args.sharpness_percentile <= 100):
        parser.error("--sharpness-percentile must be within [0, 100]")
    if not (0 <= args.texture_percentile <= 100):
        parser.error("--texture-percentile must be within [0, 100]")
    if args.max_image_dim < 0:
        parser.error("--max-image-dim must be 0 or greater")
    if args.deblur_strength < 0:
        parser.error("--deblur-strength must be 0 or greater")
    if args.mask_image_size < 32:
        parser.error("--mask-image-size must be at least 32")
    if not (0.0 <= args.mask_confidence <= 1.0):
        parser.error("--mask-confidence must be within [0, 1]")
    if args.angle_bins < 3:
        parser.error("--angle-bins must be at least 3")
    if args.sequential_overlap < 1:
        parser.error("--sequential-overlap must be at least 1")
    if args.colmap_threads < 0:
        parser.error("--colmap-threads must be 0 or greater")
    if not (0.0 <= args.quality_gate_min_overlap <= 1.0):
        parser.error("--quality-gate-min-overlap must be within [0, 1]")

    try:
        output_min_images, output_max_images = parse_output_image_range(args.output_image_range)
    except ValueError as error:
        parser.error(f"--output-image-range is invalid: {error}")

    if output_max_images > 0 and output_min_images > output_max_images:
        parser.error("--output-image-range must use min <= max when max is not zero")

    args.output_min_images = output_min_images
    args.output_max_images = output_max_images

    if not os.path.isfile(args.video):
        parser.error(f"Input video was not found: {args.video}")
    valid_mask_devices = {"auto", "cpu", "mps", "cuda"}
    if args.mask_device.lower() not in valid_mask_devices and not args.mask_device.lower().startswith("cuda:"):
        parser.error("--mask-device must be one of auto, cpu, mps, cuda, or cuda:<index>")
    valid_colmap_devices = {"auto", "cpu", "cuda"}
    if args.colmap_device.lower() not in valid_colmap_devices and not args.colmap_device.lower().startswith("cuda:"):
        parser.error("--colmap-device must be one of auto, cpu, cuda, or cuda:<index>")
    if args.cpu_only and normalize_colmap_device(args.colmap_device) != "auto":
        parser.error("Use either --cpu-only or --colmap-device, not both together")
    if args.disable_loop_detection and args.vocab_tree_path:
        print_warning("A vocabulary tree was provided, but loop detection is disabled, so that file will be ignored.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command line options."""

    parser = build_argument_parser()
    normalized = normalize_argv(argv)
    args = parser.parse_args(normalized)
    if args.command == "run":
        args = apply_profile_to_args(args, parser)
        args = apply_preset(args)
    validate_args(parser, args)
    return args


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Convert parsed arguments into the main config object."""

    output_root = args.output
    if not os.path.isabs(output_root):
        output_root = os.path.join(os.getcwd(), output_root)

    mask_classes = tuple(class_name.strip().lower() for class_name in args.mask_classes.split(",") if class_name.strip())
    if not mask_classes:
        mask_classes = DEFAULT_MASK_CLASSES

    resolved_colmap_device = resolve_colmap_device(args.colmap_device, cpu_only=args.cpu_only)
    colmap_parallel = resolve_colmap_parallel(args.colmap_parallel, resolved_colmap_device)
    resolved_mask_model = resolve_mask_model(args.mask_backend, args.mask_model)

    return PipelineConfig(
        video_path=os.path.abspath(args.video),
        output_root=os.path.abspath(output_root),
        extraction_fps=args.fps,
        blur_threshold=args.blur_threshold,
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        target_overlap=args.target_overlap,
        auto_tune=not args.no_auto_tune,
        sharpness_percentile=args.sharpness_percentile,
        texture_percentile=args.texture_percentile,
        max_image_dim=args.max_image_dim,
        color_mode=args.color_mode,
        deblur_strength=args.deblur_strength,
        enable_white_balance=not args.disable_white_balance,
        enable_clahe=not args.disable_clahe,
        enable_local_contrast=not args.disable_local_contrast,
        run_semantic_masking=not args.no_semantic_mask,
        strict_static_masking=args.strict_static_masking,
        mask_backend=args.mask_backend,
        mask_model=resolved_mask_model,
        mask_classes=mask_classes,
        mask_device=args.mask_device,
        mask_image_size=args.mask_image_size,
        mask_confidence=args.mask_confidence,
        colmap_device=resolved_colmap_device,
        colmap_parallel=colmap_parallel,
        colmap_num_threads=args.colmap_threads,
        matcher=args.matcher,
        sequential_overlap=args.sequential_overlap,
        loop_detection=not args.disable_loop_detection,
        vocab_tree_path=args.vocab_tree_path,
        use_gpu=resolved_colmap_device.startswith("cuda"),
        run_sfm=not args.no_colmap,
        angle_bins=args.angle_bins,
        validate_angle_coverage=args.validate_angle_coverage,
        output_min_images=args.output_min_images,
        output_max_images=args.output_max_images,
        quality_gate_min_overlap=args.quality_gate_min_overlap,
        quality_gate_fail=args.quality_gate_fail,
        preset=args.preset,
    )


def handle_special_command(args: argparse.Namespace) -> int | None:
    """Run helper commands that do not start the pipeline."""

    if args.command == "setup":
        return run_setup_wizard()
    if args.command == "profiles":
        print_saved_profiles()
        return 0
    if args.command == "settings":
        show_cli_settings()
        return 0
    if args.command == "presets":
        show_presets()
        return 0
    if args.command == "doctor":
        return run_doctor()
    if args.command == "shell":
        return None
    if args.command == "run":
        if getattr(args, "show_settings", False):
            show_effective_settings(args)
            return 0
        return None
    print_info("No command provided. Use --help to see available commands.")
    return 0
