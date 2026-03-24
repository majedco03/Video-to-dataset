"""User-facing CLI for the preprocessing framework."""

from __future__ import annotations

import argparse
import copy
import os
import re
import shlex
import shutil
import subprocess
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
from rich import box
from rich.panel import Panel
from rich.table import Table

from console import console, print_error, print_info, print_success, print_warning
from models import PipelineConfig
from pipeline_profiles import (
    delete_pipeline_profile,
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

    # Auto-migrate: add new fields present in the current parser but absent from the saved profile.
    _MIGRATION_EXCLUDED = PROFILE_EXCLUDED_FIELDS | {"videos", "dir", "dry_run"}
    new_fields = {
        key: value
        for key, value in defaults.items()
        if key not in profile_options
        and key not in _MIGRATION_EXCLUDED
        and value is not None
    }
    if new_fields:
        profile_options = {**profile_options, **new_fields}
        save_pipeline_profile(profile_name, profile_options)
        print_info(
            f"Profile '[bold]{profile_name}[/bold]' was automatically updated with "
            f"{len(new_fields)} new setting(s): {', '.join(sorted(new_fields))}"
        )

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
        print_info("No saved pipeline profiles were found on this device.")
        print_info("Use 'python preprocess.py setup' to create one.")
        return

    table = Table(title="Saved Pipeline Profiles", box=box.ROUNDED, show_lines=True)
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Preset")
    table.add_column("Masking")
    table.add_column("COLMAP")
    table.add_column("Saved at", style="dim")

    for name, entry in sorted(profiles.items()):
        options = entry.get("options", {}) if isinstance(entry, dict) else {}
        saved_at = entry.get("saved_at", "—") if isinstance(entry, dict) else "—"
        preset = options.get("preset", "balanced")
        mask_backend = options.get("mask_backend", DEFAULT_MASK_BACKEND)
        run_sfm = not options.get("no_colmap", False)
        table.add_row(
            name,
            preset,
            mask_backend,
            "[green]on[/green]" if run_sfm else "[dim]off[/dim]",
            saved_at[:19] if len(saved_at) > 19 else saved_at,
        )

    console.print()
    console.print(table)
    console.print("\n[dim]Use a saved profile with:[/dim]  python preprocess.py run input.mp4 --profile PROFILE_NAME")


def _questionary_available() -> bool:
    try:
        import questionary  # noqa: F401
        return True
    except ImportError:
        return False


def prompt_choice(question: str, options: list[tuple[str, object]], default_index: int = 0) -> object:
    """Ask the user to choose one item from a list, using questionary when available."""

    if _questionary_available():
        import questionary
        choices = [questionary.Choice(title=label, value=value) for label, value in options]
        result = questionary.select(question, choices=choices, default=choices[default_index]).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

    # Fallback: numbered list
    console.print(f"\n[bold]{question}[/bold]")
    for index, (label, _) in enumerate(options, start=1):
        marker = " [dim][default][/dim]" if index - 1 == default_index else ""
        console.print(f"  [cyan]{index}[/cyan]. {label}{marker}")
    while True:
        raw = input(f"Choose 1-{len(options)} [{default_index + 1}]: ").strip()
        if not raw:
            return options[default_index][1]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][1]
        lowered = raw.lower()
        for label, value in options:
            if lowered == str(value).lower() or lowered == label.lower():
                return value
        print_warning("Please choose one of the listed options.")


def prompt_text(question: str, default: str = "") -> str:
    """Prompt for free-form text while still showing the default value."""

    if _questionary_available():
        import questionary
        result = questionary.text(question, default=default).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

    suffix = f" [{default}]" if default else ""
    raw = input(f"{question}{suffix}: ").strip()
    return raw or default


def prompt_numeric(question: str, default: float | int, suggestions: list[float | int], cast: type) -> float | int:
    """Prompt for a numeric setting with common choices and a custom path."""

    all_values: list[float | int] = []
    for value in [default, *suggestions]:
        if value not in all_values:
            all_values.append(value)
    options: list[tuple[str, object]] = [(str(v), v) for v in all_values]
    options.append(("Enter a custom value…", "__custom__"))

    default_index = all_values.index(default)
    selected = prompt_choice(question, options, default_index=default_index)
    if selected != "__custom__":
        return cast(selected)

    while True:
        raw = prompt_text("Custom value", str(default))
        try:
            return cast(raw)
        except ValueError:
            print_warning("Please enter a valid number.")


def prompt_bool(question: str, default: bool) -> bool:
    """Prompt for a yes or no answer."""

    if _questionary_available():
        import questionary
        result = questionary.confirm(question, default=default).ask()
        if result is None:
            raise KeyboardInterrupt
        return bool(result)

    default_index = 0 if default else 1
    return bool(
        prompt_choice(question, [("Yes", True), ("No", False)], default_index=default_index)
    )


def build_wizard_defaults(parser: argparse.ArgumentParser, preset_name: str) -> dict[str, object]:
    """Start the setup wizard from CLI defaults plus the chosen preset."""

    values = get_run_parser_defaults(parser)
    values["preset"] = preset_name
    values.update(PRESET_DEFAULTS.get(preset_name, {}))
    return values


def _mask_device_default_index(device: str) -> tuple[int, str]:
    """Return (list_index, custom_default) for the masking device choice prompt."""
    d = str(device).strip().lower()
    if d == "auto":
        return 0, ""
    if d == "cpu":
        return 1, ""
    if d == "mps":
        return 2, ""
    if d == "cuda":
        return 3, ""
    return 4, str(device)  # specific CUDA index or other custom string


def _colmap_device_default_index(device: str) -> tuple[int, str]:
    """Return (list_index, custom_default) for the COLMAP device choice prompt."""
    d = str(device).strip().lower()
    if d == "auto":
        return 0, ""
    if d == "cpu":
        return 1, ""
    if d == "cuda":
        return 2, ""
    return 3, str(device)  # specific CUDA index or other custom string


def _run_wizard_prompts(values: dict) -> dict:
    """Run the shared wizard questions and return the updated values dict.

    *values* is pre-seeded with the current (or default) settings and each
    prompt uses those values as its default, so both the setup and the
    edit-profile wizards can call this identical body.
    """

    values["output"] = prompt_text("Default output folder for this profile", str(values["output"]))

    console.print()
    console.print(Panel.fit(
        "[bold]Frame extraction and quality filter[/bold]\n"
        "[dim]Controls which frames are pulled from the video and which are discarded.\n"
        "Higher FPS = more frames = better coverage but slower COLMAP.\n"
        "The blur threshold drops frames that are too shaky or out of focus.[/dim]",
        border_style="dim",
    ))
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

    # Parse current output_image_range into separate min/max for easier prompting.
    _range_str = str(values.get("output_image_range") or "0:0")
    try:
        _range_min, _range_max = parse_output_image_range(_range_str)
    except Exception:
        _range_min, _range_max = 0, 0
    console.print(
        "\n  [dim]Output image count[/dim]  "
        "[dim]— how many frames end up in the final dataset. Use 0 for no limit.[/dim]"
    )
    _range_min = int(prompt_numeric("Minimum output images (0 = no minimum)", _range_min, [0, 30, 50, 80], int))
    _range_max = int(prompt_numeric("Maximum output images (0 = no limit)",   _range_max, [0, 80, 120, 200], int))
    values["output_image_range"] = f"{_range_min}:{_range_max}"

    console.print()
    console.print(Panel.fit(
        "[bold]Image cleanup[/bold]\n"
        "[dim]Post-processing applied to each saved frame before COLMAP sees it.\n"
        "Smaller max dimension = faster COLMAP and less disk space.\n"
        "White balance, CLAHE, and local contrast improve textures in tricky lighting.[/dim]",
        border_style="dim",
    ))
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

    console.print()
    console.print(Panel.fit(
        "[bold]Semantic masking[/bold]\n"
        "[dim]Detects and paints out moving objects (people, cars, animals) so COLMAP\n"
        "cannot match features on them. Strongly recommended for outdoor scenes.\n"
        "YOLO is faster; Mask R-CNN tends to be more accurate on complex shapes.[/dim]",
        border_style="dim",
    ))
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
        mask_dev_idx, mask_dev_custom = _mask_device_default_index(str(values["mask_device"]))
        values["mask_device"] = prompt_choice(
            "Choose the masking device",
            [
                ("Auto detect the best device", "auto"),
                ("CPU", "cpu"),
                ("Apple MPS", "mps"),
                ("CUDA GPU", "cuda"),
                ("Custom device string", "__custom__"),
            ],
            default_index=mask_dev_idx,
        )
        if values["mask_device"] == "__custom__":
            values["mask_device"] = prompt_text("Enter the custom masking device", mask_dev_custom or "cuda:0")
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

    console.print()
    console.print(Panel.fit(
        "[bold]COLMAP reconstruction[/bold]\n"
        "[dim]Estimates where each camera was in 3D space using feature matching across frames.\n"
        "Sequential matcher is fast and works well for video walkthroughs.\n"
        "Exhaustive matcher is slower but handles object-centric or turntable captures better.\n"
        "GPU acceleration (CUDA) makes feature extraction significantly faster.[/dim]",
        border_style="dim",
    ))
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
            orig_vocab_tree = str(values.get("vocab_tree_path") or "")
            values["vocab_tree_path"] = prompt_choice(
                "Vocabulary tree path",
                [
                    ("Do not use a vocabulary tree", ""),
                    ("Enter a custom path", "__custom__"),
                ],
                default_index=0 if not orig_vocab_tree else 1,
            )
            if values["vocab_tree_path"] == "__custom__":
                values["vocab_tree_path"] = prompt_text("Enter the vocabulary tree path", orig_vocab_tree)
        colmap_dev_idx, colmap_dev_custom = _colmap_device_default_index(str(values["colmap_device"]))
        values["colmap_device"] = prompt_choice(
            "Choose the COLMAP device",
            [
                ("Auto detect the best device", "auto"),
                ("CPU", "cpu"),
                ("CUDA GPU", "cuda"),
                ("Custom device string", "__custom__"),
            ],
            default_index=colmap_dev_idx,
        )
        if values["colmap_device"] == "__custom__":
            values["colmap_device"] = prompt_text("Enter the custom COLMAP device", colmap_dev_custom or "cuda:0")
        values["colmap_parallel"] = prompt_choice(
            "Choose the COLMAP parallel mode",
            [
                ("Automatic", None),
                ("Force parallel on", True),
                ("Force parallel off", False),
            ],
            default_index=0 if values["colmap_parallel"] is None else (1 if values["colmap_parallel"] else 2),
        )
        thread_val = int(values["colmap_threads"]) if values["colmap_threads"] is not None else 0
        thread_options = [0, 4, 8, 16]
        thread_default_idx = thread_options.index(thread_val) if thread_val in thread_options else 4
        values["colmap_threads"] = prompt_choice(
            "Choose the COLMAP thread limit",
            [
                ("Automatic thread count", 0),
                ("Use 4 threads", 4),
                ("Use 8 threads", 8),
                ("Use 16 threads", 16),
                ("Enter a custom number", "__custom__"),
            ],
            default_index=thread_default_idx,
        )
        if values["colmap_threads"] == "__custom__":
            values["colmap_threads"] = prompt_numeric("Enter the custom thread limit", thread_val, [4, 8, 16], int)
        values["cpu_only"] = False

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
    values = _run_wizard_prompts(values)

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


def run_edit_profile_wizard(name: str | None = None) -> int:
    """Load an existing profile into the wizard so the user can update its settings."""

    parser = build_argument_parser()
    profiles = list_pipeline_profiles()

    if not profiles:
        print_info("No saved profiles found. Use 'setup' to create one first.")
        return 1

    if name is None:
        name = str(
            prompt_choice(
                "Select a profile to edit",
                [(n, n) for n in sorted(profiles.keys())],
                default_index=0,
            )
        )

    profile_options = load_pipeline_profile(name)
    if profile_options is None:
        print_error(f"Profile not found: {name}")
        return 1

    console.print()
    console.print(Panel.fit(f"[bold]Editing profile:[/bold]  [cyan]{name}[/cyan]", border_style="cyan"))

    # Merge parser defaults with the saved options so any new fields added
    # since the profile was created receive sensible fallback values.
    values = get_run_parser_defaults(parser)
    values.update(profile_options)

    while True:
        new_name = prompt_text("Profile name (press Enter to keep current)", name)
        if not validate_profile_name(new_name):
            print_warning("Use 2-64 characters with letters, numbers, dashes, or underscores.")
            continue
        if new_name != name and load_pipeline_profile(new_name) is not None:
            if not prompt_bool(f"Profile '{new_name}' already exists. Overwrite it?", default=False):
                continue
        break

    values = _run_wizard_prompts(values)

    print("\nProfile summary")
    print(f"- name: {new_name}")
    print(f"- preset: {values.get('preset', 'balanced')}")
    print(f"- output: {values.get('output', '')}")
    print(f"- masking: {'off' if values.get('no_semantic_mask') else values.get('mask_backend', 'yolo')}")
    print(f"- colmap: {'off' if values.get('no_colmap') else values.get('matcher', 'sequential')}")

    if not prompt_bool("Save updated profile?", default=True):
        print_info("Edit canceled — the original profile was not changed.")
        return 0

    save_pipeline_profile(new_name, values)
    if new_name != name:
        delete_pipeline_profile(name)
        print_success(f"Profile renamed '{name}' → '{new_name}' and saved.")
    else:
        print_success(f"Profile '{new_name}' updated.")
    print_info(f"Use it with: python preprocess.py run input.mp4 --profile {new_name}")
    return 0


def run_delete_profile(name: str | None = None) -> int:
    """Delete one saved profile, prompting the user to select and confirm."""

    profiles = list_pipeline_profiles()
    if not profiles:
        print_info("No saved profiles found.")
        return 0

    if name is None:
        name = str(
            prompt_choice(
                "Select a profile to delete",
                [(n, n) for n in sorted(profiles.keys())],
                default_index=0,
            )
        )

    if load_pipeline_profile(name) is None:
        print_error(f"Profile not found: {name}")
        return 1

    if not prompt_bool(f"Permanently delete profile '{name}'?", default=False):
        print_info("Delete canceled.")
        return 0

    delete_pipeline_profile(name)
    print_success(f"Profile '{name}' deleted.")
    return 0


def show_cli_settings() -> None:
    """Print a user-friendly overview of every available run setting."""

    parser = build_argument_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)  # noqa: SLF001
    )
    run_parser = subparsers_action.choices["run"]

    console.print()
    console.print(Panel.fit("[bold]Available Pipeline Settings[/bold]", border_style="cyan"))
    console.print("[dim]python preprocess.py run input.mp4 [OPTIONS][/dim]\n")

    for group, actions in _iter_run_parser_groups(run_parser):
        table = Table(
            title=group.title,
            box=box.SIMPLE_HEAD,
            show_lines=False,
            padding=(0, 1),
            title_style="bold yellow",
            title_justify="left",
        )
        table.add_column("Flag", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Default", style="dim")

        if group.description:
            console.print(f"  [dim]{group.description}[/dim]")

        for action in actions:
            label = _format_option_strings(action)
            details = action.help or ""
            default_value = action.default if action.default not in (None, False, argparse.SUPPRESS) else ""
            choices_str = f"  [dim]choices: {', '.join(str(c) for c in action.choices)}[/dim]" if action.choices else ""
            table.add_row(label, details + choices_str, str(default_value) if default_value != "" else "")

        console.print(table)
        console.print()

    presets_table = Table(title="Built-in Presets", box=box.SIMPLE_HEAD, title_style="bold yellow", title_justify="left", padding=(0, 1))
    presets_table.add_column("Preset", style="cyan")
    presets_table.add_column("Overrides")
    for preset_name, preset_values in PRESET_DEFAULTS.items():
        overrides = ", ".join(f"{k}={v}" for k, v in preset_values.items()) if preset_values else "[dim]uses standard defaults[/dim]"
        presets_table.add_row(preset_name, overrides)
    console.print(presets_table)


def show_effective_settings(args: argparse.Namespace) -> None:
    """Show the resolved settings for one specific run command without starting the pipeline."""

    config = build_config_from_args(args)
    console.print()
    console.print(Panel.fit("[bold]Resolved settings for this run[/bold]", border_style="cyan"))
    if getattr(args, "profile", ""):
        print_info(f"Loaded profile: [bold]{args.profile}[/bold]")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
    table.add_column("Setting", style="dim", no_wrap=True)
    table.add_column("Value")

    yes_no = lambda b: "[green]yes[/green]" if b else "[dim]no[/dim]"  # noqa: E731
    on_off = lambda b: "[green]on[/green]" if b else "[dim]off[/dim]"  # noqa: E731
    max_dim = str(config.max_image_dim) if config.max_image_dim > 0 else "[dim]original[/dim]"
    max_images = str(config.output_max_images) if config.output_max_images > 0 else "[dim]unbounded[/dim]"
    threads = str(config.colmap_num_threads) if config.colmap_num_threads > 0 else "[dim]auto[/dim]"

    table.add_row("Preset", f"[bold]{config.preset}[/bold]")
    table.add_row("Input", config.video_path)
    table.add_row("Output", config.output_root)
    table.add_row("FPS", str(config.extraction_fps))
    table.add_row("Blur threshold", str(config.blur_threshold))
    table.add_row("Overlap", f"{config.min_overlap} → {config.max_overlap} (target {config.target_overlap})")
    table.add_row("Auto tune", on_off(config.auto_tune))
    table.add_row("Sharpness pct", str(config.sharpness_percentile))
    table.add_row("Texture pct", str(config.texture_percentile))
    table.add_row("Max image dim", max_dim)
    table.add_row("Color mode", config.color_mode)
    table.add_row("Deblur strength", str(config.deblur_strength))
    table.add_row("White balance", on_off(config.enable_white_balance))
    table.add_row("CLAHE", on_off(config.enable_clahe))
    table.add_row("Local contrast", on_off(config.enable_local_contrast))
    table.add_row("Semantic masking", yes_no(config.run_semantic_masking))
    if config.run_semantic_masking:
        table.add_row("  Backend", config.mask_backend)
        table.add_row("  Model", config.mask_model)
        table.add_row("  Classes", ", ".join(config.mask_classes))
        table.add_row("  Device", config.mask_device)
        table.add_row("  Image size", str(config.mask_image_size))
        table.add_row("  Confidence", str(config.mask_confidence))
        table.add_row("  Strict masking", on_off(config.strict_static_masking))
    table.add_row("Angle coverage", f"{on_off(config.validate_angle_coverage)} ({config.angle_bins} bins)")
    table.add_row("Output range", f"{config.output_min_images} → {max_images}")
    table.add_row("Run COLMAP", yes_no(config.run_sfm))
    if config.run_sfm:
        table.add_row("  Device", config.colmap_device)
        table.add_row("  Matcher", config.matcher)
        table.add_row("  Seq overlap", str(config.sequential_overlap))
        table.add_row("  Loop detect", on_off(config.loop_detection))
        table.add_row("  Vocab tree", config.vocab_tree_path or "[dim]not set[/dim]")
        table.add_row("  Parallel", on_off(config.colmap_parallel))
        table.add_row("  Threads", threads)
        table.add_row("  GPU", yes_no(config.use_gpu))
    table.add_row("Quality gate overlap", str(config.quality_gate_min_overlap))
    table.add_row("Quality gate fail", on_off(config.quality_gate_fail))

    console.print(table)


def run_dry_run(args: argparse.Namespace) -> None:
    """Show resolved config and estimate frame count without running the pipeline."""

    import cv2 as _cv2

    config = build_config_from_args(args)
    show_effective_settings(args)

    console.print()
    console.print(Panel.fit("[bold]Dry Run — Frame Estimate[/bold]", border_style="yellow"))
    try:
        cap = _cv2.VideoCapture(config.video_path)
        if cap.isOpened():
            native_fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_s = total_frames / native_fps if native_fps > 0 else 0
            stride = max(1, int(round(native_fps / config.extraction_fps)))
            estimated_samples = max(0, total_frames // stride)
            cap.release()

            t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            t.add_column("", style="dim")
            t.add_column("")
            t.add_row("Native FPS", f"{native_fps:.2f}")
            t.add_row("Total frames", str(total_frames))
            t.add_row("Duration", f"{duration_s:.1f}s  ({duration_s/60:.1f} min)")
            t.add_row("Extraction stride", str(stride))
            t.add_row("Estimated samples", str(estimated_samples))
            t.add_row("Target output range", f"{config.output_min_images} → {config.output_max_images or '∞'}")
            console.print(t)
            print_info("No files were written. Pass [bold]--no-resume[/bold] to force a fresh run when you're ready.")
        else:
            print_warning(f"Could not open video for estimation: {config.video_path}")
    except Exception as exc:
        print_warning(f"Frame estimation failed: {exc}")


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
    parser.add_argument(
        "videos",
        nargs="+",
        metavar="video",
        help="One or more input video files (or a single directory with --dir).",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_DEFAULTS.keys()),
        default="balanced",
        help=(
            "Ready-to-use setup. Choices: "
            + ", ".join(sorted(PRESET_DEFAULTS.keys()))
            + ". 'laptop-fast' is the easiest starting point."
        ),
    )

    io_group = parser.add_argument_group("Input and output")
    io_group.add_argument(
        "--output",
        default="processed_dataset",
        help="Output folder. For batch runs each video gets its own subfolder inside this directory.",
    )
    io_group.add_argument(
        "--dir",
        default="",
        metavar="DIRECTORY",
        help="Process all video files inside this directory (overrides positional video arguments).",
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
    io_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved config and estimated frame count without running the pipeline.",
    )
    io_group.add_argument(
        "--export-format",
        choices=["colmap", "nerfstudio", "instant-ngp", "all"],
        default="colmap",
        help="Export format after COLMAP: colmap (default), nerfstudio, instant-ngp, or all.",
    )
    io_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and force a complete fresh run.",
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
            "  run            run the full pipeline\n"
            "  setup          create a saved pipeline step by step\n"
            "  edit-profile   edit a saved pipeline profile\n"
            "  delete-profile delete a saved pipeline profile\n"
            "  profiles       list saved local pipeline profiles\n"
            "  settings       list every available setting with defaults\n"
            "  presets        show the built-in presets\n"
            "  doctor         check the local environment\n"
            "  shell          open the interactive shell"
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

    edit_profile_parser = subparsers.add_parser("edit-profile", help="Edit a saved pipeline profile in the wizard.")
    edit_profile_parser.add_argument(
        "profile_name",
        nargs="?",
        default=None,
        help="Name of the profile to edit. Omit to choose interactively.",
    )

    delete_profile_parser = subparsers.add_parser("delete-profile", help="Delete a saved pipeline profile.")
    delete_profile_parser.add_argument(
        "profile_name",
        nargs="?",
        default=None,
        help="Name of the profile to delete. Omit to choose interactively.",
    )

    subparsers.add_parser("profiles", help="List locally saved pipeline profiles.")
    subparsers.add_parser("settings", help="List every available run setting with defaults.")
    subparsers.add_parser("presets", help="Show the built-in presets.")
    subparsers.add_parser("doctor", help="Check if common dependencies look available.")

    inspect_parser = subparsers.add_parser("inspect", help="Analyse an existing processed dataset folder.")
    inspect_parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="processed_dataset",
        help="Path to the processed dataset folder (default: processed_dataset).",
    )
    return parser


def normalize_argv(argv: list[str] | None = None) -> list[str]:
    """Keep the old one-line usage working by inserting the run command."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return ["shell"]

    known_commands = {"run", "shell", "setup", "edit-profile", "delete-profile", "profiles", "settings", "presets", "doctor", "inspect", "-h", "--help"}
    if args[0] not in known_commands:
        return ["run", *args]
    return args


def show_shell_help() -> None:
    """Print the short command list used inside the interactive shell."""

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    table.add_row("run <video> [options]", "run the full pipeline")
    table.add_row("<video> [options]", "same as run, short form")
    table.add_row("setup", "build a pipeline profile step by step")
    table.add_row("edit-profile [name]", "edit a saved pipeline profile")
    table.add_row("delete-profile [name]", "delete a saved pipeline profile")
    table.add_row("profiles", "list saved local pipeline profiles")
    table.add_row("settings", "list every run setting and its default")
    table.add_row("presets", "show built-in presets")
    table.add_row("doctor", "check the environment")
    table.add_row("inspect <dir>", "analyse an existing processed dataset")
    table.add_row("help", "show this shell help")
    table.add_row("exit", "leave the shell")
    console.print(Panel(table, title="[bold]Shell Commands[/bold]", border_style="cyan"))


def run_interactive_shell(execute_run_command: Callable[[argparse.Namespace], None]) -> int:
    """Keep the framework open so the user can run many commands in one session."""

    console.print()
    console.print(Panel.fit(
        "[bold green]Video to Dataset[/bold green]  [dim]interactive shell[/dim]\n"
        "[dim]Type [bold]help[/bold] to see commands · [bold]exit[/bold] to close[/dim]",
        border_style="green",
    ))

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

    table = Table(title="Built-in Presets", box=box.ROUNDED, show_lines=True)
    table.add_column("Preset", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Overrides", style="dim")

    preset_descriptions = {
        "balanced": "Safe default for most capture scenarios",
        "laptop-fast": "Lighter settings for quick runs on a laptop",
        "quality": "Higher-quality settings when detail matters most",
        "drone": "Optimised for aerial/drone footage with wide-angle motion",
        "indoor": "Tighter blur filter and CLAHE for controlled indoor scenes",
        "turntable": "Object-centric capture with tight overlap for turntable rigs",
    }

    for preset_name, preset_values in PRESET_DEFAULTS.items():
        overrides = ", ".join(f"{k}={v}" for k, v in preset_values.items()) if preset_values else "standard defaults"
        table.add_row(preset_name, preset_descriptions.get(preset_name, ""), overrides)

    console.print()
    console.print(table)


def run_inspect(dataset_dir: str) -> int:
    """Read preprocessing_report.json and display a rich summary."""

    import json as _json

    report_path = os.path.join(dataset_dir, "preprocessing_report.json")
    if not os.path.isfile(report_path):
        print_warning(f"No preprocessing_report.json found in: {dataset_dir}")
        print_info("Run the pipeline first or pass the correct dataset folder path.")
        return 1

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = _json.load(f)
    except (OSError, _json.JSONDecodeError) as exc:
        print_error(f"Could not read report: {exc}")
        return 1

    console.print()
    console.print(Panel.fit(
        f"[bold]Dataset Inspection[/bold]  [dim]{dataset_dir}[/dim]",
        border_style="cyan",
    ))

    # ── Overview ──────────────────────────────────────────────────────────────
    overview = Table(title="Overview", box=box.SIMPLE_HEAD, title_style="bold yellow", title_justify="left", padding=(0, 2))
    overview.add_column("Field", style="dim", no_wrap=True)
    overview.add_column("Value")
    overview.add_row("Timestamp", report.get("timestamp", "—"))
    overview.add_row("Input video", report.get("video_path", "—"))
    overview.add_row("Output root", report.get("output_root", "—"))
    overview.add_row("Output images", str(report.get("output_images", "—")))
    overview.add_row("Preset", report.get("parameters", {}).get("preset", "—"))
    console.print(overview)

    # ── Frame selection stats ─────────────────────────────────────────────────
    sel = report.get("selection_stats", {})
    if sel:
        sel_table = Table(title="Frame Selection", box=box.SIMPLE_HEAD, title_style="bold yellow", title_justify="left", padding=(0, 2))
        sel_table.add_column("Metric", style="dim", no_wrap=True)
        sel_table.add_column("Value")
        sel_table.add_row("Selected frames", str(sel.get("selected_frames", "—")))
        sel_table.add_row("Mean overlap", f"{sel.get('mean_overlap', 0):.3f}")
        sel_table.add_row("Min overlap", f"{sel.get('min_overlap_observed', 0):.3f}")
        sel_table.add_row("Max overlap", f"{sel.get('max_overlap_observed', 0):.3f}")
        sel_table.add_row("Overlap violations", str(sel.get("overlap_violations", "—")))
        console.print(sel_table)

    # ── Coverage ──────────────────────────────────────────────────────────────
    cov = report.get("selection_coverage", {})
    if cov:
        cov_table = Table(title="Angle Coverage", box=box.SIMPLE_HEAD, title_style="bold yellow", title_justify="left", padding=(0, 2))
        cov_table.add_column("Metric", style="dim", no_wrap=True)
        cov_table.add_column("Value")
        bins_total = cov.get("bins_total", 0)
        bins_after = cov.get("bins_covered_after", 0)
        coverage_pct = (bins_after / bins_total * 100) if bins_total else 0
        cov_table.add_row("Bins covered", f"{bins_after} / {bins_total}  ({coverage_pct:.0f}%)")
        cov_table.add_row("Added for coverage", str(cov.get("added_for_coverage", 0)))
        cov_table.add_row("Added for min range", str(cov.get("added_for_range", 0)))
        cov_table.add_row("Removed for max range", str(cov.get("removed_for_max_range", 0)))
        missing = cov.get("missing_bins_after", [])
        cov_table.add_row("Missing bins", str(missing) if missing else "[green]none[/green]")
        console.print(cov_table)

    # ── Quality gates ─────────────────────────────────────────────────────────
    qg = report.get("quality_gates", {})
    if qg.get("enabled"):
        qg_table = Table(title="Quality Gates", box=box.SIMPLE_HEAD, title_style="bold yellow", title_justify="left", padding=(0, 2))
        qg_table.add_column("Gate", style="dim", no_wrap=True)
        qg_table.add_column("Status")
        qg_table.add_column("Required", style="dim")
        qg_table.add_column("Observed", style="dim")
        for gate_name, gate in qg.get("checks", {}).items():
            if not gate.get("enabled"):
                continue
            passed = gate.get("passed", False)
            status_str = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            qg_table.add_row(
                gate_name,
                status_str,
                f"{gate.get('required', '—'):.3f}",
                f"{gate.get('observed', 0):.3f}",
            )
        console.print(qg_table)

    # ── Masking summary ───────────────────────────────────────────────────────
    masking = report.get("semantic_masking", {})
    if masking.get("summary", "").startswith("Semantic masking completed"):
        m_table = Table(title="Semantic Masking", box=box.SIMPLE_HEAD, title_style="bold yellow", title_justify="left", padding=(0, 2))
        m_table.add_column("Field", style="dim", no_wrap=True)
        m_table.add_column("Value")
        m_table.add_row("Backend", masking.get("backend", "—"))
        m_table.add_row("Model", masking.get("model", "—"))
        m_table.add_row("Device", masking.get("device", "—"))
        m_table.add_row("Masks written", str(masking.get("masks_written", "—")))
        m_table.add_row("Images with masks", str(masking.get("images_with_masked_objects", "—")))
        m_table.add_row("Total instances masked", str(masking.get("masked_instances", "—")))
        console.print(m_table)

    # ── COLMAP summary ────────────────────────────────────────────────────────
    colmap = report.get("colmap", {})
    metrics = colmap.get("metrics", {})
    if metrics:
        c_table = Table(title="COLMAP Reconstruction", box=box.SIMPLE_HEAD, title_style="bold yellow", title_justify="left", padding=(0, 2))
        c_table.add_column("Metric", style="dim", no_wrap=True)
        c_table.add_column("Value")
        c_table.add_row("Registered images", str(int(metrics.get("registered_images", 0))))
        c_table.add_row("3D points", str(int(metrics.get("points", 0))))
        c_table.add_row("Mean track length", f"{metrics.get('mean_track_length', 0):.2f}")
        c_table.add_row("Mean obs / image", f"{metrics.get('mean_observations_per_image', 0):.2f}")
        c_table.add_row("Matcher used", colmap.get("matcher_used", "—"))
        c_table.add_row("Device", colmap.get("device", "—"))
        console.print(c_table)

    print_info(f"Full report: {report_path}")
    return 0


def run_doctor() -> int:
    """Check a few common tools and packages."""

    def _check(label: str, ok: bool, ok_msg: str, fail_msg: str, version: str = "", required: bool = True) -> bool:
        status = "[green]OK[/green]" if ok else ("[red]MISSING[/red]" if required else "[yellow]OPTIONAL[/yellow]")
        table.add_row(label, status, version, ok_msg if ok else fail_msg)
        return ok

    table = Table(title="Environment Check", box=box.ROUNDED, show_lines=True)
    table.add_column("Component", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Version", style="dim")
    table.add_column("Notes")

    all_ok = True

    # Python
    import sys as _sys
    py_ver = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"
    _check("Python", True, "Environment active", "", py_ver)

    # OpenCV
    try:
        import cv2  # noqa: F401
        cv_ok = _check("OpenCV", True, "Installed", "", getattr(cv2, "__version__", ""))
    except Exception:
        cv_ok = _check("OpenCV", False, "", "pip install opencv-python")
    all_ok = all_ok and cv_ok

    # COLMAP — check presence and get version
    colmap_path = shutil.which("colmap")
    colmap_ok = bool(colmap_path)
    colmap_version = ""
    if colmap_ok:
        try:
            ver_result = subprocess.run(
                ["colmap", "--version"],
                check=False, capture_output=True, text=True,
            )
            raw = (ver_result.stdout or ver_result.stderr or "").strip()
            # COLMAP prints e.g. "COLMAP 3.9.1 (Commit ...)" — grab the number
            ver_match = re.search(r"\b(\d+\.\d+[\.\d]*)\b", raw)
            colmap_version = ver_match.group(1) if ver_match else raw[:20]
        except Exception:
            pass
    _check("COLMAP", colmap_ok, "Found in PATH", "Not found — install COLMAP and add to PATH", colmap_version, required=True)
    all_ok = all_ok and colmap_ok

    # Ultralytics (YOLO)
    try:
        import ultralytics  # noqa: F401
        _check("Ultralytics (YOLO)", True, "Installed", "", getattr(ultralytics, "__version__", ""), required=False)
    except Exception:
        _check("Ultralytics (YOLO)", False, "", "pip install ultralytics  (needed for YOLO masking)", required=False)

    # Torchvision (R-CNN)
    try:
        import torchvision  # noqa: F401
        _check("Torchvision (R-CNN)", True, "Installed", "", getattr(torchvision, "__version__", ""), required=False)
    except Exception:
        _check("Torchvision (R-CNN)", False, "", "pip install torchvision  (needed for R-CNN masking)", required=False)

    # CUDA
    cuda_ok = cuda_is_available()
    _check(
        "CUDA",
        cuda_ok,
        "NVIDIA runtime detected",
        "Not detected — CPU/MPS fallback will be used",
        required=False,
    )

    console.print()
    console.print(table)
    if all_ok:
        print_success("Core dependencies are ready.")
    else:
        print_warning("Some required dependencies are missing — see the table above.")

    return 0 if all_ok else 1


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply preset values only when the user did not override the default."""

    preset_values = PRESET_DEFAULTS.get(args.preset, {})
    if not preset_values:
        return args

    # Build the full defaults map from the run parser so every field is covered.
    parser = build_argument_parser()
    parser_defaults = get_run_parser_defaults(parser)

    for field_name, preset_value in preset_values.items():
        if not hasattr(args, field_name):
            continue
        if getattr(args, field_name) == parser_defaults.get(field_name):
            setattr(args, field_name, preset_value)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Keep argument checks in one place."""

    if args.command in {"shell", "setup", "edit-profile", "delete-profile", "profiles", "settings", "presets", "doctor", "inspect"}:
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

    # Resolve video list — either from --dir or positional args
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts", ".ts", ".flv"}
    if getattr(args, "dir", ""):
        dir_path = args.dir
        if not os.path.isdir(dir_path):
            parser.error(f"--dir path is not a directory: {dir_path}")
        args.resolved_videos = sorted(
            os.path.abspath(os.path.join(dir_path, f))
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
        )
        if not args.resolved_videos:
            parser.error(f"No video files found in directory: {dir_path}")
    else:
        missing = [v for v in args.videos if not os.path.isfile(v)]
        if missing:
            parser.error(f"Input video(s) not found: {', '.join(missing)}")
        args.resolved_videos = [os.path.abspath(v) for v in args.videos]
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


def build_config_from_args(args: argparse.Namespace, video_path: str | None = None) -> PipelineConfig:
    """Convert parsed arguments into the main config object.

    Pass *video_path* explicitly when processing a batch — otherwise the first
    resolved video (or the lone positional argument) is used.
    """

    resolved_videos = getattr(args, "resolved_videos", None)
    if video_path is None:
        if resolved_videos:
            video_path = resolved_videos[0]
        else:
            # Fallback for legacy callers that still use args.video
            video_path = getattr(args, "video", getattr(args, "videos", [""])[0])

    is_batch = resolved_videos is not None and len(resolved_videos) > 1
    base_output = args.output
    if not os.path.isabs(base_output):
        base_output = os.path.join(os.getcwd(), base_output)

    if is_batch:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_root = os.path.join(base_output, video_stem)
    else:
        output_root = base_output

    mask_classes = tuple(class_name.strip().lower() for class_name in args.mask_classes.split(",") if class_name.strip())
    if not mask_classes:
        mask_classes = DEFAULT_MASK_CLASSES

    resolved_colmap_device = resolve_colmap_device(args.colmap_device, cpu_only=args.cpu_only)
    colmap_parallel = resolve_colmap_parallel(args.colmap_parallel, resolved_colmap_device)
    resolved_mask_model = resolve_mask_model(args.mask_backend, args.mask_model)

    return PipelineConfig(
        video_path=os.path.abspath(video_path),
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
        export_format=getattr(args, "export_format", "colmap"),
        resume=not getattr(args, "no_resume", False),
        preset=args.preset,
    )


def handle_special_command(args: argparse.Namespace) -> int | None:
    """Run helper commands that do not start the pipeline."""

    if args.command == "setup":
        return run_setup_wizard()
    if args.command == "edit-profile":
        return run_edit_profile_wizard(getattr(args, "profile_name", None))
    if args.command == "delete-profile":
        return run_delete_profile(getattr(args, "profile_name", None))
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
    if args.command == "inspect":
        return run_inspect(getattr(args, "dataset_dir", "processed_dataset"))
    if args.command == "shell":
        return None
    if args.command == "run":
        if getattr(args, "show_settings", False):
            show_effective_settings(args)
            return 0
        if getattr(args, "dry_run", False):
            run_dry_run(args)
            return 0
        return None
    print_info("No command provided. Use --help to see available commands.")
    return 0
