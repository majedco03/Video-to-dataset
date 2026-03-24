from __future__ import annotations

import pytest

from cli import parse_args, parse_output_image_range


def test_parse_output_image_range_variants():
    assert parse_output_image_range("") == (0, 0)
    assert parse_output_image_range("30") == (30, 30)
    assert parse_output_image_range("40:120") == (40, 120)
    assert parse_output_image_range("60:") == (60, 0)
    assert parse_output_image_range(":90") == (0, 90)


def test_parse_args_accepts_output_image_range(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    args = parse_args(
        [
            "run",
            str(video),
            "--output-image-range",
            "80:160",
            "--angle-bins",
            "16",
            "--strict-static-masking",
            "--quality-gate-min-overlap",
            "0.75",
        ]
    )

    assert args.output_min_images == 80
    assert args.output_max_images == 160
    assert args.angle_bins == 16
    assert args.strict_static_masking is True
    assert args.quality_gate_min_overlap == 0.75


def test_parse_args_rejects_invalid_output_image_range(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    with pytest.raises(SystemExit):
        parse_args(["run", str(video), "--output-image-range", "100:20"])
