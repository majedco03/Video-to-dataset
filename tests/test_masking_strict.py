from __future__ import annotations

import numpy as np

from steps.masking import DetectedInstance, SemanticMaskingStep


def _instance(instance_idx: int, class_id: int, label: str, mask: np.ndarray, bbox: tuple[float, float, float, float]):
    return DetectedInstance(
        instance_idx=instance_idx,
        class_id=class_id,
        label=label,
        confidence=0.9,
        mask=mask,
        bbox=bbox,
        area_ratio=float(np.count_nonzero(mask) / mask.size),
        center_distance=0.2,
    )


def test_strict_masking_masks_all_instances():
    image_shape = (20, 20)
    mask_a = np.zeros(image_shape, dtype=np.uint8)
    mask_b = np.zeros(image_shape, dtype=np.uint8)
    mask_a[2:8, 2:8] = 1
    mask_b[10:16, 10:16] = 1

    instances = [
        _instance(0, 1, "person", mask_a, (2, 2, 8, 8)),
        _instance(1, 2, "car", mask_b, (10, 10, 16, 16)),
    ]

    valid_mask, masked_instances, _, _, focus_found = SemanticMaskingStep.build_colmap_ignore_mask(
        image_shape=image_shape,
        instances=instances,
        focus_class_id=1,
        previous_focus_bbox=None,
        keep_focus_subject=False,
    )

    assert masked_instances == 2
    assert focus_found is False
    assert int(np.count_nonzero(valid_mask == 0)) > 0


def test_default_masking_keeps_focus_instance():
    image_shape = (20, 20)
    mask_a = np.zeros(image_shape, dtype=np.uint8)
    mask_b = np.zeros(image_shape, dtype=np.uint8)
    mask_a[2:8, 2:8] = 1
    mask_b[10:16, 10:16] = 1

    instances = [
        _instance(0, 1, "person", mask_a, (2, 2, 8, 8)),
        _instance(1, 2, "car", mask_b, (10, 10, 16, 16)),
    ]

    _, masked_instances, _, _, focus_found = SemanticMaskingStep.build_colmap_ignore_mask(
        image_shape=image_shape,
        instances=instances,
        focus_class_id=1,
        previous_focus_bbox=None,
        keep_focus_subject=True,
    )

    assert masked_instances == 1
    assert focus_found is True
