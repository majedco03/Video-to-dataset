"""Semantic masking step."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from console import print_info, print_success, print_warning
from models import PipelineContext
from .base import PipelineStep


@dataclass(frozen=True)
class DetectedInstance:
    """One segmentation instance used while building the COLMAP ignore mask."""

    instance_idx: int
    class_id: int
    label: str
    confidence: float
    mask: np.ndarray
    bbox: Tuple[float, float, float, float]
    area_ratio: float
    center_distance: float


class SemanticMaskingStep(PipelineStep):
    """Build ignore masks so dynamic objects do not confuse reconstruction."""

    step_number = 4
    title = "Semantic masking"
    goal = "Detect likely moving objects and create masks so COLMAP can ignore them."

    @staticmethod
    def resolve_mask_device(requested_device: str) -> str:
        normalized = requested_device.strip().lower()
        if normalized == "cuda":
            normalized = "cuda:0"

        if normalized != "auto":
            try:
                import torch

                if normalized.startswith("cuda") and not torch.cuda.is_available():
                    return "cpu"
                if normalized == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    return "cpu"
            except Exception:
                if normalized != "cpu":
                    return "cpu"
            return normalized

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def build_mask_class_id_map(model_names: Dict[Any, Any], selected_classes: Tuple[str, ...]) -> Dict[int, str]:
        normalized_names = {
            int(class_id): str(class_name).strip().lower()
            for class_id, class_name in model_names.items()
        }
        selected_set = {class_name.strip().lower() for class_name in selected_classes}
        if "all" in selected_set or "*" in selected_set:
            return normalized_names
        return {
            class_id: class_name
            for class_id, class_name in normalized_names.items()
            if class_name in selected_set
        }

    @staticmethod
    def extract_instances(
        image_shape: Tuple[int, int],
        prediction: Any,
        selected_class_ids: Dict[int, str],
    ) -> List[DetectedInstance]:
        height, width = image_shape
        if prediction.boxes is None:
            return []

        class_ids = prediction.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = (
            prediction.boxes.conf.detach().cpu().numpy()
            if getattr(prediction.boxes, "conf", None) is not None
            else np.ones(len(class_ids), dtype=np.float32)
        )
        boxes = (
            prediction.boxes.xyxy.detach().cpu().numpy()
            if getattr(prediction.boxes, "xyxy", None) is not None
            else None
        )
        mask_data = (
            prediction.masks.data.detach().cpu().numpy()
            if prediction.masks is not None and getattr(prediction.masks, "data", None) is not None
            else None
        )

        image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        max_center_distance = float(np.linalg.norm(image_center)) + 1e-6
        instances: List[DetectedInstance] = []

        for instance_idx, class_id in enumerate(class_ids):
            if class_id not in selected_class_ids:
                continue

            if mask_data is not None:
                resized_mask = cv2.resize(mask_data[instance_idx], (width, height), interpolation=cv2.INTER_NEAREST)
                binary_mask = (resized_mask > 0.5).astype(np.uint8)
            else:
                binary_mask = np.zeros((height, width), dtype=np.uint8)

            if not np.any(binary_mask) and boxes is not None:
                x1, y1, x2, y2 = boxes[instance_idx]
                x1_i = max(0, min(width, int(np.floor(x1))))
                y1_i = max(0, min(height, int(np.floor(y1))))
                x2_i = max(0, min(width, int(np.ceil(x2))))
                y2_i = max(0, min(height, int(np.ceil(y2))))
                if x2_i > x1_i and y2_i > y1_i:
                    binary_mask[y1_i:y2_i, x1_i:x2_i] = 1

            if not np.any(binary_mask):
                continue

            if boxes is not None:
                x1, y1, x2, y2 = boxes[instance_idx]
            else:
                ys, xs = np.where(binary_mask > 0)
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())

            center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            center_distance = float(np.linalg.norm(center - image_center) / max_center_distance)
            area_ratio = float(np.count_nonzero(binary_mask) / max(height * width, 1))

            instances.append(
                DetectedInstance(
                    instance_idx=instance_idx,
                    class_id=int(class_id),
                    label=selected_class_ids[class_id],
                    confidence=float(confidences[instance_idx]),
                    mask=binary_mask,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    area_ratio=area_ratio,
                    center_distance=center_distance,
                )
            )

        return instances

    @staticmethod
    def compute_focus_score(instance: DetectedInstance) -> float:
        area_score = min(instance.area_ratio * 6.0, 1.0)
        center_score = max(0.0, 1.0 - instance.center_distance)
        return (0.45 * instance.confidence) + (0.35 * area_score) + (0.20 * center_score)

    @classmethod
    def choose_focus_class(cls, all_instances: List[List[DetectedInstance]]) -> Tuple[int | None, str | None]:
        class_scores: Dict[int, Dict[str, float | int | str]] = {}

        for instances in all_instances:
            if not instances:
                continue

            best_by_class: Dict[int, Tuple[DetectedInstance, float]] = {}
            for instance in instances:
                focus_score = cls.compute_focus_score(instance)
                current = best_by_class.get(instance.class_id)
                if current is None or focus_score > current[1]:
                    best_by_class[instance.class_id] = (instance, focus_score)

            for class_id, (instance, focus_score) in best_by_class.items():
                aggregate = class_scores.setdefault(
                    class_id,
                    {
                        "label": instance.label,
                        "frames": 0,
                        "score": 0.0,
                        "best": 0.0,
                    },
                )
                aggregate["frames"] = int(aggregate["frames"]) + 1
                aggregate["score"] = float(aggregate["score"]) + focus_score
                aggregate["best"] = max(float(aggregate["best"]), focus_score)

        if not class_scores:
            return None, None

        focus_class_id, focus_summary = max(
            class_scores.items(),
            key=lambda item: (
                int(item[1]["frames"]),
                float(item[1]["score"]),
                float(item[1]["best"]),
            ),
        )
        return focus_class_id, str(focus_summary["label"])

    @staticmethod
    def bbox_iou(first_box: Tuple[float, float, float, float], second_box: Tuple[float, float, float, float]) -> float:
        first_x1, first_y1, first_x2, first_y2 = first_box
        second_x1, second_y1, second_x2, second_y2 = second_box

        inter_x1 = max(first_x1, second_x1)
        inter_y1 = max(first_y1, second_y1)
        inter_x2 = min(first_x2, second_x2)
        inter_y2 = min(first_y2, second_y2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
        second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0
        return float(intersection / union)

    @classmethod
    def choose_focus_instance(
        cls,
        instances: List[DetectedInstance],
        focus_class_id: int | None,
        previous_focus_bbox: Tuple[float, float, float, float] | None,
    ) -> DetectedInstance | None:
        if focus_class_id is None:
            return None

        candidates = [instance for instance in instances if instance.class_id == focus_class_id]
        if not candidates:
            return None

        if previous_focus_bbox is None:
            return max(candidates, key=cls.compute_focus_score)

        def tracked_score(instance: DetectedInstance) -> float:
            temporal_score = cls.bbox_iou(previous_focus_bbox, instance.bbox)
            return cls.compute_focus_score(instance) + (0.35 * temporal_score)

        return max(candidates, key=tracked_score)

    @staticmethod
    def expand_instance_mask(mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        min_dim = max(1, min(image_shape))
        kernel_size = max(5, int(round(min_dim * 0.015)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    @classmethod
    def build_colmap_ignore_mask(
        cls,
        image_shape: Tuple[int, int],
        instances: List[DetectedInstance],
        focus_class_id: int | None,
        previous_focus_bbox: Tuple[float, float, float, float] | None,
    ) -> Tuple[np.ndarray, int, List[str], Tuple[float, float, float, float] | None, bool]:
        height, width = image_shape
        valid_mask = np.full((height, width), 255, dtype=np.uint8)
        masked_instances = 0
        used_labels: List[str] = []
        focus_instance = cls.choose_focus_instance(instances, focus_class_id, previous_focus_bbox)

        for instance in instances:
            if focus_instance is not None and instance.instance_idx == focus_instance.instance_idx:
                continue
            expanded_mask = cls.expand_instance_mask(instance.mask, image_shape)
            valid_mask[expanded_mask > 0] = 0
            masked_instances += 1
            used_labels.append(instance.label)

        next_focus_bbox = focus_instance.bbox if focus_instance is not None else previous_focus_bbox
        return valid_mask, masked_instances, used_labels, next_focus_bbox, focus_instance is not None

    def run(self, context: PipelineContext) -> PipelineContext:
        if not self.config.run_semantic_masking:
            return context
        if context.paths is None:
            raise RuntimeError("Output folders were not prepared before semantic masking.")

        self.announce()
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Semantic masking needs the ultralytics package.") from error

        os.makedirs(context.paths.masks, exist_ok=True)
        device = self.resolve_mask_device(self.config.mask_device)
        model = YOLO(self.config.mask_model)
        class_id_map = self.build_mask_class_id_map(model.names, self.config.mask_classes)

        if not class_id_map:
            raise RuntimeError(
                "None of the requested mask classes were found in the segmentation model. Please check --mask-classes."
            )

        print_info(
            f"Mask model: {self.config.mask_model} | device: {device} | classes: {', '.join(class_id_map.values())}"
        )

        predictions_by_image: List[Tuple[str, Tuple[int, int], List[DetectedInstance]]] = []

        for image_path in context.saved_images:
            image = cv2.imread(image_path)
            if image is None:
                print_warning(f"Could not read image for masking: {image_path}")
                continue

            prediction = model.predict(
                source=image,
                imgsz=self.config.mask_image_size,
                device=device,
                verbose=False,
                retina_masks=True,
            )[0]
            instances = self.extract_instances(
                image_shape=image.shape[:2],
                prediction=prediction,
                selected_class_ids=class_id_map,
            )
            predictions_by_image.append((image_path, image.shape[:2], instances))

        focus_class_id, focus_label = self.choose_focus_class(
            [instances for _, _, instances in predictions_by_image]
        )
        if focus_label is not None:
            print_info(f"Focused subject class kept unmasked: {focus_label}")
        else:
            print_warning("No focused subject class was detected. All detected objects in the selected classes will be masked.")

        masks_written = 0
        images_with_masked_objects = 0
        masked_instances_total = 0
        images_with_focus_subject = 0
        used_labels_overall: Dict[str, int] = {}
        previous_focus_bbox: Tuple[float, float, float, float] | None = None

        for image_path, image_shape, instances in predictions_by_image:
            valid_mask, masked_instances, used_labels, previous_focus_bbox, focus_found = self.build_colmap_ignore_mask(
                image_shape=image_shape,
                instances=instances,
                focus_class_id=focus_class_id,
                previous_focus_bbox=previous_focus_bbox,
            )

            mask_path = os.path.join(context.paths.masks, os.path.basename(image_path))
            cv2.imwrite(mask_path, valid_mask)
            masks_written += 1

            if focus_found:
                images_with_focus_subject += 1

            if masked_instances > 0:
                images_with_masked_objects += 1
                masked_instances_total += masked_instances
                for label in used_labels:
                    used_labels_overall[label] = used_labels_overall.get(label, 0) + 1

        print_success(f"Saved {masks_written} semantic mask images to: {context.paths.masks}")
        print_info(
            f"Images with masked objects: {images_with_masked_objects} | masked instances: {masked_instances_total}"
        )

        context.semantic_masking_result = {
            "summary": "Semantic masking completed.",
            "model": self.config.mask_model,
            "device": device,
            "mask_classes": list(class_id_map.values()),
            "focus_class": focus_label,
            "masks_written": masks_written,
            "images_with_masked_objects": images_with_masked_objects,
            "images_with_focus_subject": images_with_focus_subject,
            "masked_instances": masked_instances_total,
            "class_usage": used_labels_overall,
            "mask_dir": context.paths.masks,
        }
        return context
