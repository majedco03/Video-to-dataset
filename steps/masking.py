"""Semantic masking step."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from console import create_progress, print_info, print_success, print_warning, spinner
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


@dataclass(frozen=True)
class MaskingRuntime:
    """Loaded masking backend, resolved device, and label metadata."""

    backend: str
    device: str
    model: Any
    model_name: str
    class_names: Dict[int, str]


class SemanticMaskingStep(PipelineStep):
    """Build ignore masks so dynamic objects do not confuse reconstruction."""

    step_number = 4
    title = "Semantic masking"
    goal = (
        "Run instance segmentation to detect moving objects (people, vehicles, animals) "
        "and paint them out so COLMAP never matches features across them. "
        "Each mask is saved alongside its image and passed to COLMAP as an ignore region."
    )

    BACKGROUND_LABELS = {"__background__", "background", "n/a", "none"}

    @staticmethod
    def normalize_device_name(requested_device: str) -> str:
        normalized = requested_device.strip().lower()
        if normalized == "cuda":
            return "cuda:0"
        return normalized

    @classmethod
    def resolve_mask_device(cls, requested_device: str, backend: str) -> str:
        normalized = cls.normalize_device_name(requested_device)

        try:
            import torch
        except Exception:
            return "cpu"

        if normalized != "auto":
            if normalized.startswith("cuda"):
                return normalized if torch.cuda.is_available() else "cpu"
            if normalized == "mps":
                if backend == "rcnn":
                    return "cpu"
                has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                return "mps" if has_mps else "cpu"
            return normalized

        if torch.cuda.is_available():
            return "cuda:0"
        if backend != "rcnn" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def build_mask_class_id_map(cls, model_names: Dict[Any, Any], selected_classes: Tuple[str, ...]) -> Dict[int, str]:
        normalized_names = {
            int(class_id): str(class_name).strip().lower()
            for class_id, class_name in model_names.items()
            if str(class_name).strip().lower() not in cls.BACKGROUND_LABELS
        }
        selected_set = {class_name.strip().lower() for class_name in selected_classes}
        if "all" in selected_set or "*" in selected_set:
            return normalized_names
        return {
            class_id: class_name
            for class_id, class_name in normalized_names.items()
            if class_name in selected_set
        }

    @classmethod
    def extract_instances_from_arrays(
        cls,
        image_shape: Tuple[int, int],
        class_ids: np.ndarray,
        confidences: np.ndarray,
        boxes: np.ndarray | None,
        mask_data: np.ndarray | None,
        selected_class_ids: Dict[int, str],
        confidence_threshold: float,
    ) -> List[DetectedInstance]:
        height, width = image_shape
        image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        max_center_distance = float(np.linalg.norm(image_center)) + 1e-6
        instances: List[DetectedInstance] = []

        for instance_idx, class_id in enumerate(class_ids.astype(int)):
            if class_id not in selected_class_ids:
                continue

            confidence = float(confidences[instance_idx])
            if confidence < confidence_threshold:
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
                    confidence=confidence,
                    mask=binary_mask,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    area_ratio=area_ratio,
                    center_distance=center_distance,
                )
            )

        return instances

    @classmethod
    def extract_yolo_instances(
        cls,
        image_shape: Tuple[int, int],
        prediction: Any,
        selected_class_ids: Dict[int, str],
        confidence_threshold: float,
    ) -> List[DetectedInstance]:
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

        return cls.extract_instances_from_arrays(
            image_shape=image_shape,
            class_ids=class_ids,
            confidences=confidences,
            boxes=boxes,
            mask_data=mask_data,
            selected_class_ids=selected_class_ids,
            confidence_threshold=confidence_threshold,
        )

    @classmethod
    def extract_rcnn_instances(
        cls,
        image_shape: Tuple[int, int],
        output: Dict[str, Any],
        selected_class_ids: Dict[int, str],
        confidence_threshold: float,
    ) -> List[DetectedInstance]:
        labels = output.get("labels")
        if labels is None or len(labels) == 0:
            return []

        scores = output.get("scores")
        boxes = output.get("boxes")
        masks = output.get("masks")

        class_ids = labels.detach().cpu().numpy().astype(int)
        confidences = scores.detach().cpu().numpy() if scores is not None else np.ones(len(class_ids), dtype=np.float32)
        boxes_np = boxes.detach().cpu().numpy() if boxes is not None else None
        mask_data = masks.detach().cpu().numpy()[:, 0, :, :] if masks is not None else None

        return cls.extract_instances_from_arrays(
            image_shape=image_shape,
            class_ids=class_ids,
            confidences=confidences,
            boxes=boxes_np,
            mask_data=mask_data,
            selected_class_ids=selected_class_ids,
            confidence_threshold=confidence_threshold,
        )

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
        keep_focus_subject: bool,
    ) -> Tuple[np.ndarray, int, List[str], Tuple[float, float, float, float] | None, bool]:
        height, width = image_shape
        valid_mask = np.full((height, width), 255, dtype=np.uint8)
        masked_instances = 0
        used_labels: List[str] = []
        focus_instance = cls.choose_focus_instance(instances, focus_class_id, previous_focus_bbox) if keep_focus_subject else None

        for instance in instances:
            if focus_instance is not None and instance.instance_idx == focus_instance.instance_idx:
                continue
            expanded_mask = cls.expand_instance_mask(instance.mask, image_shape)
            valid_mask[expanded_mask > 0] = 0
            masked_instances += 1
            used_labels.append(instance.label)

        next_focus_bbox = focus_instance.bbox if focus_instance is not None else previous_focus_bbox
        return valid_mask, masked_instances, used_labels, next_focus_bbox, focus_instance is not None

    def load_yolo_runtime(self, device: str) -> MaskingRuntime:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("YOLO masking needs the ultralytics package.") from error

        model = YOLO(self.config.mask_model)
        class_names = {int(class_id): str(class_name) for class_id, class_name in model.names.items()}
        return MaskingRuntime(
            backend="yolo",
            device=device,
            model=model,
            model_name=self.config.mask_model,
            class_names=class_names,
        )

    def load_rcnn_runtime(self, device: str) -> MaskingRuntime:
        try:
            import torch
            from torchvision.models.detection import (
                MaskRCNN_ResNet50_FPN_V2_Weights,
                MaskRCNN_ResNet50_FPN_Weights,
                maskrcnn_resnet50_fpn,
                maskrcnn_resnet50_fpn_v2,
            )
        except ImportError as error:
            raise RuntimeError("R-CNN masking needs the torchvision package.") from error

        model_name = self.config.mask_model.strip().lower()
        model_registry = {
            "maskrcnn_resnet50_fpn": (maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights.DEFAULT),
            "maskrcnn_resnet50_fpn_v2": (maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
        }
        if model_name not in model_registry:
            supported = ", ".join(sorted(model_registry.keys()))
            raise RuntimeError(f"Unsupported R-CNN model '{self.config.mask_model}'. Supported values: {supported}")

        builder, weights = model_registry[model_name]
        model = builder(weights=weights)
        model.eval()

        runtime_device = device if device.startswith("cuda") else "cpu"
        model.to(torch.device(runtime_device))

        categories = weights.meta.get("categories", [])
        class_names = {int(index): str(label) for index, label in enumerate(categories)}
        return MaskingRuntime(
            backend="rcnn",
            device=runtime_device,
            model=model,
            model_name=model_name,
            class_names=class_names,
        )

    def load_masking_runtime(self) -> MaskingRuntime:
        device = self.resolve_mask_device(self.config.mask_device, self.config.mask_backend)
        if self.config.mask_backend == "rcnn":
            return self.load_rcnn_runtime(device)
        return self.load_yolo_runtime(device)

    def predict_instances(self, runtime: MaskingRuntime, image: np.ndarray, selected_class_ids: Dict[int, str]) -> List[DetectedInstance]:
        if runtime.backend == "yolo":
            prediction = runtime.model.predict(
                source=image,
                imgsz=self.config.mask_image_size,
                device=runtime.device,
                conf=self.config.mask_confidence,
                verbose=False,
                retina_masks=True,
            )[0]
            return self.extract_yolo_instances(
                image_shape=image.shape[:2],
                prediction=prediction,
                selected_class_ids=selected_class_ids,
                confidence_threshold=self.config.mask_confidence,
            )

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("R-CNN masking needs PyTorch.") from error

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(runtime.device)

        with torch.inference_mode():
            output = runtime.model([tensor])[0]

        return self.extract_rcnn_instances(
            image_shape=image.shape[:2],
            output=output,
            selected_class_ids=selected_class_ids,
            confidence_threshold=self.config.mask_confidence,
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        if not self.config.run_semantic_masking:
            return context
        if context.paths is None:
            raise RuntimeError("Output folders were not prepared before semantic masking.")

        self.announce()
        os.makedirs(context.paths.masks, exist_ok=True)

        with spinner(f"Loading {self.config.mask_backend.upper()} model..."):
            runtime = self.load_masking_runtime()
        requested_device = self.normalize_device_name(self.config.mask_device)
        if requested_device not in {"auto", runtime.device}:
            print_warning(
                f"Mask backend '{runtime.backend}' could not use '{self.config.mask_device}', so it will run on '{runtime.device}'."
            )

        class_id_map = self.build_mask_class_id_map(runtime.class_names, self.config.mask_classes)
        if not class_id_map:
            raise RuntimeError(
                "None of the requested mask classes were found in the selected segmentation model. Please check --mask-classes."
            )

        print_info(
            f"Mask backend: {runtime.backend} | model: {runtime.model_name} | device: {runtime.device} | "
            f"confidence: {self.config.mask_confidence:.2f} | classes: {', '.join(class_id_map.values())}"
        )

        predictions_by_image: List[Tuple[str, Tuple[int, int], List[DetectedInstance]]] = []
        with create_progress() as progress:
            task = progress.add_task("[cyan]Running inference", total=len(context.saved_images))
            for image_path in context.saved_images:
                image = cv2.imread(image_path)
                if image is None:
                    print_warning(f"Could not read image for masking: {image_path}")
                    progress.advance(task)
                    continue
                instances = self.predict_instances(runtime, image, class_id_map)
                predictions_by_image.append((image_path, image.shape[:2], instances))
                progress.advance(task)

        if not predictions_by_image:
            print_warning("No images were available for semantic masking.")
            context.semantic_masking_result = {
                "summary": "Semantic masking skipped because no readable images were found.",
                "backend": runtime.backend,
                "model": runtime.model_name,
                "device": runtime.device,
            }
            return context

        focus_class_id, focus_label = self.choose_focus_class([instances for _, _, instances in predictions_by_image])
        if focus_label is not None and not self.config.strict_static_masking:
            print_info(f"Focused subject class kept unmasked: {focus_label}")
        elif focus_label is not None:
            print_info("Strict static masking is enabled, so all selected dynamic classes will be masked.")
        else:
            print_warning("No focused subject class was detected. All detected objects in the selected classes will be masked.")

        masks_written = 0
        images_with_masked_objects = 0
        masked_instances_total = 0
        images_with_focus_subject = 0
        used_labels_overall: Dict[str, int] = {}
        previous_focus_bbox: Tuple[float, float, float, float] | None = None

        with create_progress() as progress:
            task = progress.add_task("[cyan]Writing masks", total=len(predictions_by_image))
            for image_path, image_shape, instances in predictions_by_image:
                valid_mask, masked_instances, used_labels, previous_focus_bbox, focus_found = self.build_colmap_ignore_mask(
                    image_shape=image_shape,
                    instances=instances,
                    focus_class_id=focus_class_id,
                    previous_focus_bbox=previous_focus_bbox,
                    keep_focus_subject=not self.config.strict_static_masking,
                )

                # COLMAP mask naming convention: <image_filename>.png (extension appended, not replaced).
                mask_filename = os.path.basename(image_path) + ".png"
                mask_path = os.path.join(context.paths.masks, mask_filename)
                cv2.imwrite(mask_path, valid_mask)
                masks_written += 1

                if focus_found:
                    images_with_focus_subject += 1

                if masked_instances > 0:
                    images_with_masked_objects += 1
                    masked_instances_total += masked_instances
                    for label in used_labels:
                        used_labels_overall[label] = used_labels_overall.get(label, 0) + 1

                progress.advance(task)

        print_success(f"Saved {masks_written} semantic mask images to: {context.paths.masks}")
        print_info(
            f"Images with masked objects: {images_with_masked_objects} | masked instances: {masked_instances_total}"
        )

        context.semantic_masking_result = {
            "summary": "Semantic masking completed.",
            "backend": runtime.backend,
            "model": runtime.model_name,
            "device": runtime.device,
            "mask_classes": list(class_id_map.values()),
            "mask_confidence": self.config.mask_confidence,
            "strict_static_masking": self.config.strict_static_masking,
            "focus_class": focus_label,
            "masks_written": masks_written,
            "images_with_masked_objects": images_with_masked_objects,
            "images_with_focus_subject": images_with_focus_subject,
            "masked_instances": masked_instances_total,
            "class_usage": used_labels_overall,
            "mask_dir": context.paths.masks,
        }
        return context
