"""Shared constants used across the preprocessing framework."""

EXPOSURE_SCORE_FLOOR = 0.15
JPEG_QUALITY = 95
OVERLAP_PREVIEW_MAX_DIM = 640
COLMAP_MAX_IMAGE_SIZE = 4096
DEFAULT_MASK_MODEL = "yolov8n-seg.pt"
DEFAULT_RCNN_MODEL = "maskrcnn_resnet50_fpn_v2"
DEFAULT_MASK_BACKEND = "yolo"
DEFAULT_MASK_CLASSES = ("all",)
DEFAULT_MASK_IMAGE_SIZE = 640
DEFAULT_MASK_CONFIDENCE = 0.35
DEFAULT_COLOR_MODE = "color"
DEFAULT_COLMAP_DEVICE = "auto"

PRESET_DEFAULTS = {
    "balanced": {},
    "laptop-fast": {
        "fps": 2.0,
        "max_image_dim": 1600,
        "mask_image_size": 512,
        "matcher": "sequential",
    },
    "quality": {
        "fps": 4.0,
        "max_image_dim": 2560,
        "mask_image_size": 768,
        "matcher": "sequential",
    },
}
