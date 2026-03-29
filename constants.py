"""Shared constants used across the preprocessing framework."""

EXPOSURE_SCORE_FLOOR = 0.15
JPEG_QUALITY = 95
OVERLAP_PREVIEW_MAX_DIM = 640
COLMAP_MAX_IMAGE_SIZE = 4096
DEFAULT_MASK_MODEL = "yolov8n-seg.pt"
DEFAULT_RCNN_MODEL = "maskrcnn_resnet50_fpn_v2"
DEFAULT_MASK_BACKEND = "yolo"
DEFAULT_MASK_CLASSES = ("all",)
DEFAULT_STRICT_STATIC_MASKING = False
DEFAULT_MASK_IMAGE_SIZE = 640
DEFAULT_MASK_CONFIDENCE = 0.35
DEFAULT_COLOR_MODE = "color"
DEFAULT_COLMAP_DEVICE = "auto"
DEFAULT_ANGLE_BINS = 12
DEFAULT_VALIDATE_ANGLE_COVERAGE = True
DEFAULT_OUTPUT_MIN_IMAGES = 0
DEFAULT_OUTPUT_MAX_IMAGES = 0
DEFAULT_QUALITY_GATE_MIN_OVERLAP = 0.0
DEFAULT_QUALITY_GATE_FAIL = False
DEFAULT_MASK_DYNAMIC_ONLY = False
MASK_DYNAMIC_GRID_SIZE = 20
MASK_DYNAMIC_MIN_PRESENCE = 0.4

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
    # Aerial/drone footage: wider overlap window, more FPS for fast motion.
    "drone": {
        "fps": 5.0,
        "min_overlap": 0.70,
        "max_overlap": 0.95,
        "target_overlap": 0.80,
        "max_image_dim": 2048,
        "blur_threshold": 60.0,
        "mask_image_size": 640,
        "matcher": "sequential",
        "sequential_overlap": 16,
    },
    # Controlled indoor scenes: tighter blur filter, CLAHE on.
    "indoor": {
        "fps": 3.0,
        "min_overlap": 0.82,
        "max_overlap": 0.97,
        "target_overlap": 0.90,
        "max_image_dim": 1920,
        "blur_threshold": 110.0,
        "mask_image_size": 640,
        "matcher": "sequential",
    },
    # COLMAP → 3DGS reconstruction: strict blur, Mask R-CNN, focus-subject kept.
    "colmap-3dgs": {
        "fps": 5.0,
        "blur_threshold": 70.0,
        "min_overlap": 0.65,
        "max_overlap": 0.95,
        "target_overlap": 0.78,
        "max_image_dim": 2048,
        "mask_backend": "rcnn",
        "mask_image_size": 768,
        "mask_confidence": 0.40,
        "matcher": "sequential",
        "sequential_overlap": 16,
        "quality_gate_min_overlap": 0.50,
    },
    # Object-centric turntable rig: tight overlap, more angle bins.
    "turntable": {
        "fps": 4.0,
        "min_overlap": 0.88,
        "max_overlap": 0.98,
        "target_overlap": 0.93,
        "max_image_dim": 2048,
        "blur_threshold": 100.0,
        "angle_bins": 24,
        "matcher": "exhaustive",
        "mask_image_size": 768,
    },
}
