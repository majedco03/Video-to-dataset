# Video to Dataset

Turn raw video into a clean, reconstruction-ready image dataset with optional semantic masking and COLMAP SfM.

## Why this project

3D pipelines fail when frames are blurry, redundant, poorly exposed, or full of moving objects. This tool automates the preprocessing stage so your COLMAP or NeRF workflow starts from higher-quality data.

## Key features

- Video frame sampling with quality filtering (sharpness, texture, exposure)
- Overlap-based frame selection for cleaner view spacing
- Image restoration pipeline (resize, deblur, white balance, CLAHE, local contrast)
- Optional semantic masking to ignore dynamic objects
  - YOLO backend (Ultralytics)
  - Mask R-CNN backend (Torchvision)
- Strict static-scene masking mode (mask all selected dynamic classes)
- Angle-coverage validation with automatic backfill from remaining candidates
- Output image range control with adaptive refill from least-covered angles
- Optional COLMAP sparse reconstruction + undistorted export
- CUDA-aware defaults for NVIDIA users
- Interactive setup wizard to create reusable local pipeline profiles

## Quick start

```bash
python preprocess.py --help
```

Run a first pipeline:

```bash
python preprocess.py run input.mp4 --preset laptop-fast
```

Use guided setup and save a reusable local profile:

```bash
python preprocess.py setup
python preprocess.py run input.mp4 --profile my_pipeline
```

## Installation

### Prerequisites

- Python 3.10+
- `pip`
- COLMAP in your `PATH` (only required if you run SfM stage)
- input video (e.g. `input.mp4`)

### 1) Clone

```bash
git clone https://github.com/majedco03/Video-to-dataset.git
cd Video-to-dataset
```

### 2) Virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Python packages

Install from requirements file:

```bash
pip install -r requirements.txt
```

If you prefer manual install, CPU setup:

```bash
pip install numpy opencv-python ultralytics torch torchvision
```

NVIDIA CUDA example (adjust to your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python ultralytics
```

### 4) COLMAP

macOS (Homebrew):

```bash
brew install colmap
```

Verify:

```bash
colmap -h
```

### 5) Environment check

```bash
python preprocess.py doctor
```

## CLI overview

### Main commands

- `run` — run the full pipeline
- `setup` — step-by-step interactive builder (saves a named local profile)
- `profiles` — list saved local profiles
- `settings` — print all run options and defaults
- `presets` — show built-in presets
- `doctor` — check dependency availability
- `shell` — interactive command shell

## Tutorial: end-to-end usage

### 1) Validate your environment

```bash
python preprocess.py doctor
```

### 2) Start with a safe baseline run

```bash
python preprocess.py run input.mp4 --preset laptop-fast --show-settings
python preprocess.py run input.mp4 --preset laptop-fast
```

### 3) Run a reconstruction-focused static-scene pipeline

```bash
python preprocess.py run input.mp4 \
  --preset quality \
  --strict-static-masking \
  --validate-angle-coverage \
  --angle-bins 16 \
  --output-image-range 120:220 \
  --quality-gate-min-overlap 0.80 \
  --quality-gate-fail
```

This configuration does the following:

- masks selected dynamic classes aggressively
- checks view-angle coverage and backfills missing bins from remaining candidates
- enforces a target output image count range
- fails early if overlap quality is below the threshold

### 4) Generate COLMAP outputs for NeRFStudio/3DGS preparation

```bash
python preprocess.py run input.mp4 \
  --preset quality \
  --strict-static-masking \
  --validate-angle-coverage \
  --angle-bins 16 \
  --output-image-range 120:220
```

By default COLMAP runs unless you use `--no-colmap`.

### 5) Verify the run report

Check these fields in `processed_dataset/preprocessing_report.json`:

- `output_images`
- `selection_stats.mean_overlap`
- `selection_coverage`
- `quality_gates`
- `colmap.summary`

### 6) Save and reuse your preferred setup

```bash
python preprocess.py setup
python preprocess.py run input.mp4 --profile my_pipeline
```

### Helpful examples

Basic run:

```bash
python preprocess.py run input.mp4 --preset laptop-fast
```

Use grayscale output:

```bash
python preprocess.py run input.mp4 --color-mode grayscale
```

Use Mask R-CNN masking:

```bash
python preprocess.py run input.mp4 --mask-backend rcnn --mask-model maskrcnn_resnet50_fpn_v2
```

Enable strict static-scene masking (mask all detected selected classes):

```bash
python preprocess.py run input.mp4 --strict-static-masking
```

Enforce angle coverage and a target output range (min:max):

```bash
python preprocess.py run input.mp4 --validate-angle-coverage --angle-bins 16 --output-image-range 120:180
```

Require reconstruction-readiness overlap quality gate:

```bash
python preprocess.py run input.mp4 --quality-gate-min-overlap 0.80 --quality-gate-fail
```

Skip masking and COLMAP:

```bash
python preprocess.py run input.mp4 --no-semantic-mask --no-colmap
```

Preview resolved config before running:

```bash
python preprocess.py run input.mp4 --show-settings
```

Run tests:

```bash
python -m pytest -q
```

## Interactive setup wizard and profiles

Create a profile:

```bash
python preprocess.py setup
```

The wizard asks each feature one by one, shows available choices, and saves the profile with your chosen name.

Use saved profile:

```bash
python preprocess.py run input.mp4 --profile my_pipeline
```

List saved profiles:

```bash
python preprocess.py profiles
```

Profile storage location (local device only):

- `~/.video_to_dataset/pipeline_profiles.json`

These profiles are not pushed to GitHub unless you manually copy them into the repo.

## Output structure

After a successful run (default output folder: `processed_dataset/`):

- `images/` — final processed images
- `masks/` — semantic masks for COLMAP (when enabled)
- `colmap/` — SfM outputs (when COLMAP is enabled)
- `preprocessing_report.json` — full run summary and resolved parameters

## Pipeline stages

1. Directory setup
2. Blur detection + frame filtering
3. Overlap-based frame selection
4. Radiometric cleanup/export
5. Semantic masking
6. COLMAP SfM
7. Final report + cleanup

## Troubleshooting

### COLMAP not found

Install COLMAP and ensure `colmap` works in the same terminal session.

### CUDA expected but CPU used

- Check NVIDIA driver installation
- Confirm CUDA-enabled PyTorch build
- Verify `nvidia-smi` works
- Ensure you selected CUDA options (`--mask-device cuda`, `--colmap-device cuda`)

### Missing YOLO model

Provide a model path with `--mask-model`, keep `yolov8n-seg.pt` locally, or disable masking.

### I only need clean images

```bash
python preprocess.py run input.mp4 --no-colmap
```

## License

See [LICENSE](LICENSE).
