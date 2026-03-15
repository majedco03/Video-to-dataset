# 3D Preprocessing Framework

This project turns a video into a cleaner image set for 3D reconstruction.

Recent updates add more direct control over the pipeline, including grayscale export, CUDA-aware defaults, optional parallel COLMAP execution, and support for both YOLO and Mask R-CNN based masking.

## What this project does

Starting from a video file, the pipeline:

1. samples frames from the video,
2. removes frames that are blurry or weak,
3. keeps a better-spaced sequence of views,
4. cleans and normalizes the exported images,
5. optionally masks dynamic objects with YOLO or Mask R-CNN segmentation,
6. optionally runs COLMAP for sparse reconstruction,
7. writes a final report with the run settings and results.

---

## Setup tutorial

This section is meant to be a complete first-time setup guide.

### 1. Prerequisites

Install these before running the pipeline:

- Python 3.10 or newer
- `pip`
- COLMAP available in your terminal `PATH`
- A video file to process, for example `input.mp4`

Optional but recommended:

- a Python virtual environment
- NVIDIA CUDA drivers if you want GPU acceleration on Windows or Linux
- a local YOLO checkpoint such as `yolov8n-seg.pt` if you want offline YOLO masking

### 2. Clone the repository

```bash
git clone https://github.com/majedco03/video_to_dataset.git
cd video_to_dataset
```

If you already have the project, just open the project folder and continue.

### 3. Create and activate a virtual environment

#### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

#### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4. Install Python dependencies

This project uses OpenCV, NumPy, Ultralytics, Torchvision, and PyTorch-backed inference for segmentation.

For a simple CPU setup:

```bash
pip install numpy opencv-python ultralytics torch torchvision
```

If you are using an NVIDIA GPU, install the PyTorch build that matches your CUDA version from the official PyTorch instructions, then install the remaining packages if needed.

Example flow:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python ultralytics
```

Notes:

- On Apple Silicon, PyTorch can use `mps` for the YOLO backend.
- The Mask R-CNN backend is most reliable on CPU or CUDA.
- If semantic masking is disabled, the pipeline can still run without Ultralytics and Torchvision.

### 5. Install COLMAP

COLMAP is required only if you want the reconstruction stage.

#### macOS with Homebrew

```bash
brew install colmap
```

#### Ubuntu or Debian

Install it from your package manager or build it from source if your repository version is outdated.

#### Windows

Install COLMAP from the official release or build instructions, then make sure `colmap` is available in your terminal `PATH`.

To verify the install:

```bash
colmap -h
```

### 6. Optional: model file setup

The default YOLO masking model is `yolov8n-seg.pt`.

The pipeline supports two masking backends:

- `yolo` for Ultralytics YOLO segmentation models
- `rcnn` for Torchvision Mask R-CNN models such as `maskrcnn_resnet50_fpn_v2`

You can use either approach:

- keep the model file locally in the project folder, or
- point the CLI to another checkpoint with `--mask-model`, or
- let Ultralytics resolve or download the checkpoint when supported by the runtime.

If you do not want segmentation at all, use `--no-semantic-mask`.

### 7. Verify the environment

Run:

```bash
python preprocess.py doctor
```

This checks whether Python, OpenCV, Ultralytics, Torchvision, COLMAP, and CUDA detection look correct.

### 8. Inspect available settings

Before your first run, view the available options:

```bash
python preprocess.py settings
```

To build a reusable pipeline step by step and save it locally on your device:

```bash
python preprocess.py setup
```

To list your saved local profiles later:

```bash
python preprocess.py profiles
```

To preview the exact resolved configuration for a specific command:

```bash
python preprocess.py run input.mp4 --show-settings
```

To run with a saved local profile:

```bash
python preprocess.py run input.mp4 --profile my_pipeline
```

### 9. Run the pipeline

#### Easiest first run

```bash
python preprocess.py run input.mp4 --preset laptop-fast
```

#### Short form

```bash
python preprocess.py input.mp4 --preset laptop-fast
```

#### Export grayscale images only

```bash
python preprocess.py run input.mp4 --color-mode grayscale
```

#### Skip semantic masking

```bash
python preprocess.py run input.mp4 --no-semantic-mask
```

#### Skip COLMAP and only export the cleaned dataset

```bash
python preprocess.py run input.mp4 --no-colmap
```

#### Use CUDA for masking and COLMAP when available

```bash
python preprocess.py run input.mp4 --mask-device cuda --colmap-device cuda
```

#### Use Mask R-CNN instead of YOLO

```bash
python preprocess.py run input.mp4 --mask-backend rcnn --mask-model maskrcnn_resnet50_fpn_v2
```

#### Force CPU only

```bash
python preprocess.py run input.mp4 --cpu-only --mask-device cpu
```

### 10. Check the output

After a successful run, the output folder usually contains:

- `images/` - final processed images
- `masks/` - semantic masks for COLMAP, when masking is enabled
- `colmap/` - reconstruction data, when COLMAP is enabled
- `preprocessing_report.json` - summary of the run and resolved parameters

---

## Main file to run

Use:

- `preprocess.py`

## Quick start

Show help:

- `python preprocess.py --help`

Show every available pipeline setting and its default:

- `python preprocess.py settings`

Create a saved pipeline step by step:

- `python preprocess.py setup`

List saved local profiles:

- `python preprocess.py profiles`

Open the interactive shell:

- `python preprocess.py`

Run with the laptop-friendly preset:

- `python preprocess.py run input.mp4 --preset laptop-fast`

The old short form also works:

- `python preprocess.py input.mp4 --preset laptop-fast`

Run with grayscale output and CPU-only COLMAP:

- `python preprocess.py run input.mp4 --color-mode grayscale --colmap-device cpu`

Run with CUDA-enabled masking and parallel COLMAP:

- `python preprocess.py run input.mp4 --mask-device cuda --colmap-device cuda --colmap-parallel`

Run with Mask R-CNN masking:

- `python preprocess.py run input.mp4 --mask-backend rcnn --mask-model maskrcnn_resnet50_fpn_v2`

Preview the resolved config without starting the pipeline:

- `python preprocess.py run input.mp4 --show-settings`

## Commands

### `run`
Runs the full pipeline.

### `shell`
Opens an interactive loop so you can run many commands without starting Python again.

### `setup`
Starts an interactive step-by-step builder. It asks about the pipeline features one by one, shows choices each time, and saves the final pipeline locally as a reusable profile.

### `profiles`
Lists the locally saved pipeline profiles available on the current device.

### `presets`
Shows the built-in presets.

### `settings`
Prints every run option, its default value, and the available choices in one list.

### `doctor`
Checks if common dependencies are available.

## Recommended first-run workflow

1. Activate your virtual environment.
2. Run `python preprocess.py doctor`.
3. Run `python preprocess.py setup` if you want a guided profile builder.
4. Run `python preprocess.py settings`.
5. Preview your exact command with `python preprocess.py run input.mp4 --show-settings`.
6. Start with `--preset laptop-fast` or with a saved profile.
7. Turn on stronger options only after the base run succeeds.

## Presets

- `balanced` - safe default
- `laptop-fast` - lighter settings for easier runs on a laptop
- `quality` - stronger settings when quality matters more than speed

## Useful advanced options

- `--color-mode {color,grayscale}` - choose the exported dataset style
- `--deblur-strength <float>` - tune the sharpening pass
- `--disable-white-balance` - keep original color balance
- `--disable-clahe` - skip local luminance equalization
- `--disable-local-contrast` - skip the final local contrast pass
- `--mask-backend yolo|rcnn` - choose the segmentation engine
- `--mask-device auto|cpu|mps|cuda[:index]` - pick the segmentation runtime device
- `--mask-confidence <float>` - ignore weak detections below the chosen confidence
- `--profile NAME` - load a saved local pipeline profile created by the setup wizard
- `--colmap-device auto|cpu|cuda[:index]` - choose how COLMAP runs
- `--colmap-parallel` / `--no-colmap-parallel` - control threaded COLMAP execution
- `--colmap-threads <int>` - cap COLMAP CPU thread usage

When CUDA is detected and COLMAP is left on `auto`, the pipeline now defaults to CUDA plus parallel execution.

## Project layout

- `preprocess.py` - main entry point
- `cli.py` - command-line interface
- `runner.py` - pipeline runner
- `models.py` - shared data classes
- `constants.py` - shared constants
- `console.py` - user-facing log output
- `filesystem.py` - output folder helpers
- `steps/` - one file per pipeline step

## Pipeline steps

1. Folder setup
2. Blur detection and frame filtering
3. Overlap-based frame selection
4. Image clean-up and radiometric normalization
5. Semantic masking
6. COLMAP structure-from-motion
7. Report writing and cleanup

## Troubleshooting

### `colmap` not found

Install COLMAP and make sure the `colmap` command works in the same terminal where you run the script.

### CUDA was expected but CPU is used

Check these points:

- NVIDIA drivers are installed correctly
- PyTorch was installed with a CUDA-enabled build
- `nvidia-smi` works in the terminal
- you are using `--mask-device cuda` or `--colmap-device cuda` when needed

### YOLO model file is missing

Either place `yolov8n-seg.pt` in the project folder, choose another checkpoint with `--mask-model`, or disable semantic masking.

### I want to use Mask R-CNN instead of YOLO

Use:

```bash
python preprocess.py run input.mp4 --mask-backend rcnn --mask-model maskrcnn_resnet50_fpn_v2
```

Make sure `torchvision` is installed.

### I only want the cleaned images

Use:

```bash
python preprocess.py run input.mp4 --no-colmap
```

### I want the simplest possible command

Use:

```bash
python preprocess.py run input.mp4 --preset laptop-fast
```

### I want to reuse the same pipeline later

Use the step-by-step wizard once:

```bash
python preprocess.py setup
```

Then run future videos with:

```bash
python preprocess.py run input.mp4 --profile my_pipeline
```
