"""Image restoration and final export step."""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

from console import create_progress, print_info, print_success, print_warning
from models import CandidateFrame, PipelineContext
from .base import PipelineStep


class RadiometricNormalizationStep(PipelineStep):
    """Polish the selected frames and write the final dataset."""

    step_number = 3
    title = "Image clean-up and export"
    goal = "Improve sharpness, color balance, and brightness consistency in the selected frames."

    @staticmethod
    def resize_image_if_needed(image: np.ndarray, max_dim: int) -> np.ndarray:
        if max_dim <= 0:
            return image
        height, width = image.shape[:2]
        if max(height, width) <= max_dim:
            return image
        scale = max_dim / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def deblur_unsharp(image: np.ndarray, sigma: float, amount: float) -> np.ndarray:
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
        image_float = image.astype(np.float32)
        blue, green, red = cv2.split(image_float)
        mean_b, mean_g, mean_r = np.mean(blue), np.mean(green), np.mean(red)
        mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
        blue *= mean_gray / (mean_b + 1e-6)
        green *= mean_gray / (mean_g + 1e-6)
        red *= mean_gray / (mean_r + 1e-6)
        balanced = cv2.merge([blue, green, red])
        return np.clip(balanced, 0, 255).astype(np.uint8)

    @staticmethod
    def match_luminance_contrast(image: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness_float = lightness.astype(np.float32)
        current_mean = float(np.mean(lightness_float))
        current_std = float(np.std(lightness_float) + 1e-6)
        normalized = (lightness_float - current_mean) * (ref_std / current_std) + ref_mean
        new_lightness = np.clip(normalized, 0, 255).astype(np.uint8)
        merged = cv2.merge([new_lightness, channel_a, channel_b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def match_grayscale_contrast(image: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
        image_float = image.astype(np.float32)
        current_mean = float(np.mean(image_float))
        current_std = float(np.std(image_float) + 1e-6)
        normalized = (image_float - current_mean) * (ref_std / current_std) + ref_mean
        return np.clip(normalized, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_clahe_luminance(image: np.ndarray, clip_limit: float = 1.6, tile_size: int = 8) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced_lightness = clahe.apply(lightness)
        return cv2.cvtColor(cv2.merge([enhanced_lightness, channel_a, channel_b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_clahe_grayscale(image: np.ndarray, clip_limit: float = 1.6, tile_size: int = 8) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)

    @staticmethod
    def apply_local_contrast_mask(image: np.ndarray, strength: float = 0.18) -> np.ndarray:
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=7.0, sigmaY=7.0)
        enhanced = cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def restore_color_frame(self, frame: np.ndarray, sharpness: float, median_sharpness: float) -> np.ndarray:
        relative_blur = max(0.0, min(1.0, (median_sharpness - sharpness) / (median_sharpness + 1e-6)))
        strength = max(self.config.deblur_strength, 0.0)
        sigma = max(0.3, (0.8 + 0.9 * relative_blur) * strength)
        amount = (0.45 + 0.65 * relative_blur) * strength

        restored = self.deblur_unsharp(frame, sigma=sigma, amount=amount) if amount > 0 else frame.copy()

        if self.config.enable_white_balance:
            restored = self.gray_world_white_balance(restored)
        if self.config.enable_clahe:
            restored = self.apply_clahe_luminance(restored)
        if self.config.enable_local_contrast:
            restored = self.apply_local_contrast_mask(restored)
        return restored

    def restore_grayscale_frame(self, frame: np.ndarray, sharpness: float, median_sharpness: float) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        relative_blur = max(0.0, min(1.0, (median_sharpness - sharpness) / (median_sharpness + 1e-6)))
        strength = max(self.config.deblur_strength, 0.0)
        sigma = max(0.3, (0.7 + 0.8 * relative_blur) * strength)
        amount = (0.40 + 0.55 * relative_blur) * strength

        restored = self.deblur_unsharp(gray, sigma=sigma, amount=amount) if amount > 0 else gray.copy()

        if self.config.enable_clahe:
            restored = self.apply_clahe_grayscale(restored)
        if self.config.enable_local_contrast:
            restored = self.apply_local_contrast_mask(restored)
        return restored

    def restore_frame(self, frame: np.ndarray, sharpness: float, median_sharpness: float) -> np.ndarray:
        if self.config.color_mode == "grayscale":
            return self.restore_grayscale_frame(frame, sharpness, median_sharpness)
        return self.restore_color_frame(frame, sharpness, median_sharpness)

    def normalize_export_image(self, image: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
        if image.ndim == 2:
            corrected = self.match_grayscale_contrast(image, ref_mean, ref_std)
        else:
            corrected = self.match_luminance_contrast(image, ref_mean, ref_std)
        return cv2.addWeighted(image, 0.35, corrected, 0.65, 0)

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.paths is None:
            raise RuntimeError("Output folders were not prepared before image clean-up.")

        self.announce()
        sharpness_values = np.array([item.sharpness for item in context.selected_frames], dtype=np.float32)
        median_sharpness = float(np.median(sharpness_values)) if len(sharpness_values) else 0.0

        processed_images: List[np.ndarray] = []
        kept_frames: List[CandidateFrame] = []
        total = len(context.selected_frames)
        with create_progress() as progress:
            task = progress.add_task("[cyan]Restoring frames", total=total)
            for item in context.selected_frames:
                frame = cv2.imread(item.frame_path)
                if frame is None:
                    print_warning(f"Could not read temporary frame: {item.frame_path}")
                    progress.advance(task)
                    continue
                frame = self.resize_image_if_needed(frame, max_dim=self.config.max_image_dim)
                processed_images.append(self.restore_frame(frame, item.sharpness, median_sharpness))
                kept_frames.append(item)
                progress.advance(task)

        if not processed_images:
            raise RuntimeError("No processed images were written.")

        if processed_images[0].ndim == 2:
            reference_lightness = processed_images[0].astype(np.float32)
        else:
            reference_lab = cv2.cvtColor(processed_images[0], cv2.COLOR_BGR2LAB)
            reference_lightness = reference_lab[:, :, 0].astype(np.float32)
        reference_mean = float(np.mean(reference_lightness))
        reference_std = float(np.std(reference_lightness) + 1e-6)

        output_paths: List[str] = []
        with create_progress() as progress:
            task = progress.add_task("[cyan]Exporting images", total=len(processed_images))
            for output_index, image in enumerate(processed_images):
                corrected = self.normalize_export_image(image, reference_mean, reference_std)
                out_path = os.path.join(context.paths.images, f"frame_{output_index:05d}.png")
                cv2.imwrite(out_path, corrected)
                output_paths.append(out_path)
                progress.advance(task)

        print_success(f"Saved {len(output_paths)} processed images to: {context.paths.images}")
        print_info(f"Frames skipped during save: {len(context.selected_frames) - len(kept_frames)}")
        print_info(f"Export color mode: {self.config.color_mode}")
        context.saved_images = output_paths
        return context
