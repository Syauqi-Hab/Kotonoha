"""Text Background Inpainter for Manga Pages.

Provides non-destructive background cleaning inside speech bubbles using
either LaMa ONNX (Large Mask Inpainting) or advanced OpenCV Telea / Navier-Stokes.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from src.core.models import SpeechBubble, BoundingBox
from src.core.config import config


class MangaInpainter:
    """Cleans Japanese text from manga panels and speech bubbles."""

    def __init__(self, model_path: Optional[Path | str] = None):
        self.model_path = Path(model_path or config.lama_model_path)
        self.ort_session = None
        self._init_lama_onnx()

    def _init_lama_onnx(self) -> None:
        """Attempts to load the LaMa ONNX model if present."""
        if self.model_path.exists():
            try:
                import onnxruntime as ort
                # Use CPU or CUDA provider if available
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                available_providers = ort.get_available_providers()
                chosen_providers = [p for p in providers if p in available_providers]
                self.ort_session = ort.InferenceSession(
                    str(self.model_path),
                    providers=chosen_providers or ["CPUExecutionProvider"]
                )
                print(f"[Inpainter] Loaded LaMa ONNX model from: {self.model_path}")
            except Exception as e:
                print(f"[Inpainter] Failed to load ONNX model ({e}). Using OpenCV fallback.")
                self.ort_session = None
        else:
            # Model not found yet; OpenCV fallback will be used
            self.ort_session = None

    def inpaint_bubbles(
        self,
        image: np.ndarray,
        bubbles: List[SpeechBubble]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inpaints and cleans text inside all given speech bubbles.

        Returns:
            Tuple of (cleaned_image, binary_mask)
        """
        h, w = image.shape[:2]
        # Create full-page binary mask (255 = inpaint target, 0 = keep original)
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # Build mask from text strokes inside each bubble's bounding box
        for bubble in bubbles:
            bx, by, bw, bh = int(bubble.bbox.x), int(bubble.bbox.y), int(bubble.bbox.width), int(bubble.bbox.height)
            bx = max(0, bx)
            by = max(0, by)
            bw = min(w - bx, bw)
            bh = min(h - by, bh)

            if bw <= 0 or bh <= 0:
                continue

            roi = image[by:by+bh, bx:bx+bw]
            bubble_mask = self._generate_text_stroke_mask(roi)
            full_mask[by:by+bh, bx:bx+bw] = np.maximum(full_mask[by:by+bh, bx:bx+bw], bubble_mask)

        # Run inpainting
        if self.ort_session is not None:
            cleaned = self._inpaint_with_lama(image, full_mask)
        else:
            cleaned = self._inpaint_with_opencv(image, full_mask, bubbles)

        return cleaned, full_mask

    def _generate_text_stroke_mask(self, roi: np.ndarray) -> np.ndarray:
        """Extracts precise text stroke mask inside a speech bubble ROI."""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # In manga, text is usually dark (< 150) on bright bubble (> 200)
        # Adaptive thresholding highlights stroke boundaries
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )

        # Simple threshold for deep black text pixels
        _, dark_thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        # Combine both
        combined_mask = cv2.bitwise_or(adaptive, dark_thresh)

        # Dilate slightly (2-3 pixels) to cover text edges and antialiasing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        # Exclude outermost 5% margin to prevent erasing bubble borders
        h, w = roi.shape[:2]
        margin_y = max(2, int(h * 0.04))
        margin_x = max(2, int(w * 0.04))
        dilated_mask[:margin_y, :] = 0
        dilated_mask[-margin_y:, :] = 0
        dilated_mask[:, :margin_x] = 0
        dilated_mask[:, -margin_x:] = 0

        return dilated_mask

    def _inpaint_with_opencv(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bubbles: List[SpeechBubble]
    ) -> np.ndarray:
        """High-quality OpenCV inpainting with smart color blending for manga bubbles."""
        # Standard Telea inpainting
        cleaned = cv2.inpaint(image, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

        # For large white bubbles, blend with dominant interior background color
        for bubble in bubbles:
            bx, by, bw, bh = int(bubble.bbox.x), int(bubble.bbox.y), int(bubble.bbox.width), int(bubble.bbox.height)
            h, w = image.shape[:2]
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(w - bx, bw), min(h - by, bh)
            if bw <= 0 or bh <= 0:
                continue

            roi_mask = mask[by:by+bh, bx:bx+bw]
            roi_cleaned = cleaned[by:by+bh, bx:bx+bw]
            roi_orig = image[by:by+bh, bx:bx+bw]

            # Sample non-masked border pixels to find bubble background color
            border_mask = np.zeros_like(roi_mask)
            border_mask[:5, :] = 255
            border_mask[-5:, :] = 255
            border_mask[:, :5] = 255
            border_mask[:, -5:] = 255
            sample_mask = cv2.bitwise_and(border_mask, cv2.bitwise_not(roi_mask))

            if np.count_nonzero(sample_mask) > 10:
                bg_color = cv2.mean(roi_orig, mask=sample_mask)[:3]
                # If background is bright (typical manga speech bubble), smooth the masked area
                if sum(bg_color) / 3.0 > 200:
                    bg_fill = np.full_like(roi_cleaned, bg_color, dtype=np.uint8)
                    # Blend slightly with Telea output to preserve subtle shading
                    blended_roi = cv2.addWeighted(roi_cleaned, 0.3, bg_fill, 0.7, 0)
                    # Apply only where text was masked
                    np.copyto(roi_cleaned, blended_roi, where=(roi_mask[:, :, None] > 0))

        return cleaned

    def _inpaint_with_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Runs LaMa ONNX inference on image and mask."""
        if self.ort_session is None:
            return image

        h, w = image.shape[:2]
        # Pad to multiple of 8 for LaMa
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        mask_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        # Normalize image to [0, 1] RGB
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_norm = (mask_pad > 0).astype(np.float32)[:, :, None]

        # Convert to NCHW format
        img_tensor = np.transpose(img_rgb, (2, 0, 1))[None, ...]
        mask_tensor = np.transpose(mask_norm, (2, 0, 1))[None, ...]

        input_names = [inp.name for inp in self.ort_session.get_inputs()]
        inputs = {input_names[0]: img_tensor, input_names[1]: mask_tensor}

        outputs = self.ort_session.run(None, inputs)
        res = outputs[0][0]
        res = np.clip(np.transpose(res, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        # Unpad back to original size
        return res_bgr[:h, :w]

