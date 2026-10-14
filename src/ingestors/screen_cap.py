"""Screen Capture Ingestor for Desktop Manga Reading.

Captures arbitrary screen regions for instant real-time translation overlays.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import ImageGrab
from src.core.models import MangaPage, BoundingBox


class ScreenCaptureIngestor:
    """Captures desktop screen regions for real-time translation."""

    @staticmethod
    def capture_region(bbox: Tuple[int, int, int, int]) -> Tuple[MangaPage, np.ndarray]:
        """Captures screen region defined by (x, y, width, height)."""
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h

        # Grab bounding box using PIL
        pil_image = ImageGrab.grab(bbox=(x, y, x2, y2), all_screens=True)
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        page = MangaPage(
            page_number=1,
            file_path="screen_capture",
            width=w,
            height=h,
            source_type="screencap"
        )
        return page, img_bgr

    @staticmethod
    def capture_full_screen() -> Tuple[MangaPage, np.ndarray]:
        """Captures entire primary display."""
        pil_image = ImageGrab.grab(all_screens=True)
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        page = MangaPage(
            page_number=1,
            file_path="screen_full",
            width=w,
            height=h,
            source_type="screencap"
        )
        return page, img_bgr

