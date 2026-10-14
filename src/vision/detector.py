"""Speech Bubble Detector for Manga Pages using Computer Vision (OpenCV).

Detects speech bubbles via morphological operations, contour hierarchy,
and aspect ratio heuristics.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import cv2
import numpy as np
from src.core.models import BoundingBox, SpeechBubble
from src.vision.reading_order import sort_manga_bubbles


class BubbleDetector:
    """Detects speech bubble candidates in manga pages."""

    def __init__(
        self,
        min_area_ratio: float = 0.001,   # Minimum 0.1% of page
        max_area_ratio: float = 0.35,    # Maximum 35% of page
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def detect_bubbles(
        self,
        image_input: np.ndarray | str,
        sort_reading_order: bool = True
    ) -> List[SpeechBubble]:
        """Detects speech bubbles in an image (path or numpy array).

        Returns a list of SpeechBubble instances sorted in manga reading order.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from: {image_input}")
        else:
            image = image_input

        height, width = image.shape[:2]
        page_area = float(width * height)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Binarize for white/bright speech bubble regions
        # Speech bubbles in manga are characteristically near-white (> 230)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        # 2. Morphological closing to seal bubble outlines and text gaps
        kernel_size = max(5, int(min(width, height) * 0.01))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Find outer contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes: List[BoundingBox] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (page_area * self.min_area_ratio) or area > (page_area * self.max_area_ratio):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h) if h > 0 else 0.0

            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue

            # Check if region contains dark text pixels (internal contrast check)
            roi_gray = gray[y:y+h, x:x+w]
            if roi_gray.size == 0:
                continue

            dark_pixel_ratio = np.count_nonzero(roi_gray < 160) / float(roi_gray.size)
            # A valid speech bubble should contain text (between 1% and 40% dark pixels)
            if 0.01 <= dark_pixel_ratio <= 0.45:
                # Add slight padding around detected bubble
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.05)
                bx = max(0, x - pad_x)
                by = max(0, y - pad_y)
                bw = min(width - bx, w + pad_x * 2)
                bh = min(height - by, h + pad_y * 2)

                candidate_boxes.append(
                    BoundingBox(x=float(bx), y=float(by), width=float(bw), height=float(bh))
                )

        # 4. Non-Maximum Suppression (Merge overlapping or nested boxes)
        merged_boxes = self._suppress_overlapping_boxes(candidate_boxes)

        bubbles = [
            SpeechBubble(bbox=box) for box in merged_boxes
        ]

        if sort_reading_order:
            bubbles = sort_manga_bubbles(bubbles, float(width), float(height))

        return bubbles

    def _suppress_overlapping_boxes(self, boxes: List[BoundingBox], iou_thresh: float = 0.3) -> List[BoundingBox]:
        """Filters out heavily overlapping bounding boxes, preserving the tighter/clearer one."""
        if not boxes:
            return []

        # Sort by area descending
        sorted_boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
        kept_boxes: List[BoundingBox] = []

        for box in sorted_boxes:
            overlap = False
            for kept in kept_boxes:
                # Calculate Intersection over Union (IoU)
                ix1 = max(box.x, kept.x)
                iy1 = max(box.y, kept.y)
                ix2 = min(box.x2, kept.x2)
                iy2 = min(box.y2, kept.y2)

                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                intersection = iw * ih

                if intersection > 0:
                    union = box.area + kept.area - intersection
                    iou = intersection / union if union > 0 else 0
                    # Also check containment ratio
                    containment = intersection / box.area
                    if iou > iou_thresh or containment > 0.7:
                        overlap = True
                        break

            if not overlap:
                kept_boxes.append(box)

        return kept_boxes

