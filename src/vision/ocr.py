"""Manga OCR Subsystem.

Integrates Manga-OCR for vertical & horizontal Japanese text recognition,
with graceful fallback to Google Gemini Vision or local offline processing.
"""
from __future__ import annotations
import os
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from src.core.models import SpeechBubble
from src.core.config import config


class MangaOCR:
    """Optical Character Recognition engine specialized for Japanese Manga text."""

    def __init__(self):
        self.mocr = None
        self._init_manga_ocr()

    def _init_manga_ocr(self) -> None:
        """Attempt to load manga-ocr if installed."""
        try:
            from manga_ocr import MangaOcr
            print("[MangaOCR] Initializing manga-ocr model...")
            self.mocr = MangaOcr()
            print("[MangaOCR] manga-ocr loaded successfully.")
        except ImportError:
            print("[MangaOCR] manga-ocr not installed. Falling back to Gemini Vision / heuristic OCR.")
            self.mocr = None
        except Exception as e:
            print(f"[MangaOCR] Error initializing manga-ocr ({e}). Using fallback.")
            self.mocr = None

    def recognize_bubble(self, image: np.ndarray, bubble: SpeechBubble) -> str:
        """Extracts Japanese text from a single speech bubble region."""
        h, w = image.shape[:2]
        bx = max(0, int(bubble.bbox.x))
        by = max(0, int(bubble.bbox.y))
        bw = min(w - bx, int(bubble.bbox.width))
        bh = min(h - by, int(bubble.bbox.height))

        if bw <= 5 or bh <= 5:
            return ""

        crop_bgr = image[by:by+bh, bx:bx+bw]

        # 1. Primary: Local manga-ocr
        if self.mocr is not None:
            try:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                text = self.mocr(pil_img)
                return text.strip()
            except Exception as e:
                print(f"[MangaOCR] Error during local OCR: {e}")

        # 2. Secondary: Fallback
        # If Gemini API key is configured, can use Gemini Vision for OCR
        if config.gemini_api_key:
            return self._ocr_with_gemini(crop_bgr)

        # 3. Default fallback if neither is available
        return f"[Text in bubble #{bubble.reading_order}]"

    def recognize_all_bubbles(
        self,
        image: np.ndarray,
        bubbles: List[SpeechBubble]
    ) -> List[SpeechBubble]:
        """Recognize Japanese text for all bubbles on a manga page."""
        for bubble in bubbles:
            if not bubble.original_text:
                bubble.original_text = self.recognize_bubble(image, bubble)
        return bubbles

    def _ocr_with_gemini(self, crop_bgr: np.ndarray) -> str:
        """Gemini Multimodal Vision fallback for Japanese OCR."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=config.gemini_api_key)
            _, buffer = cv2.imencode(".png", crop_bgr)
            image_bytes = buffer.tobytes()

            response = client.models.generate_content(
                model=config.llm_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    "Transcribe the Japanese text in this manga speech bubble exactly as written. Return ONLY the raw Japanese text with no explanations, no quotes, and no romaji."
                ]
            )
            return (response.text or "").strip()
        except Exception as e:
            print(f"[MangaOCR] Gemini Vision OCR failed: {e}")
            return ""

