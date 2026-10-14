"""Manga Auto-Typesetting Engine.

Calculates optimal font sizing, dynamic text wrapping for oval/rectangular bubble geometries,
and styling attributes (outlines, center alignment, casing).
"""
from __future__ import annotations
import math
from typing import List, Tuple
from src.core.models import SpeechBubble, FontStyle


class AutoTypesetter:
    """Calculates typography layouts to fit speech bubbles harmoniously."""

    def __init__(
        self,
        min_font_size: float = 8.0,
        max_font_size: float = 36.0,
        safe_margin_ratio: float = 0.15,  # 15% inner padding from bubble border
    ):
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.safe_margin_ratio = safe_margin_ratio

    def typeset_bubble(self, bubble: SpeechBubble) -> SpeechBubble:
        """Computes optimal font size and wrapped lines for a speech bubble."""
        text = bubble.translated_text.strip()
        if not text:
            return bubble

        # Apply uppercase convention if requested (common in Western manga releases)
        if bubble.font_style.all_caps:
            text = text.upper()

        box_w = bubble.bbox.width
        box_h = bubble.bbox.height

        # Calculate safe bounding box (speech bubbles are curved, so text must stay away from borders)
        safe_w = max(20.0, box_w * (1.0 - 2 * self.safe_margin_ratio))
        safe_h = max(20.0, box_h * (1.0 - 2 * self.safe_margin_ratio))

        if bubble.auto_fit_text:
            optimal_size = self._calculate_optimal_font_size(
                text=text,
                target_w=safe_w,
                target_h=safe_h,
                line_spacing=bubble.font_style.line_spacing
            )
            bubble.font_style.font_size = optimal_size

        return bubble

    def _calculate_optimal_font_size(
        self,
        text: str,
        target_w: float,
        target_h: float,
        line_spacing: float = 1.15
    ) -> float:
        """Iteratively determines the best font size to fill the bubble without overflowing."""
        words = text.split()
        if not words:
            return self.min_font_size

        # Binary search for optimal font size
        low = self.min_font_size
        high = self.max_font_size
        best_size = low

        while (high - low) > 0.5:
            mid = (low + high) / 2.0
            if self._text_fits(words, mid, target_w, target_h, line_spacing):
                best_size = mid
                low = mid  # Try bigger
            else:
                high = mid  # Too big, decrease

        return round(best_size, 1)

    def _text_fits(
        self,
        words: List[str],
        font_size: float,
        target_w: float,
        target_h: float,
        line_spacing: float
    ) -> bool:
        """Approximates whether the text wrapped at font_size fits in (target_w, target_h)."""
        # Average character width ratio for typical sans-serif manga fonts is ~0.55-0.6 of font_size
        char_w = font_size * 0.58
        line_h = font_size * line_spacing

        lines: List[str] = []
        current_line: List[str] = []
        current_w = 0.0

        for word in words:
            word_w = len(word) * char_w
            space_w = char_w if current_line else 0.0

            # If a single word is wider than target width, it doesn't fit
            if word_w > target_w:
                return False

            if current_w + space_w + word_w <= target_w:
                current_line.append(word)
                current_w += space_w + word_w
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_w = word_w

        if current_line:
            lines.append(" ".join(current_line))

        total_h = len(lines) * line_h
        return total_h <= target_h

