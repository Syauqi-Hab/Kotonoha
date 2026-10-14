"""Manga Reading Order Sorting Engine.

Standard Manga reading order is Right-to-Left (RTL), Top-to-Bottom (TTB).
This module groups speech bubbles into narrative panels/bands and assigns
consistent reading order indices (1, 2, 3...).
"""
from __future__ import annotations
from typing import List, Tuple
from src.core.models import SpeechBubble


def sort_manga_bubbles(
    bubbles: List[SpeechBubble],
    page_width: float = 1000.0,
    page_height: float = 1400.0,
) -> List[SpeechBubble]:
    """Sorts speech bubbles according to standard Manga reading order:

    Right-to-Left (RTL) and Top-to-Bottom (TTB).
    
    Algorithm:
    1. Clusters bubbles into horizontal panel bands based on vertical overlap.
    2. Sorts bands from top to bottom (TTB).
    3. Within each band, sorts bubbles from right to left (RTL).
    4. Reassigns 1-indexed `reading_order` sequentially.
    """
    if not bubbles:
        return []

    if len(bubbles) == 1:
        bubbles[0].reading_order = 1
        return bubbles

    # Make a copy to avoid mutating external lists unexpectedly
    working_bubbles = list(bubbles)

    # Sort primarily by Top-to-Bottom with a panel threshold
    # Estimate average bubble height for thresholding
    avg_height = sum(b.bbox.height for b in working_bubbles) / len(working_bubbles)
    band_tolerance = max(avg_height * 0.8, 60.0)

    # Sort bubbles by Y center initially
    sorted_by_y = sorted(working_bubbles, key=lambda b: b.bbox.center[1])

    # Group into horizontal bands
    bands: List[List[SpeechBubble]] = []
    current_band: List[SpeechBubble] = [sorted_by_y[0]]

    for bubble in sorted_by_y[1:]:
        band_y_center = sum(b.bbox.center[1] for b in current_band) / len(current_band)
        # If bubble vertical center is close enough to current band
        if abs(bubble.bbox.center[1] - band_y_center) < band_tolerance:
            current_band.append(bubble)
        else:
            bands.append(current_band)
            current_band = [bubble]

    if current_band:
        bands.append(current_band)

    # Within each band, sort Right-to-Left (X descending)
    # If X is very close, sort Top-to-Bottom (Y ascending)
    final_ordered: List[SpeechBubble] = []
    for band in bands:
        band.sort(key=lambda b: (-b.bbox.center[0], b.bbox.center[1]))
        final_ordered.extend(band)

    # Re-assign sequential reading order (1-indexed)
    for idx, bubble in enumerate(final_ordered, start=1):
        bubble.reading_order = idx

    return final_ordered


def renumber_bubbles(bubbles: List[SpeechBubble]) -> List[SpeechBubble]:
    """Sort bubbles by their current `reading_order` and ensure continuous 1..N indices."""
    sorted_bubbles = sorted(bubbles, key=lambda b: b.reading_order)
    for idx, b in enumerate(sorted_bubbles, start=1):
        b.reading_order = idx
    return sorted_bubbles

