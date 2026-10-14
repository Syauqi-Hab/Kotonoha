"""Unit tests for AutoTypesetter typography and text fitting."""
from src.core.models import SpeechBubble, BoundingBox
from src.translation.typesetter import AutoTypesetter


def test_auto_typesetter_font_size():
    typesetter = AutoTypesetter(min_font_size=8.0, max_font_size=32.0)

    # Large bubble with short text should receive large font size
    bubble_large = SpeechBubble(
        bbox=BoundingBox(x=100, y=100, width=300, height=300),
        translated_text="YES!"
    )
    typesetter.typeset_bubble(bubble_large)
    assert bubble_large.font_style.font_size > 20.0

    # Small bubble with long text should scale down font size
    bubble_small = SpeechBubble(
        bbox=BoundingBox(x=100, y=100, width=80, height=60),
        translated_text="This is a really long sentence that needs to wrap and scale down."
    )
    typesetter.typeset_bubble(bubble_small)
    assert bubble_small.font_style.font_size <= 12.0

