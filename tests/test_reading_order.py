"""Unit tests for Manga Reading Order (RTL, TTB) sorting engine."""
from src.core.models import SpeechBubble, BoundingBox
from src.vision.reading_order import sort_manga_bubbles


def test_manga_reading_order_two_panels():
    # Top panel bubbles
    b_top_right = SpeechBubble(
        id="top_right",
        bbox=BoundingBox(x=600, y=100, width=100, height=80)
    )
    b_top_left = SpeechBubble(
        id="top_left",
        bbox=BoundingBox(x=150, y=110, width=100, height=80)
    )

    # Bottom panel bubbles
    b_bottom_right = SpeechBubble(
        id="bottom_right",
        bbox=BoundingBox(x=550, y=700, width=100, height=80)
    )
    b_bottom_left = SpeechBubble(
        id="bottom_left",
        bbox=BoundingBox(x=120, y=710, width=100, height=80)
    )

    # Supply in scrambled order
    input_bubbles = [b_bottom_left, b_top_left, b_bottom_right, b_top_right]

    sorted_bubbles = sort_manga_bubbles(input_bubbles, page_width=800, page_height=1200)

    # Expected order: top_right (#1), top_left (#2), bottom_right (#3), bottom_left (#4)
    ordered_ids = [b.id for b in sorted_bubbles]
    assert ordered_ids == ["top_right", "top_left", "bottom_right", "bottom_left"]
    assert [b.reading_order for b in sorted_bubbles] == [1, 2, 3, 4]

