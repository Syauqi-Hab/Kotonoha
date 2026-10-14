"""Unit tests for Kotonoha core data models and context engine."""
import pytest
from src.core.models import BoundingBox, SpeechBubble, MangaPage, TranslationContext
from src.core.context_engine import ContextEngine


def test_bounding_box_geometry():
    b1 = BoundingBox(x=10, y=20, width=100, height=50)
    assert b1.x2 == 110
    assert b1.y2 == 70
    assert b1.area == 5000
    assert b1.center == (60.0, 45.0)

    # Intersects
    b2 = BoundingBox(x=50, y=30, width=80, height=60)
    assert b1.intersects(b2) is True

    # Non-overlapping
    b3 = BoundingBox(x=200, y=200, width=50, height=50)
    assert b1.intersects(b3) is False


def test_context_engine_glossary():
    engine = ContextEngine()
    engine.add_glossary_term("センパイ", "Senpai", category="honorific", notes="Senior")
    assert len(engine.context.character_glossary) == 1

    # Update existing
    engine.add_glossary_term("センパイ", "Senpai (Senior)", category="honorific")
    assert len(engine.context.character_glossary) == 1
    assert engine.context.character_glossary[0].translation == "Senpai (Senior)"

    # Remove
    assert engine.remove_glossary_term("センパイ") is True
    assert len(engine.context.character_glossary) == 0


def test_manga_page_bubble_sorting():
    page = MangaPage(page_number=1, width=800, height=1200)
    b1 = SpeechBubble(bbox=BoundingBox(x=100, y=100, width=50, height=50), reading_order=2)
    b2 = SpeechBubble(bbox=BoundingBox(x=200, y=100, width=50, height=50), reading_order=1)
    page.bubbles = [b1, b2]

    page.sort_bubbles_reading_order()
    assert page.bubbles[0].reading_order == 1
    assert page.bubbles[1].reading_order == 2

