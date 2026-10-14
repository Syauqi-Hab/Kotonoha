"""Standard unittest test suite for Kotonoha."""
import unittest
import numpy as np
import cv2

from src.core.models import BoundingBox, SpeechBubble, MangaPage, TranslationContext
from src.core.context_engine import ContextEngine
from src.vision.reading_order import sort_manga_bubbles
from src.vision.detector import BubbleDetector
from src.vision.inpainter import MangaInpainter
from src.translation.typesetter import AutoTypesetter


class TestKotonohaSuite(unittest.TestCase):

    def test_bounding_box_geometry(self):
        b1 = BoundingBox(x=10, y=20, width=100, height=50)
        self.assertEqual(b1.x2, 110)
        self.assertEqual(b1.y2, 70)
        self.assertEqual(b1.area, 5000)
        self.assertEqual(b1.center, (60.0, 45.0))

        b2 = BoundingBox(x=50, y=30, width=80, height=60)
        self.assertTrue(b1.intersects(b2))

        b3 = BoundingBox(x=200, y=200, width=50, height=50)
        self.assertFalse(b1.intersects(b3))

    def test_context_engine_glossary(self):
        engine = ContextEngine()
        engine.add_glossary_term("センパイ", "Senpai", category="honorific", notes="Senior")
        self.assertEqual(len(engine.context.character_glossary), 1)

        engine.add_glossary_term("センパイ", "Senpai (Senior)", category="honorific")
        self.assertEqual(len(engine.context.character_glossary), 1)
        self.assertEqual(engine.context.character_glossary[0].translation, "Senpai (Senior)")

        self.assertTrue(engine.remove_glossary_term("センパイ"))
        self.assertEqual(len(engine.context.character_glossary), 0)

    def test_manga_reading_order_two_panels(self):
        b_top_right = SpeechBubble(id="top_right", bbox=BoundingBox(x=600, y=100, width=100, height=80))
        b_top_left = SpeechBubble(id="top_left", bbox=BoundingBox(x=150, y=110, width=100, height=80))
        b_bottom_right = SpeechBubble(id="bottom_right", bbox=BoundingBox(x=550, y=700, width=100, height=80))
        b_bottom_left = SpeechBubble(id="bottom_left", bbox=BoundingBox(x=120, y=710, width=100, height=80))

        input_bubbles = [b_bottom_left, b_top_left, b_bottom_right, b_top_right]
        sorted_bubbles = sort_manga_bubbles(input_bubbles, page_width=800, page_height=1200)

        ordered_ids = [b.id for b in sorted_bubbles]
        self.assertEqual(ordered_ids, ["top_right", "top_left", "bottom_right", "bottom_left"])
        self.assertEqual([b.reading_order for b in sorted_bubbles], [1, 2, 3, 4])

    def test_auto_typesetter(self):
        typesetter = AutoTypesetter(min_font_size=8.0, max_font_size=32.0)

        bubble_large = SpeechBubble(
            bbox=BoundingBox(x=100, y=100, width=300, height=300),
            translated_text="YES!"
        )
        typesetter.typeset_bubble(bubble_large)
        self.assertGreater(bubble_large.font_style.font_size, 20.0)

        bubble_small = SpeechBubble(
            bbox=BoundingBox(x=100, y=100, width=80, height=60),
            translated_text="This is a really long sentence that needs to wrap and scale down."
        )
        typesetter.typeset_bubble(bubble_small)
        self.assertLessEqual(bubble_small.font_style.font_size, 12.0)

    def test_detector_and_inpainter(self):
        # Create a mock synthetic manga image with a white speech bubble and text
        img = np.full((400, 400, 3), 200, dtype=np.uint8)
        # White speech bubble
        cv2.ellipse(img, (200, 200), (80, 60), 0, 0, 360, (255, 255, 255), -1)
        # Dark text lines inside bubble
        cv2.putText(img, "TEST", (160, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        detector = BubbleDetector()
        bubbles = detector.detect_bubbles(img)
        self.assertGreaterEqual(len(bubbles), 1)

        inpainter = MangaInpainter()
        cleaned, mask = inpainter.inpaint_bubbles(img, bubbles)
        self.assertEqual(cleaned.shape, img.shape)
        self.assertEqual(mask.shape, img.shape[:2])


if __name__ == "__main__":
    unittest.main()

