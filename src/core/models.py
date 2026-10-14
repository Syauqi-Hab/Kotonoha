"""Core data models for Kotonoha manga translation suite."""
from __future__ import annotations
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import uuid

class BoundingBox(BaseModel):
    """Bounding box coordinates for a detected speech bubble or text region."""
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Returns (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Returns (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x2, self.y2)

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this bounding box overlaps another bounding box."""
        return not (
            self.x2 < other.x or
            self.x > other.x2 or
            self.y2 < other.y or
            self.y > other.y2
        )


class FontStyle(BaseModel):
    """Typography styling for manga typesetting."""
    font_family: str = "Anime Ace"
    font_size: float = 14.0
    color: str = "#000000"
    outline_color: str = "#FFFFFF"
    outline_width: float = 2.0
    bold: bool = True
    italic: bool = False
    all_caps: bool = True
    line_spacing: float = 1.15
    alignment: Literal["center", "left", "right"] = "center"


class SpeechBubble(BaseModel):
    """Represents a single detected or user-drawn speech bubble on a manga page."""
    id: str = Field(default_factory=lambda: f"bubble_{uuid.uuid4().hex[:8]}")
    bbox: BoundingBox
    reading_order: int = 0
    original_text: str = ""
    translated_text: str = ""
    speaker: str = "Unknown"
    tone: str = "Normal"  # e.g., "Scream", "Whisper", "Monologue", "Normal"
    confidence: float = 1.0
    font_style: FontStyle = Field(default_factory=FontStyle)
    is_edited: bool = False
    auto_fit_text: bool = True


class MangaPage(BaseModel):
    """Represents a single manga page with its non-destructive layer data."""
    id: str = Field(default_factory=lambda: f"page_{uuid.uuid4().hex[:8]}")
    page_number: int = 1
    file_path: Optional[str] = None
    width: int = 0
    height: int = 0
    bubbles: List[SpeechBubble] = Field(default_factory=list)
    is_inpainted: bool = False
    is_translated: bool = False
    source_type: Literal["image", "pdf", "web", "screencap", "cbz"] = "image"

    def get_bubble_by_id(self, bubble_id: str) -> Optional[SpeechBubble]:
        for b in self.bubbles:
            if b.id == bubble_id:
                return b
        return None

    def sort_bubbles_reading_order(self) -> None:
        """Sort bubbles in place by reading order index."""
        self.bubbles.sort(key=lambda b: b.reading_order)


class GlossaryEntry(BaseModel):
    """Entity or term entry in the session memory glossary."""
    term: str
    translation: str
    category: Literal["character", "honorific", "jargon", "location", "relationship"] = "character"
    notes: str = ""


class TranslationContext(BaseModel):
    """Session-level context engine maintaining narrative continuity and glossary."""
    title: str = "Untitled Manga"
    current_chapter: str = "Chapter 1"
    character_glossary: List[GlossaryEntry] = Field(default_factory=list)
    honorific_policy: Literal["preserve", "localize", "omit"] = "preserve"
    summary_history: List[str] = Field(default_factory=list)
    system_notes: str = "Maintain natural manga flow, punchy dialogue, and accurate character voices."

