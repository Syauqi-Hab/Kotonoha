"""Context-Aware LLM Translation Pipeline for Kotonoha.

Groups manga speech bubbles by page into structured LLM prompts with spatial context,
reading order, character glossaries, and honorific rules.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.core.models import MangaPage, SpeechBubble, GlossaryEntry
from src.core.context_engine import ContextEngine
from src.core.config import config


class BubbleTranslationResult(BaseModel):
    """Structured translation item returned from LLM."""
    bubble_id: str
    original_text: str
    translated_text: str
    speaker: str = "Unknown"
    tone: str = "Normal"  # Normal, Scream, Whisper, Internal Monologue, Narration


class DiscoveredEntity(BaseModel):
    """Entity detected by LLM to enrich session memory."""
    term: str
    translation: str
    category: str = "character"
    notes: str = ""


class PageTranslationResponse(BaseModel):
    """Full structured response schema for a manga page."""
    translations: List[BubbleTranslationResult]
    page_summary: str = Field(
        description="A brief 1-2 sentence narrative summary of what happens on this page for session continuity."
    )
    discovered_entities: List[DiscoveredEntity] = Field(
        default_factory=list,
        description="Any new character names, terms, or honorific relations detected."
    )


class LLMTranslator:
    """Translates manga speech bubbles preserving context, dialogue dynamics, and character consistency."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.gemini_api_key
        self.model_name = model_name or config.llm_model

    def translate_page(
        self,
        page: MangaPage,
        context_engine: ContextEngine
    ) -> MangaPage:
        """Translates all bubbles on a manga page using structured LLM context."""
        if not page.bubbles:
            return page

        # Ensure bubbles are in reading order
        page.sort_bubbles_reading_order()

        # Check if API key is provided
        if not self.api_key:
            return self._mock_translate_page(page, context_engine)

        try:
            return self._translate_page_with_gemini(page, context_engine)
        except Exception as e:
            print(f"[LLMTranslator] Gemini translation call error: {e}. Falling back to rule-based.")
            return self._mock_translate_page(page, context_engine)

    def _translate_page_with_gemini(
        self,
        page: MangaPage,
        context_engine: ContextEngine
    ) -> MangaPage:
        """Calls Google Gemini with structured JSON output schema."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Build prompt payload
        system_instruction = (
            "You are an expert, professional Japanese-to-English manga localizer and translation director.\n"
            "Translate dialogue naturally, capturing character tone, emotion, subtext, and manga punchiness.\n"
            "Preserve honorifics according to the project policy unless natural English idiom demands otherwise.\n"
            "Dialogue bubbles are ordered according to standard Manga reading order (Right-to-Left, Top-to-Bottom).\n"
            "Maintain strict consistency with the provided character names and session glossary.\n"
        )

        context_prompt = context_engine.format_llm_context_prompt()

        bubbles_payload = []
        for b in page.bubbles:
            bubbles_payload.append({
                "bubble_id": b.id,
                "reading_order": b.reading_order,
                "bounding_box": {
                    "x": round(b.bbox.x, 1),
                    "y": round(b.bbox.y, 1),
                    "w": round(b.bbox.width, 1),
                    "h": round(b.bbox.height, 1)
                },
                "japanese_text": b.original_text or "[Empty bubble or unreadable]"
            })

        user_prompt = (
            f"{context_prompt}\n\n"
            f"### PAGE {page.page_number} SPEECH BUBBLES:\n"
            f"{json.dumps(bubbles_payload, ensure_ascii=False, indent=2)}\n\n"
            "Translate each speech bubble into natural, punchy English manga dialogue. "
            "Identify the speaker and emotional tone. "
            "Provide a brief narrative summary of the page for continuity."
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=PageTranslationResponse,
                temperature=0.3,
            ),
        )

        if not response.text:
            raise ValueError("Empty response received from Gemini.")

        parsed_data = PageTranslationResponse.model_validate_json(response.text)

        # Map translations back to bubbles
        res_map = {t.bubble_id: t for t in parsed_data.translations}
        for b in page.bubbles:
            if b.id in res_map:
                res = res_map[b.id]
                b.translated_text = res.translated_text
                b.speaker = res.speaker
                b.tone = res.tone

        # Update session memory
        if parsed_data.page_summary:
            context_engine.append_summary(f"Page {page.page_number}: {parsed_data.page_summary}")

        for entity in parsed_data.discovered_entities:
            context_engine.add_glossary_term(
                term=entity.term,
                translation=entity.translation,
                category=entity.category,
                notes=entity.notes
            )

        page.is_translated = True
        return page

    def _mock_translate_page(
        self,
        page: MangaPage,
        context_engine: ContextEngine
    ) -> MangaPage:
        """Offline or demo translation fallback when no API key is provided."""
        print("[LLMTranslator] Running offline / demo translation mode.")
        terms_map = context_engine.get_terms_dict()

        for b in page.bubbles:
            if not b.translated_text:
                raw_text = b.original_text or f"Bubble #{b.reading_order}"
                # Apply glossary substitutions
                translated = raw_text
                for src_term, target_term in terms_map.items():
                    translated = translated.replace(src_term, target_term)

                # Simulated English manga localization
                if "bubble" in translated.lower() or not b.original_text:
                    b.translated_text = f"Dialogue Line #{b.reading_order}!"
                else:
                    b.translated_text = f"[TL] {translated}"

                b.speaker = b.speaker if b.speaker != "Unknown" else f"Speaker {b.reading_order}"
                b.tone = "Normal"

        page.is_translated = True
        context_engine.append_summary(f"Page {page.page_number}: Read dialogue sequence across {len(page.bubbles)} bubbles.")
        return page

