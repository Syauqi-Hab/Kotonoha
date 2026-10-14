"""Context-Stitching Memory Engine for Kotonoha.

Manages narrative continuity, character names, honorific rules,
and chapter-wide glossaries across page translation calls.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.core.models import TranslationContext, GlossaryEntry


class ContextEngine:
    """Session-level context manager that maintains consistency across pages."""

    def __init__(self, context: Optional[TranslationContext] = None):
        self.context = context or TranslationContext()

    def add_glossary_term(
        self,
        term: str,
        translation: str,
        category: str = "character",
        notes: str = ""
    ) -> GlossaryEntry:
        """Add or update an entity/term in the glossary."""
        # Check if already exists
        for entry in self.context.character_glossary:
            if entry.term.lower() == term.lower():
                entry.translation = translation
                entry.category = category  # type: ignore
                entry.notes = notes
                return entry

        new_entry = GlossaryEntry(
            term=term,
            translation=translation,
            category=category,  # type: ignore
            notes=notes
        )
        self.context.character_glossary.append(new_entry)
        return new_entry

    def remove_glossary_term(self, term: str) -> bool:
        """Remove a term from the glossary."""
        initial_len = len(self.context.character_glossary)
        self.context.character_glossary = [
            e for e in self.context.character_glossary if e.term.lower() != term.lower()
        ]
        return len(self.context.character_glossary) < initial_len

    def get_terms_dict(self) -> Dict[str, str]:
        """Return simple dictionary mapping source term -> target translation."""
        return {entry.term: entry.translation for entry in self.context.character_glossary}

    def append_summary(self, summary_line: str) -> None:
        """Append page narrative summary to context memory."""
        if summary_line:
            self.context.summary_history.append(summary_line)

    def format_llm_context_prompt(self) -> str:
        """Formats session glossary and narrative memory for LLM system prompts."""
        lines = [
            "### SESSION CONTEXT & GLOSSARY (Strict Consistency Required):",
            f"- Manga Title: {self.context.title}",
            f"- Chapter: {self.context.current_chapter}",
            f"- Honorifics Policy: {self.context.honorific_policy} (e.g. keep '-san', '-kun', '-chan' or adapt according to character warmth)",
        ]

        if self.context.character_glossary:
            lines.append("\n**Character Names & Key Terminology:**")
            for entry in self.context.character_glossary:
                note_str = f" ({entry.notes})" if entry.notes else ""
                lines.append(f"  * {entry.term} -> {entry.translation} [{entry.category}]{note_str}")

        if self.context.summary_history:
            lines.append("\n**Preceding Narrative Context:**")
            # Include last 5 page summaries for recency
            for s in self.context.summary_history[-5:]:
                lines.append(f"  * {s}")

        if self.context.system_notes:
            lines.append(f"\n**Tone & Style Directives:** {self.context.system_notes}")

        return "\n".join(lines)

    def save_to_file(self, file_path: Path | str) -> None:
        """Save context store to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.context.model_dump(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> "ContextEngine":
        """Load context store from JSON file."""
        path = Path(file_path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(context=TranslationContext(**data))

