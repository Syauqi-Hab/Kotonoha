"""UI Inspector Panels for Kotonoha.

Contains:
1. LayerControlPanel: OpenToonz-style layer visibility and opacity sliders.
2. BubbleInspectorPanel: Speech bubble text, speaker, emotion, and typography editor.
3. SessionMemoryPanel: Glossary manager, honorifics, and narrative continuity log.
"""
from __future__ import annotations
from typing import Optional, List
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QSlider,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QTabWidget,
    QColorDialog,
    QFontComboBox,
)
from PySide6.QtGui import QColor, QFont

from src.core.models import SpeechBubble, MangaPage, GlossaryEntry
from src.core.context_engine import ContextEngine


class LayerControlPanel(QWidget):
    """Controls the visibility and opacity of the 4 OpenToonz canvas layers."""

    visibility_changed = Signal(str, bool)
    opacity_changed = Signal(str, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Layer Stack (OpenToonz 4-Layer)")
        g_layout = QVBoxLayout(group)

        self.layers = [
            ("typeset", "Layer 3: Typeset Text Overlay"),
            ("boxes", "Layer 2: Bounding Boxes / OCR"),
            ("inpaint", "Layer 1: Inpaint / Cleaned"),
            ("base", "Layer 0: Base Source Image"),
        ]

        self.vis_boxes = {}
        self.sliders = {}

        for layer_key, layer_label in self.layers:
            row = QHBoxLayout()

            cb = QCheckBox(layer_label)
            cb.setChecked(True)
            cb.toggled.connect(lambda val, k=layer_key: self.visibility_changed.emit(k, val))
            self.vis_boxes[layer_key] = cb

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(100)
            slider.setFixedWidth(80)
            slider.valueChanged.connect(lambda val, k=layer_key: self.opacity_changed.emit(k, val / 100.0))
            self.sliders[layer_key] = slider

            row.addWidget(cb)
            row.addStretch()
            row.addWidget(slider)
            g_layout.addLayout(row)

        layout.addWidget(group)


class BubbleInspectorPanel(QWidget):
    """Edits text, speaker, emotion tone, and typography for the selected speech bubble."""

    bubble_updated = Signal(SpeechBubble)
    bubble_deleted = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_bubble: Optional[SpeechBubble] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Bubble Header
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel("<b>Select a bubble to inspect</b>")
        self.btn_delete = QPushButton("🗑 Delete")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._on_delete)
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_delete)
        layout.addLayout(header_layout)

        # Form Layout
        form = QFormLayout()

        # Japanese OCR Text
        self.txt_jp = QTextEdit()
        self.txt_jp.setMaximumHeight(65)
        self.txt_jp.setPlaceholderText("Raw Japanese OCR text...")
        form.addRow("Original (JP):", self.txt_jp)

        # Translated English Text
        self.txt_en = QTextEdit()
        self.txt_en.setMaximumHeight(80)
        self.txt_en.setPlaceholderText("English translated dialogue...")
        form.addRow("Typeset (EN):", self.txt_en)

        # Speaker & Tone
        speaker_row = QHBoxLayout()
        self.edit_speaker = QLineEdit()
        self.edit_speaker.setPlaceholderText("Speaker name")

        self.combo_tone = QComboBox()
        self.combo_tone.addItems(["Normal", "Scream", "Whisper", "Internal Monologue", "Narration"])
        speaker_row.addWidget(self.edit_speaker)
        speaker_row.addWidget(self.combo_tone)
        form.addRow("Speaker / Tone:", speaker_row)

        # Typography Styling
        typo_group = QGroupBox("Manga Typography Styling")
        t_layout = QFormLayout(typo_group)

        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont("Anime Ace"))
        t_layout.addRow("Font Family:", self.font_combo)

        size_row = QHBoxLayout()
        self.spin_size = QDoubleSpinBox()
        self.spin_size.setRange(6.0, 72.0)
        self.spin_size.setValue(14.0)

        self.chk_autofit = QCheckBox("Auto-Fit")
        self.chk_autofit.setChecked(True)
        size_row.addWidget(self.spin_size)
        size_row.addWidget(self.chk_autofit)
        t_layout.addRow("Font Size:", size_row)

        style_flags_row = QHBoxLayout()
        self.chk_bold = QCheckBox("Bold")
        self.chk_bold.setChecked(True)
        self.chk_caps = QCheckBox("ALL-CAPS")
        self.chk_caps.setChecked(True)
        self.combo_align = QComboBox()
        self.combo_align.addItems(["center", "left", "right"])
        style_flags_row.addWidget(self.chk_bold)
        style_flags_row.addWidget(self.chk_caps)
        style_flags_row.addWidget(self.combo_align)
        t_layout.addRow("Style / Align:", style_flags_row)

        form.addRow(typo_group)
        layout.addLayout(form)

        # Apply Button
        self.btn_apply = QPushButton("Apply Bubble Changes")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(self.btn_apply)
        layout.addStretch()

    def load_bubble(self, bubble: Optional[SpeechBubble]) -> None:
        self.current_bubble = bubble
        if not bubble:
            self.lbl_title.setText("<b>No bubble selected</b>")
            self.btn_delete.setEnabled(False)
            self.btn_apply.setEnabled(False)
            self.txt_jp.clear()
            self.txt_en.clear()
            self.edit_speaker.clear()
            return

        self.lbl_title.setText(f"<b>Bubble #{bubble.reading_order}</b> <font color='#A0AEC0'>({bubble.id})</font>")
        self.btn_delete.setEnabled(True)
        self.btn_apply.setEnabled(True)

        self.txt_jp.setPlainText(bubble.original_text)
        self.txt_en.setPlainText(bubble.translated_text)
        self.edit_speaker.setText(bubble.speaker)

        idx = self.combo_tone.findText(bubble.tone)
        if idx >= 0:
            self.combo_tone.setCurrentIndex(idx)

        self.font_combo.setCurrentFont(QFont(bubble.font_style.font_family))
        self.spin_size.setValue(bubble.font_style.font_size)
        self.chk_autofit.setChecked(bubble.auto_fit_text)
        self.chk_bold.setChecked(bubble.font_style.bold)
        self.chk_caps.setChecked(bubble.font_style.all_caps)

        align_idx = self.combo_align.findText(bubble.font_style.alignment)
        if align_idx >= 0:
            self.combo_align.setCurrentIndex(align_idx)

    def _on_apply(self) -> None:
        if not self.current_bubble:
            return

        self.current_bubble.original_text = self.txt_jp.toPlainText()
        self.current_bubble.translated_text = self.txt_en.toPlainText()
        self.current_bubble.speaker = self.edit_speaker.text().strip() or "Unknown"
        self.current_bubble.tone = self.combo_tone.currentText()
        self.current_bubble.auto_fit_text = self.chk_autofit.isChecked()

        self.current_bubble.font_style.font_family = self.font_combo.currentFont().family()
        self.current_bubble.font_style.font_size = self.spin_size.value()
        self.current_bubble.font_style.bold = self.chk_bold.isChecked()
        self.current_bubble.font_style.all_caps = self.chk_caps.isChecked()
        self.current_bubble.font_style.alignment = self.combo_align.currentText()  # type: ignore
        self.current_bubble.is_edited = True

        self.bubble_updated.emit(self.current_bubble)

    def _on_delete(self) -> None:
        if self.current_bubble:
            self.bubble_deleted.emit(self.current_bubble.id)
            self.load_bubble(None)


class SessionMemoryPanel(QWidget):
    """Manages the live context glossary and narrative memory history."""

    def __init__(self, context_engine: ContextEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.context_engine = context_engine
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()

        # Tab 1: Glossary
        glossary_tab = QWidget()
        g_layout = QVBoxLayout(glossary_tab)

        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Original Term", "Translation", "Category", "Notes"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        g_layout.addWidget(self.table)

        # Add Entry Row
        add_row = QHBoxLayout()
        self.in_term = QLineEdit()
        self.in_term.setPlaceholderText("e.g. センパイ")
        self.in_trans = QLineEdit()
        self.in_trans.setPlaceholderText("e.g. Senpai")
        self.in_cat = QComboBox()
        self.in_cat.addItems(["character", "honorific", "jargon", "location", "relationship"])

        btn_add = QPushButton("+ Add")
        btn_add.clicked.connect(self._add_term)

        add_row.addWidget(self.in_term)
        add_row.addWidget(self.in_trans)
        add_row.addWidget(self.in_cat)
        add_row.addWidget(btn_add)
        g_layout.addLayout(add_row)

        btn_delete_term = QPushButton("Remove Selected Term")
        btn_delete_term.clicked.connect(self._delete_term)
        g_layout.addWidget(btn_delete_term)

        tabs.addTab(glossary_tab, "Glossary & NER")

        # Tab 2: Narrative Memory Summary
        memory_tab = QWidget()
        m_layout = QVBoxLayout(memory_tab)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setPlaceholderText("Narrative context summaries across pages will appear here...")
        m_layout.addWidget(self.txt_summary)

        tabs.addTab(memory_tab, "Narrative Continuity")
        layout.addWidget(tabs)

        self.refresh_ui()

    def refresh_ui(self) -> None:
        """Reloads table from context_engine."""
        glossary = self.context_engine.context.character_glossary
        self.table.setRowCount(len(glossary))

        for row, entry in enumerate(glossary):
            self.table.setItem(row, 0, QTableWidgetItem(entry.term))
            self.table.setItem(row, 1, QTableWidgetItem(entry.translation))
            self.table.setItem(row, 2, QTableWidgetItem(entry.category))
            self.table.setItem(row, 3, QTableWidgetItem(entry.notes))

        # Refresh narrative summaries
        summaries = self.context_engine.context.summary_history
        self.txt_summary.setPlainText("\n\n".join(summaries))

    def _add_term(self) -> None:
        term = self.in_term.text().strip()
        trans = self.in_trans.text().strip()
        cat = self.in_cat.currentText()
        if term and trans:
            self.context_engine.add_glossary_term(term, trans, category=cat)
            self.in_term.clear()
            self.in_trans.clear()
            self.refresh_ui()

    def _delete_term(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            item = self.table.item(row, 0)
            if item:
                self.context_engine.remove_glossary_term(item.text())
                self.refresh_ui()

