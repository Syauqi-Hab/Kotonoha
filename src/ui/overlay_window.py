"""Floating Screen Overlay Mode for Kotonoha.

Provides a transparent, always-on-top, click-through screen translation overlay
for real-time raw manga reading directly on desktop browsers or readers.
"""
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np

from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QGraphicsDropShadowEffect,
)

from src.core.models import MangaPage, SpeechBubble, BoundingBox
from src.ingestors.screen_cap import ScreenCaptureIngestor
from src.vision.detector import BubbleDetector
from src.vision.ocr import MangaOCR
from src.translation.llm_translator import LLMTranslator
from src.core.context_engine import ContextEngine


class SnipOverlayWidget(QWidget):
    """Full-screen transparent snip widget allowing user to drag a rectangle."""

    snip_completed = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._start_pos: Optional[QPoint] = None
        self._current_pos: Optional[QPoint] = None

    def start_snip(self) -> None:
        # Cover all available virtual desktop space
        self.showFullScreen()
        self._start_pos = None
        self._current_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        # Dim background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self._start_pos and self._current_pos:
            r = QRect(self._start_pos, self._current_pos).normalized()
            # Clear snipped area to look transparent
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(r, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Border around snip
            painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
            painter.drawRect(r)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.globalPosition().toPoint()
            self._current_pos = self._start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self._start_pos:
            self._current_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._start_pos and self._current_pos:
            r = QRect(self._start_pos, self._current_pos).normalized()
            self.hide()
            if r.width() > 20 and r.height() > 20:
                self.snip_completed.emit(r.x(), r.y(), r.width(), r.height())
            self._start_pos = None
            self._current_pos = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()


class FloatingOverlayWindow(QWidget):
    """Floating toolbar and overlay container for real-time desktop manga translation."""

    export_to_main = Signal(MangaPage, np.ndarray)

    def __init__(self, context_engine: ContextEngine):
        super().__init__()
        self.context_engine = context_engine
        self.detector = BubbleDetector()
        self.ocr = MangaOCR()
        self.translator = LLMTranslator()

        self.last_page: Optional[MangaPage] = None
        self.last_image: Optional[np.ndarray] = None
        self.snip_rect: Optional[Tuple[int, int, int, int]] = None

        self._drag_start: Optional[QPoint] = None
        self.is_click_through = False

        self.snip_widget = SnipOverlayWidget()
        self.snip_widget.snip_completed.connect(self.on_snip_done)

        self._init_ui()

    def _init_ui(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(380, 110)

        # Style sheet
        self.setStyleSheet("""
            QFrame#mainCard {
                background-color: rgba(22, 22, 28, 235);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
            }
            QLabel {
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QPushButton {
                background-color: #2D3748;
                color: #FFFFFF;
                border: 1px solid #4A5568;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4A5568;
                border-color: #00D2FF;
            }
            QPushButton#btnSnip {
                background-color: #3182CE;
                border-color: #63B3ED;
            }
            QPushButton#btnSnip:hover {
                background-color: #2B6CB0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        card = QFrame()
        card.setObjectName("mainCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)

        # Title bar
        title_row = QHBoxLayout()
        title_label = QLabel("<b>KOTONOHA</b> <font color='#00D2FF'>Floating Overlay</font>")
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(22, 22)
        btn_close.clicked.connect(self.hide)

        title_row.addWidget(title_label)
        title_row.addStretch()
        title_row.addWidget(btn_close)
        card_layout.addLayout(title_row)

        # Status
        self.status_label = QLabel("Click 'Snip Region' to select raw manga panel.")
        self.status_label.setStyleSheet("color: #A0AEC0; font-size: 11px;")
        card_layout.addWidget(self.status_label)

        # Controls row
        btn_row = QHBoxLayout()
        self.btn_snip = QPushButton("✂ Snip Region")
        self.btn_snip.setObjectName("btnSnip")
        self.btn_snip.clicked.connect(self.snip_widget.start_snip)

        self.btn_send_main = QPushButton("⎘ Send to Studio")
        self.btn_send_main.setEnabled(False)
        self.btn_send_main.clicked.connect(self.send_to_main_studio)

        btn_row.addWidget(self.btn_snip)
        btn_row.addWidget(self.btn_send_main)
        card_layout.addLayout(btn_row)

        layout.addWidget(card)

    def on_snip_done(self, x: int, y: int, w: int, h: int) -> None:
        self.snip_rect = (x, y, w, h)
        self.status_label.setText(f"Capturing ({w}x{h}). Detecting bubbles...")

        # 1. Grab screen
        page, img_bgr = ScreenCaptureIngestor.capture_region((x, y, w, h))

        # 2. Detect speech bubbles
        bubbles = self.detector.detect_bubbles(img_bgr)
        page.bubbles = bubbles

        # 3. OCR bubbles
        self.ocr.recognize_all_bubbles(img_bgr, bubbles)

        # 4. Context Translate
        self.translator.translate_page(page, self.context_engine)

        self.last_page = page
        self.last_image = img_bgr
        self.btn_send_main.setEnabled(True)

        self.status_label.setText(
            f"Translated {len(bubbles)} bubble(s)! Send to Studio to edit."
        )

    def send_to_main_studio(self) -> None:
        if self.last_page and self.last_image is not None:
            self.export_to_main.emit(self.last_page, self.last_image)
            self.hide()

    # Drag window by background
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_start:
            self.move(event.globalPosition().toPoint() - self._drag_start)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_start = None

