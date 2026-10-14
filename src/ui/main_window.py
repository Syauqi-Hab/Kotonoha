"""Main Window for Kotonoha Manga Translation & Typesetting Suite.

Integrates:
- OpenToonz-style multi-layer canvas
- Document / Page navigator
- Layer stack manager
- Bubble inspector & typesetter
- Session context memory & glossary
- Ingestors (PDF, Web, Screen Snip, Images)
- Pipeline controls (Auto-detect, OCR, Inpaint, LLM Translate, Typeset)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap, QImage, QPainter
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QLabel,
)

from src.core.models import MangaPage, SpeechBubble, BoundingBox
from src.core.context_engine import ContextEngine
from src.core.config import config
from src.vision.detector import BubbleDetector
from src.vision.reading_order import sort_manga_bubbles
from src.vision.ocr import MangaOCR
from src.vision.inpainter import MangaInpainter
from src.translation.llm_translator import LLMTranslator
from src.translation.typesetter import AutoTypesetter
from src.ingestors.pdf_reader import PDFReader
from src.ingestors.cbz_reader import CBZReader
from src.ingestors.web_fetcher import WebMangaFetcher
from src.ui.canvas import MangaCanvas, numpy_to_qpixmap
from src.ui.panels import LayerControlPanel, BubbleInspectorPanel, SessionMemoryPanel
from src.ui.overlay_window import FloatingOverlayWindow


class MainWindow(QMainWindow):
    """Main Application Window for Kotonoha."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{config.app_name} - Manga Translation & Typesetting Suite")
        self.resize(1400, 900)

        # Core Engines
        self.context_engine = ContextEngine()
        self.detector = BubbleDetector()
        self.ocr = MangaOCR()
        self.inpainter = MangaInpainter()
        self.translator = LLMTranslator()
        self.typesetter = AutoTypesetter()
        self.web_fetcher = WebMangaFetcher()

        # Document State
        self.pages: List[MangaPage] = []
        self.page_images: List[np.ndarray] = []
        self.cleaned_images: List[Optional[np.ndarray]] = []
        self.current_page_idx: int = -1

        # Sub-windows
        self.overlay_window = FloatingOverlayWindow(self.context_engine)
        self.overlay_window.export_to_main.connect(self.on_overlay_page_received)

        self._init_ui()
        self._apply_theme()
        self._load_sample_page()

    def _init_ui(self) -> None:
        # Central Splitter Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. Left Sidebar: Page Navigator
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)

        left_header = QLabel("<b>PAGES NAVIGATOR</b>")
        self.page_list = QListWidget()
        self.page_list.setIconSize(QSize(80, 110))
        self.page_list.currentRowChanged.connect(self._on_page_selected)

        btn_add_page = QPushButton("+ Add Image Page")
        btn_add_page.clicked.connect(self.action_open_image)

        left_layout.addWidget(left_header)
        left_layout.addWidget(self.page_list)
        left_layout.addWidget(btn_add_page)
        main_splitter.addWidget(left_panel)

        # 2. Center: OpenToonz 4-Layer Canvas
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.canvas = MangaCanvas()
        self.canvas.bubble_selected.connect(self._on_bubble_selected)
        self.canvas.bubble_changed.connect(self._on_bubble_changed)
        self.canvas.bubble_created.connect(self._on_bubble_created)

        center_layout.addWidget(self.canvas)
        main_splitter.addWidget(center_panel)

        # 3. Right Sidebar: Inspector Panels
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.layer_panel = LayerControlPanel()
        self.layer_panel.visibility_changed.connect(self.canvas.set_layer_visibility)
        self.layer_panel.opacity_changed.connect(self.canvas.set_layer_opacity)

        self.inspector_panel = BubbleInspectorPanel()
        self.inspector_panel.bubble_updated.connect(self._on_bubble_inspected_update)
        self.inspector_panel.bubble_deleted.connect(self._on_bubble_deleted)

        self.memory_panel = SessionMemoryPanel(self.context_engine)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.layer_panel)
        right_splitter.addWidget(self.inspector_panel)
        right_splitter.addWidget(self.memory_panel)

        right_layout.addWidget(right_splitter)
        main_splitter.addWidget(right_panel)

        # Splitter sizing ratio
        main_splitter.setSizes([200, 850, 350])
        main_layout.addWidget(main_splitter)

        # Build Menu & Toolbar
        self._create_menus()
        self._create_toolbar()

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a manga image, PDF, or snip a screen region.")

    def _create_menus(self) -> None:
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        act_open_img = QAction("Open Manga Image...", self)
        act_open_img.setShortcut(QKeySequence.StandardKey.Open)
        act_open_img.triggered.connect(self.action_open_image)
        file_menu.addAction(act_open_img)

        act_open_pdf = QAction("Open Manga PDF...", self)
        act_open_pdf.triggered.connect(self.action_open_pdf)
        file_menu.addAction(act_open_pdf)

        act_open_cbz = QAction("Open Comic Archive (.cbz, .zip)...", self)
        act_open_cbz.triggered.connect(self.action_open_cbz)
        file_menu.addAction(act_open_cbz)

        act_fetch_web = QAction("Fetch from Web URL...", self)
        act_fetch_web.triggered.connect(self.action_fetch_web)
        file_menu.addAction(act_fetch_web)

        file_menu.addSeparator()

        act_overlay = QAction("Launch Floating Screen Overlay", self)
        act_overlay.setShortcut("F9")
        act_overlay.triggered.connect(self.action_toggle_overlay)
        file_menu.addAction(act_overlay)

        file_menu.addSeparator()

        act_export_page = QAction("Export Current Typeset Page...", self)
        act_export_page.setShortcut(QKeySequence.StandardKey.Save)
        act_export_page.triggered.connect(self.action_export_page)
        file_menu.addAction(act_export_page)

        act_export_cbz = QAction("Export All to CBZ Archive...", self)
        act_export_cbz.triggered.connect(self.action_export_cbz)
        file_menu.addAction(act_export_cbz)

        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Pipeline Menu
        pipe_menu = menubar.addMenu("&Pipeline")

        act_auto_all = QAction("⚡ Run 1-Click Translation Pipeline", self)
        act_auto_all.setShortcut("F5")
        act_auto_all.triggered.connect(self.action_run_full_pipeline)
        pipe_menu.addAction(act_auto_all)

        pipe_menu.addSeparator()

        act_detect = QAction("1. Detect Speech Bubbles", self)
        act_detect.triggered.connect(self.action_detect_bubbles)
        pipe_menu.addAction(act_detect)

        act_sort = QAction("2. Sort Manga Reading Order (RTL)", self)
        act_sort.triggered.connect(self.action_sort_bubbles)
        pipe_menu.addAction(act_sort)

        act_ocr = QAction("3. Run Japanese OCR", self)
        act_ocr.triggered.connect(self.action_run_ocr)
        pipe_menu.addAction(act_ocr)

        act_inpaint = QAction("4. Inpaint Text Backgrounds", self)
        act_inpaint.triggered.connect(self.action_inpaint_page)
        pipe_menu.addAction(act_inpaint)

        act_translate = QAction("5. Translate Dialogue with LLM", self)
        act_translate.triggered.connect(self.action_translate_page)
        pipe_menu.addAction(act_translate)

        act_typeset = QAction("6. Auto-Typeset Translated Text", self)
        act_typeset.triggered.connect(self.action_typeset_page)
        pipe_menu.addAction(act_typeset)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Tool actions
        self.act_tool_select = QAction("Pointer (Select)", self)
        self.act_tool_select.setCheckable(True)
        self.act_tool_select.setChecked(True)
        self.act_tool_select.triggered.connect(lambda: self._set_tool("select"))
        toolbar.addAction(self.act_tool_select)

        self.act_tool_box = QAction("Draw Bubble Box", self)
        self.act_tool_box.setCheckable(True)
        self.act_tool_box.triggered.connect(lambda: self._set_tool("add_box"))
        toolbar.addAction(self.act_tool_box)

        toolbar.addSeparator()

        # One-Click Full Pipeline Action
        act_pipeline = QAction("⚡ 1-Click Translate Page", self)
        act_pipeline.triggered.connect(self.action_run_full_pipeline)
        toolbar.addAction(act_pipeline)

        toolbar.addSeparator()

        act_inpaint = QAction("Clean / Inpaint", self)
        act_inpaint.triggered.connect(self.action_inpaint_page)
        toolbar.addAction(act_inpaint)

        act_sort = QAction("Sort RTL Order", self)
        act_sort.triggered.connect(self.action_sort_bubbles)
        toolbar.addAction(act_sort)

        toolbar.addSeparator()

        act_overlay = QAction("Screen Snip Overlay", self)
        act_overlay.triggered.connect(self.action_toggle_overlay)
        toolbar.addAction(act_overlay)

    def _set_tool(self, tool_name: str) -> None:
        self.canvas.active_tool = tool_name
        self.act_tool_select.setChecked(tool_name == "select")
        self.act_tool_box.setChecked(tool_name == "add_box")
        self.status_bar.showMessage(f"Active Tool: {tool_name.replace('_', ' ').title()}")

    def _apply_theme(self) -> None:
        """Applies OpenToonz-inspired dark studio palette."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1A1A20;
            }
            QToolBar {
                background-color: #24242C;
                border-bottom: 1px solid #33333E;
                padding: 4px;
                spacing: 6px;
            }
            QToolBar QToolButton {
                background-color: #2D2D38;
                color: #FFFFFF;
                border: 1px solid #3E3E4C;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QToolBar QToolButton:hover {
                background-color: #3E3E50;
                border-color: #00D2FF;
            }
            QToolBar QToolButton:checked {
                background-color: #0088CC;
                border-color: #00D2FF;
            }
            QSplitter::handle {
                background-color: #2B2B36;
            }
            QListWidget {
                background-color: #1E1E26;
                color: #E2E8F0;
                border: 1px solid #2D2D3A;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background-color: #2B6CB0;
                color: #FFFFFF;
            }
            QGroupBox {
                color: #A0AEC0;
                font-weight: bold;
                border: 1px solid #2D2D3A;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #252530;
                color: #FFFFFF;
                border: 1px solid #3A3A4C;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #3182CE;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2B6CB0;
            }
            QStatusBar {
                background-color: #18181E;
                color: #A0AEC0;
                border-top: 1px solid #2D2D38;
            }
        """)

    # Page Management
    def add_page(self, page: MangaPage, image_bgr: np.ndarray, cleaned_bgr: Optional[np.ndarray] = None) -> None:
        self.pages.append(page)
        self.page_images.append(image_bgr)
        self.cleaned_images.append(cleaned_bgr)

        # Add thumbnail item
        item = QListWidgetItem(f"Page {page.page_number}")
        # Generate thumbnail
        thumb_h, thumb_w = image_bgr.shape[:2]
        scale = 100.0 / max(thumb_h, thumb_w)
        thumb = cv2.resize(image_bgr, (int(thumb_w * scale), int(thumb_h * scale)))
        item.setIcon(QIcon(numpy_to_qpixmap(thumb)))
        self.page_list.addItem(item)

        if self.current_page_idx == -1:
            self.page_list.setCurrentRow(0)

    def _on_page_selected(self, index: int) -> None:
        if 0 <= index < len(self.pages):
            self.current_page_idx = index
            page = self.pages[index]
            base_img = self.page_images[index]
            clean_img = self.cleaned_images[index]
            self.canvas.load_page(page, base_img, clean_img)
            self.inspector_panel.load_bubble(None)
            self.status_bar.showMessage(f"Loaded Page {page.page_number} ({page.width}x{page.height} px)")

    # Pipeline Actions
    def action_detect_bubbles(self) -> None:
        """Detect speech bubbles in current page."""
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]
        image = self.page_images[self.current_page_idx]

        self.status_bar.showMessage("Detecting speech bubbles...")
        bubbles = self.detector.detect_bubbles(image, sort_reading_order=True)
        page.bubbles = bubbles
        self.canvas.refresh_bubbles()
        self.status_bar.showMessage(f"Detected {len(bubbles)} speech bubble(s).")

    def action_sort_bubbles(self) -> None:
        """Sort bubbles Right-to-Left, Top-to-Bottom."""
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]
        page.bubbles = sort_manga_bubbles(page.bubbles, float(page.width), float(page.height))
        self.canvas.refresh_bubbles()
        self.status_bar.showMessage("Sorted speech bubbles in Manga reading order (RTL, TTB).")

    def action_run_ocr(self) -> None:
        """Recognize Japanese text."""
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]
        image = self.page_images[self.current_page_idx]

        self.status_bar.showMessage("Running OCR on bubbles...")
        self.ocr.recognize_all_bubbles(image, page.bubbles)
        self.canvas.refresh_bubbles()
        self.status_bar.showMessage(f"OCR completed for {len(page.bubbles)} bubbles.")

    def action_inpaint_page(self) -> None:
        """Clean Japanese text background."""
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]
        image = self.page_images[self.current_page_idx]

        self.status_bar.showMessage("Inpainting speech bubbles...")
        cleaned, _ = self.inpainter.inpaint_bubbles(image, page.bubbles)
        self.cleaned_images[self.current_page_idx] = cleaned
        page.is_inpainted = True
        self.canvas.inpaint_layer_item.setPixmap(numpy_to_qpixmap(cleaned))
        self.canvas.inpaint_layer_item.setVisible(True)
        self.status_bar.showMessage("Inpainting completed.")

    def action_translate_page(self) -> None:
        """Context-aware LLM translation."""
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]

        self.status_bar.showMessage("Translating dialogue with Gemini Context LLM...")
        self.translator.translate_page(page, self.context_engine)
        self.action_typeset_page()
        self.memory_panel.refresh_ui()
        self.status_bar.showMessage(f"Translated page {page.page_number} with narrative continuity.")

    def action_typeset_page(self) -> None:
        """Auto-fit and typeset translated text."""
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]

        for bubble in page.bubbles:
            self.typesetter.typeset_bubble(bubble)

        self.canvas.refresh_bubbles()
        self.status_bar.showMessage("Typesetting applied.")

    def action_run_full_pipeline(self) -> None:
        """One-click complete translation suite."""
        if self.current_page_idx == -1:
            QMessageBox.information(self, "No Page Loaded", "Please load a manga page first.")
            return

        self.status_bar.showMessage("Executing 1-Click Translation Pipeline...")
        page = self.pages[self.current_page_idx]
        image = self.page_images[self.current_page_idx]

        # 1. Detect if empty
        if not page.bubbles:
            page.bubbles = self.detector.detect_bubbles(image, sort_reading_order=True)

        # 2. Sort reading order
        page.bubbles = sort_manga_bubbles(page.bubbles, float(page.width), float(page.height))

        # 3. OCR
        self.ocr.recognize_all_bubbles(image, page.bubbles)

        # 4. Inpaint
        cleaned, _ = self.inpainter.inpaint_bubbles(image, page.bubbles)
        self.cleaned_images[self.current_page_idx] = cleaned
        page.is_inpainted = True
        self.canvas.inpaint_layer_item.setPixmap(numpy_to_qpixmap(cleaned))
        self.canvas.inpaint_layer_item.setVisible(True)

        # 5. Translate
        self.translator.translate_page(page, self.context_engine)

        # 6. Typeset
        for bubble in page.bubbles:
            self.typesetter.typeset_bubble(bubble)

        self.canvas.refresh_bubbles()
        self.memory_panel.refresh_ui()
        self.status_bar.showMessage(f"Pipeline complete! Translated {len(page.bubbles)} bubbles on Page {page.page_number}.")

    # File Ingestion Handlers
    def action_open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Manga Image", "", "Manga Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                h, w = img.shape[:2]
                page = MangaPage(
                    page_number=len(self.pages) + 1,
                    file_path=file_path,
                    width=w,
                    height=h,
                    source_type="image"
                )
                self.add_page(page, img)

    def action_open_pdf(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Manga PDF", "", "PDF Manga Documents (*.pdf)"
        )
        if file_path:
            try:
                reader = PDFReader(file_path)
                for i in range(reader.page_count):
                    page, img = reader.load_manga_page(i)
                    self.add_page(page, img)
                self.status_bar.showMessage(f"Imported {reader.page_count} pages from PDF.")
            except Exception as e:
                QMessageBox.critical(self, "PDF Error", f"Failed to load PDF: {e}")

    def action_open_cbz(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Comic Archive", "", "Comic Archives (*.cbz *.zip)"
        )
        if file_path:
            try:
                reader = CBZReader(file_path)
                for i in range(reader.page_count):
                    page, img = reader.load_manga_page(i)
                    self.add_page(page, img)
                self.status_bar.showMessage(f"Imported {reader.page_count} pages from CBZ archive.")
            except Exception as e:
                QMessageBox.critical(self, "CBZ Error", f"Failed to load CBZ: {e}")

    def action_fetch_web(self) -> None:
        url, ok = QInputDialog.getText(self, "Fetch Web Manga", "Enter Manga Image Panel URL:")
        if ok and url.strip():
            try:
                page, img = self.web_fetcher.fetch_page_from_url(url.strip(), page_number=len(self.pages) + 1)
                self.add_page(page, img)
                self.status_bar.showMessage(f"Imported manga page from: {url}")
            except Exception as e:
                QMessageBox.critical(self, "Web Fetch Error", f"Failed to fetch manga from URL: {e}")

    def action_toggle_overlay(self) -> None:
        if self.overlay_window.isVisible():
            self.overlay_window.hide()
        else:
            self.overlay_window.show()

    def on_overlay_page_received(self, page: MangaPage, img_bgr: np.ndarray) -> None:
        page.page_number = len(self.pages) + 1
        self.add_page(page, img_bgr)
        self.action_typeset_page()
        self.status_bar.showMessage("Received and typeset screen capture page from Overlay Mode!")

    def action_export_page(self) -> None:
        """Renders final non-destructive composite (Cleaned/Base + Typeset text) to file."""
        if self.current_page_idx == -1:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Typeset Page", f"kotonoha_page_{self.current_page_idx + 1}.png", "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if not file_path:
            return

        page = self.pages[self.current_page_idx]
        base_img = self.cleaned_images[self.current_page_idx]
        if base_img is None:
            base_img = self.page_images[self.current_page_idx]

        # Render QPixmap composition
        pixmap = numpy_to_qpixmap(base_img)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Draw typeset layer items
        for item in self.canvas.typeset_group.childItems():
            item.paint(painter, None)

        painter.end()
        pixmap.save(file_path)
        QMessageBox.information(self, "Export Successful", f"Saved typeset manga page to:\n{file_path}")

    def _render_page_composite(self, page_idx: int) -> np.ndarray:
        """Renders cleaned/base image + typeset overlay into an RGB/BGR numpy array."""
        base_img = self.cleaned_images[page_idx]
        if base_img is None:
            base_img = self.page_images[page_idx]

        pixmap = numpy_to_qpixmap(base_img)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        if page_idx == self.current_page_idx:
            for item in self.canvas.typeset_group.childItems():
                item.paint(painter, None)
        else:
            from src.ui.canvas import TypesetItem
            for bubble in self.pages[page_idx].bubbles:
                if bubble.translated_text:
                    t_item = TypesetItem(bubble)
                    painter.save()
                    painter.translate(bubble.bbox.x, bubble.bbox.y)
                    t_item.paint(painter, None)
                    painter.restore()

        painter.end()

        qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_BGR888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        arr = np.array(ptr, dtype=np.uint8).reshape((h, w, 3))
        return arr.copy()

    def action_export_cbz(self) -> None:
        """Exports all loaded pages as a standardized .cbz comic archive."""
        if not self.pages:
            QMessageBox.information(self, "No Pages", "No manga pages loaded to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Pages to CBZ Archive", "manga_chapter.cbz", "Comic Book Zip (*.cbz)"
        )
        if not file_path:
            return

        pages_data = []
        for idx, page in enumerate(self.pages):
            rendered_bgr = self._render_page_composite(idx)
            pages_data.append((page.page_number, rendered_bgr))

        try:
            CBZReader.export_pages_to_cbz(file_path, pages_data)
            QMessageBox.information(
                self, "CBZ Export Successful",
                f"Successfully exported {len(pages_data)} translated pages to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CBZ: {e}")

    # Canvas Interaction Callbacks
    def _on_bubble_selected(self, bubble_id: str) -> None:
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]
        bubble = page.get_bubble_by_id(bubble_id)
        self.inspector_panel.load_bubble(bubble)

    def _on_bubble_changed(self, bubble: SpeechBubble) -> None:
        self.inspector_panel.load_bubble(bubble)

    def _on_bubble_created(self, bubble: SpeechBubble) -> None:
        self.inspector_panel.load_bubble(bubble)
        self.status_bar.showMessage(f"Created new speech bubble #{bubble.reading_order}")

    def _on_bubble_inspected_update(self, bubble: SpeechBubble) -> None:
        self.typesetter.typeset_bubble(bubble)
        self.canvas.refresh_bubbles()
        self.canvas.select_bubble_by_id(bubble.id)

    def _on_bubble_deleted(self, bubble_id: str) -> None:
        if self.current_page_idx == -1:
            return
        page = self.pages[self.current_page_idx]
        page.bubbles = [b for b in page.bubbles if b.id != bubble_id]
        # Renumber
        for idx, b in enumerate(page.bubbles, start=1):
            b.reading_order = idx
        self.canvas.refresh_bubbles()
        self.status_bar.showMessage("Deleted speech bubble.")

    def _load_sample_page(self) -> None:
        """Generates a default manga sample page for immediate preview."""
        h, w = 1200, 800
        # Light manga paper background
        sample_img = np.full((h, w, 3), 245, dtype=np.uint8)

        # Draw manga panel borders (OpenCV)
        cv2.rectangle(sample_img, (50, 50), (w - 50, int(h * 0.48)), (20, 20, 20), 4)
        cv2.rectangle(sample_img, (50, int(h * 0.52)), (w - 50, h - 50), (20, 20, 20), 4)

        # Panel 1: Top Right Bubble (Reading Order #1)
        cv2.ellipse(sample_img, (600, 180), (120, 90), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(sample_img, (600, 180), (120, 90), 0, 0, 360, (20, 20, 20), 3)

        # Panel 1: Top Left Bubble (Reading Order #2)
        cv2.ellipse(sample_img, (220, 260), (110, 80), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(sample_img, (220, 260), (110, 80), 0, 0, 360, (20, 20, 20), 3)

        # Panel 2: Bottom Bubble (Reading Order #3)
        cv2.ellipse(sample_img, (400, 800), (150, 100), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(sample_img, (400, 800), (150, 100), 0, 0, 360, (20, 20, 20), 3)

        # Pre-seed sample bubbles
        b1 = SpeechBubble(
            bbox=BoundingBox(x=490, y=100, width=220, height=160),
            reading_order=1,
            original_text="待って、センパイ！",
            translated_text="Wait, Senpai!",
            speaker="Kohai",
            tone="Scream"
        )
        b2 = SpeechBubble(
            bbox=BoundingBox(x=120, y=190, width=200, height=140),
            reading_order=2,
            original_text="何だ、もう時間がないぞ。",
            translated_text="What is it? We're out of time.",
            speaker="Senpai",
            tone="Normal"
        )
        b3 = SpeechBubble(
            bbox=BoundingBox(x=260, y=710, width=280, height=180),
            reading_order=3,
            original_text="これを受け取ってください…！",
            translated_text="Please accept this...!",
            speaker="Kohai",
            tone="Whisper"
        )

        sample_page = MangaPage(
            page_number=1,
            file_path="sample_demo_page.png",
            width=w,
            height=h,
            bubbles=[b1, b2, b3],
            source_type="image"
        )

        # Pre-seed glossary
        self.context_engine.add_glossary_term("センパイ", "Senpai", category="honorific", notes="Upperclassman")
        self.context_engine.add_glossary_term("コウハイ", "Kohai", category="character", notes="Junior student")

        self.add_page(sample_page, sample_img)
        self.action_typeset_page()

