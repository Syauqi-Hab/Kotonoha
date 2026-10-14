"""OpenToonz-Inspired Multi-Layer Non-Destructive Canvas for Kotonoha.

Features:
- Layer 0: Base Source Image (QGraphicsPixmapItem)
- Layer 1: Inpaint / Cleaned Background (QGraphicsPixmapItem)
- Layer 2: Bounding Box & OCR Vectors (QGraphicsItemGroup with resize handles & order badges)
- Layer 3: Typeset Text Overlay (QGraphicsItemGroup with manga outline typography)
- Smooth Pan & Zoom, Layer Opacity & Visibility control
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainterPath,
    QCursor,
    QTransform,
    QWheelEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsPixmapItem,
    QGraphicsItemGroup,
    QGraphicsTextItem,
    QGraphicsPathItem,
    QWidget,
)

from src.core.models import MangaPage, SpeechBubble, BoundingBox, FontStyle
from src.vision.reading_order import renumber_bubbles


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Converts OpenCV BGR or BGRA numpy array to QPixmap."""
    if img is None or img.size == 0:
        return QPixmap()

    if len(img.shape) == 2:  # Grayscale
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    elif len(img.shape) == 3:
        h, w, ch = img.shape
        if ch == 3:  # BGR
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)
        elif ch == 4:  # BGRA
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            return QPixmap.fromImage(qimg)

    return QPixmap()


class HandleItem(QGraphicsRectItem):
    """Interactive resize handle on the corners of a bounding box."""

    def __init__(self, parent_box: "BubbleGraphicsItem", corner_id: str):
        super().__init__(-5, -5, 10, 10, parent=parent_box)
        self.parent_box = parent_box
        self.corner_id = corner_id  # 'tl', 'tr', 'bl', 'br'
        self.setBrush(QBrush(QColor(50, 150, 255)))
        self.setPen(QPen(QColor(255, 255, 255), 1.5))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(10)

        # Set appropriate cursor
        if corner_id in ("tl", "br"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.parent_box.handle_moved(self.corner_id, self.pos())


class BubbleGraphicsItem(QGraphicsRectItem):
    """Interactive vector bounding box representing a manga speech bubble."""

    def __init__(self, bubble: SpeechBubble, canvas: "MangaCanvas"):
        super().__init__(bubble.bbox.x, bubble.bbox.y, bubble.bbox.width, bubble.bbox.height)
        self.bubble = bubble
        self.canvas = canvas

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Normal styling
        self.default_pen = QPen(QColor(0, 160, 255, 200), 2.0, Qt.PenStyle.DashLine)
        self.selected_pen = QPen(QColor(255, 170, 0, 255), 2.5, Qt.PenStyle.SolidLine)
        self.hover_pen = QPen(QColor(0, 220, 255, 255), 2.0, Qt.PenStyle.SolidLine)
        self.setPen(self.default_pen)
        self.setBrush(QBrush(QColor(0, 160, 255, 25)))

        # Create 4 corner resize handles
        self.handles: Dict[str, HandleItem] = {
            "tl": HandleItem(self, "tl"),
            "tr": HandleItem(self, "tr"),
            "bl": HandleItem(self, "bl"),
            "br": HandleItem(self, "br"),
        }
        self.update_handle_positions()
        self.set_handles_visible(False)

    def set_handles_visible(self, visible: bool) -> None:
        for handle in self.handles.values():
            handle.setVisible(visible)

    def update_handle_positions(self) -> None:
        r = self.rect()
        self.handles["tl"].setPos(r.topLeft())
        self.handles["tr"].setPos(r.topRight())
        self.handles["bl"].setPos(r.bottomLeft())
        self.handles["br"].setPos(r.bottomRight())

    def handle_moved(self, corner_id: str, new_pos: QPointF) -> None:
        r = self.rect()
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()

        if corner_id == "tl":
            x1 = min(new_pos.x(), x2 - 10)
            y1 = min(new_pos.y(), y2 - 10)
        elif corner_id == "tr":
            x2 = max(new_pos.x(), x1 + 10)
            y1 = min(new_pos.y(), y2 - 10)
        elif corner_id == "bl":
            x1 = min(new_pos.x(), x2 - 10)
            y2 = max(new_pos.y(), y1 + 10)
        elif corner_id == "br":
            x2 = max(new_pos.x(), x1 + 10)
            y2 = max(new_pos.y(), y1 + 10)

        self.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.update_handle_positions()
        self.sync_model_bbox()

    def sync_model_bbox(self) -> None:
        scene_rect = self.sceneBoundingRect()
        self.bubble.bbox.x = scene_rect.x()
        self.bubble.bbox.y = scene_rect.y()
        self.bubble.bbox.width = scene_rect.width()
        self.bubble.bbox.height = scene_rect.height()
        self.canvas.bubble_changed.emit(self.bubble)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)

        # Draw Reading Order Badge at top-right
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r = self.rect()
        badge_radius = 12
        badge_center = QPointF(r.right() - badge_radius - 2, r.top() + badge_radius + 2)

        # Badge circle
        badge_color = QColor(255, 140, 0) if self.isSelected() else QColor(30, 144, 255)
        painter.setBrush(QBrush(badge_color))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(badge_center, badge_radius, badge_radius)

        # Badge text (Reading order number)
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        badge_text = str(self.bubble.reading_order)
        painter.drawText(
            QRectF(badge_center.x() - badge_radius, badge_center.y() - badge_radius, badge_radius * 2, badge_radius * 2),
            Qt.AlignmentFlag.AlignCenter,
            badge_text
        )

        painter.restore()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            is_selected = bool(value)
            self.setPen(self.selected_pen if is_selected else self.default_pen)
            self.set_handles_visible(is_selected)
            if is_selected:
                self.canvas.bubble_selected.emit(self.bubble.id)
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.sync_model_bbox()
        return super().itemChange(change, value)


class TypesetItem(QGraphicsItem):
    """Renders manga-styled typeset text with dynamic outlines and wrapping."""

    def __init__(self, bubble: SpeechBubble):
        super().__init__()
        self.bubble = bubble
        self.setPos(bubble.bbox.x, bubble.bbox.y)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.bubble.bbox.width, self.bubble.bbox.height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        text = self.bubble.translated_text.strip()
        if not text:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        style = self.bubble.font_style
        font = QFont(style.font_family, int(style.font_size))
        font.setBold(style.bold)
        font.setItalic(style.italic)
        painter.setFont(font)

        fm = QFontMetrics(font)
        display_text = text.upper() if style.all_caps else text

        rect = self.boundingRect()
        # Inset safe margin
        pad_x = rect.width() * 0.12
        pad_y = rect.height() * 0.12
        safe_rect = rect.adjusted(pad_x, pad_y, -pad_x, -pad_y)

        # Wrap text into lines
        words = display_text.split()
        lines: List[str] = []
        current_line: List[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            if fm.horizontalAdvance(test_line) <= safe_rect.width():
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []

        if current_line:
            lines.append(" ".join(current_line))

        # Calculate vertical centering
        line_height = int(fm.height() * style.line_spacing)
        total_height = len(lines) * line_height
        start_y = safe_rect.top() + max(0.0, (safe_rect.height() - total_height) / 2.0) + fm.ascent()

        outline_color = QColor(style.outline_color)
        fill_color = QColor(style.color)

        for i, line in enumerate(lines):
            line_w = fm.horizontalAdvance(line)
            # Alignment
            if style.alignment == "center":
                start_x = safe_rect.left() + max(0.0, (safe_rect.width() - line_w) / 2.0)
            elif style.alignment == "right":
                start_x = safe_rect.right() - line_w
            else:
                start_x = safe_rect.left()

            y = start_y + i * line_height

            path = QPainterPath()
            path.addText(QPointF(start_x, y), font, line)

            # Draw outline stroke
            if style.outline_width > 0:
                pen = QPen(outline_color, style.outline_width * 2)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.strokePath(path, pen)

            # Draw fill
            painter.fillPath(path, QBrush(fill_color))

        painter.restore()


class MangaCanvas(QGraphicsView):
    """OpenToonz-style 4-layer non-destructive manga composition canvas."""

    bubble_selected = Signal(str)
    bubble_changed = Signal(SpeechBubble)
    bubble_created = Signal(SpeechBubble)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # 4 Discrete Layers
        self.base_layer_item = QGraphicsPixmapItem()
        self.inpaint_layer_item = QGraphicsPixmapItem()
        self.boxes_group = QGraphicsItemGroup()
        self.typeset_group = QGraphicsItemGroup()

        # Z-Stack (OpenToonz order: Base -> Inpaint -> Boxes -> Typeset)
        self.base_layer_item.setZValue(0)
        self.inpaint_layer_item.setZValue(1)
        self.boxes_group.setZValue(2)
        self.typeset_group.setZValue(3)

        self.scene.addItem(self.base_layer_item)
        self.scene.addItem(self.inpaint_layer_item)
        self.scene.addItem(self.boxes_group)
        self.scene.addItem(self.typeset_group)

        # Canvas View Settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(QColor(28, 28, 34)))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # State
        self.current_page: Optional[MangaPage] = None
        self.base_image_np: Optional[np.ndarray] = None
        self.cleaned_image_np: Optional[np.ndarray] = None
        self.active_tool: str = "select"  # "select", "add_box"
        self._is_panning = False
        self._pan_start = QPointF()
        self._new_box_start: Optional[QPointF] = None
        self._temp_rect_item: Optional[QGraphicsRectItem] = None

    # Layer Visibility & Opacity
    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        if layer_name == "base":
            self.base_layer_item.setVisible(visible)
        elif layer_name == "inpaint":
            self.inpaint_layer_item.setVisible(visible)
        elif layer_name == "boxes":
            self.boxes_group.setVisible(visible)
        elif layer_name == "typeset":
            self.typeset_group.setVisible(visible)

    def set_layer_opacity(self, layer_name: str, opacity: float) -> None:
        opacity = max(0.0, min(1.0, opacity))
        if layer_name == "base":
            self.base_layer_item.setOpacity(opacity)
        elif layer_name == "inpaint":
            self.inpaint_layer_item.setOpacity(opacity)
        elif layer_name == "boxes":
            self.boxes_group.setOpacity(opacity)
        elif layer_name == "typeset":
            self.typeset_group.setOpacity(opacity)

    def load_page(self, page: MangaPage, base_image: np.ndarray, cleaned_image: Optional[np.ndarray] = None) -> None:
        """Loads a MangaPage into the canvas and renders all 4 layers."""
        self.current_page = page
        self.base_image_np = base_image
        self.cleaned_image_np = cleaned_image

        # 1. Base Layer
        base_pixmap = numpy_to_qpixmap(base_image)
        self.base_layer_item.setPixmap(base_pixmap)
        self.scene.setSceneRect(0, 0, base_pixmap.width(), base_pixmap.height())

        # 2. Inpaint Layer
        if cleaned_image is not None:
            self.inpaint_layer_item.setPixmap(numpy_to_qpixmap(cleaned_image))
            self.inpaint_layer_item.setVisible(True)
        else:
            self.inpaint_layer_item.setPixmap(QPixmap())
            self.inpaint_layer_item.setVisible(False)

        # 3. Vector Bounding Boxes & 4. Typeset Text
        self.refresh_bubbles()
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def refresh_bubbles(self) -> None:
        """Re-draws all vector bubble boxes and typeset text items."""
        # Clear existing items
        for item in list(self.boxes_group.childItems()):
            self.scene.removeItem(item)
        for item in list(self.typeset_group.childItems()):
            self.scene.removeItem(item)

        if not self.current_page:
            return

        for bubble in self.current_page.bubbles:
            # Box Item
            box_item = BubbleGraphicsItem(bubble, self)
            self.boxes_group.addToGroup(box_item)

            # Typeset Item
            if bubble.translated_text:
                text_item = TypesetItem(bubble)
                self.typeset_group.addToGroup(text_item)

    def select_bubble_by_id(self, bubble_id: str) -> None:
        """Programmatically select a bubble."""
        for item in self.boxes_group.childItems():
            if isinstance(item, BubbleGraphicsItem):
                if item.bubble.id == bubble_id:
                    item.setSelected(True)
                else:
                    item.setSelected(False)

    # Mouse & Pan/Zoom Controls
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Smooth zooming with mouse wheel."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Middle Click or Space+LeftClick initiates Pan
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self.active_tool == "add_box" and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self._new_box_start = scene_pos
            self._temp_rect_item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
            self._temp_rect_item.setPen(QPen(QColor(0, 255, 120), 2, Qt.PenStyle.DashLine))
            self.scene.addItem(self._temp_rect_item)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self.active_tool == "add_box" and self._new_box_start and self._temp_rect_item:
            current_scene = self.mapToScene(event.pos())
            rect = QRectF(self._new_box_start, current_scene).normalized()
            self._temp_rect_item.setRect(rect)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._is_panning and (event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton):
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if self.active_tool == "add_box" and self._new_box_start and self._temp_rect_item:
            current_scene = self.mapToScene(event.pos())
            rect = QRectF(self._new_box_start, current_scene).normalized()
            self.scene.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
            self._new_box_start = None

            # Only create if box has a minimum size (e.g. 15x15)
            if rect.width() >= 15 and rect.height() >= 15 and self.current_page:
                new_bubble = SpeechBubble(
                    bbox=BoundingBox(
                        x=rect.x(),
                        y=rect.y(),
                        width=rect.width(),
                        height=rect.height()
                    ),
                    reading_order=len(self.current_page.bubbles) + 1
                )
                self.current_page.bubbles.append(new_bubble)
                self.refresh_bubbles()
                self.select_bubble_by_id(new_bubble.id)
                self.bubble_created.emit(new_bubble)

            event.accept()
            return

        super().mouseReleaseEvent(event)

