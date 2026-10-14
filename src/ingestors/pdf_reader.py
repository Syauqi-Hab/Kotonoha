"""PDF Manga Ingestor using PyMuPDF (fitz).

Renders PDF manga pages into high-resolution images for the Base Canvas.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from src.core.models import MangaPage


class PDFReader:
    """Extracts and renders manga pages from PDF files."""

    def __init__(self, file_path: Path | str, dpi: int = 200):
        self.file_path = Path(file_path)
        self.dpi = dpi
        self.zoom = dpi / 72.0  # Default PDF resolution is 72 points per inch
        self.doc = None
        self._open_document()

    def _open_document(self) -> None:
        """Opens the PDF document with PyMuPDF."""
        try:
            import fitz
            self.doc = fitz.open(str(self.file_path))
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is not installed. Please run: pip install PyMuPDF")
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF file {self.file_path}: {e}")

    @property
    def page_count(self) -> int:
        return len(self.doc) if self.doc else 0

    def render_page_to_numpy(self, page_index: int) -> np.ndarray:
        """Renders a PDF page to a BGR numpy array suitable for OpenCV / Canvas."""
        if not self.doc or page_index < 0 or page_index >= len(self.doc):
            raise IndexError(f"Page index {page_index} out of range (0..{self.page_count - 1})")

        import fitz
        page = self.doc[page_index]
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # PyMuPDF samples are RGB
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Convert RGB to BGR for OpenCV consistency
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    def load_manga_page(self, page_index: int) -> Tuple[MangaPage, np.ndarray]:
        """Loads a PDF page and constructs a MangaPage model."""
        img_bgr = self.render_page_to_numpy(page_index)
        h, w = img_bgr.shape[:2]

        manga_page = MangaPage(
            page_number=page_index + 1,
            file_path=str(self.file_path),
            width=w,
            height=h,
            source_type="pdf",
        )
        return manga_page, img_bgr

    def close(self) -> None:
        if self.doc:
            self.doc.close()
            self.doc = None

