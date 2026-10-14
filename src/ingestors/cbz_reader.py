"""CBZ (Comic Book Zip) Manga Ingestor & Exporter.

Reads and extracts manga pages directly from .cbz and .zip comic archives,
with natural page number sorting and in-memory image decoding.
Also supports exporting translated chapters back to .cbz archives.
"""
from __future__ import annotations
import re
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.core.models import MangaPage


def natural_sort_key(s: str):
    """Sort strings containing numbers in human order (e.g., page 2 before page 10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class CBZReader:
    """Reads pages from a .cbz or .zip comic archive."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}

    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self.image_entries: List[str] = []
        self._inspect_archive()

    def _inspect_archive(self) -> None:
        """Finds all valid manga image files in archive, sorted naturally."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CBZ file not found: {self.file_path}")

        if not zipfile.is_zipfile(str(self.file_path)):
            raise ValueError(f"File is not a valid zip/cbz archive: {self.file_path}")

        with zipfile.ZipFile(self.file_path, "r") as zf:
            all_files = zf.namelist()
            valid_images = [
                f for f in all_files
                if not f.startswith("__MACOSX/") and not f.startswith(".")
                and Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            self.image_entries = sorted(valid_images, key=natural_sort_key)

    @property
    def page_count(self) -> int:
        return len(self.image_entries)

    def read_page_to_numpy(self, page_index: int) -> np.ndarray:
        """Decodes an image from the CBZ archive into a BGR numpy array."""
        if page_index < 0 or page_index >= len(self.image_entries):
            raise IndexError(f"Page index {page_index} out of range (0..{self.page_count - 1})")

        entry_name = self.image_entries[page_index]
        with zipfile.ZipFile(self.file_path, "r") as zf:
            image_bytes = zf.read(entry_name)

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError(f"Failed to decode image from archive entry: {entry_name}")

        return img_bgr

    def load_manga_page(self, page_index: int) -> Tuple[MangaPage, np.ndarray]:
        """Loads a page from the archive and constructs a MangaPage model."""
        img_bgr = self.read_page_to_numpy(page_index)
        h, w = img_bgr.shape[:2]
        entry_name = self.image_entries[page_index]

        manga_page = MangaPage(
            page_number=page_index + 1,
            file_path=f"{self.file_path.name}#{entry_name}",
            width=w,
            height=h,
            source_type="cbz"
        )
        return manga_page, img_bgr

    @staticmethod
    def export_pages_to_cbz(
        output_cbz_path: Path | str,
        pages_data: List[Tuple[int, np.ndarray]]
    ) -> None:
        """Exports a list of (page_number, image_bgr) as a standardized .cbz archive."""
        out_path = Path(output_cbz_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for page_num, img_bgr in pages_data:
                filename = f"page_{page_num:03d}.png"
                success, buffer = cv2.imencode(".png", img_bgr)
                if success:
                    zf.writestr(filename, buffer.tobytes())
