"""Kotonoha (コトノハ) - Context-Aware Manga Translation & Typesetting Suite.

Application Entry Point.
"""
import sys
import argparse
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from src.core.config import config
from src.ui.main_window import MainWindow
from src.ui.overlay_window import FloatingOverlayWindow
from src.core.context_engine import ContextEngine


def main():
    parser = argparse.ArgumentParser(description="Kotonoha - Manga Translation & Typesetting Suite")
    parser.add_argument("--overlay", action="store_true", help="Launch directly into Floating Screen Overlay mode")
    parser.add_argument("--file", type=str, default=None, help="Manga image or PDF to load upon launch")
    args = parser.parse_args()

    # Enable High DPI pixmaps
    app = QApplication(sys.argv)
    app.setApplicationName(config.app_name)
    app.setApplicationVersion(config.version)

    context_engine = ContextEngine()

    if args.overlay:
        window = FloatingOverlayWindow(context_engine)
        window.show()
    else:
        window = MainWindow()
        window.show()

        if args.file and Path(args.file).exists():
            file_path = str(Path(args.file).resolve())
            ext = Path(file_path).suffix.lower()
            try:
                if ext == ".pdf":
                    from src.ingestors.pdf_reader import PDFReader
                    reader = PDFReader(file_path)
                    for i in range(reader.page_count):
                        page, img = reader.load_manga_page(i)
                        window.add_page(page, img)
                elif ext in (".cbz", ".zip"):
                    from src.ingestors.cbz_reader import CBZReader
                    reader = CBZReader(file_path)
                    for i in range(reader.page_count):
                        page, img = reader.load_manga_page(i)
                        window.add_page(page, img)
                else:
                    import cv2
                    from src.core.models import MangaPage
                    img = cv2.imread(file_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        page = MangaPage(page_number=1, file_path=file_path, width=w, height=h)
                        window.add_page(page, img)
            except Exception as e:
                print(f"[Main] Error loading file {file_path}: {e}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

