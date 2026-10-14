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
            ext = Path(args.file).suffix.lower()
            if ext == ".pdf":
                window.action_open_pdf()
            else:
                window.action_open_image()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

