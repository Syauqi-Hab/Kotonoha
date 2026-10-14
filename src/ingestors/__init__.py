"""Ingestors package initialization."""
from src.ingestors.pdf_reader import PDFReader
from src.ingestors.web_fetcher import WebMangaFetcher
from src.ingestors.screen_cap import ScreenCaptureIngestor
from src.ingestors.cbz_reader import CBZReader

__all__ = [
    "PDFReader",
    "WebMangaFetcher",
    "ScreenCaptureIngestor",
    "CBZReader",
]
