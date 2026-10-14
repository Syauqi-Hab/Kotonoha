"""Ingestors package initialization."""
from src.ingestors.pdf_reader import PDFReader
from src.ingestors.web_fetcher import WebMangaFetcher
from src.ingestors.screen_cap import ScreenCaptureIngestor

__all__ = [
    "PDFReader",
    "WebMangaFetcher",
    "ScreenCaptureIngestor",
]

