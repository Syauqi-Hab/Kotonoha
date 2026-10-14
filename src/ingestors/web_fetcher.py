"""Web Manga Fetcher using HTTPX.

Downloads manga panels and pages from web URLs, decoding them directly into
MangaPage structures and image arrays.
"""
from __future__ import annotations
from typing import Tuple, List, Optional
import cv2
import numpy as np
import httpx
from src.core.models import MangaPage


class WebMangaFetcher:
    """Asynchronous and synchronous manga panel fetcher from web URLs."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": "https://manga.example.com/",
    }

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def fetch_page_from_url(self, url: str, page_number: int = 1) -> Tuple[MangaPage, np.ndarray]:
        """Synchronously downloads an image from a URL and converts to MangaPage."""
        with httpx.Client(headers=self.DEFAULT_HEADERS, timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            image_bytes = response.content

        # Decode image from buffer
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError(f"Failed to decode image from URL: {url}")

        h, w = img_bgr.shape[:2]
        page = MangaPage(
            page_number=page_number,
            file_path=url,
            width=w,
            height=h,
            source_type="web"
        )
        return page, img_bgr

    async def fetch_page_from_url_async(self, url: str, page_number: int = 1) -> Tuple[MangaPage, np.ndarray]:
        """Asynchronously downloads an image from a URL and converts to MangaPage."""
        async with httpx.AsyncClient(headers=self.DEFAULT_HEADERS, timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            image_bytes = response.content

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError(f"Failed to decode image from URL: {url}")

        h, w = img_bgr.shape[:2]
        page = MangaPage(
            page_number=page_number,
            file_path=url,
            width=w,
            height=h,
            source_type="web"
        )
        return page, img_bgr

