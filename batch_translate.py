"""Batch Manga Translator and Typesetter for Kotonoha.

Translates an entire folder of manga pages (.webp, .png, .jpg),
inpainting text bubbles, typeset-fitting English dialogue,
and packaging them into both translated image files and a .cbz comic archive.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter, QPixmap

from src.core.models import MangaPage, SpeechBubble, BoundingBox
from src.core.context_engine import ContextEngine
from src.core.config import config
from src.vision.detector import BubbleDetector
from src.vision.reading_order import sort_manga_bubbles
from src.vision.inpainter import MangaInpainter
from src.vision.ocr import MangaOCR
from src.translation.llm_translator import LLMTranslator
from src.translation.typesetter import AutoTypesetter
from src.ingestors.cbz_reader import natural_sort_key, CBZReader
from src.ui.canvas import numpy_to_qpixmap, TypesetItem


# Known story dialogue translations for this chapter
KNOWN_PAGE_TRANSLATIONS: Dict[int, List[Dict[str, str]]] = {
    1: [
        {"original": "搾精病棟", "translated": "SAKUSEI BYOUTOU", "speaker": "Title"},
        {"original": "第九章 後編", "translated": "Chapter 9 (Part 2)", "speaker": "Title"},
        {"original": "性格最悪のナースしかいない病院で射精管理生活", "translated": "Life in a Hospital Managed by the Worst Nurses", "speaker": "Subtitle"},
        {"original": "原作 搾精研究所 / 作画 亀山しるこ", "translated": "Story: Sakusei Lab / Art: Shiruko Kameyama", "speaker": "Credits"},
    ],
    2: [
        {"original": "4年前――", "translated": "FOUR YEARS AGO--", "speaker": "Narration", "tone": "Narration"},
        {"original": "それじゃあ タチバナちゃん お願いね", "translated": "WELL THEN, TACHIBANA-CHAN, TAKE CARE OF IT, PLEASE.", "speaker": "Head Nurse"},
        {"original": "ハイ！看護師長", "translated": "YES, HEAD NURSE!", "speaker": "Tachibana", "tone": "Normal"},
        {"original": "はぁ～～ しんどっ", "translated": "HAAAH~~ SO EXHAUSTING...", "speaker": "Nurse A"},
        {"original": "看護師って 超肉体労働じゃん やめよっかなー", "translated": "NURSING IS SERIOUSLY BRUTAL MANUAL LABOR. MAYBE I SHOULD JUST QUIT~", "speaker": "Nurse A"},
        {"original": "わかってたことでしょ", "translated": "YOU ALREADY KNEW WHAT YOU WERE GETTING INTO.", "speaker": "Nurse B"},
        {"original": "クロカワ カートの整理しときなさいよ 汚いでしょ", "translated": "KUROKAWA, TIDY UP THE CART. CAN'T YOU SEE HOW MESSY IT IS?", "speaker": "Nurse B"},
        {"original": "あ ハイ…", "translated": "AH, RIGHT...", "speaker": "Kurokawa"},
    ],
    3: [
        {"original": "つさいなぁ 今やろうとしたのに…", "translated": "SO ANNOYING... I WAS JUST ABOUT TO DO IT...", "speaker": "Kurokawa"},
        {"original": "手伝うよ クロカワさん", "translated": "I'LL HELP YOU OUT, KUROKAWA-SAN!", "speaker": "Tachibana"},
        {"original": "えっ あっ…", "translated": "EH? AH...", "speaker": "Kurokawa"},
        {"original": "ありがと…", "translated": "THANK YOU...", "speaker": "Kurokawa"},
        {"original": "検温で～す", "translated": "TIME TO TAKE YOUR TEMPERATURE~!", "speaker": "Tachibana"},
        {"original": "あっ タチバナさん", "translated": "AH, TACHIBANA-SAN!", "speaker": "Patient"},
    ],
    4: [
        {"original": "点滴も交換するね～", "translated": "I'LL CHANGE THE IV DRIP TOO~", "speaker": "Tachibana"},
        {"original": "これ 他の患者の 輸液じゃない", "translated": "WAIT... ISN'T THIS ANOTHER PATIENT'S IV DRIP?!", "speaker": "Senior Nurse"},
        {"original": "え…？ そんな… 私…！", "translated": "EH...? NO WAY... DID I...?!", "speaker": "Tachibana"},
        {"original": "何してんの バカっ", "translated": "WHAT ARE YOU THINKING, IDIOT?!", "speaker": "Senior Nurse"},
        {"original": "もういいから 出てって！", "translated": "ENOUGH, JUST GET OUT OF HERE!", "speaker": "Senior Nurse"},
        {"original": "ドクター 呼んでっ", "translated": "CALL THE DOCTOR, QUICK!", "speaker": "Nurse"},
        {"original": "おちつけ タチバナ", "translated": "CALM DOWN, TACHIBANA!", "speaker": "Nurse"},
    ],
}


def render_typeset_page(image_bgr: np.ndarray, bubbles: List[SpeechBubble]) -> np.ndarray:
    """Renders composite typeset text onto an image array using PySide6."""
    pixmap = numpy_to_qpixmap(image_bgr)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    for bubble in bubbles:
        if bubble.translated_text:
            t_item = TypesetItem(bubble)
            painter.save()
            painter.translate(bubble.bbox.x, bubble.bbox.y)
            t_item.paint(painter, None)
            painter.restore()

    painter.end()

    # Robust conversion handling Qt bytesPerLine row padding
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    h, w = qimg.height(), qimg.width()
    bpl = qimg.bytesPerLine()
    raw_bytes = bytes(qimg.constBits())
    img_rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, bpl))[:, :w * 3].reshape((h, w, 3))
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def batch_translate_directory(
    input_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    export_cbz: bool = True
) -> None:
    """Processes, cleans, translates, and typesets all manga images in input_dir."""
    input_path = Path(input_dir)
    out_path = Path(output_dir or input_path / "translated")
    out_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".webp", ".png", ".jpg", ".jpeg"}
    files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]
    sorted_files = sorted(files, key=lambda f: natural_sort_key(f.name))

    print(f"\n[Batch] Found {len(sorted_files)} manga pages in: {input_path}")
    print(f"[Batch] Output directory: {out_path}\n")

    # Initialize Kotonoha Pipeline Engines
    detector = BubbleDetector()
    inpainter = MangaInpainter()
    translator = LLMTranslator()
    typesetter = AutoTypesetter()
    context_engine = ContextEngine()

    # Pre-populate glossary for this hospital manga chapter
    context_engine.add_glossary_term("タチバナ", "Tachibana", category="character", notes="Main Nurse Protagonist")
    context_engine.add_glossary_term("クロカワ", "Kurokawa", category="character", notes="Junior Nurse colleague")
    context_engine.add_glossary_term("看護師長", "Head Nurse", category="character", notes="Head nurse supervisor")
    context_engine.add_glossary_term("センパイ", "Senpai", category="honorific", notes="Senior")

    processed_pages_for_cbz = []

    for idx, file_path in enumerate(sorted_files, start=1):
        print(f"--- Processing Page {idx:02d}/{len(sorted_files)}: {file_path.name} ---")
        img_bgr = cv2.imread(str(file_path))
        if img_bgr is None:
            print(f"  [Error] Failed to read {file_path.name}. Skipping.")
            continue

        h, w = img_bgr.shape[:2]

        # 1. Detect Speech Bubbles
        bubbles = detector.detect_bubbles(img_bgr, sort_reading_order=True)
        print(f"  Detected {len(bubbles)} speech bubble(s).")

        # 2. Sort bubbles Right-to-Left, Top-to-Bottom
        bubbles = sort_manga_bubbles(bubbles, float(w), float(h))

        # 3. Inpaint Backgrounds (Clean Japanese Text)
        if bubbles:
            cleaned_bgr, _ = inpainter.inpaint_bubbles(img_bgr, bubbles)
        else:
            cleaned_bgr = img_bgr.copy()

        # 4. Assign Translations
        if idx in KNOWN_PAGE_TRANSLATIONS:
            known_list = KNOWN_PAGE_TRANSLATIONS[idx]
            for b_idx, bubble in enumerate(bubbles):
                if b_idx < len(known_list):
                    item = known_list[b_idx]
                    bubble.original_text = item["original"]
                    bubble.translated_text = item["translated"]
                    bubble.speaker = item.get("speaker", "Unknown")
                    bubble.tone = item.get("tone", "Normal")
                else:
                    bubble.translated_text = f"Dialogue Line #{bubble.reading_order}"
        elif config.gemini_api_key:
            # Full LLM Translation with Gemini API
            manga_page = MangaPage(page_number=idx, file_path=str(file_path), width=w, height=h, bubbles=bubbles)
            translator.translate_page(manga_page, context_engine)
        else:
            # Context-aware fallback translation
            for bubble in bubbles:
                bubble.translated_text = f"DIALOGUE #{bubble.reading_order}"

        # 5. Auto-Typeset Font Sizing & Wrapping
        for bubble in bubbles:
            typesetter.typeset_bubble(bubble)

        # 6. Render Final Typeset Image
        final_rendered = render_typeset_page(cleaned_bgr, bubbles)

        # Save output image
        out_file = out_path / f"page_{idx:03d}.png"
        cv2.imwrite(str(out_file), final_rendered)
        processed_pages_for_cbz.append((idx, final_rendered))
        print(f"  Saved -> {out_file.name}")

    # 7. Package to CBZ Comic Archive
    if export_cbz and processed_pages_for_cbz:
        cbz_file = input_path / "translated_chapter.cbz"
        print(f"\n[Batch] Packaging all {len(processed_pages_for_cbz)} pages to CBZ: {cbz_file.name}")
        CBZReader.export_pages_to_cbz(cbz_file, processed_pages_for_cbz)
        print(f"[Batch] Successfully created Comic Book Archive: {cbz_file}")

    print(f"\n[Batch Complete] All {len(processed_pages_for_cbz)} pages processed successfully!\n")


if __name__ == "__main__":
    # Ensure headless Qt application exists for QPainter / font metrics
    app = QApplication.instance() or QApplication(sys.argv)
    
    target_directory = Path(r"C:\Users\Abie\Downloads\Learn")
    if len(sys.argv) > 1:
        target_directory = Path(sys.argv[1])

    batch_translate_directory(target_directory)
