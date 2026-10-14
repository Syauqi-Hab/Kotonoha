# Kotonoha (コトノハ) 📖✨
**Context-Aware Manga Translation & Typesetting Suite**

Kotonoha adalah aplikasi desktop profesional untuk menerjemahkan dan melakukan *typesetting* manga secara otomatis dan interaktif. Berbeda dengan translator standar perkata, Kotonoha memahami kontinuitas narasi, relasi karakter, honorifik Jepang (*-san*, *-kun*, *-senpai*), dan alur baca komik Jepang (Right-to-Left, Top-to-Bottom).

---

## 🎨 Arsitektur Non-Destructive 4-Layer (OpenToonz-Inspired)

Kotonoha menggunakan sistem canvas 4 lapis yang tidak merusak gambar asli (*non-destructive*):
1. **Layer 0: Base Layer (Source):** Gambar manga asli / panel PDF / hasil tangkapan layar.
2. **Layer 1: Inpaint Mask Layer:** Lapisan bersih yang menghapus teks bahasa Jepang menggunakan LaMa ONNX / OpenCV Telea tanpa merusak background.
3. **Layer 2: Bounding Box & OCR Layer:** Bentuk vektor interaktif penanda balon teks lengkap dengan urutan baca (*reading order badges* 1, 2, 3...) dan teks OCR.
4. **Layer 3: Typeset Text Overlay Layer:** Teks terjemahan yang dibungkus (*wrapped*) secara dinamis, auto-fit proporsional, dan di-styling dengan gaya manga typography (outline/stroke, bold, centered).

Setiap layer memiliki kontrol visibilitas (toggle mata) dan slider opasitas tersendiri di panel kanan.

---

## 🚀 Fitur Utama

- **Studio Canvas:** Pan, zoom halus (scroll wheel), seleksi balon teks, drag & resize handle vektor balon bicara.
- **Urutan Baca Manga Otomatis:** Algoritma pengurutan **Kanan ke Kiri (RTL)** dan **Atas ke Bawah (TTB)**.
- **Deteksi Balon Otomatis:** Deteksi kontur OpenCV pintar untuk menemukan balon teks.
- **Inpainting Teks:** Pembersihan teks berbasis LaMa ONNX dan smart fallback OpenCV Telea & Navier-Stokes.
- **Context-Stitching LLM Translation:** Menerjemahkan balon bicara per-halaman sekaligus dengan Google Gemini API, menjaga suara karakter, emosi (*whisper, scream, monologue*), dan kontinuitas cerita.
- **Session Glossary & NER Memory:** Panel memori istilah untuk konsistensi nama karakter dan honorifik.
- **Floating Screen Overlay Mode:** Mode overlay transparan (F9) untuk membaca dan menerjemahkan manga mentah (*raw*) langsung di browser atau aplikasi reader lain di layar komputer.
- **Multi-Format Ingestion:** Dukungan gambar (PNG, JPG, WEBP), Manga PDF beresolusi tinggi (PyMuPDF), Web URL fetcher (HTTPX), dan Screen Region Snipper.
- **Ekspor Non-Destructive:** Ekspor langsung halaman hasil typeset bersih siap rilis tanpa garis bantu vektor.

---

## 📦 Instalasi & Persiapan

1. **Clone / Buka Direktori:**
   ```bash
   cd "C:\Abie\Passion Project\Kotonoha"
   ```

2. **Instal Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Konfigurasi Gemini API Key (Opsional untuk LLM cerdas):**
   Setel environment variable `GEMINI_API_KEY`:
   - Windows PowerShell:
     ```powershell
     $env:GEMINI_API_KEY="AIzaSy..."
     ```
   - Atau buat file `.env` di root project:
     ```env
     GEMINI_API_KEY=AIzaSy...
     ```
   *(Catatan: Jika tanpa API key, Kotonoha tetap berjalan dengan mulus menggunakan offline demo translation mode)*.

---

## 🖥️ Cara Menjalankan

### 1. Menjalankan Studio Lengkap:
```bash
python main.py
```

### 2. Membuka File Tertentu Saat Startup:
```bash
python main.py --file "path/to/manga_page.png"
# atau
python main.py --file "path/to/manga_chapter.pdf"
```

### 3. Menjalankan Langsung Mode Floating Overlay:
```bash
python main.py --overlay
```

---

## 📁 Struktur Proyek
```
kotonoha/
├── assets/
│   ├── fonts/               # Font manga (Anime Ace, Wild Words, dll)
│   ├── icons/               # Ikon UI
│   └── models/              # Model LaMa ONNX inpainter
├── src/
│   ├── core/
│   │   ├── config.py        # Pengaturan aplikasi & model
│   │   ├── models.py        # Model Pydantic (SpeechBubble, MangaPage, Context)
│   │   └── context_engine.py# Session Glossary & Narrative Memory
│   ├── ingestors/
│   │   ├── pdf_reader.py    # PyMuPDF engine
│   │   ├── web_fetcher.py   # HTTPX async downloader
│   │   └── screen_cap.py    # Screen region grabber
│   ├── vision/
│   │   ├── detector.py      # OpenCV speech bubble contour detector
│   │   ├── reading_order.py # Manga RTL & TTB sorting algorithm
│   │   ├── inpainter.py     # LaMa ONNX & OpenCV inpainter
│   │   └── ocr.py           # Manga-OCR & Gemini Vision fallback
│   ├── translation/
│   │   ├── llm_translator.py# Structured JSON Gemini translation pipeline
│   │   └── typesetter.py    # Auto-fit font size & word wrapping
│   └── ui/
│       ├── canvas.py        # OpenToonz-style 4-layer PySide6 Canvas
│       ├── panels.py        # Layer Stack, Bubble Inspector, Glossary Panel
│       ├── overlay_window.py# Floating transparent screen snip & translate
│       └── main_window.py   # Main Studio Window
├── tests/
│   └── test_suite.py        # Unittest suite
├── requirements.txt
├── main.py
└── README.md
```

---

## 🧪 Menjalankan Pengujian (Tests)
```bash
python -m unittest tests/test_suite.py
```
Semua test (geometri bounding box, RTL reading order, auto-typesetting text-fitting, deteksi balon, dan inpainting) telah diverifikasi dan berjalan sukses.

