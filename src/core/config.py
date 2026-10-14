"""Core configuration and settings for Kotonoha."""
import os
from pathlib import Path
from pydantic import BaseModel, Field

# Root directory of the application
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
ICONS_DIR = ASSETS_DIR / "icons"
CACHE_DIR = PROJECT_ROOT / ".kotonoha_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)
ICONS_DIR.mkdir(parents=True, exist_ok=True)

class AppConfig(BaseModel):
    """Global configuration settings for Kotonoha."""
    app_name: str = "Kotonoha (コトノハ)"
    version: str = "1.0.0"
    
    # Translation LLM settings
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    llm_model: str = "gemini-2.5-flash"
    source_lang: str = "Japanese"
    target_lang: str = "English"
    
    # Typesetting & Fonts
    default_font_family: str = "Anime Ace"
    fallback_font_family: str = "Comic Sans MS"
    default_font_size: float = 14.0
    default_text_color: str = "#000000"
    default_outline_color: str = "#FFFFFF"
    default_outline_width: float = 2.0
    
    # Inpainting model
    lama_model_path: Path = ASSETS_DIR / "models" / "lama.onnx"
    use_opencv_fallback: bool = True
    
    # Reading Order Rule: "rtl" (Right-to-Left, Top-to-Bottom)
    reading_order: str = "rtl"

# Global instance
config = AppConfig()

