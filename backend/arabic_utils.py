import arabic_reshaper
from bidi.algorithm import get_display
import re

# Arabic Unicode block ranges
ARABIC_RANGES = [
    ("\u0600", "\u06FF"),  # Arabic
    ("\u0750", "\u077F"),  # Arabic Supplement
    ("\u08A0", "\u08FF"),  # Arabic Extended-A
    ("\uFB50", "\uFDFF"),  # Arabic Presentation Forms-A
    ("\uFE70", "\uFEFF"),  # Arabic Presentation Forms-B
]


def contains_arabic(text: str) -> bool:
    """Check if string contains Arabic characters."""
    if not text:
        return False
    return any(start <= ch <= end for start, end in ARABIC_RANGES for ch in text)


def reshape_for_display(text: str) -> str:
    """
    Reshape Arabic text for display (bidi-aware).
    Does NOT affect embeddings (keep embeddings raw).
    """
    if not contains_arabic(text):
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def clean_text(text: str) -> str:
    """Remove excessive whitespace/newlines for consistency."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()
 