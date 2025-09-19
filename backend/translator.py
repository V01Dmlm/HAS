# backend/translator.py
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import logging
from typing import List, Union

# Fix random seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger("Translator")
logging.basicConfig(level=logging.INFO)

class Translator:
    def __init__(self, silent: bool = False):
        """
        Initialize English ↔ Arabic translator with robust handling.
        silent: if True, suppress warnings (good for production)
        """
        self.silent = silent
        self.to_english = GoogleTranslator(source='auto', target='en')
        self.to_arabic = GoogleTranslator(source='en', target='ar')
        self.cache = {}  # simple in-memory cache to prevent duplicate translations

    def translate_to_english(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Translate text to English only if not already English.
        Returns original text on failure or if text is too short.
        Supports batch translation.
        """
        if isinstance(text, list):
            return [self.translate_to_english(t) for t in text]

        t = text.strip() if text else ""
        if not t or len(t) < 2:
            return text

        if t in self.cache:
            return self.cache[t]

        try:
            lang = detect(t)
            if lang != 'en':
                translated = self.to_english.translate(t)
                self.cache[t] = translated
                return translated
            return t
        except Exception as e:
            if not self.silent:
                logger.warning(f"⚠️ Translation to English failed: {e}")
            return text

    def translate_to_arabic(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Translate English text to Arabic.
        Returns original text on failure or if text is too short.
        Supports batch translation.
        """
        if isinstance(text, list):
            return [self.translate_to_arabic(t) for t in text]

        t = text.strip() if text else ""
        if not t or len(t) < 2:
            return text

        if t in self.cache:
            return self.cache[t]

        try:
            translated = self.to_arabic.translate(t)
            self.cache[t] = translated
            return translated
        except Exception as e:
            if not self.silent:
                logger.warning(f"⚠️ Translation to Arabic failed: {e}")
            return text

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        Returns language code (e.g., 'en', 'ar').
        Defaults to 'en' if detection fails.
        """
        t = text.strip() if text else ""
        if not t:
            return "en"
        try:
            return detect(t)
        except Exception as e:
            if not self.silent:
                logger.warning(f"⚠️ Language detection failed: {e}")
            return "en"
