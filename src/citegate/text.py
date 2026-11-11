"""Unicode-aware normalisation shared by retrieval, metrics and the reference agents.

English, French and Arabic are handled by one pipeline: NFKD, combining marks removed (French
accents, Arabic diacritics), Arabic-Indic digits mapped to ASCII, light Arabic letter folding.
"""

from __future__ import annotations

import re
import unicodedata

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_FOLD = str.maketrans({"ـ": None, "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    """Lower-case, strip accents and diacritics, fold Arabic variants and digits."""
    text = text.translate(_DIGITS)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.translate(_ARABIC_FOLD).lower()


_RAW_STOPWORDS = (
    "the a an is are of for at to in on and or be must can how what where does do i it by this "
    "that with from not "
    "le la les de du des un une et est sont pour au aux en dans que qui quel quelle quels "
    "combien ou il se ce par sur avec doivent faut ne pas je cette "
    "في من على إلى عن هي هو أن ما كم هل أو بخصوص"
)
STOPWORDS: frozenset[str] = frozenset(normalize(w) for w in _RAW_STOPWORDS.split())

_EN_HINTS = frozenset(
    [
        "the",
        "is",
        "for",
        "how",
        "what",
        "where",
        "can",
        "are",
        "at",
        "must",
        "takes",
        "does",
        "i",
        "a",
        "this",
        "in",
    ]
)
_FR_HINTS = frozenset(
    [
        "le",
        "la",
        "les",
        "du",
        "des",
        "est",
        "pour",
        "quel",
        "quels",
        "combien",
        "ou",
        "au",
        "une",
        "et",
        "sont",
        "doivent",
        "avoir",
        "prend",
        "faut",
        "je",
        "ne",
        "pas",
        "dans",
        "cette",
    ]
)


def tokens(text: str, *, drop_stop: bool = False) -> list[str]:
    """Normalised word tokens, optionally without stop-words."""
    found = _WORD.findall(normalize(text))
    return [t for t in found if t not in STOPWORDS] if drop_stop else found


def detect_lang(text: str) -> str:
    """Tiny script and stop-word language detector for ``en``, ``fr`` and ``ar``."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    arabic = sum(1 for ch in letters if "\u0600" <= ch <= "\u06ff")
    if arabic / len(letters) > 0.3:
        return "ar"
    toks = set(tokens(text))
    return "fr" if len(toks & _FR_HINTS) > len(toks & _EN_HINTS) else "en"
