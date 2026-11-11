from __future__ import annotations

from citegate.text import detect_lang, normalize, tokens


def test_normalize_folds_digits_accents_and_arabic_variants() -> None:
    assert normalize("٢٥") == "25"
    assert normalize("Où ÊTRE") == "ou etre"
    assert normalize("أين") == normalize("اين")
    assert normalize("مَدرَسة") == normalize("مدرسه")


def test_tokens_drop_stopwords() -> None:
    assert tokens("The fee for passport", drop_stop=True) == ["fee", "passport"]


def test_detect_lang() -> None:
    assert detect_lang("How much is the fee?") == "en"
    assert detect_lang("Quel est le tarif pour le passeport ?") == "fr"
    assert detect_lang("كم رسوم تجديد جواز السفر؟") == "ar"
    assert detect_lang("1234") == "en"
