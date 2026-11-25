"""Letter pairs that OCR confuses: the allowlist for automatic fixes against a reference.

Only for OCR'd texts. A typed text's errors come from the keyboard, not from letter shapes: TYPED_PAIRS is where keyboard slips will go
(empty for now, so typed texts get no letter-swap fixes).

For OCR, two sources, per script (the project's typeface: square or Rashi):
- hand-picked pairs: TYPICAL for every script, plus RASHI_SCRIPT for Rashi-script projects;
- pairs whose letters look alike, from letter_similarity.json (experiments/letter_similarity.py), scoring at least GLYPH_CUTOFF. These
  include two-for-one confusions such as וז/ח.

A pair is unordered: ב/כ allows ב→כ and כ→ב.
"""
import json
from functools import lru_cache
from pathlib import Path

SCRIPTS = ("square", "rashi")
SOURCES = ("ocr", "typed", "mixed")      # mixed: source unclear, OCR and typing/conversion patterns both
TYPED_PAIRS: list[str] = []
TYPICAL = ["ה/ח", "ד/ר", "ס/ם", "י/ו", "ב/כ", "ג/נ", "ו/ז", "ת/ח", "ו/ן"]
RASHI_SCRIPT = ["א/ח", "ל/נ", "ו/נ", "א/מ", "צ/ל"]
GLYPH_CUTOFF = 0.70
SIMILARITY = Path(__file__).with_name("letter_similarity.json")


def pair_key(a: str, b: str) -> str:
    return "/".join(sorted((a, b)))


def _keys(pairs: list[str]) -> set[str]:
    return {pair_key(*p.split("/")) for p in pairs}


@lru_cache(maxsize=None)
def similarity() -> dict[str, dict[str, float]]:
    return json.loads(SIMILARITY.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def allowed_pairs(script: str = "square", source: str = "ocr") -> frozenset[str]:
    if script not in SCRIPTS:
        raise ValueError(f"unknown script {script!r}; use one of {', '.join(SCRIPTS)}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; use one of {', '.join(SOURCES)}")
    if source == "typed":
        return frozenset(_keys(TYPED_PAIRS))
    hand = _keys(TYPICAL + (RASHI_SCRIPT if script == "rashi" else []))
    glyph = {k for k, v in similarity()[script].items() if v >= GLYPH_CUTOFF}
    return frozenset(hand | glyph)
