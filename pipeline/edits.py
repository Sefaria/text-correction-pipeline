"""Applying accepted fixes to segment text.

A candidate addresses its words by character offsets [c0, c1) in the ORIGINAL segment text. Several fixes in one segment are applied
together, right to left, on the original text, so one fix never shifts another's offsets. A reviewer's free-text override replaces the
whole segment and supersedes span fixes in that segment.
"""
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class SpanFix:
    c0: int
    c1: int
    text: str            # replacement for original[c0:c1]


def apply_fixes(original: str, fixes: Iterable[SpanFix]) -> str:
    """Apply non-overlapping span fixes to `original`. Overlapping fixes: the one starting later wins; the earlier is dropped."""
    out = original
    last_start: Optional[int] = None
    for f in sorted(fixes, key=lambda f: (f.c0, f.c1), reverse=True):
        if last_start is not None and f.c1 > last_start:
            continue
        out = out[:f.c0] + f.text + out[f.c1:]
        last_start = f.c0
    return _tidy(out)


def _tidy(text: str) -> str:
    """Deleting a word can leave a doubled space; collapse runs of spaces (not other whitespace) and strip the ends."""
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def replacement_for(original_words: str, fix: str) -> str:
    """The text that replaces a flagged span, keeping markup the fix left out: if the flagged raw text was '<b>שמנים</b>' and the fix is
    'שמנים', keep the tags around it. Only leading and trailing tag runs are carried over."""
    import re
    m = re.match(r"^((?:<[^>]+>)*)(.*?)((?:<[^>]+>)*)$", original_words, re.S)
    if not m or "<" in fix:
        return fix
    lead, _, trail = m.groups()
    return f"{lead}{fix}{trail}"
