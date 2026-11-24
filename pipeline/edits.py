"""Applying accepted fixes to segment text.

A candidate addresses its words by character offsets [c0, c1) in the ORIGINAL segment text. Several fixes in one segment are applied
together, right to left, on the original text, so one fix never shifts another's offsets. A reviewer's free-text override replaces the
whole segment and supersedes span fixes in that segment.
"""
import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class SpanFix:
    c0: int
    c1: int
    text: str            # replacement for original[c0:c1]


def apply_fixes(original: str, fixes: Iterable[SpanFix]) -> str:
    """Apply non-overlapping span fixes to `original`. Overlapping fixes: the one starting later wins; the earlier is dropped.
    Only the edited spots change: a deletion drops one of the spaces it would leave doubled, the rest of the text is kept byte for byte."""
    out = original
    last_start: Optional[int] = None
    for f in sorted(fixes, key=lambda f: (f.c0, f.c1), reverse=True):
        if last_start is not None and f.c1 > last_start:
            continue
        before, after = out[:f.c0], out[f.c1:]
        if f.text == "":
            if before.endswith(" ") and (after.startswith(" ") or not after):
                before = before[:-1]
            elif not before and after.startswith(" "):
                after = after[1:]
        out = before + f.text + after
        last_start = f.c0
    return out


_EDGES = re.compile(r"^((?:<[^>]+>)*)(.*?)((?:<[^>]+>)*)$", re.S)


def replacement_for(original_words: str, fix: str) -> Optional[str]:
    """The text that replaces a flagged span, keeping markup the fix left out: if the flagged raw text was '<b>שמנים</b>' and the fix is
    'שמנים', keep the tags around it. Only leading and trailing tag runs are carried over. A span with markup inside it (e.g.
    '<b>שתים</b> שלש') has no safe place to put the fix, since dropping or keeping the inner tags can unbalance the HTML: None, and the
    reviewer edits the segment text instead. A fix that contains markup itself is used as written."""
    if "<" in fix:
        return fix
    m = _EDGES.match(original_words)
    if not m:
        return fix
    lead, middle, trail = m.groups()
    if "<" in middle:
        return None
    return f"{lead}{fix}{trail}"
