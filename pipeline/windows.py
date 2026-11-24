"""Windows of consecutive segments, as the detector saw them in training.

Ported from translation-engine experiments/version_quality_from_history/scope_sample.py `windows()`: consecutive segments of one section
are joined into windows of up to WINDOW words; a longer segment is cut into WINDOW-word pieces. Tokens are pipeline.tokens.words() of
each segment (whitespace chunks, also split after <br>), so (segment index, token index) maps straight back to the raw text, HTML included.
"""
from dataclasses import dataclass

from .tokens import words

WINDOW = 350


@dataclass
class Window:
    id: int
    tokens: list[str]
    pos: list[tuple[int, int]]          # for each token: (segment index in Version.segments, token index within that segment)


def section_of(ref: str) -> str:
    """The section a segment ref belongs to: the ref without its last address component."""
    if ":" in ref:
        return ref.rsplit(":", 1)[0]
    return ref.rsplit(" ", 1)[0]


def build_windows(segments: list[tuple[str, str]], window: int = WINDOW) -> list[Window]:
    out: list[Window] = []
    cur_t: list[str] = []
    cur_p: list[tuple[int, int]] = []
    sec = None

    def flush():
        nonlocal cur_t, cur_p
        if cur_t:
            out.append(Window(len(out), cur_t, cur_p))
        cur_t, cur_p = [], []

    for s, (ref, text) in enumerate(segments):
        toks = words(text)
        if not toks:
            continue
        if cur_t and (section_of(ref) != sec or len(cur_t) + len(toks) > window):
            flush()
        sec = section_of(ref)
        start = 0
        while len(toks) - start > window:              # a very long segment: cut it into pieces
            flush()
            cur_t, cur_p = toks[start:start + window], [(s, i) for i in range(start, start + window)]
            flush()
            start += window
        cur_t += toks[start:]
        cur_p += [(s, i) for i in range(start, len(toks))]
    flush()
    return out
