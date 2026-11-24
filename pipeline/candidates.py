"""Candidate errors: flagged words (detector score >= threshold) merged into spans, with context.

Adjacent flagged tokens of the same segment become one candidate, since a wrongly joined or split word is flagged on several tokens.
A candidate addresses its words as (segment index, token range) in the raw segment text, so a fix replaces exactly those tokens.
"""
from dataclasses import asdict, dataclass

from .text import display_word
from .tokens import tokenize
from .windows import Window

CONTEXT = 50


@dataclass
class Candidate:
    id: int
    seg: int                    # index into Version.segments
    ref: str
    i1: int                     # token range [i1, i2) within pipeline.tokens.tokenize(segment text)
    i2: int
    c0: int                     # the same span as character offsets in the raw segment text
    c1: int
    words: str                  # the flagged tokens, raw
    score: float                # highest detector score in the span
    before: str                 # up to CONTEXT display words before the span (crossing segment boundaries)
    after: str

    def dict(self) -> dict:
        return asdict(self)

    def excerpt(self, n: int = 25) -> str:
        """Context with the span marked ⟦like this⟧, as the LLM judges see it."""
        b = self.before.split()[-n:]
        a = self.after.split()[:n]
        return " ".join(b + [f"⟦{display_word(self.words) or self.words}⟧"] + a)


class CandidateFinder:
    """Turns detector scores into candidates. Built once per version; `find` can be called on successive batches of windows
    (windows never split a candidate except inside a very long segment, where the span is simply cut at the window edge)."""

    def __init__(self, segments: list[tuple[str, str]], context: int = CONTEXT):
        self.segments = segments
        self.context = context
        self.spans = [tokenize(t) for _, t in segments]
        self.flat: list[str] = []
        self.flat_at: dict[tuple[int, int], int] = {}
        for s, ((_, text), sp) in enumerate(zip(segments, self.spans)):
            for i, (a, b) in enumerate(sp):
                self.flat_at[(s, i)] = len(self.flat)
                self.flat.append(display_word(text[a:b]))

    def find(self, windows: list[Window], scores: dict[int, list[float]], threshold: float, start_id: int = 0) -> list[Candidate]:
        flagged: dict[tuple[int, int], float] = {}
        for w in windows:
            for (s, i), x in zip(w.pos, scores[w.id]):
                if x >= threshold:
                    flagged[(s, i)] = x
        out: list[Candidate] = []
        for (s, i) in sorted(flagged):
            if out and out[-1].seg == s and out[-1].i2 == i:               # extend the previous span
                c = out[-1]
                c.i2 = i + 1
                c.score = max(c.score, flagged[(s, i)])
                continue
            out.append(Candidate(start_id + len(out), s, self.segments[s][0], i, i + 1, 0, 0, "", flagged[(s, i)], "", ""))
        flat, ctx = self.flat, self.context
        for c in out:
            c.c0, c.c1 = self.spans[c.seg][c.i1][0], self.spans[c.seg][c.i2 - 1][1]
            c.words = self.segments[c.seg][1][c.c0:c.c1]
            a, b = self.flat_at[(c.seg, c.i1)], self.flat_at[(c.seg, c.i2 - 1)] + 1
            c.before = " ".join(w for w in flat[max(0, a - ctx):a] if w)
            c.after = " ".join(w for w in flat[b:b + ctx] if w)
            c.score = round(c.score, 4)
        return out


def find_candidates(segments: list[tuple[str, str]], windows: list[Window], scores: dict[int, list[float]],
                    threshold: float, context: int = CONTEXT) -> list[Candidate]:
    return CandidateFinder(segments, context).find(windows, scores, threshold)
