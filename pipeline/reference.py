"""Compare candidates with an uploaded reference text (another edition, a scan transcription, a manuscript transcription).

1. Load: adapters by file extension turn the upload into plain text. Plain text is the main case; others are best effort.
2. Index: the reference's words, reduced to letters only (pipeline.text.match_key), indexed by 4-grams.
3. Locate: the 4-grams of ~15 words of Sefaria context on each side of the candidate vote for where the candidate sits in the reference;
   a location just after the previous candidate's breaks ties, since candidates come in book order.
4. Align: word-level SequenceMatcher between the Sefaria window and the reference region maps the flagged words to reference words.

Outcomes: "agrees" (same letters: the detector was probably wrong), "differs" (the reference reading becomes the proposed fix),
"not_found".
"""
import html
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .text import TAG, display_word, match_key

N = 4
SIDE = 15
MIN_VOTES = 2


# ---- loading -------------------------------------------------------------------------------------------------------------------------

def _utf16(data: bytes) -> str | None:
    """The UTF-16 codec, if the bytes look like UTF-16: a BOM, or (BOM-less) nearly every high byte is 0x00 or 0x05 (ASCII or Hebrew; no
    text in UTF-8 or cp1255 has NUL or 0x05 at every other byte). Trying UTF-16 blindly is wrong: it decodes almost any even-length bytes without error, e.g. a cp1255 file."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    sample = data[:4000]
    if len(data) % 2 or len(sample) < 2:
        return None
    for enc, hi in (("utf-16-le", sample[1::2]), ("utf-16-be", sample[0::2])):
        if sum(b in (0x00, 0x05) for b in hi) > 0.9 * len(hi):
            return enc
    return None


def _txt(data: bytes) -> str:
    """UTF-16 when the bytes look like it (checked first: ASCII in UTF-16 is also valid UTF-8), then UTF-8, then Windows Hebrew (cp1255)."""
    encs = [_utf16(data), "utf-8-sig", "cp1255"]
    for enc in filter(None, encs):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _html(data: bytes) -> str:
    t = _txt(data)
    t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>", "\n", t)
    return html.unescape(TAG.sub(" ", t))


def _docx(data: bytes) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)


def _csv(data: bytes) -> str:
    """A Sefaria version CSV (or any CSV): the last column of each row after any header rows."""
    import csv
    rows = list(csv.reader(io.StringIO(_txt(data))))
    start = 5 if rows and rows[0] and rows[0][0].strip().lower() == "index title" else 0
    return "\n".join(r[-1] for r in rows[start:] if r)


LOADERS = {".txt": _txt, ".text": _txt, ".md": _txt, ".html": _html, ".htm": _html, ".docx": _docx, ".pdf": _pdf, ".csv": _csv}


def load_text(filename: str, data: bytes) -> str:
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ".txt"
    if ext not in LOADERS:
        raise ValueError(f"unsupported reference file type {ext}; use one of {', '.join(sorted(LOADERS))}")
    return LOADERS[ext](data)


# ---- alignment -----------------------------------------------------------------------------------------------------------------------

@dataclass
class Alignment:
    status: str                       # "agrees" | "differs" | "not_found"
    ref_words: str = ""               # the reference words aligned to the flagged span (display form)
    excerpt: str = ""                 # reference context with those words marked ⟦…⟧ (⟦⟧ alone when the reference has nothing there)
    position: Optional[int] = None    # reference word index of the span start
    votes: int = 0


class Reference:
    def __init__(self, text: str):
        self.words = [w for w in (display_word(t) for t in text.split()) if w]
        keys = [match_key(w) for w in self.words]
        self.kept = [i for i, k in enumerate(keys) if k]          # word indices with letters
        self.keys = [keys[i] for i in self.kept]
        self.index: dict[tuple, list[int]] = defaultdict(list)
        for p in range(len(self.keys) - N + 1):
            self.index[tuple(self.keys[p:p + N])].append(p)
        self.last: Optional[int] = None                           # position of the previous located candidate (in key space)

    def _locate(self, before: list[str], after: list[str], span_len: int) -> tuple[Optional[int], int]:
        votes: Counter = Counter()
        for side, offset0 in ((before, -len(before)), (after, span_len)):
            for j in range(len(side) - N + 1):
                gram = tuple(side[j:j + N])
                hits = self.index.get(gram, ())
                if len(hits) > 50:                                  # formulaic phrase: no information
                    continue
                for p in hits:
                    votes[(p - (offset0 + j)) // 3] += 1            # bucket: tolerate small insertions/deletions
        if not votes:
            return None, 0
        best = max(votes.values())
        cands = [b for b, v in votes.items() if v == best]
        if self.last is not None and len(cands) > 1:
            cands.sort(key=lambda b: (b * 3 < self.last, abs(b * 3 - self.last)))
        return cands[0] * 3, best

    def align(self, before: str, span: str, after: str) -> Alignment:
        b = [k for k in (match_key(w) for w in before.split()) if k][-SIDE:]
        a = [k for k in (match_key(w) for w in after.split()) if k][:SIDE]
        s = [k for k in (match_key(w) for w in display_word(span).split()) if k]
        anchor, votes = self._locate(b, a, len(s))
        if anchor is None or votes < MIN_VOTES:
            return Alignment("not_found", votes=votes)
        lo = max(0, anchor - len(b) - 10)
        hi = min(len(self.keys), anchor + len(s) + len(a) + 10)
        mine = b + s + a
        sm = SequenceMatcher(None, mine, self.keys[lo:hi], autojunk=False)
        s0, s1 = len(b), len(b) + len(s)
        r0 = r1 = None
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            # reference range covering mine[s0:s1]
            if i2 <= s0 or i1 >= s1:
                if tag == "insert" and i1 == s0 == s1:              # empty span (punctuation only)
                    r0 = lo + j1 if r0 is None else r0; r1 = lo + j2
                continue
            if tag == "equal":
                j_start = j1 + max(0, s0 - i1); j_end = j1 + min(i2, s1) - i1
            else:
                j_start, j_end = j1, j2
            r0 = lo + j_start if r0 is None else min(r0, lo + j_start)
            r1 = lo + j_end if r1 is None else max(r1, lo + j_end)
        if r0 is None:                                              # nothing aligned to the span: reference omits it
            # place an empty marker at the boundary after the matched `before`
            r0 = r1 = lo + next((j2 for tag, i1, i2, j1, j2 in sm.get_opcodes() if i2 >= s0), 0)
        assert r1 is not None                                       # r0 and r1 are always set together
        self.last = r0
        ref_keys = self.keys[r0:r1]
        w0 = self.kept[r0] if r0 < len(self.kept) else len(self.words)
        w1 = self.kept[r1 - 1] + 1 if r1 > r0 else w0
        ref_words = " ".join(self.words[w0:w1])
        ctx = 20
        marked = self.words[max(0, w0 - ctx):w0] + [f"⟦{ref_words}⟧"] + self.words[w1:w1 + ctx]
        status = "agrees" if "".join(ref_keys) == "".join(s) and s else "differs"
        return Alignment(status, ref_words, " ".join(marked), w0, votes)
