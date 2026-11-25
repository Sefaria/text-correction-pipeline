"""Automatic fixes against a reference text: OCR letter swaps and word spacing; and the letter-swap map of a version.

1. Fix (`auto_fix`): the flagged Sefaria words and the reference reading have the same letters except for substitutions of up to 2
   letters for up to 2 letters, each one an allowed OCR pair (pipeline.letter_pairs: hand-picked plus look-alike letters, per script),
   and/or the words are divided differently (לאבא / לא בא). The fix is Sefaria's text with just those letters replaced and the reference's
   word division. Punctuation, gershayim and markup stay Sefaria's. Final forms follow the new word division. Anything else (letters
   added or dropped, a pair that is not allowed, niqqud in the span) gets no automatic fix.
2. Gershayim (`gershayim_moved`): a proposed fix that only moves a geresh/gershayim (ר״הי → רה״י) is not worth making.
3. Map (`build_map`): the whole version aligned with the reference block by block, counting the letter substitutions between words the
   alignment pairs up. Shown to the reviewer as evidence of which swaps the version has; the fix rule uses the allowlist, not the counts
   (frequent substitutions include spelling and censorship differences: יהודא/יהודה, עכו״ם/גוי).
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import accumulate
from typing import Optional

from .letter_pairs import pair_key
from .reference import Reference
from .text import MAQAF, NIQQUD, TAG, match_key, model_word
from .tokens import tokenize, words as tokens

MAX_SIDE = 2            # letters on either side of one substitution
MAX_SUBS = 2            # substitutions in one word pair for the map to count it as an OCR swap
BLOCK = 60              # Sefaria words per alignment block
SLACK = 10              # extra reference words on each side of a block
MIN_BLOCK_VOTES = 3
EXAMPLES = 3
STORED_PAIRS = 40       # most frequent pairs kept in the stored map; the rest only count towards the total

_HEB = re.compile("[א-ת]")
_BOUNDARY = re.compile(rf"[\s{MAQAF}]")
_FOLD = str.maketrans("ךםןףץ", "כמנפצ")
_UNFOLD = str.maketrans("כמנפצ", "ךםןףץ")
_GERSH = str.maketrans({'"': "״", "'": "׳"})


def letters(text: str) -> str:
    """Hebrew letters only, final forms kept."""
    return "".join(_HEB.findall(model_word(text).replace(" ", "")))


YOD_VAV = frozenset("וי")
ANY_LETTER = frozenset("אבגדהוזחטיכךלמםנןסעפףצץקרשת")


def _sm_ops(a: str, b: str, indel: frozenset = frozenset()) -> Optional[list[tuple]]:
    """The diff's alignment of letters `a` with `b`, if it is substitutions only (see _alignments)."""
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(("equal", i1, i2, j1, j2))
            continue
        if tag in ("insert", "delete") and len(a[i1:i2] + b[j1:j2]) == 1 and (a[i1:i2] + b[j1:j2]) in indel:
            out.append((tag, i1, i2, j1, j2))
            continue
        if tag != "replace" or i2 - i1 > MAX_SIDE or j2 - j1 > MAX_SIDE:
            return None
        fx, fy = a[i1:i2].translate(_FOLD), b[j1:j2].translate(_FOLD)
        if len(fx) != len(fy) and set(fx) & set(fy):     # ם→מו is a letter added, not a swap
            return None
        out.append(("equal" if fx == fy else "replace", i1, i2, j1, j2))
    return out


def split_pairs(x: str, y: str) -> list[str]:
    """The letter pairs of one substitution. Same length: letter by letter, since adjacent swaps come out of the diff as one block
    (שכתובה/שבחובה: כת→בח is כ/ב and ת/ח); letters that differ only in final form are left out. Otherwise (two for one, וז/ח): the
    whole block."""
    if len(x) == len(y):
        return [pair_key(p, q) for p, q in zip(x, y) if p.translate(_FOLD) != q.translate(_FOLD)]
    return [pair_key(x, y)]


def allowed_sub(x: str, y: str, allowed) -> bool:
    """Is the substitution x→y made of allowed pairs (as a whole, or letter by letter)?"""
    return pair_key(x, y) in allowed or all(k in allowed for k in split_pairs(x, y))


def _positional(a: str, b: str, i0: int = 0, j0: int = 0) -> Optional[list[tuple]]:
    """Letter i of `a` against letter i of `b` (same length): runs of equal letters and substitutions of up to MAX_SIDE."""
    same = [x.translate(_FOLD) == y.translate(_FOLD) for x, y in zip(a, b)]
    out, k = [], 0
    while k < len(a):
        e = k
        while e < len(a) and same[e] == same[k]:
            e += 1
        if not same[k] and e - k > MAX_SIDE:
            return None
        out.append(("equal" if same[k] else "replace", i0 + k, i0 + e, j0 + k, j0 + e))
        k = e
    return out


def _alignments(a: str, b: str, indel: frozenset = frozenset()) -> list[list[tuple]]:
    """Ways to turn letters `a` into `b` by substitutions only, as opcodes (tag, i1, i2, j1, j2): tag "equal" (also a final-form
    difference alone) or "replace" (up to MAX_SIDE letters for up to MAX_SIDE). indel: letters that may also be added ("insert") or
    dropped ("delete"), one at most; the caller decides where that is acceptable. Empty if there is none.
    First the diff's alignment, then letter-by-letter ones, fewest changed letters first: the diff matches the longest runs, so in
    בכצל/בכלל it pairs the two ל's and reads "drop צ, add ל" instead of the swap צ→ל."""
    out = []
    first = _sm_ops(a, b, indel)
    if first is not None:
        out.append(first)
    alts = []
    if len(a) == len(b):
        alts.append(_positional(a, b))
    elif indel and len(a) == len(b) + 1:                          # Sefaria has an extra letter at k
        for k in (k for k, ch in enumerate(a) if ch in indel):
            left, right = _positional(a[:k], b[:k]), _positional(a[k + 1:], b[k:], k + 1, k)
            if left is not None and right is not None:
                alts.append(left + [("delete", k, k + 1, k, k)] + right)
    elif indel and len(b) == len(a) + 1:                          # Sefaria lacks the letter at b[k]
        for k in (k for k, ch in enumerate(b) if ch in indel):
            left, right = _positional(a[:k], b[:k]), _positional(a[k:], b[k + 1:], k, k + 1)
            if left is not None and right is not None:
                alts.append(left + [("insert", k, k, k, k + 1)] + right)
    changed = lambda ops: sum(i2 - i1 if t == "replace" else 0 if t == "equal" else 1 for t, i1, i2, _, _ in ops)  # noqa: E731
    out += sorted((x for x in alts if x is not None and x != first), key=changed)
    return [ops for ops in out if not _shuffles_yod_vav(ops, a, b)]


def _shuffles_yod_vav(ops: list[tuple], a: str, b: str) -> bool:
    """Does an added/dropped ו/י sit next to a substitution of only ו/י? Then the two together just move a ו/י around, and can stand
    for what the י/ו rule refuses: ותלו→ותלוי read as "add ו inside, swap the last ו→י" is a letter added at the end."""
    for n, (tag, *_rest) in enumerate(ops):
        if tag not in ("insert", "delete"):
            continue
        for t, i1, i2, j1, j2 in (ops[m] for m in (n - 1, n + 1) if 0 <= m < len(ops)):
            if t == "replace" and set(a[i1:i2] + b[j1:j2]) <= {"ו", "י"}:
                return True
    return False


def _ops(a: str, b: str, indel: frozenset = frozenset()) -> Optional[list[tuple]]:
    """The first of _alignments, or None."""
    alts = _alignments(a, b, indel)
    return alts[0] if alts else None


def letter_diff(a: str, b: str) -> Optional[list[tuple[int, int, str, str]]]:
    """[(i1, i2, a[i1:i2], replacement)] turning letters `a` into `b` by substitutions only (see _ops); None if that is not possible."""
    ops = _ops(a, b)
    return None if ops is None else [(i1, i2, a[i1:i2], b[j1:j2]) for t, i1, i2, j1, j2 in ops if t == "replace"]


# ---- the fix -------------------------------------------------------------------------------------------------------------------------

def _word_final(s: str, i: int) -> bool:
    """Is the Hebrew letter at s[i] the last letter of its word (nothing but non-letters before the next space, maqaf or the end)?"""
    for ch in s[i + 1:]:
        if ch.isspace() or ch == MAQAF:
            return True
        if _HEB.match(ch):
            return False
    return True


_MARKS = re.compile(r"""[״׳"'()\[\]{}<>]""")
_ABBR = set("״׳\"'")


def _in_abbreviation(chars: list[str], i: int) -> bool:
    """Is chars[i] in a word with a geresh or gershayim (an abbreviation)?"""
    s, e = i, i
    while s > 0 and not chars[s - 1].isspace():
        s -= 1
    while e < len(chars) and not chars[e].isspace():
        e += 1
    return any(ch in _ABBR for ch in chars[s:e])
KINDS = ("OCR swaps", "transposed", "spacing", "י/ו", "doubled letter", "missing/extra letter")


def _transposed(x: str, y: str) -> bool:
    """Two neighbouring letters in the other order (הזכהר / הזכרה: הר / רה)."""
    return len(x) == len(y) == 2 and x.translate(_FOLD) == y[::-1].translate(_FOLD) and x[0] != x[1]


def auto_fix(words: str, ref_words: str, allowed: set[str] | frozenset[str],
             known: set[str] | frozenset[str] | None = None, ocr: bool = True,
             typed: bool = False) -> Optional[tuple[str, str]]:
    """(fixed raw words, kind) for the flagged raw `words` against the reference reading, or None if the rule does not apply.
    kind: "OCR swaps", "transposed", "spacing", "י/ו", "doubled letter", "missing/extra letter", or several joined by " + ".

    allowed: the OCR letter pairs (empty for a typed text). Typed/converted texts (typed): also two neighbouring letters in the other
    order (הזכהר→הזכרה), and with `known`, one added or dropped letter of any kind (see below).

    known: the reference's words (match_key forms). Needed for any added or dropped letter, of which there may be one:
    - OCR (ocr): a י or ו (OCR drops thin letters and reads specks as them: שאנם→שאינם, טעמאי→טעמא), but only
    - inside a word or as a stray last letter on Sefaria's side, never at the start of a word (a ו there is a conjunction) and never
      added at the end (אמר→אמרו is grammar);
    - when Sefaria's word never appears in the reference: a real word (הכי/היכי, לשתוף/לשתף) is a reading or a spelling, not damage;
    - when the reference reading has no gershayim, quotes or brackets (ס"ד, (אחרות)): the fix keeps Sefaria's punctuation, so an
      abbreviation would lose its gershayim, and brackets mark an editor's emendation.
    - Typed (typed), any letter, under the same gershayim/brackets condition: a doubled letter dropped (הההרים→ההרים, not the word's
      first); or, when Sefaria's word never appears in the reference, a letter missing inside a word (שעים→שערים), also where a space
      took its place (תקי ות→תקיעות), or an extra letter inside a word (הברהכה→הברכה)."""
    if not ref_words or NIQQUD.search(words):
        return None
    tags = [m.span() for m in TAG.finditer(words)]
    in_tag = lambda i: any(s <= i < e for s, e in tags)  # noqa: E731
    pos = [i for i, ch in enumerate(words) if _HEB.match(ch) and not in_tag(i)]
    ref = [w for w in (letters(x) for x in _BOUNDARY.split(TAG.sub(" ", ref_words))) if w]
    a, b = "".join(words[i] for i in pos), "".join(ref)
    if not a or not b:
        return None
    starts = set(accumulate(len(w) for w in ref[:-1]))            # letter indices in b where a reference word starts
    sef_starts = {0} | {k for k in range(1, len(a))
                        if any(_BOUNDARY.match(words[i]) and not in_tag(i) for i in range(pos[k - 1] + 1, pos[k]))}

    def token_key(i: int) -> str:
        """match_key of the Sefaria token holding raw index i."""
        return next(match_key(words[s:e]) for s, e in tokenize(words) if s <= i < e)

    def _indel_kind(tag: str, i1: int, j1: int, letter: str) -> Optional[str]:
        """The kind of one added/dropped letter if the source's rules allow it where it is, else None."""
        last_of_word = lambda k: k == len(a) - 1 or k + 1 in sef_starts  # noqa: E731
        if tag == "delete":                                       # Sefaria's letter a[i1] goes
            if i1 in sef_starts:
                return None
            unknown = token_key(pos[i1]) not in known
            if ocr and letter in YOD_VAV and unknown:             # inside, or a stray last letter
                return "י/ו"
            if typed and a[i1].translate(_FOLD) == a[i1 - 1].translate(_FOLD):
                return "doubled letter"
            if typed and unknown and not last_of_word(i1):
                return "missing/extra letter"
            return None
        if i1 >= len(a):                                          # added after the last letter
            return None
        if i1 not in sef_starts:                                  # inside a word
            unknown = token_key(pos[i1]) not in known
            if ocr and letter in YOD_VAV and unknown:
                return "י/ו"
            if typed and unknown:
                return "missing/extra letter"
            return None
        # at a Sefaria word start: only where a space took the letter's place (תקי ות / תקיעות)
        if typed and i1 > 0 and j1 not in starts and j1 + 1 not in starts and j1 > 0:
            gap = words[pos[i1 - 1] + 1:pos[i1]]
            if not gap.strip() and token_key(pos[i1 - 1]) + token_key(pos[i1]) not in known:
                return "missing/extra letter"
        return None

    def attempt(ops: list[tuple]) -> Optional[tuple[str, str]]:
        indel_kind = None
        indel_ops = [o for o in ops if o[0] in ("insert", "delete")]
        if indel_ops:
            if len(indel_ops) > 1 or known is None or _MARKS.search(TAG.sub("", ref_words)):
                return None
            tag, i1, i2, j1, j2 = indel_ops[0]
            indel_kind = _indel_kind(tag, i1, j1, (a[i1:i2] or b[j1:j2]))
            if indel_kind is None:
                return None
        kinds: set[str] = {indel_kind} if indel_kind else set()
        units = []                                                    # (first, last sef letter index, replacement or None, b index)
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                units += [(k, k, None, j1 + k - i1) for k in range(i1, i2)]
                continue
            if tag == "delete":                                       # drop Sefaria's letter
                units.append((i1, i1, "", None))
                continue
            if tag == "insert":                                       # add the reference's letter before Sefaria's letter i1
                units.append(("insert", b[j1:j2], j1))
                continue
            if allowed_sub(a[i1:i2], b[j1:j2], allowed):
                kinds.add("OCR swaps")
            elif typed and _transposed(a[i1:i2], b[j1:j2]):
                kinds.add("transposed")
            else:
                return None
            if pos[i2 - 1] - pos[i1] != i2 - i1 - 1 or any(j in starts for j in range(j1 + 1, j2)):
                return None                                           # something sits inside the swapped letters, or a word starts there
            units.append((i1, i2 - 1, b[j1:j2], j1))

        out, length, recheck = [], 0, []                              # recheck: output indices whose final form may change
        cut, last = 0, None                                           # raw index after the previous unit; output index of its last letter
        pending = ""                                                  # a letter to add before the next unit
        for unit in units:
            if unit[0] == "insert":
                _, pending, pending_j = unit
                continue
            first, end, rep, j = unit
            if rep == "":                                             # dropped letter: never a word's first, so no boundary before it
                out.append(words[cut:pos[first]]); length += len(out[-1])
                if last is not None:
                    recheck.append(last)
                cut = pos[first] + 1
                continue
            if pending:
                j = pending_j
            gap = words[cut:pos[first]]
            if last is not None:
                sef_b = any(_BOUNDARY.match(ch) and not in_tag(cut + n) for n, ch in enumerate(gap))
                ref_b = j in starts
                if sef_b and not ref_b:                               # join: only a plain space can go
                    if gap.strip():
                        return None
                    gap = ""
                    kinds.add("spacing")
                    recheck.append(last)
                elif ref_b and not sef_b:                             # split
                    gap = gap + " "
                    kinds.add("spacing")
                    recheck.append(last)
            out.append(gap); length += len(gap)
            text = rep if rep is not None else words[pos[first]]
            if rep is not None:
                recheck.extend(range(length + len(pending), length + len(pending) + len(text)))
            if pending:
                recheck.extend(range(length, length + len(pending)))
                text, pending = pending + text, ""
            out.append(text); length += len(text)
            last, cut = length - 1, pos[end] + 1
        out.append(words[cut:])
        if not kinds or ("transposed" in kinds and indel_kind):   # בדבר→דבר as "swap בד→דב, drop ב" removes a first letter
            return None
        fixed = list("".join(out))
        for i in recheck:
            if _in_abbreviation(fixed, i):                        # ע"כ, א"נ: the reference's form, not a final one
                continue
            fixed[i] = fixed[i].translate(_UNFOLD) if _word_final("".join(fixed), i) else fixed[i].translate(_FOLD)
        result = "".join(fixed)
        if letters(result).translate(_FOLD) != b.translate(_FOLD):
            return None
        return result, " + ".join(k for k in KINDS if k in kinds)

    indel = (ANY_LETTER if typed else YOD_VAV if ocr else frozenset()) if known is not None else frozenset()
    for ops in _alignments(a, b, indel):
        got = attempt(ops)
        if got is not None:
            return got
    return None


def _prefix_groups(acr: str, ref: list[str]) -> Optional[int]:
    """How many reference words the acronym letters `acr` stand for, each group of letters opening its word in order
    (ר״הי / רשות היחיד, ג״ה / גירסת הספרים); the fewest, at least 2. None if they don't."""
    def go(s: str, i: int) -> Optional[int]:
        if not s:
            return 0
        if i >= len(ref):
            return None
        for n in range(1, len(s) + 1):
            if ref[i].translate(_FOLD).startswith(s[:n].translate(_FOLD)):
                rest = go(s[n:], i + 1)
                if rest is not None:
                    return rest + 1
        return None
    k = go(acr, 0)
    return k if k is not None and k >= 2 else None


def abbreviation_of(words: str, ref_words: str) -> bool:
    """Is every difference between the flagged words and the reference reading an abbreviation in Sefaria's text? A word with a final
    geresh whose letters open the reference's word (פפ׳ / פפא, המד׳ / המדה), or a gershayim acronym whose letters open consecutive
    reference words (ג״ה / גירסת הספרים). Other words must have the same letters. Such a candidate is not an error."""
    if not ref_words:
        return False
    ref = [w for w in (letters(x) for x in _BOUNDARY.split(TAG.sub(" ", ref_words))) if w]
    i, used = 0, False
    for tok in (TAG.sub("", t).translate(_GERSH) for t in words.split()):
        let = letters(tok)
        if not let:
            continue
        if i < len(ref) and let.translate(_FOLD) == ref[i].translate(_FOLD):
            i += 1
            continue
        if re.search("[א-ת]׳[^א-ת]*$", tok) and "״" not in tok and i < len(ref) and len(ref[i]) > len(let) \
                and ref[i].translate(_FOLD).startswith(let.translate(_FOLD)):
            i, used = i + 1, True
            continue
        k = _prefix_groups(let, ref[i:]) if "״" in tok else None
        if k is None:
            return False
        i, used = i + k, True
    return used and i == len(ref)


def _word_starts(words: list[str]) -> set[int]:
    return set(accumulate(len(w) for w in words[:-1]))


def _letter_words(text: str) -> list[str]:
    return [w for w in (letters(x) for x in _BOUNDARY.split(TAG.sub(" ", text))) if w]


def spacing_confirmed(words: str, fix: str, ref_words: str) -> bool:
    """Does the reference divide words where `fix` re-divides Sefaria's (same letters, other spaces), even where the reference's letters
    differ (לאהיה → לא היה against "לא הוה"; בעלמ'ליקני → בעלמ' ליקני against "בעלמא ליקני")? Each changed boundary must fall between
    two letters the reference has too, with the reference dividing there exactly when the fix does."""
    sw, fw, rw = _letter_words(words), _letter_words(fix), _letter_words(ref_words)
    a = "".join(sw)
    if not a or not rw or "".join(fw).translate(_FOLD) != a.translate(_FOLD):
        return False
    ss, fs, rs = _word_starts(sw), _word_starts(fw), _word_starts(rw)
    changed = ss ^ fs
    if not changed:
        return False
    b = "".join(rw)
    to_b = {}
    for i, j, n in SequenceMatcher(None, a.translate(_FOLD), b.translate(_FOLD), autojunk=False).get_matching_blocks():
        to_b.update({i + k: j + k for k in range(n)})
    for k in changed:
        if k - 1 not in to_b or k not in to_b:
            return False
        if any(to_b[k - 1] < r <= to_b[k] for r in rs) != (k in fs):
            return False
    return True


def abbreviation_ref(words: str, ref_words: str) -> Optional[str]:
    """The reference reading cut to Sefaria's abbreviations: where Sefaria writes a word with a final geresh and the reference writes it
    in full, only the reference's first letters (as many as Sefaria's): חחמו' against תחמוד → תחמו. Compared with that, an abbreviation
    with an OCR error is fixed and stays an abbreviation (תחמו'). None when no word is cut or the words don't pair up."""
    toks = [t for t in (TAG.sub("", x).translate(_GERSH) for x in words.split()) if letters(t)]
    rw = _letter_words(ref_words)
    if len(toks) != len(rw):
        return None
    out, cut = [], False
    for t, r in zip(toks, rw):
        n = len(letters(t))
        if re.search("[א-ת]׳[^א-ת]*$", t) and "״" not in t and len(r) > n:
            out.append(r[:n])
            cut = True
        else:
            out.append(r)
    return " ".join(out) if cut else None


def gershayim_moved(words: str, fix: Optional[str]) -> bool:
    """Does `fix` differ from `words` only in where a geresh/gershayim stands (ר״הי → רה״י)? ASCII " and ' count as gershayim/geresh."""
    if not fix:
        return False
    w, f = TAG.sub("", words).translate(_GERSH).strip(), TAG.sub("", fix).translate(_GERSH).strip()
    marks = lambda s: sorted(ch for ch in s if ch in "״׳")  # noqa: E731
    strip = lambda s: re.sub("[״׳]", "", s)                  # noqa: E731
    return w != f and bool(marks(w)) and marks(w) == marks(f) and strip(w) == strip(f)


# ---- the map -------------------------------------------------------------------------------------------------------------------------

@dataclass
class SwapMap:
    counts: dict[str, int] = field(default_factory=dict)           # "ב/כ" -> times seen (both directions)
    examples: dict[str, list[str]] = field(default_factory=dict)   # "ב/כ" -> ["סבר→כבר", ...] (Sefaria → reference)
    words_compared: int = 0
    dropped: int = 0                                               # substitutions of pairs left out of a stored map
    blocks: int = 0
    blocks_located: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values()) + self.dropped

    def add(self, sef_word: str, ref_word: str, a: str, b: str):
        subs = letter_diff(a, b)
        if not subs or len(subs) > MAX_SUBS:
            return
        for _, _, x, y in subs:
            for k in split_pairs(x, y):
                self.counts[k] = self.counts.get(k, 0) + 1
                ex = self.examples.setdefault(k, [])
                if len(ex) < EXAMPLES:
                    ex.append(f"{sef_word}→{ref_word}")

    def to_dict(self) -> dict:
        ranked = sorted(self.counts.items(), key=lambda kv: -kv[1])
        pairs = [{"pair": k, "count": n, "examples": self.examples.get(k, [])} for k, n in ranked[:STORED_PAIRS]]
        return {"pairs": pairs, "total": self.total, "words_compared": self.words_compared, "blocks": self.blocks,
                "blocks_located": self.blocks_located}

    @classmethod
    def from_dict(cls, d: dict) -> "SwapMap":
        m = cls(words_compared=d.get("words_compared", 0), blocks=d.get("blocks", 0), blocks_located=d.get("blocks_located", 0))
        for p in d.get("pairs", []):
            m.counts[p["pair"]] = p["count"]
            m.examples[p["pair"]] = p.get("examples", [])
        m.dropped = d.get("total", 0) - sum(m.counts.values())
        return m


def build_map(segment_texts: list[str], ref: Reference) -> SwapMap:
    """Align the whole version with the reference and count the letter substitutions between words the alignment pairs up."""
    sef = [(w, match_key(w), letters(w)) for t in segment_texts for w in (TAG.sub("", x).strip() for x in tokens(t))]
    sef = [x for x in sef if x[1]]
    keys = [k for _, k, _ in sef]
    m = SwapMap()
    last = None
    for b0 in range(0, len(sef), BLOCK):
        block = keys[b0:b0 + BLOCK]
        m.blocks += 1
        ref.last = last
        anchor, votes = ref._locate([], block, 0)
        if anchor is None or votes < MIN_BLOCK_VOTES:
            continue
        m.blocks_located += 1
        lo, hi = max(0, anchor - SLACK), min(len(ref.keys), anchor + len(block) + SLACK)
        last = anchor
        sm = SequenceMatcher(None, block, ref.keys[lo:hi], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                m.words_compared += i2 - i1
            elif tag == "replace" and i2 - i1 == j2 - j1:
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    m.words_compared += 1
                    w, _, a = sef[b0 + i]
                    rw = ref.words[ref.kept[lo + j]]
                    m.add(w, rw, a, letters(rw))
    ref.last = None
    return m
