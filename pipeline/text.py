"""Text normalization shared by the detector, the reference aligner and the LLM excerpts.

TAG and NIQQUD are vendored from Sefaria/translation-engine experiments/version_quality_from_history/classify.py (via ML-Workflows
app/error_detection/steps/vendor/normalize.py). The detector was trained with exactly these patterns: re-sync rather than edit.
"""
import re

TAG = re.compile(r"&lt;.*?&gt;|<[^>]*>|&nbsp;|&[a-z]+;")
NIQQUD = re.compile("[֑-ׇֽֿׁׂׅׄ]")          # points and cantillation, excluding maqaf and sof pasuq
MAQAF = "־"
HEBREW = re.compile("[א-ת]")


def model_word(token: str) -> str:
    """What the detector sees for one whitespace token: tags and niqqud removed, maqaf split. Punctuation and quote marks stay."""
    return TAG.sub("", NIQQUD.sub("", token)).replace(MAQAF, " ").strip()


def display_word(token: str) -> str:
    """A token as a reader sees it: tags removed, niqqud kept."""
    return TAG.sub("", token).strip()


_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
_NON_LETTERS = re.compile(r"[^א-תa-zA-Z0-9]")


def match_key(token: str) -> str:
    """Aggressive normalization for comparing against a reference text: only letters, final forms folded.
    Two editions that differ only in niqqud, punctuation, geresh/gershayim or final-letter forms compare equal."""
    return _NON_LETTERS.sub("", model_word(token).replace(" ", "")).translate(_FINALS)
