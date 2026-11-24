"""Tokens of a raw segment, with character offsets.

A token is a whitespace-separated chunk, further split after a line-break tag (<br>, <br/>): Sefaria texts often write
"הא'</b><br>תורק." with no space, and with the tags stripped the detector reads it as two words wrongly joined. The tag stays with the
preceding token. Offsets let a fix replace exactly the flagged characters, leaving the rest of the segment (HTML included) untouched.
"""
import re

_CHUNK = re.compile(r"\S+")
_BREAK = re.compile(r"<br\s*/?>", re.I)


def tokenize(text: str) -> list[tuple[int, int]]:
    """[(start, end)] character spans of the tokens of `text`."""
    out = []
    for m in _CHUNK.finditer(text):
        s, e = m.start(), m.end()
        cut = s
        for b in _BREAK.finditer(text, s, e):
            if b.end() < e:
                out.append((cut, b.end()))
                cut = b.end()
        out.append((cut, e))
    return out


def words(text: str) -> list[str]:
    return [text[s:e] for s, e in tokenize(text)]
