"""How alike Hebrew letters look, per script: the source of pipeline/letter_similarity.json.

    pip install pillow
    python -m experiments.letter_similarity SQUARE_FONT[,SQUARE_FONT...] RASHI_FONT[,...]   -> pipeline/letter_similarity.json

Used with Frank Ruhl Libre + David Libre (square) and Noto Rashi Hebrew (Rashi script), all from github.com/google/fonts (OFL).

Each letter (final forms included) is drawn on a common baseline, so ascenders (ל) and descenders (ק ן ך) count. The ink is blurred a
little, standing in for worn type and scan noise, and two shapes are compared by fuzzy overlap (sum of min / sum of max), at the best
of a few small horizontal shifts. Pairs of two letters are compared with single letters too (וז against ח), laid out right to left (needs Pillow with raqm).
Scores of several fonts for one script are averaged.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LETTERS = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
SIZE = 96
H, W = 2 * SIZE, 3 * SIZE
BASE = int(SIZE * 1.3)
BLUR = SIZE / 30
SHIFTS = range(-SIZE // 12, SIZE // 12 + 1, 2)
KEEP = 0.5                                             # scores stored (the app uses a higher cutoff)
OUT = Path(__file__).resolve().parent.parent / "pipeline" / "letter_similarity.json"


def render(font: ImageFont.FreeTypeFont, text: str) -> np.ndarray:
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    x0, _, x1, _ = d.textbbox((0, 0), text, font=font, anchor="ls", direction="rtl")
    d.text(((W - (x1 - x0)) / 2 - x0, BASE), text, fill=255, font=font, anchor="ls", direction="rtl")
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(BLUR)), dtype=np.float32) / 255
    return a


def sim(a: np.ndarray, b: np.ndarray) -> float:
    best = 0.0
    for s in SHIFTS:
        bb = np.roll(b, s, axis=1)
        best = max(best, float(np.minimum(a, bb).sum() / np.maximum(a, bb).sum()))
    return best


def scores(font_paths: list[str]) -> dict[str, float]:
    total: dict[str, float] = {}
    for path in font_paths:
        font = ImageFont.truetype(path, SIZE)
        one = {ch: render(font, ch) for ch in LETTERS}
        two = {a + b: render(font, a + b) for a, b in itertools.product(LETTERS, repeat=2)}
        for x, y in itertools.combinations(LETTERS, 2):
            k = "/".join(sorted((x, y)))
            total[k] = total.get(k, 0) + sim(one[x], one[y]) / len(font_paths)
        for xy, img in two.items():
            for z in LETTERS:
                if z in xy:
                    continue
                k = "/".join(sorted((xy, z)))
                total[k] = total.get(k, 0) + sim(img, one[z]) / len(font_paths)
    return total


def main(square: str, rashi: str):
    out = {}
    for name, fonts in (("square", square.split(",")), ("rashi", rashi.split(","))):
        s = scores(fonts)
        out[name] = {k: round(v, 3) for k, v in sorted(s.items(), key=lambda kv: -kv[1]) if v >= KEEP}
        print(name, "top single-letter pairs:", [(k, out[name][k]) for k in out[name] if len(k) == 3][:25])
        print(name, "top two-for-one:", [(k, out[name][k]) for k in out[name] if len(k) == 4][:15])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
