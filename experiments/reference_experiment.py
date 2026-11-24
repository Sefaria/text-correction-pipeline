"""Phase 0, reference text: does an uploaded reference cut false positives without dropping real errors, and does the reference-aware
LLM prompt beat the plain one on the candidates where the reference differs?

    python -m experiments.reference_experiment build      -> output/reference/<slug>.txt (one simulated reference per eval version)
    python -m experiments.reference_experiment run ARM    -> output/reference/answers.<ARM>.jsonl and a summary

Sefaria rarely has two overlapping editions of these works (it stores complementary parts as separate versions), so each reference is
simulated from the Sefaria text itself:
  * gold errors (Opus labels) are corrected to the gold fix, except 10% that the reference shares (a common misprint);
  * 15% of gold-correct flagged words get an edition variant in the reference (full/defective spelling), and 5% a reference OCR error;
  * elsewhere: HTML and niqqud stripped, punctuation changed, lines reflowed, 5% of segments missing, 2% of words spelling-varied,
    0.3% of words with OCR-style letter swaps.
This tests the alignment mechanics and the adjudication prompt; it is not a substitute for a real second edition.
"""
import asyncio
import collections
import json
import random
import re
import sys
from pathlib import Path

from pipeline.llm_filter import Item, Judge, Usage
from pipeline.reference import Reference
from pipeline.text import NIQQUD, TAG

from .run_arms import ARMS, OUT, eval_items

REF = OUT / "reference"
SWAPS = [("ד", "ר"), ("ה", "ח"), ("ב", "כ"), ("ו", "י"), ("ס", "ם"), ("ן", "ו")]


def plene(word: str, rng: random.Random) -> str:
    """Add or drop a vav/yod inside a word: a spelling difference between editions."""
    if len(word) < 3:
        return word
    inner = [i for i in range(1, len(word) - 1) if word[i] in "וי"]
    if inner and rng.random() < 0.5:
        i = rng.choice(inner)
        return word[:i] + word[i + 1:]
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + rng.choice("וי") + word[i:]


def ocr(word: str, rng: random.Random) -> str:
    opts = [(i, b) for i, ch in enumerate(word) for a, b in SWAPS + [(y, x) for x, y in SWAPS] if ch == a]
    if not opts:
        return word
    i, b = rng.choice(opts)
    return word[:i] + b + word[i + 1:]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", NIQQUD.sub("", TAG.sub(" ", text))).strip()


def build():
    REF.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    gold = {}
    for p in (OUT / "gold").glob("*.jsonl"):
        for l in open(p):
            g = json.loads(l); gold[g["id"]] = g
    rows = eval_items()
    by_slug = collections.defaultdict(list)
    for r in rows:
        by_slug[r["id"].split("#")[0]].append(r)
    plan = {}
    for slug, items in by_slug.items():
        v = json.load(open(OUT / "versions" / f"{slug}.json"))
        segs = [list(s) for s in v["segments"]]
        # span edits per segment on the raw text (offsets from the candidate), applied right to left
        edits = collections.defaultdict(list)
        for r in items:
            g = gold.get(r["id"])
            if not g or g["label"] == "unsure":
                continue
            kind = None
            if g["label"] == "error":
                if rng.random() < 0.10:
                    kind = "shared_misprint"
                else:
                    kind = "corrected"; edits[r["seg"]].append((r["c0"], r["c1"], g["fix"]))
            else:
                x = rng.random()
                if x < 0.15:
                    kind = "variant"; edits[r["seg"]].append((r["c0"], r["c1"], " ".join(plene(w, rng) for w in r["words"].split())))
                elif x < 0.20:
                    kind = "ref_ocr"; edits[r["seg"]].append((r["c0"], r["c1"], " ".join(ocr(w, rng) for w in r["words"].split())))
                else:
                    kind = "same"
            plan[r["id"]] = kind
        protected = {(r["seg"]) for r in items}
        lines = []
        for s, (ref, text) in enumerate(segs):
            for c0, c1, rep in sorted(edits.get(s, []), reverse=True):
                text = text[:c0] + rep + text[c1:]
            if s not in protected and rng.random() < 0.05:
                continue                                              # passage missing from the reference
            ws = clean(text).split()
            for i, w in enumerate(ws):
                if s in protected:
                    break                                             # keep candidate segments' noise to the planned edit
                x = rng.random()
                if x < 0.02:
                    ws[i] = plene(w, rng)
                elif x < 0.023:
                    ws[i] = ocr(w, rng)
            t = " ".join(ws).replace(":", ".").replace(",", "")
            lines.append(t)
        # reflow into ~12-word lines
        flat = " ".join(lines).split()
        (REF / f"{slug}.txt").write_text("\n".join(" ".join(flat[i:i + 12]) for i in range(0, len(flat), 12)), encoding="utf-8")
    json.dump(plan, open(REF / "plan.json", "w"), indent=0)
    print(collections.Counter(plan.values()))


def run(arm: str):
    gold = {}
    for p in (OUT / "gold").glob("*.jsonl"):
        for l in open(p):
            g = json.loads(l); gold[g["id"]] = g
    plan = json.load(open(REF / "plan.json"))
    rows = [r for r in eval_items() if r["id"] in plan]
    refs = {}
    aligned = {}
    for r in sorted(rows, key=lambda r: (r["id"].split("#")[0], r["cand"])):
        slug = r["id"].split("#")[0]
        if slug not in refs:
            refs[slug] = Reference((REF / f"{slug}.txt").read_text(encoding="utf-8"))
        aligned[r["id"]] = refs[slug].align(r["before"], r["words"], r["after"])
    st = collections.Counter((gold[i]["label"], a.status) for i, a in aligned.items())
    print("gold label x alignment:", dict(st))
    print("plan x alignment:", dict(collections.Counter((plan[i], a.status) for i, a in aligned.items())))
    differs = [r for r in rows if aligned[r["id"]].status == "differs"]
    items = [Item(r["id"], f'{r["ref"]} ({r["version"]})', r["excerpt"], aligned[r["id"]].excerpt) for r in differs]
    spec = ARMS[arm]
    usage = Usage()
    out = asyncio.run(Judge(spec, usage=usage).judge(items))
    with open(REF / f"answers.{arm}.jsonl", "w", encoding="utf-8") as f:
        for v in out:
            f.write(json.dumps({**v.__dict__, "ref_words": aligned[v.key].ref_words}, ensure_ascii=False) + "\n")
    print(arm, "with reference on", len(items), "differs items", usage.by_model)


if __name__ == "__main__":
    if sys.argv[1] != "score":
        {"build": lambda: build(), "run": lambda: run(sys.argv[2])}[sys.argv[1]]()


def score(arm: str):
    """Whole pipeline with a reference (agrees -> dismissed, differs -> reference-aware LLM) vs the LLM alone, against the gold labels."""
    from pipeline.text import match_key
    gold = {}
    for p in (OUT / "gold").glob("*.jsonl"):
        for l in open(p):
            g = json.loads(l); gold[g["id"]] = g
    plan = json.load(open(REF / "plan.json"))
    ids = [i for i in plan if gold[i]["label"] in ("error", "correct")]
    withref = {a["key"]: a for a in map(json.loads, open(REF / f"answers.{arm}.jsonl"))}
    noref = {a["key"]: a for a in map(json.loads, open(OUT / "arms" / f"{arm}.jsonl"))}

    def tally(decide, fix):
        c = collections.Counter(); fo = fn = 0
        for i in ids:
            g = gold[i]["label"]; v = decide(i); c[(g, v)] += 1
            if g == v == "error":
                f = fix(i)
                if f is not None:
                    fn += 1; fo += match_key(f) == match_key(gold[i]["fix"])
        te, fe, tc, fc = c[("error", "error")], c[("error", "correct")], c[("correct", "correct")], c[("correct", "error")]
        return f"recall {te / (te + fe):.3f}  fp_removed {tc / (tc + fc):.3f}  queue {te + fc}/{len(ids)}  fix_match {fo / max(1, fn):.3f}"

    v = lambda a: (a or {}).get("verdict") or "error"
    print("LLM only (no reference):       ", tally(lambda i: v(noref.get(i)), lambda i: (noref.get(i) or {}).get("fix")))
    print("reference + LLM (agrees=drop): ", tally(lambda i: v(withref.get(i)) if i in withref else "correct",
                                                  lambda i: (withref.get(i) or {}).get("fix") or (withref.get(i) or {}).get("ref_words")))
    print("reference only (differs=keep): ", tally(lambda i: "error" if i in withref else "correct",
                                                  lambda i: (withref.get(i) or {}).get("ref_words")))
    sub = [i for i in ids if i in withref]
    c = collections.Counter((plan[i], gold[i]["label"], v(withref[i]), v(noref.get(i))) for i in sub)
    print("on 'differs' items (plan, gold, with-ref verdict, no-ref verdict):")
    for k, n in sorted(c.items(), key=lambda kv: -kv[1]):
        print("  ", k, n)


if __name__ == "__main__" and sys.argv[1] == "score":
    score(sys.argv[2])
