"""Automatic decisions on the Phase 0 reference experiment: how many reviews do they save, and how many wrong decisions do they make?

    python -m experiments.reference_experiment build      # simulated references (needs output/versions/, see build_eval_set)
    python -m experiments.ocr_swaps_experiment [ARM]      # default arm flash-b8; uses the recorded answers, no LLM calls

Each eval item is aligned with its version's simulated reference (current aligner), then the app's rules (app.review.auto_decide) are
applied with the recorded LLM answers: the reference-aware answers where the reference differs, the plain answers elsewhere (as the
app does). Run once per script. Scored against the Opus gold labels: an accepted fix is right when gold says error and the fix matches the
gold fix (letters and word division); a gershayim dismissal is right when gold says correct.

Caveat: the simulated reference is the Sefaria text with the flagged gold errors corrected, plus noise (spelling variants, OCR swaps
from a fixed list, punctuation). It tests the rules and the LLM gate against a noisy reference; it is not a real second edition.
"""
import collections
import json
import sys

from pipeline.letter_pairs import SCRIPTS, allowed_pairs
from pipeline.ocr_swaps import auto_fix, gershayim_moved
from pipeline.reference import Reference
from pipeline.text import match_key

from .reference_experiment import REF
from .run_arms import OUT, eval_items


def words_key(s: str) -> str:
    """Letters and word division."""
    return " ".join(k for k in (match_key(w) for w in s.split()) if k)


def main(arm: str = "flash-b8"):
    gold = {}
    for p in (OUT / "gold").glob("*.jsonl"):
        for line in open(p):
            g = json.loads(line); gold[g["id"]] = g
    plan = json.load(open(REF / "plan.json"))
    withref = {a["key"]: a for a in map(json.loads, open(REF / f"answers.{arm}.jsonl"))}
    noref = {a["key"]: a for a in map(json.loads, open(OUT / "arms" / f"{arm}.jsonl"))}
    rows = sorted((r for r in eval_items() if r["id"] in plan and gold[r["id"]]["label"] in ("error", "correct")),
                  key=lambda r: (r["id"].split("#")[0], r["cand"]))
    refs: dict[str, Reference] = {}
    known: dict[str, set[str]] = {}
    aligned = {}
    for r in rows:
        slug = r["id"].split("#")[0]
        if slug not in refs:
            text = (REF / f"{slug}.txt").read_text(encoding="utf-8")
            refs[slug] = Reference(text)
            known[slug] = {k for k in (match_key(w) for w in text.split()) if k}
        aligned[r["id"]] = refs[slug].align(r["before"], r["words"], r["after"])

    print(f"arm {arm}: {len(rows)} items, {sum(gold[r['id']]['label'] == 'error' for r in rows)} gold errors")
    for script in SCRIPTS:
        allowed = allowed_pairs(script)
        tally, wrong = collections.Counter(), []
        for r in rows:
            i, a, g = r["id"], aligned[r["id"]], gold[r["id"]]
            ans = withref.get(i) if a.status == "differs" else noref.get(i)
            if not ans or ans.get("verdict") != "error":
                continue
            tally["LLM error"] += 1
            if gershayim_moved(r["words"], ans.get("fix")):
                ok = g["label"] == "correct"
                tally["gershayim dismissed, " + ("right" if ok else "WRONG (gold: error)")] += 1
                if not ok:
                    wrong.append((i, "gershayim", r["words"], ans.get("fix"), g["fix"]))
                continue
            if a.status not in ("differs", "agrees") or not a.ref_words:
                continue
            if ans.get("fix") and match_key(ans["fix"]) != match_key(a.ref_words):
                tally["skipped: LLM fix differs from reference"] += 1
                continue
            got = auto_fix(r["words"], a.ref_words, allowed, known[i.split("#")[0]])
            if got is None:
                continue
            fix, kind = got
            ok = g["label"] == "error" and words_key(fix) == words_key(g["fix"])
            tally[f"accepted ({kind}), " + ("right" if ok else "WRONG")] += 1
            if not ok:
                wrong.append((i, plan[i], g["label"], r["words"], fix, g["fix"], g["reason"]))
        print(f"  script {script}:")
        for k, n in sorted(tally.items()):
            print(f"    {k:48} {n}")
        for w in wrong:
            print("    WRONG", *w)


if __name__ == "__main__":
    main(*sys.argv[1:])
