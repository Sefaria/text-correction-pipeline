"""Phase 0 scoring: every arm (and combinations) against the Opus gold labels.

    python -m experiments.report      -> prints tables, writes experiments/output/scores.json

A failed call (no verdict) counts as "error": the candidate stays in the human queue, which is the safe direction.
"""
import collections
import json
from pathlib import Path

from pipeline.llm_filter import PRICES
from pipeline.text import match_key

from .run_arms import ARMS, OUT, eval_items


def load():
    gold = {}
    for p in (OUT / "gold").glob("*.jsonl"):
        for l in open(p):
            g = json.loads(l); gold[g["id"]] = g
    rows = {r["id"]: r for r in eval_items() if gold.get(r["id"], {}).get("label") in ("error", "correct")}
    arms = {}
    for name in ARMS:
        p = OUT / "arms" / f"{name}.jsonl"
        if p.exists():
            arms[name] = {a["key"]: a for a in map(json.loads, open(p))}
    usage = collections.defaultdict(lambda: [0, 0, 0])
    for u in map(json.loads, open(OUT / "arms" / "usage.jsonl")):
        for m, (i, o, c) in u["by_model"].items():
            x = usage[u["arm"]]; x[0] += i; x[1] += o; x[2] += u["items"]
            x.append(m)
    return gold, rows, arms, usage


def verdict(a):
    return a.get("verdict") or "error" if a else "error"


def score(ids, gold, decide, fixes=None):
    c = collections.Counter()
    fix_ok = fix_n = 0
    for i in ids:
        g, v = gold[i]["label"], decide(i)
        c[(g, v)] += 1
        if fixes and g == "error" and v == "error" and gold[i]["fix"] is not None:
            fx = fixes(i)
            if fx is not None:
                fix_n += 1
                fix_ok += match_key(fx) == match_key(gold[i]["fix"])
    te, fe = c[("error", "error")], c[("error", "correct")]
    tc, fc = c[("correct", "correct")], c[("correct", "error")]
    return {
        "n": len(ids), "gold_errors": te + fe,
        "error_recall": round(te / max(1, te + fe), 3),            # real errors kept in the queue
        "fp_removed": round(tc / max(1, tc + fc), 3),              # false positives the filter takes out
        "queue_precision": round(te / max(1, te + fc), 3),         # share of what the reviewer sees that is a real error
        "queue_share": round((te + fc) / max(1, len(ids)), 3),
        "fix_match": round(fix_ok / max(1, fix_n), 3) if fixes else None,
    }


def main():
    gold, rows, arms, usage = load()
    ids = sorted(rows)
    groups = {"all": ids,
              "error-heavy": [i for i in ids if rows[i]["why"] == "errors"],
              "fp-heavy": [i for i in ids if rows[i]["why"] == "false_positives"]}
    deciders = {}
    for name, ans in arms.items():
        deciders[name] = (lambda i, a=ans: verdict(a.get(i)), lambda i, a=ans: (a.get(i) or {}).get("fix"))
    if "flash-low-b8" in arms and "sonnet-low-b8" in arms:
        f, s = arms["flash-low-b8"], arms["sonnet-low-b8"]
        deciders["drop only if flash-low & sonnet-low agree correct"] = (
            lambda i: "correct" if verdict(f.get(i)) == verdict(s.get(i)) == "correct" else "error",
            lambda i: (s.get(i) or {}).get("fix"))
        deciders["cascade flash-low -> sonnet-low (flash correct@>=0.8 drops)"] = (
            lambda i: "correct" if verdict(f.get(i)) == "correct" and (f.get(i) or {}).get("confidence", 0) >= 0.8 else verdict(s.get(i)),
            lambda i: (s.get(i) or {}).get("fix"))
    if "sonnet-low-b8" in arms:
        s = arms["sonnet-low-b8"]
        for t in (0.6, 0.8):
            deciders[f"sonnet-low-b8, drop only if correct@>={t}"] = (
                lambda i, t=t: "correct" if verdict(s.get(i)) == "correct" and (s.get(i) or {}).get("confidence", 0) >= t else "error",
                lambda i: (s.get(i) or {}).get("fix"))
    out = {}
    print(f"gold: {len(ids)} items ({collections.Counter(gold[i]['label'] for i in ids)})\n")
    for gname, gids in groups.items():
        print(f"== {gname} ({len(gids)} items, {sum(gold[i]['label'] == 'error' for i in gids)} real errors)")
        print(f"   {'decider':62s} recall  fp_rm  q_prec  q_share  fix")
        out[gname] = {}
        for d, (dec, fx) in deciders.items():
            m = score(gids, gold, dec, fx)
            out[gname][d] = m
            print(f"   {d:62s} {m['error_recall']:.3f}  {m['fp_removed']:.3f}  {m['queue_precision']:.3f}   {m['queue_share']:.3f}   {m['fix_match']}")
        print()
    print("== cost (from logged usage)")
    for arm, (i, o, n, *models) in usage.items():
        m = models[0]
        if m in PRICES:
            pi, po = PRICES[m]
            cost = i / 1e6 * pi + o / 1e6 * po
            print(f"   {arm:16s} {n:4d} items  in {i:>8d}  out {o:>8d}  ${cost:6.3f}  = ${1000 * cost / n:5.2f} per 1k candidates")
    json.dump(out, open(OUT / "scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
