"""Anchor the Opus gold labels on human ones: the 26 hand-labelled flagged words and 50 controls (25 words editors corrected,
25 unedited words) from translation-engine experiments/version_quality_from_history/output/llm_judge/items.jsonl.

    python -m experiments.human_check
"""
import asyncio
import collections
import json

from pipeline.llm_filter import Item, Judge, ModelSpec

from .run_arms import ARMS, OUT

SPECS = {"opus-high-b8": ModelSpec("anthropic", "claude-opus-5-5", "high", batch=8),
         **{k: ARMS[k] for k in ("flash-b8", "flash-low-b8", "sonnet-low-b8", "haiku-b8")}}


def main():
    rows = [json.loads(l) for l in open(OUT / "human_items.jsonl")]
    rows = [r for r in rows if r["truth"] in ("real", "variant") and not r.get("yerushalmi")]
    truth = {r["id"]: "error" if r["truth"] == "real" else "correct" for r in rows}
    items = [Item(r["id"], f'{r["ref"]} ({r["version"]})', " ".join(f"⟦{w}⟧" if f else w for w, f in r.get("wide") or r["words"]))
             for r in rows]
    sets = collections.Counter(r["set"] for r in rows)
    print("items", len(items), dict(sets))

    async def go():
        return {k: await Judge(s).judge(items) for k, s in SPECS.items()}
    res = asyncio.run(go())
    out = {}
    for k, vs in res.items():
        by = collections.defaultdict(collections.Counter)
        for v in vs:
            r = next(x for x in rows if x["id"] == v.key)
            by[r["set"]][(truth[v.key], v.verdict or "error")] += 1
        acc = sum(c[(t, t)] for c in by.values() for t in ("error", "correct")) / len(vs)
        out[k] = {s: {f"{t}->{v}": n for (t, v), n in c.items()} for s, c in by.items()} | {"accuracy": round(acc, 3)}
        print(f"{k:14s} acc {acc:.3f}  " + "  ".join(f"{s}: {dict(c)}" for s, c in by.items()))
    json.dump(out, open(OUT / "human_check.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
