"""Phase 0: run each LLM configuration ("arm") over the evaluation set; answers cached in experiments/output/arms/<label>.jsonl.

    python -m experiments.run_arms [ARM ...]        (default: all arms)

Items whose flagged span contains a <br> tag are skipped: the offset tokenizer (pipeline/tokens.py) no longer produces them.
"""
import asyncio
import json
import sys
from pathlib import Path

from pipeline.llm_filter import Item, Judge, ModelSpec, Usage

OUT = Path(__file__).resolve().parent / "output"
ARMS = {
    "flash-b1": ModelSpec("gemini", "gemini-3.8-flash", batch=1),
    "flash-b8": ModelSpec("gemini", "gemini-3.8-flash", batch=8),
    "flash-low-b8": ModelSpec("gemini", "gemini-3.8-flash", "low", batch=8),
    "flash-medium-b8": ModelSpec("gemini", "gemini-3.8-flash", "medium", batch=8),
    "sonnet-low-b1": ModelSpec("anthropic", "claude-sonnet-5-5", "low", batch=1),
    "sonnet-low-b8": ModelSpec("anthropic", "claude-sonnet-5-5", "low", batch=8),
    "sonnet-medium-b8": ModelSpec("anthropic", "claude-sonnet-5-5", "medium", batch=8),
    "haiku-b8": ModelSpec("anthropic", "claude-haiku-5-5", "medium", batch=8),
}


def eval_items() -> list[dict]:
    rows = [json.loads(l) for l in open(OUT / "evalset.jsonl")]
    return [r for r in rows if "<br" not in r["words"].lower()]


async def run_arm(name: str, spec: ModelSpec, rows: list[dict]):
    path = OUT / "arms" / f"{name}.jsonl"
    path.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["key"] for l in open(path)} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        print(name, "cached"); return
    items = [Item(r["id"], f'{r["ref"]} ({r["version"]})', r["excerpt"]) for r in todo]
    usage = Usage()
    judge = Judge(spec, concurrency=8, usage=usage)
    f = open(path, "a", encoding="utf-8")

    def write(vs):
        for v in vs:
            f.write(json.dumps(v.__dict__, ensure_ascii=False) + "\n")
        f.flush()

    await judge.judge(items, on_done=write)
    f.close()
    with open(OUT / "arms" / "usage.jsonl", "a") as u:
        u.write(json.dumps({"arm": name, "items": len(items), **usage.__dict__}) + "\n")
    print(name, len(items), "items", usage.by_model, flush=True)


async def main(names):
    rows = eval_items()
    print("items", len(rows))
    await asyncio.gather(*(run_arm(n, ARMS[n], rows) for n in names))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(ARMS)))
