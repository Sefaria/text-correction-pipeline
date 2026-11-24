"""Phase 0 evaluation set: detector candidates from the worst-quality versions in the translation-scope quality pass.

    python -m experiments.build_eval_set [PER_VERSION]      -> experiments/output/evalset.jsonl, versions/<slug>.json, gold_in/<slug>.jsonl

Versions come from translation-engine experiments/version_quality_from_history/output/scope/version_report.csv: six with many confirmed
errors and two where the detector's flags were almost all false positives (precision <= 0.05). Each version is fetched in full, scored,
and up to PER_VERSION candidates are sampled at random (seed 0).
"""
import json
import random
import re
import sys
from pathlib import Path

from pipeline.candidates import find_candidates
from pipeline.detector import Detector
from pipeline.sefaria_client import fetch_version
from pipeline.windows import build_windows

OUT = Path(__file__).resolve().parent / "output"
VERSIONS = [
    # (index title, version title, why)
    ("Chiddushei Ramban on Megillah", "Chiddushei HaRamban, Jerusalem 1928-29", "errors"),
    ("Ritva on Makkot", "Hamisha Shitot, Sulzbach 1761.", "errors"),
    ("Minchat Shai on Psalms", "Minchat Shai", "errors"),
    ("Bekhor Shor", "Bekhor Shor, Leipzig, 1856", "errors"),
    ("Rashi on Nedarim", "Vilna Edition", "errors"),
    ("Sforno on Song of Songs", "Chamesh Megillot, Warsaw 1875", "errors"),
    ("Metzudat Zion on Psalms", "On Your Way", "false_positives"),
    ("Ibn Ezra on Song of Songs", "Ibn Ezra's commentary on the Canticles", "false_positives"),
]


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main(per_version: int = 80):
    OUT.joinpath("versions").mkdir(parents=True, exist_ok=True)
    det = Detector()
    rng = random.Random(0)
    rows = []
    for title, vt, why in VERSIONS:
        v = fetch_version(title, vt)
        ws = build_windows(v.segments)
        scores = det.score(ws)
        cands = find_candidates(v.segments, ws, scores, det.threshold)
        json.dump({"title": title, "versionTitle": vt, "words": v.words, "segments": v.segments,
                   "candidates": [c.dict() for c in cands]},
                  open(OUT / "versions" / f"{slug(title)}.json", "w"), ensure_ascii=False)
        pick = rng.sample(cands, min(per_version, len(cands)))
        for c in sorted(pick, key=lambda c: c.id):
            rows.append({**c.dict(), "cand": c.id, "id": f"{slug(title)}#{c.id}", "title": title, "version": vt, "why": why,
                         "excerpt": c.excerpt(25)})
        print(f"{title}: {v.words} words, {len(cands)} candidates ({1000 * len(cands) / max(v.words, 1):.1f}/1k), sampled {len(pick)}",
              flush=True)
    with open(OUT / "evalset.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # one input file per version for the Opus gold labellers (they see the context, not the detector score)
    OUT.joinpath("gold_in").mkdir(exist_ok=True)
    by_slug: dict[str, list] = {}
    for r in rows:
        by_slug.setdefault(r["id"].split("#")[0], []).append(r)
    for s, rs in by_slug.items():
        with open(OUT / "gold_in" / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({"id": r["id"], "source": f'{r["ref"]} ({r["version"]})', "before": r["before"],
                                    "flagged": r["words"], "after": r["after"]}, ensure_ascii=False) + "\n")
    print("eval rows", len(rows))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
