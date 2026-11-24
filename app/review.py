"""Review state: what each segment says now, the decisions that got it there, undo, history and the modtools CSV.

A segment's current text = its override if a reviewer wrote one, else the original with every accepted candidate's fix applied
(pipeline.edits.apply_fixes on original offsets). Accepting a candidate in an overridden segment patches the override text where the
flagged words still are (nearest occurrence to their original position). Every action appends an event holding the state before and
after, so undo is exact and the history list can show old -> new text.
"""
import json

from pipeline.csv_export import export_csv, export_filename
from pipeline.edits import SpanFix, apply_fixes, replacement_for

from . import db

FILTERS = {
    "todo": "decision = 'pending' AND bucket IN ('likely_error', 'unchecked')",
    "accepted": "decision = 'accepted'",
    "rejected": "decision = 'rejected'",
    "overridden": "decision = 'overridden'",
    "likely_ok": "bucket = 'likely_ok'",
    "dismissed": "bucket = 'dismissed'",
    "all": "1 = 1",
}
ORDER = "ORDER BY seg, c0"


class ReviewError(ValueError):
    pass


def _segment(c, pid: int, seg: int) -> dict:
    return db.row(c.execute("SELECT * FROM segments WHERE project_id = ? AND idx = ?", (pid, seg)).fetchone())


def segment_text(c, pid: int, seg: int) -> str:
    s = _segment(c, pid, seg)
    if s["override"] is not None:
        return s["override"]
    fixes = [SpanFix(r["c0"], r["c1"], r["applied_fix"]) for r in c.execute(
        "SELECT c0, c1, applied_fix FROM candidates WHERE project_id = ? AND seg = ? AND decision = 'accepted'", (pid, seg))]
    return apply_fixes(s["original"], fixes) if fixes else s["original"]


def _patch_override(text: str, words: str, near: int, replacement: str) -> str | None:
    hits, i = [], text.find(words)
    while i >= 0:
        hits.append(i); i = text.find(words, i + 1)
    if not hits:
        return None
    at = min(hits, key=lambda h: abs(h - near))
    return text[:at] + replacement + text[at + len(words):]


def counts(pid: int) -> dict:
    c = db.conn()
    out = {k: c.execute(f"SELECT COUNT(*) FROM candidates WHERE project_id = ? AND {w}", (pid,)).fetchone()[0] for k, w in FILTERS.items()}
    out["changed_segments"] = len(changed_segments(pid))
    return out


def list_candidates(pid: int, flt: str) -> list[dict]:
    where = FILTERS.get(flt)
    if where is None:
        raise ReviewError(f"unknown filter {flt}")
    rows = db.conn().execute(f"SELECT id, seg, ref, words, score, bucket, decision, proposed_fix, align_status FROM candidates "
                             f"WHERE project_id = ? AND {where} {ORDER}", (pid,)).fetchall()
    return [dict(r) for r in rows]


def candidate_detail(pid: int, cid: int) -> dict:
    c = db.conn()
    cand = db.row(c.execute("SELECT * FROM candidates WHERE project_id = ? AND id = ?", (pid, cid)).fetchone())
    if not cand:
        raise ReviewError("no such candidate")
    seg = _segment(c, pid, cand["seg"])
    current = segment_text(c, pid, cand["seg"])
    fix = cand["proposed_fix"]
    proposed = None
    if fix is not None and cand["decision"] == "pending":
        proposed = _with_fix(c, pid, cand, replacement_for(cand["words"], fix))
    others = [dict(r) for r in c.execute("SELECT id, words, decision, c0, c1 FROM candidates WHERE project_id = ? AND seg = ? AND id != ? "
                                         "ORDER BY c0", (pid, cand["seg"], cid))]
    return {**cand, "original": seg["original"], "current": current, "overridden": seg["override"] is not None,
            "proposed_text": proposed, "others_in_segment": others}


def _with_fix(c, pid: int, cand: dict, replacement: str) -> str | None:
    """The segment text if this candidate's span is replaced (on top of the other accepted fixes / the override)."""
    s = _segment(c, pid, cand["seg"])
    if s["override"] is not None:
        return _patch_override(s["override"], cand["words"], cand["c0"], replacement)
    fixes = [SpanFix(r["c0"], r["c1"], r["applied_fix"]) for r in c.execute(
        "SELECT c0, c1, applied_fix FROM candidates WHERE project_id = ? AND seg = ? AND decision = 'accepted' AND id != ?",
        (pid, cand["seg"], cand["id"]))]
    return apply_fixes(s["original"], fixes + [SpanFix(cand["c0"], cand["c1"], replacement)])


def _state(c, pid: int, cand: dict) -> dict:
    s = _segment(c, pid, cand["seg"])
    return {"decision": cand["decision"], "applied_fix": cand["applied_fix"], "override": s["override"]}


def _set_state(c, pid: int, cand: dict, st: dict, user: str | None):
    c.execute("UPDATE candidates SET decision = ?, applied_fix = ?, decided_by = ?, decided_at = ? WHERE project_id = ? AND id = ?",
              (st["decision"], st["applied_fix"], user, db.now(), pid, cand["id"]))
    c.execute("UPDATE segments SET override = ? WHERE project_id = ? AND idx = ?", (st["override"], pid, cand["seg"]))


def decide(pid: int, cid: int, action: str, user: str, text: str | None = None) -> dict:
    """action: accept (proposed fix, or `text` as the replacement for the flagged words), reject, override (`text` = whole segment),
    reset (back to pending)."""
    with db.tx() as c:
        cand = db.row(c.execute("SELECT * FROM candidates WHERE project_id = ? AND id = ?", (pid, cid)).fetchone())
        if not cand:
            raise ReviewError("no such candidate")
        prev = _state(c, pid, cand)
        old_text = segment_text(c, pid, cand["seg"])
        nxt = dict(prev)
        if action == "accept":
            fix = text if text is not None else cand["proposed_fix"]
            if fix is None:
                raise ReviewError("no proposed fix to accept; edit the text instead")
            rep = replacement_for(cand["words"], fix.strip())
            if prev["override"] is not None:
                patched = _patch_override(prev["override"], cand["words"], cand["c0"], rep)
                if patched is None:
                    raise ReviewError("the flagged words are no longer in this segment's edited text; edit the segment instead")
                nxt.update(decision="accepted", applied_fix=rep, override=patched)
            else:
                nxt.update(decision="accepted", applied_fix=rep)
        elif action == "reject":
            nxt.update(decision="rejected", applied_fix=None)     # an earlier accepted fix drops out of the recomputed text
        elif action == "override":
            if text is None or not text.strip():
                raise ReviewError("override needs the full segment text (an empty segment would blank it on Sefaria)")
            nxt.update(decision="overridden", applied_fix=None, override=text.strip())
        elif action == "reset":
            nxt.update(decision="pending", applied_fix=None)
        else:
            raise ReviewError(f"unknown action {action}")
        _set_state(c, pid, cand, nxt, user)
        new_text = segment_text(c, pid, cand["seg"])
        c.execute("INSERT INTO events (project_id, cand_id, seg, action, prev, next, old_text, new_text, user, at) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (pid, cid, cand["seg"], action, db.dumps(prev), db.dumps(nxt), old_text, new_text, user, db.now()))
        c.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (db.now(), pid))
    return candidate_detail(pid, cid)


def undo(pid: int, user: str) -> dict | None:
    """Undo the latest not-yet-undone event of this project. Returns the candidate it touched."""
    with db.tx() as c:
        ev = db.row(c.execute("SELECT * FROM events WHERE project_id = ? AND undone = 0 ORDER BY id DESC LIMIT 1", (pid,)).fetchone())
        if not ev:
            return None
        cand = db.row(c.execute("SELECT * FROM candidates WHERE project_id = ? AND id = ?", (pid, ev["cand_id"])).fetchone())
        _set_state(c, pid, cand, json.loads(ev["prev"]), user)
        c.execute("UPDATE events SET undone = 1 WHERE id = ?", (ev["id"],))
    return candidate_detail(pid, ev["cand_id"])


def history(pid: int, limit: int = 20) -> list[dict]:
    rows = db.conn().execute(
        "SELECT e.id, e.cand_id, e.action, e.old_text, e.new_text, e.user, e.at, e.undone, c.ref, c.words "
        "FROM events e JOIN candidates c ON c.project_id = e.project_id AND c.id = e.cand_id "
        "WHERE e.project_id = ? ORDER BY e.id DESC LIMIT ?", (pid, limit)).fetchall()
    return [dict(r) for r in rows]


def changed_segments(pid: int) -> list[tuple[str, str]]:
    c = db.conn()
    segs = [r[0] for r in c.execute(
        "SELECT DISTINCT seg FROM candidates WHERE project_id = ? AND decision = 'accepted' "
        "UNION SELECT idx FROM segments WHERE project_id = ? AND override IS NOT NULL ORDER BY 1", (pid, pid))]
    out = []
    for s in segs:
        seg = _segment(c, pid, s)
        text = segment_text(c, pid, s)
        if text != seg["original"]:
            out.append((seg["ref"], text))
    return out


def export(pid: int) -> tuple[str, str]:
    p = db.get_project(pid)
    body = export_csv(p["title"], p["version_title"], p["version_source"] or "", p["version_notes"] or "", changed_segments(pid))
    return export_filename(p["title"], p["version_title"]), body
