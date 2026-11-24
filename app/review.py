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


class NotFound(ReviewError):
    pass


def _segment(c, pid: int, seg: int) -> dict:
    s = db.row(c.execute("SELECT * FROM segments WHERE project_id = ? AND idx = ?", (pid, seg)).fetchone())
    if s is None:
        raise NotFound("no such segment")
    return s


def _candidate(c, pid: int, cid: int) -> dict:
    cand = db.row(c.execute("SELECT * FROM candidates WHERE project_id = ? AND id = ?", (pid, cid)).fetchone())
    if not cand:
        raise NotFound("no such candidate")
    return cand


def _accepted_fixes(c, pid: int, seg: int, exclude: int | None = None) -> list[SpanFix]:
    return [SpanFix(r["c0"], r["c1"], r["applied_fix"]) for r in c.execute(
        "SELECT c0, c1, applied_fix FROM candidates WHERE project_id = ? AND seg = ? AND decision = 'accepted' AND id IS NOT ?",
        (pid, seg, exclude))]


def segment_text(c, pid: int, seg: int) -> str:
    s = _segment(c, pid, seg)
    if s["override"] is not None:
        return s["override"]
    fixes = _accepted_fixes(c, pid, seg)
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
    cand = _candidate(c, pid, cid)
    seg = _segment(c, pid, cand["seg"])
    current = segment_text(c, pid, cand["seg"])
    fix = cand["proposed_fix"]
    proposed = None
    if fix is not None and cand["decision"] == "pending":
        rep = replacement_for(cand["words"], fix)
        proposed = _with_fix(c, pid, cand, rep) if rep is not None else None
    others = [dict(r) for r in c.execute("SELECT id, words, decision, c0, c1 FROM candidates WHERE project_id = ? AND seg = ? AND id != ? "
                                         "ORDER BY c0", (pid, cand["seg"], cid))]
    return {**cand, "original": seg["original"], "current": current, "overridden": seg["override"] is not None,
            "proposed_text": proposed, "others_in_segment": others}


def _with_fix(c, pid: int, cand: dict, replacement: str) -> str | None:
    """The segment text if this candidate's span is replaced (on top of the other accepted fixes / the override)."""
    s = _segment(c, pid, cand["seg"])
    if s["override"] is not None:
        return _patch_override(s["override"], cand["words"], cand["c0"], replacement)
    return apply_fixes(s["original"], _accepted_fixes(c, pid, cand["seg"], exclude=cand["id"]) + [SpanFix(cand["c0"], cand["c1"], replacement)])


def _state(c, pid: int, cand: dict) -> dict:
    s = _segment(c, pid, cand["seg"])
    return {"decision": cand["decision"], "applied_fix": cand["applied_fix"], "override": s["override"]}


def _set_state(c, pid: int, cand: dict, st: dict, user: str | None):
    c.execute("UPDATE candidates SET decision = ?, applied_fix = ?, decided_by = ?, decided_at = ? WHERE project_id = ? AND id = ?",
              (st["decision"], st["applied_fix"], user, db.now(), pid, cand["id"]))
    c.execute("UPDATE segments SET override = ? WHERE project_id = ? AND idx = ?", (st["override"], pid, cand["seg"]))


def _undo_in_override(c, pid: int, cand: dict, prev: dict) -> str | None:
    """The segment override once this candidate's own change is taken out of it (for reject and reset).

    Without an override the recomputed text drops the fix by itself. With one, the override was written on top of the accepted fixes, so
    an accepted candidate's fix is reversed in it (its fix text back to the flagged words, nearest its position); and the candidate
    that wrote the override takes it away, leaving the original plus the accepted fixes, unless another candidate's edit shares it."""
    ov = prev["override"]
    if ov is None:
        return None
    if cand["decision"] == "accepted":
        patched = _patch_override(ov, prev["applied_fix"], cand["c0"], cand["words"]) if prev["applied_fix"] else None
        if patched is None:
            raise ReviewError("this fix can no longer be found in the segment's edited text; edit the segment instead")
        return patched
    if cand["decision"] == "overridden":
        others = c.execute("SELECT COUNT(*) FROM candidates WHERE project_id = ? AND seg = ? AND decision = 'overridden' AND id != ?",
                           (pid, cand["seg"], cand["id"])).fetchone()[0]
        if others:
            raise ReviewError("another candidate's edit of this segment includes this one; edit the segment instead")
        return None
    return ov


def decide(pid: int, cid: int, action: str, user: str, text: str | None = None) -> dict:
    """action: accept (proposed fix, or `text` as the replacement for the flagged words), reject, override (`text` = whole segment),
    reset (back to pending)."""
    with db.tx() as c:
        cand = _candidate(c, pid, cid)
        prev = _state(c, pid, cand)
        old_text = segment_text(c, pid, cand["seg"])
        nxt = dict(prev)
        if action == "accept":
            if text is None and cand["decision"] != "pending":
                # re-accepting the proposal would silently replace a fix the reviewer already chose (or typed)
                raise ReviewError("this candidate is already decided; reset it or edit the fix instead")
            fix = text if text is not None else cand["proposed_fix"]
            if fix is None:
                raise ReviewError("no proposed fix to accept; edit the text instead")
            rep = replacement_for(cand["words"], fix.strip())
            if rep is None:
                raise ReviewError("the flagged words contain markup, so the fix can't be placed safely; edit the segment instead")
            if prev["override"] is not None:
                # the edited text holds this candidate's earlier fix if it was accepted before, else the flagged words
                here = prev["applied_fix"] if cand["decision"] == "accepted" else cand["words"]
                patched = _patch_override(prev["override"], here, cand["c0"], rep) if here else None
                if patched is None:
                    raise ReviewError("the flagged words are no longer in this segment's edited text; edit the segment instead")
                nxt.update(decision="accepted", applied_fix=rep, override=patched)
            else:
                nxt.update(decision="accepted", applied_fix=rep)
        elif action == "reject":
            nxt.update(decision="rejected", applied_fix=None,     # an earlier accepted fix drops out of the recomputed text
                       override=_undo_in_override(c, pid, cand, prev))
        elif action == "override":
            if text is None or not text.strip():
                raise ReviewError("override needs the full segment text (an empty segment would blank it on Sefaria)")
            nxt.update(decision="overridden", applied_fix=None, override=text.strip())
        elif action == "reset":
            nxt.update(decision="pending", applied_fix=None, override=_undo_in_override(c, pid, cand, prev))
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
    """Undo this user's latest not-yet-undone event in this project (projects are shared, so never another reviewer's).
    Returns the candidate it touched."""
    with db.tx() as c:
        ev = db.row(c.execute("SELECT * FROM events WHERE project_id = ? AND user = ? AND undone = 0 ORDER BY id DESC LIMIT 1",
                              (pid, user)).fetchone())
        if not ev:
            return None
        cand = _candidate(c, pid, ev["cand_id"])
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
    p = db.project(pid)
    body = export_csv(p["title"], p["version_title"], p["version_source"] or "", p["version_notes"] or "", changed_segments(pid))
    return export_filename(p["title"], p["version_title"]), body
