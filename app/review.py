"""Review state: what each segment says now, the decisions that got it there, undo, history and the modtools CSV.

A segment's current text = its override if a reviewer wrote one, else the original with every accepted candidate's fix applied
(pipeline.edits.apply_fixes on original offsets). Accepting a candidate in an overridden segment patches the override text where the
flagged words still are (nearest occurrence to their original position). Every action appends an event holding the state before and
after, so undo is exact and the history list can show old -> new text.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from pipeline.csv_export import export_csv, export_filename
from pipeline.edits import SpanFix, apply_fixes, replacement_for
from pipeline.letter_pairs import allowed_pairs
from collections import Counter

from pipeline.ocr_swaps import abbreviation_of, abbreviation_ref, auto_fix, gershayim_moved, spacing_confirmed
from pipeline.reference import Reference
from pipeline.text import display_word, match_key
from pipeline.tokens import tokenize

from . import db

AUTO = "auto: "                        # decided_by prefix of automatic decisions (auto_decide); a reviewer's later action replaces it

FILTERS = {
    "todo": "decision = 'pending' AND bucket IN ('likely_error', 'unchecked')",
    "accepted": "decision = 'accepted'",
    "auto": f"decision != 'pending' AND decided_by LIKE '{AUTO}%'",
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


def decide(pid: int, cid: int, action: str, user: str, text: str | None = None, only_pending: bool = False) -> dict:
    """action: accept (proposed fix, or `text` as the replacement for the flagged words), reject, override (`text` = whole segment),
    reset (back to pending). only_pending: refuse if the candidate was decided meanwhile (automatic decisions never replace a reviewer's)."""
    with db.tx() as c:
        cand = _candidate(c, pid, cid)
        if only_pending and cand["decision"] != "pending":
            raise ReviewError("this candidate is already decided")
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


@lru_cache(maxsize=4)
def _reference_counts(path: str, mtime: float) -> Counter:
    """How often each word (match_key form) occurs in the reference (cached per file version)."""
    return Counter(k for k in (match_key(w) for w in Path(path).read_text(encoding="utf-8").split()) if k)


def _reference_words(path: str, mtime: float) -> frozenset[str]:
    """The reference's words, as match_key forms."""
    return frozenset(_reference_counts(path, mtime))


@lru_cache(maxsize=8)
def _version_counts(pid: int) -> Counter:
    """How often each word (match_key form) occurs in the version's original text."""
    return Counter(k for (t,) in db.conn().execute("SELECT original FROM segments WHERE project_id = ?", (pid,))
                   for k in (match_key(w) for w in t.split()) if k)


def _vocabulary_backs(words: str, fix: str, ref_counts: Counter, version: Counter) -> bool:
    """Without a reference reading: does the vocabulary back the LLM's fix? Every word the fix removes occurs nowhere else (not in the
    reference, only here in the version), and every word it brings in occurs somewhere (חופס → תופס; באריכותופירש → באריכות ופירש)."""
    old = Counter(k for k in (match_key(w) for w in words.split()) if k)
    new = Counter(k for k in (match_key(w) for w in fix.split()) if k)
    gone, came = old - new, new - old
    return bool(gone or came) and all(ref_counts[k] == 0 and version[k] <= old[k] for k in gone) \
        and all(ref_counts[k] + version[k] > 0 for k in came)


@lru_cache(maxsize=2)
def _reference(path: str, mtime: float) -> Reference:
    """The reference, indexed for alignment (cached per file version)."""
    return Reference(Path(path).read_text(encoding="utf-8"))


def _neighbour_fix(c, pid: int, r: dict, ref: Reference, fix_args: tuple) -> tuple[str, str] | None:
    """(new segment text, kind) when the flagged words and one neighbouring word, taken together, differ from the reference only in
    where the space between them falls (plus what auto_fix allows): אל ⟦אהוא⟧ → אלא הוא, ⟦הי⟧ העיר → היה עיר. The neighbour must be
    in the same segment and not part of another candidate. In a segment already edited as a whole (by a reviewer, or by this rule for
    another candidate: העי ⟦רהיה⟧ and ⟦הי⟧ העיר), the words are replaced where they still stand in the edited text. None otherwise."""
    seg = _segment(c, pid, r["seg"])
    orig = seg["original"]
    spans = tokenize(orig)
    mine = [n for n, (s, e) in enumerate(spans) if s < r["c1"] and e > r["c0"]]
    if not mine:
        return None
    before, after = r["before"].split(), r["after"].split()
    for n in (mine[0] - 1, mine[-1] + 1):
        if not 0 <= n < len(spans):
            continue
        s, e = spans[n]
        word = display_word(orig[s:e])
        if not word or c.execute("SELECT 1 FROM candidates WHERE project_id = ? AND seg = ? AND id != ? AND c0 < ? AND c1 > ?",
                                 (pid, r["seg"], r["id"], e, s)).fetchone():
            continue
        if n < mine[0]:
            if not before or before[-1] != word:
                continue
            e0, e1, ctx = s, r["c1"], (" ".join(before[:-1]), r["after"])
        else:
            if not after or after[0] != word:
                continue
            e0, e1, ctx = r["c0"], e, (r["before"], " ".join(after[1:]))
        ext = orig[e0:e1]
        a = ref.align(ctx[0], ext, ctx[1])
        if a.status not in ("differs", "agrees") or not a.ref_words:   # same letters ("agrees") is the space moved
            continue
        got = auto_fix(ext, a.ref_words, *fix_args)
        if got is None or "spacing" not in got[1]:
            continue
        if seg["override"] is not None:
            base = seg["override"]
            hits = [m.start() for m in re.finditer(re.escape(ext), base)]
            if not hits:
                continue
            at = min(hits, key=lambda h: abs(h - e0))             # where the words still stand, nearest their original place
            return base[:at] + got[0] + base[at + len(ext):], got[1]
        others = [f for f in _accepted_fixes(c, pid, r["seg"]) if f.c1 <= e0 or f.c0 >= e1]
        return apply_fixes(orig, others + [SpanFix(e0, e1, got[0])]), got[1]
    return None


def auto_decide(pid: int, ids: list[int] | None = None) -> dict:
    """Automatic decisions on pending candidates the LLM calls errors (of `ids`, or all):
    - reject, as "auto: gershayim", when the LLM's fix only moves a geresh/gershayim (pipeline.ocr_swaps.gershayim_moved);
    - reject, as "auto: abbreviation", when Sefaria abbreviates what the reference writes in full (פפ׳ / פפא; ג״ה / גירסת הספרים):
      not an error, whatever the LLM proposed (pipeline.ocr_swaps.abbreviation_of);
    - accept, as "auto: <kinds>", when the reference reading differs from Sefaria's only in ways the project's text source allows
      (pipeline.ocr_swaps.auto_fix): OCR letter swaps for its script, word division, one י/ו (OCR); transposed neighbouring letters, a
      doubled letter, one missing or extra letter (typed/converted); both for a mixed source. If the LLM proposed a fix that differs
      from the reference reading, the candidate stays with the reviewer.
    - typed/converted and mixed sources: when the space between the flagged words and a neighbouring word is in the wrong place
      (אל ⟦אהוא⟧ → אלא הוא), edit the segment, as "auto: <kinds>" with decision "overridden" (_neighbour_fix).
    Each decision is an ordinary event: it shows in the history, and a reviewer can reset it. Returns counts by decided_by."""
    p = db.project(pid)
    source = p["source"] or "ocr"
    allowed = allowed_pairs(p["script"] or "square", source)
    path = db.ref_path(pid)
    known = _reference_words(str(path), path.stat().st_mtime) if path.exists() else None
    fix_args = (allowed, known, source in ("ocr", "mixed"), source in ("typed", "mixed"))
    rows = [dict(r) for r in db.conn().execute(
        "SELECT id, seg, c0, c1, before, words, after, ref_words, align_status, llm_fix FROM candidates WHERE project_id = ? "
        "AND decision = 'pending' AND bucket = 'likely_error' AND llm_verdict = 'error' ORDER BY seg, c0", (pid,))]
    if ids is not None:
        wanted = set(ids)
        rows = [r for r in rows if r["id"] in wanted]
    done: dict[str, int] = {}
    for r in rows:
        if gershayim_moved(r["words"], r["llm_fix"]):
            action, fix, who = "reject", None, AUTO + "gershayim"
        elif r["align_status"] == "differs" and abbreviation_of(r["words"], r["ref_words"]):
            action, fix, who = "reject", None, AUTO + "abbreviation"
        elif r["align_status"] in ("differs", "agrees") and r["ref_words"]:
            llm_fix = r["llm_fix"]
            got = None
            if not llm_fix or match_key(llm_fix) == match_key(r["ref_words"]):
                got = auto_fix(r["words"], r["ref_words"], *fix_args)
                cut = abbreviation_ref(r["words"], r["ref_words"]) if got is None else None
                if cut:                                           # חחמו' / תחמוד: fix the abbreviation, keep it one
                    got = auto_fix(r["words"], cut, *fix_args)
            if got is None and llm_fix and spacing_confirmed(r["words"], llm_fix, r["ref_words"]):
                got = auto_fix(r["words"], llm_fix, frozenset(), None, False, False)   # Sefaria's text, the LLM's spaces
            if got is not None:
                action, (fix, kind) = "accept", got
            elif llm_fix and match_key(llm_fix) != match_key(r["ref_words"]):
                continue
            elif fix_args[3] and path.exists():
                got = _neighbour_fix(db.conn(), pid, r, _reference(str(path), path.stat().st_mtime), fix_args)
                if got is None:
                    continue
                action, (fix, kind) = "override", got
            else:
                continue
            who = AUTO + kind
        elif r["align_status"] == "not_found" and r["llm_fix"] and path.exists():
            # the reference lacks this passage: the LLM's fix, if it is an allowed kind and the vocabulary backs it
            got = auto_fix(r["words"], r["llm_fix"], fix_args[0], None, fix_args[2], fix_args[3])
            if got is None or not _vocabulary_backs(r["words"], got[0], _reference_counts(str(path), path.stat().st_mtime),
                                                    _version_counts(pid)):
                continue
            action, (fix, kind) = "accept", got
            who = AUTO + kind + " (no reference)"
        else:
            continue
        try:
            decide(pid, r["id"], action, who, fix, only_pending=True)
            done[who] = done.get(who, 0) + 1
        except ReviewError:                 # decided meanwhile, or the fix can't be placed: leave it to the reviewer
            pass
    return done


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
