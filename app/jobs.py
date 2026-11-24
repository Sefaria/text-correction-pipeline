"""Background processing: fetch -> detect -> align -> llm -> ready.

Two threads:
  * the detector thread takes queued projects one at a time (the detector uses every CPU) and runs fetch, detect and align;
  * the LLM thread runs an asyncio loop where every project in the `llm` stage has its own task, under one global concurrency limit.

Every stage writes its results as it goes and records where it is (stage, stage_done/total, detect_next_window), so a restart or
redeploy resumes a project where it stopped. Candidates are saved after every detector batch, so reviewing can start before the end.
"""
import asyncio
import json
import logging
import os
import threading
import time
import traceback

from pipeline import sefaria_client
from pipeline.candidates import CandidateFinder, excerpt
from pipeline.llm_filter import Item, Judge, ModelSpec, Stop, Usage
from pipeline.reference import Reference
from pipeline.windows import build_windows

from . import db

log = logging.getLogger("jobs")
DETECT_BLOCK = 64                     # windows per saved batch
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "16"))
# Default LLM pass (Phase 0, experiments/REPORT.md): Gemini 3.8 Flash, default thinking, 8 candidates per call.
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "gemini:gemini-3.8-flash::8")


class Cancelled(Exception):
    pass


class Worker:
    def __init__(self, detector_factory=None):
        self._detector = None
        self._detector_factory = detector_factory
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.llm_loop: asyncio.AbstractEventLoop | None = None
        self.llm_running: set[int] = set()
        self.sem = asyncio.Semaphore(LLM_CONCURRENCY)       # one limit across all projects' LLM calls (binds to the loop on first use)

    # ---- lifecycle ------------------------------------------------------------------------------------------------------------------
    def start(self):
        db.init()
        self.recover()
        threading.Thread(target=self._detect_loop, name="detector", daemon=True).start()
        threading.Thread(target=self._llm_thread, name="llm", daemon=True).start()

    @staticmethod
    def recover():
        """Put projects left mid-run by a restart back where the threads will pick them up."""
        # a cancel that was pending when the process stopped is final: the detector loop never picks up a cancel_requested project
        db.conn().execute("UPDATE projects SET status = 'cancelled', stage_started = NULL "
                          "WHERE status IN ('running', 'queued') AND cancel_requested = 1")
        # resume: anything mid-detection goes back to the detector queue, mid-llm projects are picked up by the LLM loop
        db.conn().execute("UPDATE projects SET status = 'queued' WHERE status = 'running' AND stage IN ('queued', 'fetch', 'detect', 'align')")

    def detector(self):
        if self._detector is None:
            if self._detector_factory:
                self._detector = self._detector_factory()
            else:
                from pipeline.detector import Detector
                self._detector = Detector()
        return self._detector

    def queue_position(self, pid: int) -> int | None:
        ids = [r[0] for r in db.conn().execute("SELECT id FROM projects WHERE status = 'queued' ORDER BY id")]
        return ids.index(pid) + 1 if pid in ids else None

    # ---- detector thread ------------------------------------------------------------------------------------------------------------
    def _detect_loop(self):
        while not self.stop.is_set():
            p = db.row(db.conn().execute("SELECT * FROM projects WHERE status = 'queued' AND cancel_requested = 0 ORDER BY id LIMIT 1").fetchone())
            if not p:
                self.wake.wait(5); self.wake.clear()
                continue
            pid = p["id"]
            try:
                db.update_project(pid, status="running")
                self._run_front(p)
            except Cancelled:
                db.update_project(pid, status="cancelled", stage_started=None)
            except Exception as e:
                log.exception("project %s failed", pid)
                db.update_project(pid, status="failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")

    @staticmethod
    def _cancel_requested(pid: int) -> bool:
        return bool(db.conn().execute("SELECT cancel_requested FROM projects WHERE id = ?", (pid,)).fetchone()[0])

    def _check_cancel(self, pid: int):
        if self._cancel_requested(pid):
            raise Cancelled()

    def _stage(self, pid: int, stage: str, total: int, done: int = 0):
        db.update_project(pid, stage=stage, stage_total=total, stage_done=done, stage_base=done, stage_started=time.time())

    def _segments(self, pid: int) -> list[tuple[str, str]]:
        return [(r["ref"], r["original"]) for r in db.conn().execute(
            "SELECT ref, original FROM segments WHERE project_id = ? ORDER BY idx", (pid,))]

    def _run_front(self, p: dict):
        pid = p["id"]
        stage = p["stage"]
        if stage in ("queued", "fetch"):
            self._stage(pid, "fetch", 1)
            v = sefaria_client.fetch_version(p["title"], p["version_title"])
            with db.tx() as c:
                c.execute("DELETE FROM segments WHERE project_id = ?", (pid,))
                c.executemany("INSERT INTO segments (project_id, idx, ref, original) VALUES (?, ?, ?, ?)",
                              [(pid, i, r, t) for i, (r, t) in enumerate(v.segments)])
                c.execute("UPDATE projects SET version_source = ?, version_notes = ?, words = ?, detect_next_window = 0 WHERE id = ?",
                          (v.version_source, v.version_notes, v.words, pid))
            stage = "detect"
            p = db.project(pid)
        segments = self._segments(pid)
        if stage == "detect":
            self._detect(p, segments)
            stage = "align"
        if stage == "align":
            self._align(pid)
        self._stage(pid, "llm", 0)
        db.update_project(pid, status="running")
        self._kick_llm()

    def _detect(self, p: dict, segments):
        pid = p["id"]
        det = self.detector()
        threshold = p["threshold"] if p["threshold"] is not None else det.threshold
        if p["threshold"] is None:
            db.update_project(pid, threshold=threshold)
        windows = build_windows(segments)
        finder = CandidateFinder(segments)
        start = p["detect_next_window"] or 0
        self._stage(pid, "detect", len(windows), start)
        next_id = db.conn().execute("SELECT COALESCE(MAX(id) + 1, 0) FROM candidates WHERE project_id = ?", (pid,)).fetchone()[0]
        for b in range(start, len(windows), DETECT_BLOCK):
            self._check_cancel(pid)
            block = windows[b:b + DETECT_BLOCK]
            scores = det.score(block)
            cands = finder.find(block, scores, threshold, next_id)
            next_id += len(cands)
            with db.tx() as c:
                c.executemany(
                    "INSERT OR REPLACE INTO candidates (project_id, id, seg, ref, i1, i2, c0, c1, words, score, before, after) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(pid, x.id, x.seg, x.ref, x.i1, x.i2, x.c0, x.c1, x.words, x.score, x.before, x.after) for x in cands])
                c.execute("UPDATE projects SET stage_done = ?, detect_next_window = ?, updated_at = ? WHERE id = ?",
                          (b + len(block), b + len(block), db.now(), pid))

    def _align(self, pid: int):
        path = db.ref_path(pid)
        if not path.exists():
            return
        rows = db.conn().execute("SELECT id, before, words, after FROM candidates WHERE project_id = ? ORDER BY seg, c0", (pid,)).fetchall()
        self._stage(pid, "align", len(rows))
        ref = Reference(path.read_text(encoding="utf-8"))
        out = []
        for n, r in enumerate(rows, 1):
            a = ref.align(r["before"], r["words"], r["after"])
            out.append((a.status, a.ref_words, a.excerpt, pid, r["id"]))
            if n % 200 == 0:
                db.update_project(pid, stage_done=n)
        with db.tx() as c:
            c.executemany("UPDATE candidates SET align_status = ?, ref_words = ?, ref_excerpt = ? WHERE project_id = ? AND id = ?", out)
        db.update_project(pid, stage_done=len(rows))

    # ---- LLM thread -----------------------------------------------------------------------------------------------------------------
    def _llm_thread(self):
        self.llm_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.llm_loop)
        self.llm_loop.create_task(self._llm_poll())
        self.llm_loop.run_forever()

    def _kick_llm(self):
        if self.llm_loop:
            self.llm_loop.call_soon_threadsafe(lambda: None)

    async def _llm_poll(self):
        while not self.stop.is_set():
            for (pid,) in db.conn().execute("SELECT id FROM projects WHERE status = 'running' AND stage = 'llm'").fetchall():
                if pid not in self.llm_running:
                    self.llm_running.add(pid)
                    asyncio.get_running_loop().create_task(self._llm_project(pid))
            await asyncio.sleep(2)

    async def _llm_project(self, pid: int):
        try:
            await self._llm(pid)
        except Cancelled:
            db.update_project(pid, status="cancelled", stage_started=None)
        except Exception as e:
            log.exception("llm pass failed for %s", pid)
            db.update_project(pid, status="failed", error=f"LLM pass: {type(e).__name__}: {e}")
        finally:
            self.llm_running.discard(pid)

    async def _llm(self, pid: int):
        p = db.project(pid)
        spec = ModelSpec.parse(p["llm"] or DEFAULT_LLM)
        if not p["llm"]:
            db.update_project(pid, llm=spec.spec)
        c = db.conn()
        total = c.execute("SELECT COUNT(*) FROM candidates WHERE project_id = ?", (pid,)).fetchone()[0]
        todo = [dict(r) for r in c.execute(
            "SELECT * FROM candidates WHERE project_id = ? AND bucket = 'unchecked' ORDER BY seg, c0", (pid,))]
        self._stage(pid, "llm", total, total - len(todo))
        usage = Usage.from_dict(json.loads(p["usage"] or "{}"))  # keep cost from before a restart
        judge = Judge(spec, usage=usage, should_stop=lambda: self._cancel_requested(pid))
        judge.sem = self.sem                                   # one limit across all projects
        title = f"{p['title']} ({p['version_title']})"
        done = total - len(todo)

        def item(r):
            return Item(str(r["id"]), f"{r['ref']} ({p['version_title']})", excerpt(r["before"], r["words"], r["after"]),
                        r["ref_excerpt"] if r["align_status"] == "differs" else None)

        by_id = {r["id"]: r for r in todo}

        def save(verdicts):
            nonlocal done
            rows = []
            for v in verdicts:
                r = by_id[int(v.key)]
                rows.append(classify(r, v) + (pid, r["id"]))
            with db.tx() as cx:
                cx.executemany("UPDATE candidates SET llm_verdict = ?, llm_fix = ?, llm_confidence = ?, llm_reason = ?, llm_model = ?, "
                               "llm_error = ?, proposed_fix = ?, fix_source = ?, bucket = ? WHERE project_id = ? AND id = ?", rows)
            done += len(verdicts)
            db.update_project(pid, stage_done=done, usage=db.dumps(usage.to_dict()))

        # one continuous pool (bounded by the shared semaphore): a slow call never holds up the others
        try:
            await judge.judge([item(r) for r in todo], on_done=save)
        except Stop:
            raise Cancelled()
        failed = db.conn().execute("SELECT COUNT(*) FROM candidates WHERE project_id = ? AND bucket = 'unchecked'", (pid,)).fetchone()[0]
        note = (f"{failed} candidate(s) could not be checked by the LLM (provider errors); they are in the review queue unchecked. "
                "Retry LLM check runs only those.") if failed else None
        db.update_project(pid, status="ready", stage="ready", stage_done=total - failed, stage_total=total, stage_started=None, error=note)


def classify(r: dict, v) -> tuple:
    """(llm_verdict, llm_fix, llm_confidence, llm_reason, llm_model, llm_error, proposed_fix, fix_source, bucket) for one verdict.

    Reference agrees + LLM "correct" -> dismissed. Reference agrees + LLM "error" stays in the queue: in Phase 0 (experiments/REPORT.md)
    dismissing every agreeing candidate dropped real errors the reference shared or that differ only in punctuation (recall 0.83);
    this backstop kept recall at 0.96 with the same false-positive removal.
    A failed call leaves the candidate unchecked (still in the review queue, with the reference reading as the proposal when there is
    one) so "Retry LLM check" can run it again."""
    # an empty reference reading (nothing there, or punctuation only) is not a proposal: accepting it would delete the words
    ref_fix = (r["ref_words"] or None) if r["align_status"] == "differs" else None
    if r["align_status"] == "agrees" and v.verdict == "correct":
        return ("correct", None, v.confidence, v.reason, v.model, None, None, None, "dismissed")
    if v.verdict is None:
        return (None, None, None, None, v.model, v.error, ref_fix, "reference" if ref_fix is not None else None, "unchecked")
    if v.verdict == "correct":
        return ("correct", None, v.confidence, v.reason, v.model, None, ref_fix, "reference" if ref_fix is not None else None, "likely_ok")
    # An empty LLM fix is ambiguous (delete? unknown?), so it is not proposed: the reviewer edits instead. Deleting stays an explicit edit.
    fix = v.fix or ref_fix
    source = None if fix is None else ("reference+llm" if ref_fix is not None and fix == ref_fix else
                                       "reference" if not v.fix else "llm")
    return ("error", v.fix, v.confidence, v.reason, v.model, None, fix, source, "likely_error")
