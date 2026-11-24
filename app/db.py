"""SQLite storage (one file on the Coolify volume). One connection per thread; WAL so the worker and web requests don't block each other."""
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DB_PATH = DATA_DIR / "app.db"
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    title TEXT NOT NULL,
    version_title TEXT NOT NULL,
    version_source TEXT DEFAULT '',
    version_notes TEXT DEFAULT '',
    ref_filename TEXT,                 -- uploaded reference file name (text kept in DATA_DIR/refs/<id>.txt)
    threshold REAL,
    llm TEXT,                          -- model spec label used for the LLM pass
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | running | ready | failed | cancelled
    stage TEXT DEFAULT 'queued',       -- queued | fetch | detect | align | llm | ready
    stage_done INTEGER DEFAULT 0,
    stage_total INTEGER DEFAULT 0,
    stage_started REAL,
    stage_base INTEGER DEFAULT 0,      -- stage_done when this run of the stage started (for the ETA after a resume)
    detect_next_window INTEGER DEFAULT 0,  -- resume point for the detector
    words INTEGER DEFAULT 0,
    usage TEXT DEFAULT '{}',
    error TEXT,
    cancel_requested INTEGER DEFAULT 0,
    archived INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    project_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    ref TEXT NOT NULL,
    original TEXT NOT NULL,
    override TEXT,                     -- reviewer's full-segment text, supersedes span fixes
    PRIMARY KEY (project_id, idx)
);
CREATE TABLE IF NOT EXISTS candidates (
    project_id INTEGER NOT NULL,
    id INTEGER NOT NULL,
    seg INTEGER NOT NULL,
    ref TEXT NOT NULL,
    i1 INTEGER, i2 INTEGER, c0 INTEGER NOT NULL, c1 INTEGER NOT NULL,
    words TEXT NOT NULL,
    score REAL,
    before TEXT, after TEXT,
    align_status TEXT,                 -- agrees | differs | not_found | NULL (no reference)
    ref_words TEXT, ref_excerpt TEXT,
    llm_verdict TEXT, llm_fix TEXT, llm_confidence REAL, llm_reason TEXT, llm_model TEXT, llm_error TEXT,
    proposed_fix TEXT,                 -- replacement for the flagged words (NULL: no proposal)
    fix_source TEXT,                   -- reference | llm | reference+llm
    bucket TEXT NOT NULL DEFAULT 'unchecked',   -- unchecked | likely_error | likely_ok | dismissed
    decision TEXT NOT NULL DEFAULT 'pending',   -- pending | accepted | rejected | overridden
    applied_fix TEXT,                  -- the replacement actually applied (accepted: proposed or edited)
    decided_by TEXT, decided_at REAL,
    PRIMARY KEY (project_id, id)
);
CREATE INDEX IF NOT EXISTS cand_seg ON candidates(project_id, seg);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    cand_id INTEGER,
    seg INTEGER,
    action TEXT NOT NULL,              -- accept | reject | override | edit_fix | reset
    prev TEXT NOT NULL,                -- JSON: candidate decision/applied_fix and segment override before the action
    next TEXT NOT NULL,                -- JSON: after
    old_text TEXT, new_text TEXT,      -- segment text before/after, for the history list
    user TEXT, at REAL NOT NULL,
    undone INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ev_proj ON events(project_id, id);
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=30000")
        _local.conn = c
    return c


def init():
    c = conn()
    c.executescript(SCHEMA)
    cols = {r[1] for r in c.execute("PRAGMA table_info(projects)")}
    if "stage_base" not in cols:                      # databases created before the column existed
        c.execute("ALTER TABLE projects ADD COLUMN stage_base INTEGER DEFAULT 0")
    (DATA_DIR / "refs").mkdir(parents=True, exist_ok=True)


class tx:
    """`with tx() as c:` — one IMMEDIATE transaction (serializes writers, so read-modify-write is safe)."""

    def __enter__(self):
        self.c = conn()
        self.c.execute("BEGIN IMMEDIATE")
        return self.c

    def __exit__(self, exc_type, *_):
        self.c.execute("ROLLBACK" if exc_type else "COMMIT")


def now() -> float:
    return time.time()


def row(r) -> dict | None:
    return dict(r) if r is not None else None


def update_project(pid: int, **fields):
    fields["updated_at"] = now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn().execute(f"UPDATE projects SET {cols} WHERE id = ?", [*fields.values(), pid])


def get_project(pid: int) -> dict | None:
    return row(conn().execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone())


def project(pid: int) -> dict:
    """A project that must exist (the worker's own, or one just created)."""
    p = get_project(pid)
    if p is None:
        raise KeyError(f"project {pid} does not exist")
    return p


def ref_path(pid: int) -> Path:
    return DATA_DIR / "refs" / f"{pid}.txt"


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)
