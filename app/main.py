"""FastAPI app: projects, review API, CSV export, static UI, Google sign-in restricted to one domain."""
import logging
from contextlib import asynccontextmanager
import os
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from pipeline import sefaria_client
from pipeline.csv_export import EmptySegmentError
from pipeline.reference import load_text

from . import db, review
from .jobs import DEFAULT_LLM, Worker, parse_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
WEB = Path(__file__).resolve().parent.parent / "web"
ALLOWED_DOMAIN = os.environ.get("GOOGLE_ALLOWED_DOMAIN", "sefaria.org")
DEV_USER = os.environ.get("DEV_AUTH_EMAIL")              # local development only: skip Google sign-in
MAX_REFERENCE_BYTES = 50 * 1024 * 1024

@asynccontextmanager
async def lifespan(_app):
    worker.start()
    yield


SESSION_SECRET = os.environ.get("SESSION_SECRET")
if not SESSION_SECRET:
    if not DEV_USER:
        # a known fallback key would let anyone forge a signed-in session cookie
        raise RuntimeError("SESSION_SECRET must be set (or DEV_AUTH_EMAIL, for local development only)")
    SESSION_SECRET = secrets.token_urlsafe(32)

app = FastAPI(title="Text Correction Pipeline", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET,
                   https_only=os.environ.get("SECURE_COOKIES", "true") == "true", same_site="lax", max_age=14 * 24 * 3600)
worker = Worker()

oauth = None
if os.environ.get("GOOGLE_CLIENT_ID"):
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register("google", client_id=os.environ["GOOGLE_CLIENT_ID"], client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
                   server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
                   client_kwargs={"scope": "openid email profile"})
elif not DEV_USER:
    logging.warning("Neither GOOGLE_CLIENT_ID nor DEV_AUTH_EMAIL is set: every request will be refused.")


# ---- auth ----------------------------------------------------------------------------------------------------------------------------

def user(request: Request) -> str:
    if DEV_USER:
        return DEV_USER
    email = request.session.get("email")
    if not email:
        raise HTTPException(401, "sign in required")
    return email


@app.get("/login")
async def login(request: Request):
    if not oauth:
        raise HTTPException(500, "Google sign-in is not configured")
    redirect = str(request.url_for("auth_callback"))
    if os.environ.get("PUBLIC_URL"):
        redirect = os.environ["PUBLIC_URL"].rstrip("/") + "/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect, hd=ALLOWED_DOMAIN)


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    info = token.get("userinfo") or {}
    email = (info.get("email") or "").lower()
    if not info.get("email_verified") or not email.endswith("@" + ALLOWED_DOMAIN):
        request.session.clear()
        return Response(f"Only {ALLOWED_DOMAIN} accounts can use this tool.", status_code=403)
    request.session["email"] = email
    return RedirectResponse("/")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


@app.get("/healthz")
def healthz():
    db.conn().execute("SELECT 1")
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    try:
        return {"email": user(request), "login_url": None}
    except HTTPException:
        return JSONResponse({"email": None, "login_url": "/login"}, status_code=401)


# ---- Sefaria lookups -----------------------------------------------------------------------------------------------------------------
_titles: dict = {"at": 0, "list": []}


@app.get("/api/books")
def books(q: str = "", u: str = Depends(user)):
    if time.time() - _titles["at"] > 6 * 3600:
        _titles["list"] = sefaria_client.index_titles(); _titles["at"] = time.time()
    ql = q.strip().lower()
    if not ql:
        return []
    hits = [t for t in _titles["list"] if ql in t.lower()]
    hits.sort(key=lambda t: (not t.lower().startswith(ql), len(t)))
    return hits[:25]


@app.get("/api/versions")
def versions(title: str, u: str = Depends(user)):
    return sefaria_client.hebrew_versions(title)


# ---- projects ------------------------------------------------------------------------------------------------------------------------

def _project(pid: int) -> dict:
    p = db.get_project(pid)
    if not p:
        raise HTTPException(404, "no such project")
    return p


def _progress(p: dict) -> dict:
    eta = None
    worked = (p["stage_done"] or 0) - (p.get("stage_base") or 0)       # work done in this run of the stage (not before a resume)
    if p["stage_started"] and p["stage_total"] and worked > 0:
        elapsed = time.time() - p["stage_started"]
        rate = worked / max(elapsed, 1e-6)
        if elapsed > 5:
            eta = max(0, (p["stage_total"] - p["stage_done"]) / rate)
    return {"status": p["status"], "stage": p["stage"], "done": p["stage_done"], "total": p["stage_total"], "eta_seconds": eta,
            "queue_position": worker.queue_position(p["id"]), "error": p["error"]}


def _summary(p: dict) -> dict:
    import json
    return {k: p[k] for k in ("id", "owner", "name", "title", "version_title", "ref_filename", "threshold", "llm", "words", "archived",
                              "created_at", "updated_at")} | {"progress": _progress(p), "counts": review.counts(p["id"]),
                                                              "usage": json.loads(p["usage"] or "{}")}


@app.get("/api/projects")
def projects(all: bool = False, archived: bool = False, u: str = Depends(user)):
    q = "SELECT * FROM projects WHERE archived = ?" + ("" if all else " AND owner = ?") + " ORDER BY updated_at DESC"
    rows = db.conn().execute(q, (int(archived),) if all else (int(archived), u)).fetchall()
    return [_summary(dict(r)) for r in rows]


@app.post("/api/projects")
async def create_project(title: str = Form(...), version_title: str = Form(...), name: str = Form(""),
                         threshold: float | None = Form(None), llm: str = Form(""),
                         reference: UploadFile | None = File(None), u: str = Depends(user)):
    if llm:
        parse_spec(llm)
    ref_text = None
    if reference is not None and reference.filename:
        data = await reference.read()
        if len(data) > MAX_REFERENCE_BYTES:
            raise HTTPException(413, "reference file is larger than 50 MB")
        try:
            ref_text = load_text(reference.filename, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if len(ref_text.split()) < 20:
            raise HTTPException(400, "no text could be read from the reference file")
    t = db.now()
    cur = db.conn().execute(
        "INSERT INTO projects (owner, name, title, version_title, ref_filename, threshold, llm, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (u, name.strip() or f"{title} — {version_title}", title, version_title, reference.filename if ref_text else None,
         threshold, llm or None, t, t))
    pid = cur.lastrowid
    if ref_text:
        db.ref_path(pid).write_text(ref_text, encoding="utf-8")
    worker.wake.set()
    return _summary(db.get_project(pid))


@app.get("/api/projects/{pid}")
def get_project(pid: int, u: str = Depends(user)):
    return _summary(_project(pid))


@app.get("/api/projects/{pid}/progress")
def progress(pid: int, u: str = Depends(user)):
    p = _project(pid)
    return _progress(p) | {"counts": review.counts(pid)}


class Patch(BaseModel):
    name: str | None = None
    archived: bool | None = None


@app.patch("/api/projects/{pid}")
def patch_project(pid: int, body: Patch, u: str = Depends(user)):
    _project(pid)
    fields = {k: (int(v) if isinstance(v, bool) else v) for k, v in body.model_dump().items() if v is not None}
    if fields:
        db.update_project(pid, **fields)
    return _summary(_project(pid))


@app.post("/api/projects/{pid}/cancel")
def cancel(pid: int, u: str = Depends(user)):
    p = _project(pid)
    if p["status"] == "queued":
        db.update_project(pid, status="cancelled")
    elif p["status"] == "running":
        db.update_project(pid, cancel_requested=1)
    return _progress(_project(pid))


@app.post("/api/projects/{pid}/retry")
def retry(pid: int, u: str = Depends(user)):
    p = _project(pid)
    if p["status"] not in ("failed", "cancelled"):
        raise HTTPException(409, "only failed or cancelled projects can be retried")
    status = "running" if p["stage"] == "llm" else "queued"
    db.update_project(pid, status=status, cancel_requested=0, error=None)
    worker.wake.set()
    return _progress(_project(pid))


@app.delete("/api/projects/{pid}")
def delete_project(pid: int, u: str = Depends(user)):
    p = _project(pid)
    if p["status"] == "running":
        raise HTTPException(409, "cancel the project before deleting it")
    with db.tx() as c:
        for t in ("events", "candidates", "segments"):
            c.execute(f"DELETE FROM {t} WHERE project_id = ?", (pid,))
        c.execute("DELETE FROM projects WHERE id = ?", (pid,))
    db.ref_path(pid).unlink(missing_ok=True)
    return {"deleted": pid}


# ---- review --------------------------------------------------------------------------------------------------------------------------

@app.get("/api/projects/{pid}/candidates")
def candidates(pid: int, filter: str = "todo", u: str = Depends(user)):
    _project(pid)
    try:
        return review.list_candidates(pid, filter)
    except review.ReviewError as e:
        raise HTTPException(400, str(e))


@app.get("/api/projects/{pid}/candidates/{cid}")
def candidate(pid: int, cid: int, u: str = Depends(user)):
    try:
        return review.candidate_detail(pid, cid)
    except review.ReviewError as e:
        raise HTTPException(404, str(e))


class Decision(BaseModel):
    action: str
    text: str | None = None


@app.post("/api/projects/{pid}/candidates/{cid}/decision")
def decision(pid: int, cid: int, body: Decision, u: str = Depends(user)):
    _project(pid)
    try:
        return review.decide(pid, cid, body.action, u, body.text)
    except review.ReviewError as e:
        raise HTTPException(400, str(e))


@app.post("/api/projects/{pid}/undo")
def undo(pid: int, u: str = Depends(user)):
    _project(pid)
    r = review.undo(pid, u)
    if r is None:
        raise HTTPException(409, "nothing to undo")
    return r


@app.get("/api/projects/{pid}/history")
def history(pid: int, limit: int = 20, u: str = Depends(user)):
    _project(pid)
    return review.history(pid, limit)


@app.get("/api/projects/{pid}/export.csv")
def export(pid: int, u: str = Depends(user)):
    _project(pid)
    try:
        name, body = review.export(pid)
    except EmptySegmentError as e:
        raise HTTPException(409, str(e))
    from urllib.parse import quote
    return Response(body.encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})


@app.get("/api/config")
def config(u: str = Depends(user)):
    return {"default_llm": DEFAULT_LLM, "domain": ALLOWED_DOMAIN}


# ---- UI ------------------------------------------------------------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/review")
def review_page():
    return FileResponse(WEB / "review.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
