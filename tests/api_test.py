"""API and worker tests with a fake Sefaria, detector and LLM (no network, no model)."""
import csv
import importlib
import io
import time

import pytest
from fastapi.testclient import TestClient

from pipeline import sefaria_client
from pipeline.llm_filter import Verdict
from pipeline.sefaria_client import Version

SEGMENTS = [
    ("Test Book 1:1", "ויאמר משה אל העם אל תיראו התיצבו וראו את ישועח ה' אשר יעשה לכם היום"),
    ("Test Book 1:2", "כי אשר ראיתם את מצרים היום <b>לא</b> תספו לראתם עוד עד עולם"),
    ("Test Book 2:1", "ה' ילחם לכם ואתם תחרשון ויאמר ה' אל משה מה תצעק אלי"),
]
FLAG = {"ישועח", "תספו", "תצעק"}           # the fake detector flags these words
LLM = {"ישועח": ("error", "ישועת"), "תספו": ("correct", ""), "תצעק": ("error", "תצעק")}


class FakeDetector:
    threshold = 0.5

    def score(self, windows, **_):
        return {w.id: [0.9 if t in FLAG else 0.1 for t in w.tokens] for w in windows}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEV_AUTH_EMAIL", "tester@sefaria.org")
    monkeypatch.setenv("SECURE_COOKIES", "false")
    from app import db
    importlib.reload(db)
    from app import review, jobs, main
    importlib.reload(review); importlib.reload(jobs); importlib.reload(main)
    monkeypatch.setattr(sefaria_client, "fetch_version", lambda t, v, language="he": Version(t, v, "he", "http://src", "", list(SEGMENTS)))

    async def fake_judge(self, items, on_done=None):
        out = []
        for it in items:
            word = it.excerpt.split("⟦")[1].split("⟧")[0]
            verdict, fix = LLM[word]
            out.append(Verdict(it.key, verdict, fix, 0.9, "fake", "fake"))
        if on_done:
            on_done(out)
        return out
    monkeypatch.setattr(jobs.Judge, "judge", fake_judge)
    main.worker = jobs.Worker(detector_factory=FakeDetector)
    with TestClient(main.app) as c:
        yield c


def wait_ready(client, pid, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        p = client.get(f"/api/projects/{pid}/progress").json()
        if p["status"] in ("ready", "failed"):
            return p
        time.sleep(0.2)
    raise AssertionError(f"not ready: {p}")


def test_end_to_end(client):
    ref = ("ויאמר משה אל העם אל תיראו התיצבו וראו את ישועת ה' אשר יעשה לכם היום כי אשר ראיתם את מצרים היום לא "
           "תספו לראתם עוד עד עולם ה' ילחם לכם ואתם תחרשון ויאמר ה' אל משה מה תצעק אלי")
    r = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1"},
                    files={"reference": ("ref.txt", ref.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    p = wait_ready(client, pid)
    assert p["status"] == "ready", p

    cands = client.get(f"/api/projects/{pid}/candidates?filter=all").json()
    by = {c["words"]: c for c in cands}
    assert set(by) == FLAG
    assert by["ישועח"]["bucket"] == "likely_error" and by["ישועח"]["proposed_fix"] == "ישועת"
    assert by["תספו"]["bucket"] == "dismissed"           # reference agrees and LLM says correct
    assert by["תצעק"]["bucket"] == "likely_error"        # reference agrees, but the LLM says error: backstop keeps it
    todo = client.get(f"/api/projects/{pid}/candidates?filter=todo").json()
    assert [c["words"] for c in todo] == ["ישועח", "תצעק"]

    cid = by["ישועח"]["id"]
    d = client.get(f"/api/projects/{pid}/candidates/{cid}").json()
    assert "ישועת" in d["proposed_text"] and d["align_status"] == "differs"

    # accept, then undo, then accept with an edited fix
    assert client.post(f"/api/projects/{pid}/candidates/{cid}/decision", json={"action": "accept"}).json()["decision"] == "accepted"
    assert client.post(f"/api/projects/{pid}/undo").json()["decision"] == "pending"
    r = client.post(f"/api/projects/{pid}/candidates/{cid}/decision", json={"action": "accept", "text": "ישועתו"}).json()
    assert "ישועתו" in r["current"]

    # whole-segment override on another candidate, then export
    cid2 = by["תספו"]["id"]
    new = "כי אשר ראיתם את מצרים היום <b>לא</b> תוסיפו לראתם עוד עד עולם"
    r = client.post(f"/api/projects/{pid}/candidates/{cid2}/decision", json={"action": "override", "text": new}).json()
    assert r["current"] == new
    hist = client.get(f"/api/projects/{pid}/history").json()
    assert [h["action"] for h in hist][:3] == ["override", "accept", "accept"] and hist[2]["undone"] == 1

    out = client.get(f"/api/projects/{pid}/export.csv")
    assert out.status_code == 200 and "corrections" in out.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(out.text)))
    assert [r[1] for r in rows[:3]] == ["Test Book", "V1", "he"]
    assert rows[5:] == [["Test Book 1:1", SEGMENTS[0][1].replace("ישועח", "ישועתו")], ["Test Book 1:2", new]]

    # an empty override is refused (it would blank the segment on upload)
    r = client.post(f"/api/projects/{pid}/candidates/{cid2}/decision", json={"action": "override", "text": "  "})
    assert r.status_code == 400


def test_project_management(client):
    pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1", "name": "first"}).json()["id"]
    wait_ready(client, pid)
    assert client.patch(f"/api/projects/{pid}", json={"name": "renamed"}).json()["name"] == "renamed"
    client.patch(f"/api/projects/{pid}", json={"archived": True})
    assert client.get("/api/projects").json() == []
    assert [p["id"] for p in client.get("/api/projects?archived=true").json()] == [pid]
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert client.get(f"/api/projects/{pid}").status_code == 404
