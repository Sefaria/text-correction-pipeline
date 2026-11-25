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
FAIL: set = set()                         # words whose fake LLM call fails (provider error)


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
            calls.append(word)
            if word in FAIL:
                out.append(Verdict(it.key, None, model="fake", error="ServerError: 503 UNAVAILABLE"))
                continue
            verdict, fix = LLM[word]
            out.append(Verdict(it.key, verdict, fix, 0.9, "fake", "fake"))
        if on_done:
            on_done(out)
        return out
    monkeypatch.setattr(jobs.Judge, "judge", fake_judge)
    main.worker = jobs.Worker(detector_factory=FakeDetector)
    with TestClient(main.app) as c:
        yield c


calls: list = []


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
    # typed: no OCR letter-swap fixes, so ישועח/ישועת stays for the reviewer (automatic decisions: test_automatic_decisions)
    r = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1", "source": "typed"},
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


def test_classify_empty_reference_reading_is_not_a_proposal():
    from app.jobs import classify
    r = {"ref_words": "", "align_status": "differs"}            # reference has nothing here, or only punctuation
    for v in (Verdict("1", "error", "", 0.9, "", "m"), Verdict("1", None, model="m", error="boom"), Verdict("1", "correct", model="m")):
        row = classify(r, v)
        assert row[6] is None and row[7] is None, v              # proposed_fix, fix_source: nothing to accept, so nothing deleted


def two_flag_project(client, monkeypatch):
    """A project with one segment holding two separately flagged words (ויאמינח, עבדח)."""
    seg = ("Test Book 3:1", "ויאמינח בה' ובמשה עבדח אז ישיר")
    monkeypatch.setitem(globals(), "SEGMENTS", [seg])
    monkeypatch.setitem(globals(), "FLAG", {"ויאמינח", "עבדח"})
    monkeypatch.setitem(globals(), "LLM", {"ויאמינח": ("error", "ויאמינו"), "עבדח": ("error", "עבדו")})
    pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1"}).json()["id"]
    assert wait_ready(client, pid)["status"] == "ready"
    by = {c["words"]: c["id"] for c in client.get(f"/api/projects/{pid}/candidates?filter=all").json()}
    return pid, seg[1], by["ויאמינח"], by["עבדח"]


def decide(client, pid, cid, action, text=None):
    return client.post(f"/api/projects/{pid}/candidates/{cid}/decision", json={"action": action, "text": text})


def exported(client, pid):
    return list(csv.reader(io.StringIO(client.get(f"/api/projects/{pid}/export.csv").text)))[5:]


def test_reject_and_reset_take_their_change_out_of_an_override(client, monkeypatch):
    pid, original, a, b = two_flag_project(client, monkeypatch)
    decide(client, pid, a, "accept")
    edited = "ויאמינו בה' ובמשה עבדו אז ישיר!"
    assert decide(client, pid, b, "override", edited).json()["current"] == edited
    # rejecting A reverses its fix inside the override, keeping B's hand edit
    assert decide(client, pid, a, "reject").json()["current"] == "ויאמינח בה' ובמשה עבדו אז ישיר!"
    assert exported(client, pid) == [["Test Book 3:1", "ויאמינח בה' ובמשה עבדו אז ישיר!"]]
    # resetting B (the one that wrote the override) drops the override: nothing is left to export
    r = decide(client, pid, b, "reset").json()
    assert r["decision"] == "pending" and r["current"] == original and not r["overridden"]
    assert exported(client, pid) == []


def test_reaccepting_a_decided_candidate_is_refused(client, monkeypatch):
    pid, _, a, _ = two_flag_project(client, monkeypatch)
    assert "ויאמינו!" in decide(client, pid, a, "accept", "ויאמינו!").json()["current"]
    r = decide(client, pid, a, "accept")                         # e.g. Enter pressed again on the reviewer's own fix
    assert r.status_code == 400
    assert "ויאמינו!" in client.get(f"/api/projects/{pid}/candidates/{a}").json()["current"]


def test_markup_inside_the_span_is_not_auto_fixed(client, monkeypatch):
    pid, _, a, _ = two_flag_project(client, monkeypatch)
    from app import db
    with db.tx() as c:                                           # as if the flagged span were '<b>ויאמינח</b> בה''
        c.execute("UPDATE candidates SET words = '<b>ויאמינח</b> בה''' WHERE project_id = ? AND id = ?", (pid, a))
    assert client.get(f"/api/projects/{pid}/candidates/{a}").json()["proposed_text"] is None
    assert decide(client, pid, a, "accept", "ויאמינו בה'").status_code == 400


def test_undo_only_undoes_your_own_change(client, monkeypatch):
    pid, original, a, b = two_flag_project(client, monkeypatch)
    from app import review
    decide(client, pid, b, "reject")                             # tester@sefaria.org
    review.decide(pid, a, "accept", "other@sefaria.org")         # another reviewer, later
    assert client.post(f"/api/projects/{pid}/undo").json()["id"] == b
    cands = {c["id"]: c["decision"] for c in client.get(f"/api/projects/{pid}/candidates?filter=all").json()}
    assert cands == {a: "accepted", b: "pending"}


def test_cancel_pending_at_restart_becomes_cancelled(client):
    from app import db, jobs
    pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1"}).json()["id"]
    wait_ready(client, pid)
    db.update_project(pid, status="running", stage="detect", cancel_requested=1)
    jobs.Worker.recover()
    assert db.get_project(pid)["status"] == "cancelled"
    assert client.post(f"/api/projects/{pid}/retry").status_code == 200


def test_missing_session_secret_refuses_to_start(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DEV_AUTH_EMAIL", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    from app import main
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        importlib.reload(main)
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    importlib.reload(main)


def test_failed_llm_calls_stay_unchecked_and_retry_runs_only_them(client):
    FAIL.add("תספו")
    try:
        pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1"}).json()["id"]
        p = wait_ready(client, pid)
        assert p["status"] == "ready" and "could not be checked" in p["error"]
        by = {c["words"]: c for c in client.get(f"/api/projects/{pid}/candidates?filter=all").json()}
        assert by["תספו"]["bucket"] == "unchecked" and by["ישועח"]["bucket"] == "likely_error"
        assert "תספו" in [c["words"] for c in client.get(f"/api/projects/{pid}/candidates?filter=todo").json()]
    finally:
        FAIL.clear()
    calls.clear()
    assert client.post(f"/api/projects/{pid}/retry").status_code == 200
    p = wait_ready(client, pid)
    assert p["status"] == "ready" and p["error"] is None
    assert calls == ["תספו"]                                          # only the failed candidate was re-checked
    by = {c["words"]: c for c in client.get(f"/api/projects/{pid}/candidates?filter=all").json()}
    assert by["תספו"]["bucket"] == "likely_ok"


def test_automatic_decisions(client, monkeypatch):
    """ישועח: the reference reads ישועת (ח/ת, an OCR pair) and the LLM agrees -> accepted. ראיחם: the LLM proposes a different fix than
    the reference -> stays. לאבא: same letters, the reference divides לא בא -> accepted. ר״הי: the LLM only moves the gershayim ->
    dismissed (rejected). Then a reviewer resets one, and running the automatic pass again leaves it alone."""
    base = ("ויאמר משה אל העם אל תיראו התיצבו וראו את ישועת ה' אשר יעשה לכם היום כי אשר ראיתם את מצרים היום לא "
            "בא לראתם עוד עד עולם ה' ילחם לכם ואתם תחרשון ויאמר רה״י ה' אל משה מה תצעק אלי")
    segs = [("Test Book 1:1", "ויאמר משה אל העם אל תיראו התיצבו וראו את <i>ישועח</i> ה' אשר יעשה לכם היום"),
            ("Test Book 1:2", "כי אשר ראיחם את מצרים היום לאבא לראתם עוד עד עולם"),
            ("Test Book 2:1", "ה' ילחם לכם ואתם תחרשון ויאמר ר״הי ה' אל משה מה תצעק אלי")]
    monkeypatch.setitem(globals(), "SEGMENTS", segs)
    monkeypatch.setitem(globals(), "FLAG", {"<i>ישועח</i>", "ראיחם", "לאבא", "ר״הי"})
    monkeypatch.setitem(globals(), "LLM", {"ישועח": ("error", "ישועת"), "ראיחם": ("error", "ראיתים"), "לאבא": ("error", "לא בא"),
                                           "ר״הי": ("error", "רה״י")})
    pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1"},
                      files={"reference": ("ref.txt", base.encode(), "text/plain")}).json()["id"]
    assert wait_ready(client, pid)["status"] == "ready"

    p = client.get(f"/api/projects/{pid}").json()
    assert p["script"] == "square" and {"pair": "ח/ת", "allowed": True} .items() <= p["ocr_swaps"]["pairs"][0].items()
    assert p["counts"]["auto"] == 3 and p["counts"]["todo"] == 1
    auto = {c["words"]: c for c in client.get(f"/api/projects/{pid}/candidates?filter=auto").json()}
    assert {w: c["decision"] for w, c in auto.items()} == {"<i>ישועח</i>": "accepted", "לאבא": "accepted", "ר״הי": "rejected"}
    assert {h["user"] for h in client.get(f"/api/projects/{pid}/history").json()} == {"auto: OCR swaps", "auto: spacing", "auto: gershayim"}
    assert exported(client, pid) == [["Test Book 1:1", segs[0][1].replace("ישועח", "ישועת")],
                                     ["Test Book 1:2", segs[1][1].replace("לאבא", "לא בא")]]

    # a reviewer resets one; the automatic pass never overrides a reviewer
    assert decide(client, pid, auto["<i>ישועח</i>"]["id"], "reject").json()["decision"] == "rejected"
    r = client.post(f"/api/projects/{pid}/auto?realign_reference=true").json()
    assert r["decided"] == {} and client.get(f"/api/projects/{pid}").json()["counts"]["auto"] == 2

    assert client.patch(f"/api/projects/{pid}", json={"script": "rashi"}).json()["script"] == "rashi"
    assert client.patch(f"/api/projects/{pid}", json={"script": "gothic"}).status_code == 400

    # a typed text gets no letter-swap fixes: after a reset, only the spacing fix comes back
    assert client.patch(f"/api/projects/{pid}", json={"source": "typed"}).json()["source"] == "typed"
    assert not any(s["allowed"] for s in client.get(f"/api/projects/{pid}").json()["ocr_swaps"]["pairs"])
    decide(client, pid, auto["<i>ישועח</i>"]["id"], "reset"); decide(client, pid, auto["לאבא"]["id"], "reset")
    assert client.post(f"/api/projects/{pid}/auto").json()["decided"] == {"auto: spacing": 1}


def test_typed_source_moves_a_letter_across_a_space(client, monkeypatch):
    """אל ⟦אהוא⟧: only אהוא is flagged, but the fix is in the space before it (אלא הוא): the segment is edited automatically."""
    ref = "ויאמר משה אל העם דאמר משעת הנחתו אלא הוא ר' יהושע דמתני' בר פלוגתיה דר' אליעזר ועוד דברים רבים כאן"
    monkeypatch.setitem(globals(), "SEGMENTS", [("Test Book 1:1", "ויאמר משה אל העם דאמר משעת הנחתו אל אהוא ר' יהושע דמתני' בר "
                                                                  "פלוגתיה דר' אליעזר ועוד דברים רבים כאן")])
    monkeypatch.setitem(globals(), "FLAG", {"אהוא"})
    monkeypatch.setitem(globals(), "LLM", {"אהוא": ("error", "הוא")})
    pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1", "source": "typed"},
                      files={"reference": ("ref.txt", ref.encode(), "text/plain")}).json()["id"]
    assert wait_ready(client, pid)["status"] == "ready"
    (c,) = client.get(f"/api/projects/{pid}/candidates?filter=auto").json()
    assert c["decision"] == "overridden"
    assert exported(client, pid) == [["Test Book 1:1", "ויאמר משה אל העם דאמר משעת הנחתו אלא הוא ר' יהושע דמתני' בר "
                                                       "פלוגתיה דר' אליעזר ועוד דברים רבים כאן"]]
    assert client.get(f"/api/projects/{pid}/history").json()[0]["user"] == "auto: spacing"


def test_two_cross_word_fixes_in_one_segment(client, monkeypatch):
    """העי ⟦רהיה⟧ and ⟦הי⟧ העיר in one segment: the second is fixed in the text the first one edited."""
    ref = "ומי אמר כן וכי בתוך העיר היה יער ומכל זה מוכח שלא היה עיר ויישוב שם ואין לומר שאחר כך דברים רבים"
    monkeypatch.setitem(globals(), "SEGMENTS", [("Test Book 1:1", "ומי אמר כן וכי בתוך העי רהיה יער ומכל זה מוכח שלא הי העיר "
                                                                  "ויישוב שם ואין לומר שאחר כך דברים רבים")])
    monkeypatch.setitem(globals(), "FLAG", {"רהיה", "הי"})
    monkeypatch.setitem(globals(), "LLM", {"רהיה": ("error", "היה"), "הי": ("error", "היה")})
    pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1", "source": "typed"},
                      files={"reference": ("ref.txt", ref.encode(), "text/plain")}).json()["id"]
    assert wait_ready(client, pid)["status"] == "ready"
    assert client.get(f"/api/projects/{pid}").json()["counts"]["auto"] == 2
    assert exported(client, pid) == [["Test Book 1:1", "ומי אמר כן וכי בתוך העיר היה יער ומכל זה מוכח שלא היה עיר ויישוב שם "
                                                       "ואין לומר שאחר כך דברים רבים"]]


def test_automatic_pass_without_a_reference_only_handles_gershayim(client, monkeypatch):
    monkeypatch.setitem(globals(), "SEGMENTS", [("Test Book 1:1", "ויאמר ר״הי משה לאבא")])
    monkeypatch.setitem(globals(), "FLAG", {"ר״הי", "לאבא"})
    monkeypatch.setitem(globals(), "LLM", {"ר״הי": ("error", "רה״י"), "לאבא": ("error", "לא בא")})
    pid = client.post("/api/projects", data={"title": "Test Book", "version_title": "V1"}).json()["id"]
    wait_ready(client, pid)
    by = {c["words"]: c["decision"] for c in client.get(f"/api/projects/{pid}/candidates?filter=all").json()}
    assert by == {"ר״הי": "rejected", "לאבא": "pending"}
    assert client.post(f"/api/projects/{pid}/auto").json()["decided"] == {}
