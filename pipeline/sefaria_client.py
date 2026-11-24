"""Read-only access to the public Sefaria API: book titles, a book's Hebrew versions, and a version's segments.

A version's text comes from the export CSV (download/version/<title> - <lang> - <versionTitle>.csv), which is already in the modtools
upload format with canonical segment refs, so an export of corrections round-trips exactly.
"""
import csv
import io
import os
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

BASE = os.environ.get("SEFARIA_BASE_URL", "https://www.sefaria.org").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=15.0)
HEADERS = {"User-Agent": "sefaria-text-correction-pipeline"}


@dataclass
class Version:
    title: str                       # Index title
    version_title: str
    language: str
    version_source: str = ""
    version_notes: str = ""
    segments: list[tuple[str, str]] = field(default_factory=list)   # (canonical segment ref, raw text) in book order

    @property
    def words(self) -> int:
        return sum(len(t.split()) for _, t in self.segments)


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=TIMEOUT, headers=HEADERS, follow_redirects=True)


def index_titles() -> list[str]:
    with _client() as c:
        r = c.get("/api/index/titles")
        r.raise_for_status()
        return r.json()["books"]


def hebrew_versions(title: str) -> list[dict]:
    """Hebrew versions of a book, highest priority first, with the fields the UI shows."""
    with _client() as c:
        r = c.get(f"/api/texts/versions/{quote(title)}")
        r.raise_for_status()
    out = [{"versionTitle": v["versionTitle"], "versionSource": v.get("versionSource") or "", "license": v.get("license") or "",
            "priority": v.get("priority") or 0, "isPrimary": bool(v.get("isPrimary")), "versionNotes": v.get("versionNotes") or ""}
           for v in r.json() if v.get("language") == "he"]
    return sorted(out, key=lambda v: (-float(v["priority"] or 0), not v["isPrimary"], v["versionTitle"]))


def parse_version_csv(text: str) -> Version:
    """Parse the single-book export/upload CSV: 5 header rows (label, value), then (ref, text) rows."""
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 5 or rows[0][0].strip().lower() != "index title":
        raise ValueError("not a Sefaria version CSV (expected an 'Index Title' header row)")
    head = [r[1] if len(r) > 1 else "" for r in rows[:5]]
    segments = [(r[0], r[1]) for r in rows[5:] if len(r) > 1 and r[0]]
    return Version(title=head[0], version_title=head[1], language=head[2], version_source=head[3], version_notes=head[4],
                   segments=segments)


def fetch_version(title: str, version_title: str, language: str = "he") -> Version:
    path = f"/download/version/{quote(f'{title} - {language} - {version_title}')}.csv"
    with _client() as c:
        r = c.get(path)
    if r.status_code != 200 or "csv" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"Sefaria would not export {title} / {version_title} (HTTP {r.status_code}). "
                           "Copyrighted versions cannot be downloaded through the export endpoint.")
    return parse_version_csv(r.text)
