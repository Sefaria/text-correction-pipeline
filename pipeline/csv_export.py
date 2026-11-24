"""The corrections file for Sefaria modtools' "Upload texts from CSV" (POST /api/text-upload, single-book format).

Format (Sefaria-Project sefaria/export.py `_import_versions_from_csv`): 5 header rows in column B (Index Title, Version Title, Language,
Version Source, Version Notes), then one row per segment: canonical segment ref, full new segment text.

Only changed segments are written. A row with an empty cell would BLANK that segment on upload (sefaria/tracker.py modify_bulk_text
treats "" as new text), so segments edited down to nothing are refused rather than exported.
"""
import csv
import io
from datetime import date


class EmptySegmentError(ValueError):
    pass


def export_csv(title: str, version_title: str, version_source: str, version_notes: str,
               changes: list[tuple[str, str]], language: str = "he") -> str:
    """`changes`: [(segment ref, new full text)] in book order."""
    empty = [ref for ref, text in changes if not text.strip()]
    if empty:
        raise EmptySegmentError(f"these segments would be uploaded empty, which blanks them on Sefaria: {', '.join(empty[:10])}")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Index Title", title])
    w.writerow(["Version Title", version_title])
    w.writerow(["Language", language])
    w.writerow(["Version Source", version_source])
    w.writerow(["Version Notes", version_notes])
    for ref, text in changes:
        w.writerow([ref, text])
    return buf.getvalue()


def export_filename(title: str, version_title: str, language: str = "he", today: date | None = None) -> str:
    safe = lambda s: "".join(ch for ch in s if ch not in '/\\:*?"<>|').strip()
    return f"{safe(title)} - {language} - {safe(version_title)} - corrections-{(today or date.today()).isoformat()}.csv"
