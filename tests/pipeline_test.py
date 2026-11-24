import csv
import io

import pytest

from pipeline.candidates import find_candidates
from pipeline.csv_export import EmptySegmentError, export_csv
from pipeline.edits import SpanFix, apply_fixes, replacement_for
from pipeline.reference import Reference, load_text
from pipeline.sefaria_client import parse_version_csv
from pipeline.text import match_key
from pipeline.tokens import tokenize, words
from pipeline.windows import build_windows, section_of


def test_tokenize_splits_after_br_and_keeps_offsets():
    t = "<b>הא'</b><br>תורק. שמנים <br>טובים"
    assert words(t) == ["<b>הא'</b><br>", "תורק.", "שמנים", "<br>", "טובים"]
    assert all(t[a:b] for a, b in tokenize(t))


def test_section_of():
    assert section_of("Rashi on Nedarim 2a:1:3") == "Rashi on Nedarim 2a:1"
    assert section_of("Bekhor Shor, Genesis 1:4:1") == "Bekhor Shor, Genesis 1:4"
    assert section_of("Pesach Haggadah, Kadesh 3") == "Pesach Haggadah, Kadesh"


def test_windows_respect_sections_and_size():
    segs = [("B 1:1", "א " * 200), ("B 1:2", "ב " * 200), ("B 2:1", "ג " * 10), ("B 2:2", "ד " * 800)]
    ws = build_windows(segs, window=350)
    for w in ws:
        assert len(w.tokens) <= 350
        assert len({section_of(segs[s][0]) for s, _ in w.pos}) == 1
    # every token appears exactly once, in order
    flat = [p for w in ws for p in w.pos]
    assert flat == [(s, i) for s, (_, t) in enumerate(segs) for i in range(len(t.split()))]


def test_candidates_merge_adjacent_and_offsets():
    segs = [("B 1:1", "אחד <b>שתים</b> שלש ארבע"), ("B 1:2", "חמש שש")]
    ws = build_windows(segs)
    scores = {ws[0].id: [0.1, 0.9, 0.8, 0.1, 0.9, 0.1]}
    cs = find_candidates(segs, ws, scores, 0.5)
    assert [(c.seg, c.i1, c.i2) for c in cs] == [(0, 1, 3), (1, 0, 1)]
    c = cs[0]
    assert segs[0][1][c.c0:c.c1] == "<b>שתים</b> שלש" == c.words
    assert c.before == "אחד" and c.after.startswith("ארבע")
    assert "⟦" in c.excerpt()


def test_apply_fixes_multiple_and_deletion():
    t = "<b>אחד</b> שתים שלש ארבע"
    fixes = [SpanFix(11, 15, "שנים"), SpanFix(16, 20, "")]
    assert apply_fixes(t, fixes) == "<b>אחד</b> שנים ארבע"
    assert replacement_for("<b>שתים</b>", "שנים") == "<b>שנים</b>"


def test_match_key_ignores_niqqud_punct_finals():
    assert match_key("בְּרֵאשִׁית,") == match_key("בראשית")
    assert match_key('ר"ל') == match_key("רל")
    assert match_key("שלום") != match_key("שלוס")


def test_reference_alignment_agrees_differs_not_found():
    base = ("ויאמר משה אל העם אל תיראו התיצבו וראו את ישועת ה' אשר יעשה לכם היום כי אשר ראיתם את מצרים "
            "היום לא תספו לראתם עוד עד עולם ה' ילחם לכם ואתם תחרשון ויאמר ה' אל משה מה תצעק אלי דבר אל בני ישראל ויסעו")
    ref = Reference("מבוא כלשהו\n" + base.replace("ישועת", "יְשׁוּעַת"))
    toks = base.split()
    i = toks.index("ישועת")
    a = ref.align(" ".join(toks[:i]), "ישועת", " ".join(toks[i + 1:]))
    assert a.status == "agrees"
    a = ref.align(" ".join(toks[:i]), "ישועח", " ".join(toks[i + 1:]))
    assert a.status == "differs" and a.ref_words == "יְשׁוּעַת" and "⟦יְשׁוּעַת⟧" in a.excerpt
    j = toks.index("תחרשון")                      # a wrongly split word: two tokens flagged
    a = ref.align(" ".join(toks[:j]), "תחר שון", " ".join(toks[j + 1:]))
    assert a.status == "agrees"
    a = ref.align("שורה שאינה קיימת בכלל במקור הזה", "מילה", "ועוד דברים שאינם שם כלל ועיקר")
    assert a.status == "not_found"


def test_load_text_html_and_csv():
    assert "שלום" in load_text("a.html", "<p>שלום<br>עולם</p>".encode())
    c = "Index Title,X\nVersion Title,V\nLanguage,he\nVersion Source,s\nVersion Notes,\nX 1:1,שלום עולם\n"
    assert load_text("a.csv", c.encode()).strip() == "שלום עולם"


def test_export_csv_round_trip_and_no_empty_cells():
    out = export_csv("Rashi on Nedarim", "Vilna Edition", "http://x", "", [("Rashi on Nedarim 2a:1:1", 'טקסט, עם "מרכאות"')])
    v = parse_version_csv(out)
    assert (v.title, v.version_title, v.language) == ("Rashi on Nedarim", "Vilna Edition", "he")
    assert v.segments == [("Rashi on Nedarim 2a:1:1", 'טקסט, עם "מרכאות"')]
    # the importer's header logic: single-book format when A1 is not "version title"
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0][0].strip().lower() != "version title" and [r[1] for r in rows[:5]][:3] == ["Rashi on Nedarim", "Vilna Edition", "he"]
    with pytest.raises(EmptySegmentError):
        export_csv("T", "V", "", "", [("T 1:1", "  ")])
