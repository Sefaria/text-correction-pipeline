import asyncio
import csv
import io

import pytest

from pipeline.candidates import find_candidates
from pipeline.csv_export import EmptySegmentError, export_csv
from pipeline.edits import SpanFix, apply_fixes, replacement_for
from pipeline.llm_filter import Item, Judge, ModelSpec, Stop, Verdict
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



def test_apply_fixes_leaves_whitespace_outside_the_edit_alone():
    t = " אחד  שתים שלש "
    assert apply_fixes(t, [SpanFix(6, 10, "שנים")]) == " אחד  שנים שלש "      # double space and outer spaces untouched
    assert apply_fixes(t, [SpanFix(6, 10, "")]) == " אחד  שלש "               # deletion drops one of the spaces it leaves
    assert apply_fixes("אחד שתים", [SpanFix(4, 8, "")]) == "אחד"
    assert apply_fixes("אחד שתים", [SpanFix(0, 3, "")]) == "שתים"


def test_replacement_for_refuses_markup_inside_the_span():
    assert replacement_for("<b>שתים</b> שלש", "שנים שלש") is None
    assert replacement_for("שתים<br>שלש", "שנים שלש") is None
    assert replacement_for("<i><b>שתים</b></i>", "שנים") == "<i><b>שנים</b></i>"
    assert replacement_for("<b>שתים</b> שלש", "<b>שנים</b> שלש") == "<b>שנים</b> שלש"     # markup in the fix: used as written


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


def test_load_text_encodings():
    heb = "שלום עולם"
    even = heb + "!"                       # cp1255 is one byte per char: an even-length file used to decode as UTF-16 garbage
    assert len(even.encode("cp1255")) % 2 == 0
    assert load_text("a.txt", even.encode("cp1255")) == even
    assert load_text("a.txt", heb.encode("cp1255")) == heb
    assert load_text("a.txt", heb.encode("utf-16")) == heb                          # with BOM
    assert load_text("a.txt", heb.encode("utf-16-le")) == heb                       # BOM-less
    assert load_text("a.txt", "hello world".encode("utf-16-le")) == "hello world"


def test_judge_waits_for_running_groups_on_stop():
    """After Stop, no group may still be calling the model once judge() has raised."""
    stop = {"now": False}
    running, finished = set(), []

    class FakeJudge(Judge):
        def __init__(self):
            self.spec = ModelSpec("gemini", "m", None, 1)
            self.should_stop = lambda: stop["now"]
            self.sem = asyncio.Semaphore(2)

        async def _call(self, items):
            async with self.sem:
                if self.should_stop():
                    raise Stop()
                running.add(items[0].key)
                if items[0].key == "0":
                    await asyncio.sleep(0.01)          # "1" is mid-call when this one turns stop on
                    running.discard("0")
                    stop["now"] = True
                    raise Stop()
                await asyncio.sleep(0.05)
                running.discard(items[0].key)
                finished.append(items[0].key)
                return [Verdict(items[0].key, "correct")]

    async def go():
        with pytest.raises(Stop):
            await FakeJudge().judge([Item(str(i), "s", "e") for i in range(10)])
        return set(running)

    assert asyncio.run(go()) == set()       # the call running beside the one that stopped finished before judge() raised
    assert finished == ["1"]                # and no group started after the stop


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


def test_overload_backoff_waits_longer():
    from pipeline.llm_filter import backoff, overloaded
    over = RuntimeError("ServerError: 503 UNAVAILABLE. This model is currently experiencing high demand")
    assert overloaded(over) and not overloaded(ValueError("Expecting value: line 1 column 1"))
    assert backoff(4, over) >= 30 and backoff(4, ValueError("x")) <= 8 * 1.25


# ---- automatic fixes -------------------------------------------------------------------------------------------------------------

from pipeline.letter_pairs import allowed_pairs  # noqa: E402
from pipeline.ocr_swaps import (SwapMap, abbreviation_of, abbreviation_ref, auto_fix, build_map,  # noqa: E402
                                gershayim_moved, letter_diff, spacing_confirmed)


def test_letter_diff_substitutions_only():
    assert letter_diff("סבר", "כבר") == [(0, 1, "ס", "כ")]
    assert letter_diff("וזלק", "חלק") == [(0, 2, "וז", "ח")]
    assert letter_diff("שלומ", "שלום") == []                  # final form alone is not a difference
    assert letter_diff("שלום", "שלומו") is None                 # letters added: a variant, not a swap
    assert letter_diff("שלום", "שלומו"[:3] + "מו") is None      # ם→מו is a letter added too
    assert letter_diff("אבגדה", "אזחטה") is None                 # a substitution longer than 2 letters


def test_allowed_pairs_by_script():
    sq, ra = allowed_pairs("square"), allowed_pairs("rashi")
    assert {"ב/כ", "ה/ח", "ם/ס", "ו/י", "ח/ת", "ו/ן"} <= sq and {"ב/כ", "ם/ס", "ו/ן"} <= ra     # hand-picked, every script
    assert "ל/נ" in ra and "ל/נ" not in sq                                      # Rashi script only
    assert "וז/ח" in ra                                                          # look-alike two-for-one, from the glyph map
    assert "א/ה" not in sq | ra                                                  # spelling (יהודא/יהודה), not OCR
    assert allowed_pairs("rashi", "typed") == frozenset()                        # typed texts: keyboard slips, not yet listed


def fix(words, ref, script="square"):
    got = auto_fix(words, ref, allowed_pairs(script))
    return got and got[0]


def test_auto_fix_swaps():
    assert fix("<b>ובית</b>,", "וכית") == "<b>וכית</b>,"
    assert fix("וזלק", "חלק.", "rashi") == "חלק"
    assert fix("שלוס", "שלום") == "שלום"
    assert fix("דקירה", "דקורה") == "דקורה"
    assert fix("סאיו", "סאין") == "סאין"                        # ו/ן (final nun)
    assert fix("שכתובה.", "שבחובה,") == "שבחובה."               # adjacent swaps (כ/ב, ת/ח) come out of the diff as one block
    assert fix("שכאובה", "שבחובה") is None                      # ...and each letter must be an allowed pair (א/ח: Rashi only)
    assert fix("שכאובה", "שבחובה", "rashi") == "שבחובה"
    # the diff pairs the two ל's (drop צ, add ל); the letter-by-letter alignment finds the swap צ→ל (Rashi script only)
    assert fix("בכצל", "בכלל", "rashi") == "בכלל" and fix("בכצל", "בכלל") is None
    assert fix("יהודא", "יהודה") is None                     # א/ה is not an OCR pair
    assert fix("ולא", "ונא") is None and fix("ולא", "ונא", "rashi") == "ונא"
    assert fix("וּבית", "וכית") is None                       # niqqud in the span: left to the reviewer
    assert fix("ובית", "וכיתו") is None                       # also a letter added
    assert fix("ובית", "") is None and fix("ובית", "ובית") is None


def test_auto_fix_spacing():
    assert auto_fix("לאבא", "לא בא", allowed_pairs()) == ("לא בא", "spacing")
    assert fix("ותדע שאלו לאבא אלא", "ותדע שאלו לא בא אלא") == "ותדע שאלו לא בא אלא"
    assert fix("היה מחיצה", "היהמחיצה") == "היהמחיצה"         # joining works too
    assert fix("שלומו", "שלום ו") == "שלום ו"                  # final form follows the new division
    assert fix("כמו כן", "כמוכן") == "כמוכן"
    assert fix("<b>לא</b>בא", "לא בא") == "<b>לא</b> בא"
    assert fix("לא, בא", "לאבא") is None                      # only a plain space can be dropped
    assert auto_fix("לאכא", "לא בא", allowed_pairs()) == ("לא בא", "OCR swaps + spacing")
    assert fix("על־פי", "על פי") is None                      # maqaf and space both divide words: no change


def test_auto_fix_one_yod_or_vav():
    known = {match_key(w) for w in "הכי היכי לשתוף לשתף הא היא אמר אמרו".split()}
    f = lambda w, r: (auto_fix(w, r, allowed_pairs("rashi"), known) or (None, None))  # noqa: E731
    assert f("שאנם", "שאינם") == ("שאינם", "י/ו")                 # dropped inside a word
    assert f("טעמאי", "טעמא") == ("טעמא", "י/ו")                  # stray last letter
    assert f("לביתוי,", "לביתו") == ("לביתו,", "י/ו")
    assert f("חים", "חיים")[0] == "חיים" and f("וכין", "וכיון")[0] == "וכיון"
    assert f("דשמוחל", "דשמואל")[1] == "OCR swaps"
    assert f("ובעילה", "ובעי ליה") == ("ובעי ליה", "spacing + י/ו")
    assert f("ודבקעה", "דבקעה") == (None, None)                    # a word's first letter (the conjunction ו)
    assert f("אמר", "אמרו") == (None, None) and f("כתב", "כתבי") == (None, None)   # added at the end: grammar
    assert f("הכי", "היכי") == (None, None)                         # Sefaria's word is a real word in the reference
    assert f("לשתוף", "לשתף") == (None, None)
    assert f("סיד", 'ס"ד') == (None, None) and f("אחריות", "(אחרות)") == (None, None)   # gershayim, brackets
    assert f("שאנם", "שאינמו") == (None, None)                      # two letters added
    assert auto_fix("שאנם", "שאינם", allowed_pairs("rashi")) is None  # no reference words given: rule off


def test_yod_vav_shuffle_is_not_the_yod_vav_rule():
    known: set = set()
    f = lambda w, r: auto_fix(w, r, allowed_pairs("rashi"), known)  # noqa: E731
    assert f("ותלו", "ותלוי") is None                              # = a letter added at the end
    assert f("מכחל", "מכותל") == ("מכותל", "OCR swaps + י/ו")       # a real swap next to the added ו is fine


def test_abbreviation_of():
    assert abbreviation_of("פפ׳", "פפא") and abbreviation_of("לרב פפ׳", "לרב פפא,")
    assert abbreviation_of("המד׳ השלישי׳", "המדה השלישית") and abbreviation_of("משו'", "משום")
    assert abbreviation_of("ג״ה", "גירסת הספרים") and abbreviation_of("ר״הי", "רשות היחיד")
    assert not abbreviation_of("ומרי׳", "אומרים")                    # a letter missing: an error with a geresh
    assert not abbreviation_of("חושיבנ׳", "חושבנא") and not abbreviation_of("פט׳", "פפא")
    assert not abbreviation_of("פפא", "פפא") and not abbreviation_of("לרחב״ן", "לרחבן")
    assert not abbreviation_of("החי צונו׳", "החיצונות")              # the error is the space
    assert not abbreviation_of("פפ׳ בכצל", "פפא בכלל")               # an abbreviation and an error


def test_auto_fix_typed_patterns():
    known = {match_key(w) for w in "האי היא אל לא שעה ההר".split()}
    typed = lambda w, r: auto_fix(w, r, allowed_pairs("square", "typed"), known, ocr=False, typed=True)  # noqa: E731
    assert typed("הזכהר", "הזכרה") == ("הזכרה", "transposed")
    assert typed("לורמ", "לומר") == ("לומר", "transposed")           # final forms follow
    assert typed("הההרים", "ההרים") == ("ההרים", "doubled letter")
    assert typed("שעים", "שערים") == ("שערים", "missing/extra letter")
    assert typed("תקי ות", "תקיעות") == ("תקיעות", "spacing + missing/extra letter")   # a space took the letter's place
    assert typed("הברהכה", "הברכה") == ("הברכה", "missing/extra letter")
    assert typed("אל אהוא", "אלא הוא") == ("אלא הוא", "spacing")
    assert typed("שעה", "שערה") is None                              # Sefaria's word is a word of the reference
    assert typed("השופרו", "השופרות") is None                         # missing at the end: not inside the word
    assert typed("א", "לא") is None                                   # missing at the start
    assert typed("בדבר", "דבר") is None                               # an extra first letter (a prefix)
    assert typed("הפלה", "תפלה") is None                              # an OCR swap: not a typed pattern
    mixed = lambda w, r: auto_fix(w, r, allowed_pairs("rashi", "mixed"), known, ocr=True, typed=True)  # noqa: E731
    assert mixed("הן", "הר") is None and mixed("בכצל", "בכלל") == ("בכלל", "OCR swaps")
    assert mixed("הזכהר", "הזכרה") == ("הזכרה", "transposed") and mixed("טעמאי", "טעמא")[1] == "י/ו"
    ocr = lambda w, r: auto_fix(w, r, allowed_pairs("rashi", "ocr"), known)  # noqa: E731
    assert ocr("הזכהר", "הזכרה") is None and ocr("שעים", "שערים") is None   # typed patterns are off for OCR


def test_abbreviations_keep_their_letter_forms_and_marks():
    assert fix('ע"ב', 'ע"כ') == 'ע"כ' and fix('א"ג', 'א"נ') == 'א"נ'       # not ע"ך / א"ן
    assert fix('רמב"ס', 'רמב"ם') == 'רמב"ם'                                  # the reference's final form where it has one
    cut = abbreviation_ref("ולא חחמו'", "ולא תחמוד")
    assert cut == "ולא תחמו" and fix("ולא חחמו'", cut) == "ולא תחמו'"        # the abbreviation is fixed and kept
    assert abbreviation_ref("ולא חחמוד", "ולא תחמוד") is None


def test_spacing_confirmed_by_the_reference():
    assert spacing_confirmed("לאהיה", "לא היה", "לא הוה")                    # the reference's letters differ, its space agrees
    assert spacing_confirmed("בעלמ'ליקני", "בעלמ' ליקני", "בעלמא ליקני")
    assert not spacing_confirmed("לאהיה", "לאה יה", "לא הוה")                # the space is elsewhere in the reference
    assert not spacing_confirmed("לאהיה", "לא הוה", "לא הוה")                # the fix changes letters: not spacing only
    assert not spacing_confirmed("לא היה", "לא היה", "לא הוה")               # nothing to change


def test_vocabulary_backs_a_fix_without_a_reference():
    from collections import Counter
    from app.review import _vocabulary_backs
    ref, ver = Counter({"תופס": 4, "ופירש": 2}), Counter({"חופס": 1, "באריכותופירש": 1, "באריכות": 2, "שזה": 3})
    assert _vocabulary_backs("שזה חופס", "שזה תופס", ref, ver)
    assert _vocabulary_backs("באריכותופירש", "באריכות ופירש", ref, ver)
    assert not _vocabulary_backs("שזה", "שזו", ref, ver)                     # שזה is a word elsewhere
    assert not _vocabulary_backs("חופס", "חופש", ref, ver)                   # חופש occurs nowhere


def test_gershayim_moved():
    assert gershayim_moved("ר״הי", "רה״י") and gershayim_moved('ר"הי', "רה״י")
    assert not gershayim_moved("לרחב״ן", "לרחבן")             # removed, not moved
    assert not gershayim_moved("ר״הי", "ר״הי") and not gershayim_moved("ר״הי", None)
    assert not gershayim_moved("ר״הי", "רה״ח")                # letters changed too


def test_reference_maps_a_replaced_run_word_for_word():
    base = "תחת הקורה דעלמא מתיר כאן פי' או משום דקורה משום היכירא והיכירא מלגאו בעינן או משום דקורה משום מחיצה וסל דחודו הפנימי"
    ref = Reference(base)
    sef = base.split()
    i = sef.index("דקורה")
    a = ref.align(" ".join(sef[:i]), "דקירה", "משו׳ " + " ".join(sef[i + 2:]))
    assert a.status == "differs" and a.ref_words == "דקורה"


def test_reference_narrows_to_the_matching_word_when_word_order_differs():
    tail = " אתה עושה כניסתו ויציאתו כשני ראשים וכאילו הוא מפולש שנמצא"
    ref = Reference("סגי כמפולש דעלמא דהא אם תמצא לומר שאין צריך בעקמימותו כלום" + tail)
    a = ref.align("סגי כמפולש דעלמא דהא אם תמצא לומר שאין בעקמומיתו צריך", "כלים", tail)
    assert a.status == "differs" and a.ref_words == "כלום"
    a = ref.align("סגי כמפולש דעלמא דהא אם תמצא לומר שאין צריך בעקמימותו", "כלום", tail)
    assert a.status == "agrees" and a.ref_words == "כלום"
    split = Reference("ותדע שאלו לא בא אלא לומר שאין צריך בעקמימותו כלום" + tail)
    assert split.align("ותדע שאלו", "לאבא", "אלא לומר שאין צריך בעקמימותו כלום" + tail).ref_words == "לא בא"


def test_swap_map_counts_substitutions():
    base = ("ויאמר משה אל העם אל תיראו התיצבו וראו את ישועת ה' אשר יעשה לכם היום כי אשר ראיתם את מצרים "
            "היום לא תספו לראתם עוד עד עולם ה' ילחם לכם ואתם תחרשון ויאמר ה' אל משה מה תצעק אלי דבר אל בני ישראל ויסעו "
            "ואתה הרם את מטך ונטה את ידך על הים ובקעהו ויבאו בני ישראל בתוך הים ביבשה ואני הנני מחזק את לב מצרים ויבאו אחריהם")
    sef = base.replace("ישועת", "ישועח").replace("ראיתם", "ראיחם").replace("לראתם", "לראחם").replace("ובקעהו", "ובקעחו")
    sef = sef.replace("בתוך", "בתור").replace("אחריהם", "אחריחם") + " " + base     # longer than one block
    ref = Reference(base.replace("ידך", "ידיך") + " " + base)             # a spelling variant (letter added) is not counted
    m = build_map([sef], ref)
    assert m.counts == {"ח/ת": 3, "ה/ח": 2, "ך/ר": 1}
    back = SwapMap.from_dict(m.to_dict())
    assert back.counts == m.counts and back.total == 6 and m.to_dict()["pairs"][0]["examples"][0] == "ישועח→ישועת"
