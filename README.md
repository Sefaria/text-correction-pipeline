# Text Correction Pipeline

A web tool for Sefaria text experts. It does four things:

1. Pick a Hebrew version on Sefaria and run the [Hebrew error detector](https://huggingface.co/Sefaria/BEREL_3.0-hebrew-error-detector) over all of it.
2. Filter the detector's many false positives with an LLM pass, and compare with an uploaded reference text if there is one.
3. Review the remaining candidate corrections one at a time, keyboard first.
4. Export the accepted corrections as a CSV for modtools' **Upload texts from CSV**.

```
fetch version CSV ─▶ windows ─▶ detector (CPU) ─▶ candidates ─▶ [reference alignment] ─▶ LLM check ─▶ review ─▶ modtools CSV
 sefaria.org          350 words   BEREL 3.0 v2       spans          optional upload           Gemini Flash
```

## How a candidate is triaged

| Reference | LLM | Where it goes |
|---|---|---|
| none uploaded | error | **To review**, with the LLM's fix |
| none uploaded | correct | *Likely OK*: hidden from the queue, still viewable |
| differs at that spot | error | **To review**, with the fix (reference reading, checked by the LLM) |
| differs | correct (an edition variant) | *Likely OK* |
| agrees with Sefaria | correct | *Dismissed*: hidden, still viewable |
| agrees | error | **To review**, labelled "reference agrees, LLM thinks error" |

The last row is a backstop. In testing, dismissing every candidate the reference agreed with lost real errors: misprints both editions share, and fixes that only change punctuation. Recall fell from 0.96 to 0.83. See `experiments/REPORT.md`.

The model choice and these rules come from the Phase 0 experiment in `experiments/`. That experiment compared models against Opus gold labels on 532 flagged words from the 8 worst versions in the translation-scope quality pass, and checked against human labels.

### Automatic decisions

Each project records how its text was made: **OCR**, **typed / converted** (typing errors, or damage from old software), or
**mixed / unclear** (both sets of rules), and for OCR the edition's **script** (square or Rashi). Both can be
changed on the dashboard. As the LLM pass saves each batch, these rules run on the candidates the LLM calls errors:

| Rule | Decision | When |
|---|---|---|
| OCR swaps | accept, as `auto: OCR swaps` | The reference reads differently, and every letter difference is an allowed pair: up to 2 letters for up to 2 (ב/כ, וז/ח) |
| Spacing | accept, as `auto: spacing` | Same letters, words divided differently (לאבא / לא בא). Can combine with OCR swaps |
| One י/ו (OCR only) | accept, as `auto: י/ו` | One י or ו dropped inside a word or stray at its end on Sefaria's side (שאנם→שאינם, טעמאי→טעמא). Never a word's first letter, never added at the end (אמר/אמרו), only when Sefaria's word never appears in the reference (so not הכי/היכי or לשתוף/לשתף), and not when the reference reading has gershayim or brackets (ס"ד, (אחרות)). Can combine with the rules above |
| Transposed (typed) | accept, as `auto: transposed` | Two neighbouring letters in the other order (הזכהר→הזכרה, ואניו→ואינו). Not combined with an added or dropped letter |
| Doubled letter (typed) | accept, as `auto: doubled letter` | Sefaria repeats a letter (הההרים→ההרים), not the word's first |
| Missing/extra letter (typed) | accept, as `auto: missing/extra letter` | One letter of any kind missing inside a word (שעים→שערים), also where a space took its place (תקי ות→תקיעות), or one extra inside a word (הברהכה→הברכה); only when Sefaria's word never appears in the reference |
| Space across two words (typed) | edit the segment, as `auto: spacing` | Only one word was flagged, but the error is in the space next to it (אל ⟦אהוא⟧→אלא הוא, העי ⟦רהיה⟧→העיר היה). The neighbouring word must be in the same segment and not part of another candidate |
| Spacing the reference confirms | accept, as `auto: spacing` | The LLM's fix only moves spaces, and the reference divides there, even where its letters differ (לאהיה→לא היה against "לא הוה"; בעלמ'ליקני→בעלמ' ליקני, the abbreviation kept). Sefaria's letters and marks stay |
| Abbreviation with an error | accept, as `auto: OCR swaps` (etc.) | Sefaria's abbreviation is compared with the start of the reference's full word, so it is fixed and stays an abbreviation (חחמו'→תחמו' against תחמוד). Letters in abbreviations keep the reference's form: ע"כ, א"נ, not ע"ך, א"ן |
| No reference here | accept, as `auto: … (no reference)` | The aligner found no reading for this passage: the LLM's fix, if it is an allowed kind (OCR swaps, spacing, transposed) and the vocabulary backs it: the words it removes occur nowhere else, in the reference or the version, and the words it brings in occur somewhere (חופס→תופס, באריכותופירש→באריכות ופירש) |
| Abbreviation | reject, as `auto: abbreviation` | Sefaria abbreviates what the reference writes in full (פפ׳/פפא, ג״ה/גירסת הספרים): not an error |
| Gershayim | reject, as `auto: gershayim` | The LLM's fix only moves a geresh or gershayim (ר״הי → רה״י): not worth changing |

For the first two rules, any fix the LLM proposed must match the reference reading. The fix changes only those letters and spaces in
Sefaria's text, so punctuation, gershayim and markup stay as they are, and final forms follow the new word division. Spans with niqqud
are left to the reviewer.

**Allowed pairs** (`pipeline/letter_pairs.py`) are the only pairs the OCR rule accepts:
- *Hand-picked:* ה/ח ד/ר ס/ם י/ו ב/כ ג/נ ו/ז ת/ח ו/ן for every script, plus א/ח ל/נ ו/נ א/מ צ/ל for Rashi script.
- *Look-alike letters:* `experiments/letter_similarity.py` draws every letter in square fonts (Frank Ruhl Libre, David Libre) and a Rashi
  font (Noto Rashi Hebrew), then scores how much each pair's strokes overlap, two-for-one pairs included. Pairs scoring at least 0.70
  are added: in square script ט/ס ד/ך כ/פ; in Rashi script כ/פ ן/ץ ב/פ ק/ת ח/ר וז/ח. The scores are in `pipeline/letter_similarity.json`.
- *Typed / converted texts* get no letter-swap fixes (`TYPED_PAIRS` is empty, ready for keyboard slips); *mixed* uses the OCR pairs.

The align stage also counts the letter substitutions between the whole version and its reference. The dashboard shows the allowed pairs
among them. The counts are only evidence: frequent substitutions include spelling and censorship differences (יהודא/יהודה, עכו״ם/גוי),
so they never allow a pair by themselves.

Automatic decisions are ordinary decisions. They show in the history, the **Automatic decisions** filter lists them, and a reviewer can
reset or change any of them. Automatic decisions never override a decision a reviewer has made. **Run automatic fixes** on the dashboard
runs the rules over a whole project, for example after changing its script. For projects created before these rules,
`POST /api/projects/{id}/auto?realign_reference=true` first re-aligns them with the current aligner, without new LLM calls.

On the Phase 0 reference experiment (`experiments/ocr_swaps_experiment.py`), the rules accepted 47 fixes in square script and 56 in Rashi
script, out of 225 candidates the LLM flagged, all matching the gold fix. The one-י/ו rule adds 10 more right and 1 wrong: a deliberate
rare spelling (איבך) that the simulated reference spelled אבך. A rare word can't be told from OCR damage by counting, so expect the
occasional spelling change from this rule.

## Review shortcuts

Shortcuts use physical keys, so they work with a Hebrew keyboard layout.

| Key | Action |
|---|---|
| `J` / `K` | Next / previous |
| `A` or `Enter` | Accept the proposed fix and go to the next candidate |
| `R` or `X` | Reject (not an error) and go to the next candidate |
| `E` | Edit the replacement for the marked words; `Ctrl+Enter` saves |
| `S` | Edit the whole segment's text (HTML kept as is) |
| `U` or `Ctrl+Z` | Undo the last change |
| `F` | Cycle the filter |
| `Shift+E` | Export CSV |
| `?` | Show or hide the shortcuts and history sidebar |

Every action is saved on the server immediately, in an append-only event log. That log drives the history sidebar and undo.

## The export

The export is the single-book modtools format: 5 header rows (Index Title, Version Title, Language, Version Source, Version Notes), then `ref,text` rows. Only changed segments are included, each with its full new text. Refs come straight from Sefaria's own export, so they are canonical.

The uploader (`sefaria/tracker.py` `modify_bulk_text`) has two behaviours to know about:
- Segments missing from the CSV are left untouched.
- An empty cell blanks that segment. So the app refuses an empty edit, and the exporter refuses to write one.

## Projects and progress

A project is one version, plus an optional reference file. Several projects can exist at once.
- The detector runs one project at a time, because it uses every CPU core.
- LLM passes for different projects run at the same time, sharing one concurrency limit (`LLM_CONCURRENCY`, default 16).
- Each stage saves as it goes. After a restart or redeploy, a project continues where it stopped.
- Candidates can be reviewed while detection and the LLM pass are still running.

The dashboard and review page show each stage's progress as a count, a bar and a time estimate.

## Run locally

```sh
uv venv --python 3.12 && uv pip install -e . pytest
export GEMINI_API_KEY=...              # default LLM pass; ANTHROPIC_API_KEY if you pick a Claude model
DEV_AUTH_EMAIL=you@sefaria.org SECURE_COOKIES=false DATA_DIR=./data .venv/bin/uvicorn app.main:app --port 8077
.venv/bin/python -m pytest             # unit and API tests (fake detector and LLM, no network)
```

`DEV_AUTH_EMAIL` skips Google sign-in. Never set it on a deployed instance.

## Deploy (Coolify, Sefaria Dev Server)

This follows the translation-review app: one container, one replica, and a volume for state.

1. Create a Coolify application from this repo, branch `main`, using the root `Dockerfile`. Internal port `8080`, health check `/healthz`.
2. Add a persistent volume at `/data`, writable by UID 10001. It holds the SQLite database and the uploaded reference texts.
3. Set the environment:

   | Variable | Value |
   |---|---|
   | `GEMINI_API_KEY` | Gemini key (the default LLM pass) |
   | `ANTHROPIC_API_KEY` | optional; only for projects that choose a Claude model |
   | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | OAuth client. Its redirect URI is `https://<host>/auth/callback` |
   | `GOOGLE_ALLOWED_DOMAIN` | `sefaria.org` (the default) |
   | `PUBLIC_URL` | `https://<host>`, so the OAuth redirect is right behind the proxy |
   | `SESSION_SECRET` | random, at least 32 characters (required: the app will not start without it unless `DEV_AUTH_EMAIL` is set) |
   | `DEFAULT_LLM` | optional; default `gemini:gemini-3.8-flash::8` (provider:model:effort:batch) |
   | `LLM_CONCURRENCY` | optional; default 16 |

4. Resources: 4 CPUs and about 4 GB of RAM. The detector scores about 2,000 words a second on an M-series Mac; expect somewhat slower on the server, so a 1M-word version takes roughly 10–30 minutes.
5. Run **exactly one replica**. The job worker lives in the web process.
6. After deploying, check four things:
   - `/healthz` returns ok.
   - Signing in with an account outside sefaria.org is refused.
   - A small version runs end to end.
   - A project survives a redeploy.

**LLM cost.** The default Gemini 3.8 Flash pass costs about $5.50 per 1,000 candidates at the 2027 list price, and about half that until 2026-12-31. At 10 flags per 1,000 words, a 50,000-word version costs about $3. Per-project cost is shown on the dashboard.

## Verified upload

On 2026-11-24 an exported CSV was uploaded through modtools on the talmudit cauldron. Only the edited segment changed: of the 35
segments of Abarbanel on Haggai, the one with the accepted fix, byte for byte. The Hebrew decoded correctly. For a version with HTML
markup, try a cauldron first anyway.

## Layout

```
pipeline/     shared logic: sefaria_client, tokens, windows, detector, candidates, reference, llm_filter, edits, csv_export
app/          FastAPI app (main), SQLite (db), job worker (jobs), review state / undo / export (review)
web/          plain HTML + JS modules, no build step
experiments/  Phase 0: eval set, Opus gold labels, model arms, reference experiment, REPORT.md
tests/        pytest
```
