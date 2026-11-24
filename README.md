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
   | `SESSION_SECRET` | random, at least 32 characters |
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

## Before relying on an export

Upload a corrections CSV to a **cauldron sandbox** first, not prod. Then check two things:
- Only the edited segments changed.
- The upload decodes. `text_upload_api` passes the uploaded file straight to `csv.reader`, which I could not verify from the code alone.

## Layout

```
pipeline/     shared logic: sefaria_client, tokens, windows, detector, candidates, reference, llm_filter, edits, csv_export
app/          FastAPI app (main), SQLite (db), job worker (jobs), review state / undo / export (review)
web/          plain HTML + JS modules, no build step
experiments/  Phase 0: eval set, Opus gold labels, model arms, reference experiment, REPORT.md
tests/        pytest
```
