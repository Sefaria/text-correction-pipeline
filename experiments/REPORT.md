# Phase 0: which LLM should filter the detector's flags?

**Decision for the app's default:** Gemini 3.8 Flash at its default thinking level, 8 candidates per call (`gemini:gemini-3.8-flash::8`).
- **Accuracy:** it keeps 93% of real errors, removes 95% of false positives, and its proposed fix matches the gold fix 84% of the time.
- **Human labels:** it scored best of all models, including Opus.
- **Cost:** about $5.50 per 1,000 candidates at the 2027 list price.

Sonnet 5.5 did clearly worse on every measure.

With a reference text, dismiss a candidate only when the reference agrees **and** the LLM calls it correct. That keeps 96% of real errors.

## Setup

**Candidates.** The detector (BEREL 3.0 hebrew-error-detector v2, threshold 0.536) was run over 8 full versions from the translation-scope quality pass:
- 6 with many confirmed errors: Chiddushei Ramban on Megillah, Ritva on Makkot, Minchat Shai on Psalms, Bekhor Shor (Leipzig), Rashi on Nedarim, and Sforno on Song of Songs.
- 2 where almost every flag was a false positive: Metzudat Zion on Psalms, and Ibn Ezra on Song of Songs.

Up to 80 candidates were sampled per version (`build_eval_set.py`), 622 in total.

**Gold labels.** One Opus subagent per version labelled each candidate as error, correct or unsure, with a fix. They saw ±50 words of context and could look up quoted sources on the Sefaria API. Result: 217 error, 388 correct, 17 unsure. Unsure items are left out of scoring.

**Removed from the eval.** 73 items were dropped because their flagged span contained a `<br>` (see the finding below). That leaves **532 scored items: 217 errors and 315 correct.**

**Human anchor.** The 26 hand-labelled flagged words and 50 controls from translation-engine `llm_judge` (`human_check.py`).

**Arms.** Each candidate is shown as an excerpt of about 25 words each side, with the flagged words marked ⟦…⟧. The prompt is adapted from translation-engine `llm_judge.py`, and the output is JSON with a schema (`run_arms.py`). A failed call counts as "error", so the candidate stays in the queue.

## Results against the Opus gold labels (532 items)

| Arm | Real errors kept | False positives removed | Reviewer's queue that is real | Fix matches gold | $ per 1k candidates |
|---|---|---|---|---|---|
| **Flash, default thinking, 8 per call** | **0.926** | **0.946** | **0.922** | **0.841** | 5.47 |
| Flash, default thinking, 1 per call | 0.954 | 0.917 | 0.888 | 0.850 | 5.35 |
| Flash, medium thinking, 8 per call | 0.935 | 0.930 | 0.902 | 0.862 | 5.63 (same thinking-token use as default: no saving) |
| Flash, low thinking, 8 per call | 0.880 | 0.943 | 0.914 | 0.759 | 0.69 |
| Haiku 5.5, 8 per call | 0.710 | 0.924 | 0.865 | 0.714 | 0.16 |
| Sonnet 5.5, low effort, 1 per call | 0.940 | 0.448 | 0.540 | 0.618 | 3.16 |
| Sonnet 5.5, low effort, 8 per call | 0.834 | 0.676 | 0.640 | 0.497 | 1.34 |
| Sonnet 5.5, medium effort, 8 per call | 0.816 | 0.622 | 0.598 | 0.531 | 1.36 |
| Drop only if Flash-low and Sonnet-low both say correct | 0.954 | 0.657 | 0.657 | 0.435 | 2.03 |

**Prices.** Flash is costed at the 2027 list price ($1.50 in / $7.50 out per MTok, thinking tokens included). Until 2026-12-31 it is half that. Most of Flash's cost and latency is thinking tokens, about 9k per call at the default level.

**The two false-positive-heavy versions** (87 items, 8 real errors): Flash default removed 99% of the false positives and kept 6 of the 8 errors. Sonnet low removed 82% and kept 5. These texts are glossaries of rare Biblical words and Ibn Ezra's grammatical vocabulary, which is what fools the detector. Flash handles them much better.

## Against human labels (76 items)

| Model | Accuracy | Hand-labelled flags (26) | Editor-corrected words (25) | Unedited words (25) |
|---|---|---|---|---|
| Flash, default thinking | **0.895** | 24 right | 19 caught | 25 right |
| Opus 5.5, high effort | 0.868 | 22 | 19 | 25 |
| Flash, low thinking | 0.803 | 23 | 15 | 23 |
| Haiku 5.5 | 0.763 | 20 | 16 | 22 |
| Sonnet 5.5, low effort | 0.750 | 21 | 14 | 22 |

Opus tracks the human labels closely, so it is a reasonable gold standard. Flash default agrees with the humans at least as well as Opus does. Sonnet's weak spot is calling correct words errors. That agrees with the earlier finding in translation-engine that Sonnet alone scored 0.92 on the 26 hand labels, but did worse once the controls were added.

## Reference text (`reference_experiment.py`)

**How the test works.** Sefaria rarely has two overlapping editions of these works; it stores complementary parts as separate versions. So each reference was simulated from the Sefaria text:
- Gold errors were corrected in the reference, except 10%, which kept the same misprint.
- 15% of correct flagged words got a spelling variant in the reference, and 5% got an OCR error of the reference's own.
- Everywhere else, HTML and niqqud were stripped, punctuation changed, lines reflowed, 5% of passages dropped, 2% of words respelled and 0.3% given letter swaps.

This tests the mechanics. It does not stand in for a real second edition.

**Alignment** (4-gram voting, then word-level SequenceMatcher):
- It placed every candidate.
- The only "agrees" results on real errors were the planned shared misprints, plus 7 fixes that change only punctuation (alignment ignores punctuation on purpose, because editions differ there).

| Pipeline | Real errors kept | False positives removed | Fix matches gold |
|---|---|---|---|
| LLM only (no reference) | 0.926 | 0.946 | 0.841 |
| Reference: dismiss everything that agrees; LLM on the rest | 0.829 | 0.994 | 0.983 |
| **Reference + LLM: dismiss only if the reference agrees *and* the LLM says correct** | **0.959** | **0.946** | ≈0.98 where the reference differs |

The app uses the last rule. Candidates where the reference agrees but the LLM says error stay in the queue, labelled as such.

## Finding: line-break tags were a large source of false positives

With whitespace tokenization, `הא'</b><br>תורק.` is one token. Once its tags are stripped, it reads as two words wrongly joined, and the detector flags it. In Ibn Ezra on Song of Songs, 191 of 227 flags were this.

`pipeline/tokens.py` now also splits tokens after `<br>`, which took that version from 227 flags to 68. The translation-scope quality pass used the same tokenization, so its precision figures for HTML-heavy versions are probably understated.

## Caveats

- **Gold labels are Opus, not humans.** On the human set, Opus agrees 87%, and its misses are corrected words it judged to be acceptable variants. Recall figures may be off by a few points either way.
- **Proposed fixes:** about 1 in 6 accepted proposals from Flash would need editing. Fix-match with a reference is much higher.
- **Phase 0 used 8 versions.** Re-run `run_arms.py` with a new eval set if the app is pointed at a very different genre, such as liturgy or Kabbalah.

## Reproduce

```sh
python -m experiments.build_eval_set                 # detector over the 8 versions -> output/evalset.jsonl
# gold labels: one Opus subagent per output/gold_in/<version>.jsonl -> output/gold/<version>.jsonl
python -m experiments.run_arms                       # every arm, cached
python -m experiments.report                         # tables
python -m experiments.human_check                    # human anchor
python -m experiments.reference_experiment build && python -m experiments.reference_experiment run flash-b8 \
  && python -m experiments.reference_experiment score flash-b8
```
