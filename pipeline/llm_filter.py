"""LLM pass over detector candidates: is the marked word really a transcription error, and if so, what is the fix?

Prompt adapted from translation-engine experiments/version_quality_from_history/llm_judge.py (SYSTEM), extended to multi-word spans,
to batches (several candidates of one passage in one call), and to a reference-text variant.

Providers: Anthropic (Claude) and Google (Gemini), both through their SDKs, both asked for JSON with a schema.
"""
import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

SYSTEM = """You check digitized Hebrew and Aramaic texts (rabbinic literature, commentaries, responsa, liturgy) for transcription errors.
Each item is a short excerpt with one word or a few adjacent words marked ⟦like this⟧. Decide whether the marked text, as written, is an
error in the digitized text:

error: a typo, a scanning (OCR) misreading such as ד/ר, ה/ח, ב/כ, ו/י, ס/ם, ן/ו confusions, two words wrongly joined or one word wrongly
       split, stray or missing characters, or a garbled abbreviation. The printed source would not have had it this way.
correct: a valid word, an abbreviation or acronym, a legitimate spelling convention (full or defective spelling), Aramaic, a proper name,
         a rare word (commentaries and glossaries define rare words on purpose), a quotation as it appears in its source, or a variant
         reading that the source plausibly had.

An automatic detector chose these words and it flags many correct words, so do not assume the marked text is wrong. The excerpt may be
cut off at either end. Other words in it may also have errors; judge only the marked text.
For an error, `fix` is the corrected text that replaces exactly the marked words (it may have more or fewer words). Otherwise `fix` is empty.
`reason` is one short English phrase."""

REFERENCE_SYSTEM = SYSTEM + """

You also get the corresponding passage from a reference copy of the same work (another edition, a scan transcription, or a manuscript
transcription), with the words that align to the marked text marked ⟦like this⟧. The reference differs from the digitized text there.
The reference can itself contain errors, and editions legitimately differ. Decide whether the digitized text has an error:
use the reference as strong evidence, but call it `correct` when the digitized reading is a valid variant and the reference's reading is
not clearly what the digitized edition printed. When the digitized text is wrong, the fix is usually the reference's reading, adjusted to
the digitized text's own spelling and abbreviation conventions."""

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "n": {"type": "integer"},
        "verdict": {"type": "string", "enum": ["error", "correct"]},
        "fix": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["n", "verdict", "fix", "confidence", "reason"],
    "additionalProperties": False,
}
BATCH_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["items"],
    "additionalProperties": False,
}

# USD per million tokens (input, output; Gemini output includes thinking tokens). Anthropic: first-party list prices (2026-11).
# Gemini 3.8 Flash: $1.50 / $7.50 from 2027-01-01 (promotional $0.75 / $3.75 until then); override with GEMINI_FLASH_PRICE="in,out".
PRICES = {
    "gemini-3.8-flash": (1.50, 7.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}
if os.environ.get("GEMINI_FLASH_PRICE"):
    _i, _o = os.environ["GEMINI_FLASH_PRICE"].split(",")
    PRICES["gemini-3.8-flash"] = (float(_i), float(_o))


@dataclass
class ModelSpec:
    provider: str                     # "anthropic" | "gemini"
    model: str
    effort: Optional[str] = None      # Anthropic output_config.effort; Gemini thinking_level (minimal|low|medium|high)
    batch: int = 1                    # candidates per call

    @property
    def spec(self) -> str:
        """provider:model:effort:batch, the form app.jobs.parse_spec reads back."""
        return f"{self.provider}:{self.model}:{self.effort or ''}:{self.batch}"

    @property
    def label(self) -> str:
        return f"{self.model}{'/' + self.effort if self.effort else ''}{'/b' + str(self.batch) if self.batch > 1 else ''}"


@dataclass
class Item:
    """One thing to judge. `excerpt` has the span marked ⟦…⟧; `reference` (optional) has the aligned reference words marked."""
    key: str
    source: str
    excerpt: str
    reference: Optional[str] = None


@dataclass
class Verdict:
    key: str
    verdict: Optional[str]            # "error" | "correct" | None (failed)
    fix: str = ""
    confidence: Optional[float] = None
    reason: str = ""
    model: str = ""
    error: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, i: int, o: int):
        self.input_tokens += i; self.output_tokens += o; self.calls += 1
        m = self.by_model.setdefault(model, [0, 0, 0])
        m[0] += i; m[1] += o; m[2] += 1

    def cost(self) -> Optional[float]:
        total = 0.0
        for model, (i, o, _) in self.by_model.items():
            if model not in PRICES:
                return None
            pi, po = PRICES[model]
            total += i / 1e6 * pi + o / 1e6 * po
        return round(total, 4)


def render(items: list[Item]) -> str:
    parts = []
    for n, it in enumerate(items, 1):
        s = f"Item {n}\nSource: {it.source}\nDigitized text:\n{it.excerpt}"
        if it.reference is not None:
            s += f"\nReference:\n{it.reference}"
        parts.append(s)
    head = f"Judge each of the {len(items)} items below. Return one entry per item, with `n` = the item number.\n\n"
    return head + "\n\n".join(parts)


class Stop(Exception):
    """Raised inside judge() when `should_stop()` turns true; calls already running finish, no new ones start."""


class Judge:
    def __init__(self, spec: ModelSpec, concurrency: int = 8, usage: Optional[Usage] = None, should_stop=None):
        self.spec = spec
        self.should_stop = should_stop
        self.sem = asyncio.Semaphore(concurrency)
        self.usage = usage or Usage()
        if spec.provider == "anthropic":
            import anthropic
            self.client = anthropic.AsyncAnthropic(max_retries=4)
        elif spec.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        else:
            raise ValueError(spec.provider)

    async def _anthropic(self, system: str, user: str) -> dict:
        kw = {}
        if self.spec.effort:
            kw["output_config"] = {"effort": self.spec.effort, "format": {"type": "json_schema", "schema": BATCH_SCHEMA}}
        else:
            kw["output_config"] = {"format": {"type": "json_schema", "schema": BATCH_SCHEMA}}
        r = await self.client.messages.create(model=self.spec.model, max_tokens=16000, system=system,
                                              messages=[{"role": "user", "content": user}], **kw)
        self.usage.add(self.spec.model, r.usage.input_tokens, r.usage.output_tokens)
        if r.stop_reason == "refusal":
            raise RuntimeError("refusal")
        text = next(b.text for b in r.content if b.type == "text")
        return json.loads(text)

    async def _gemini(self, system: str, user: str) -> dict:
        from google.genai import types
        thinking = types.ThinkingConfig(thinking_level=self.spec.effort.upper()) if self.spec.effort else None
        cfg = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                          response_json_schema=BATCH_SCHEMA, thinking_config=thinking)
        r = await self.client.aio.models.generate_content(model=self.spec.model, contents=user, config=cfg)
        um = r.usage_metadata
        out_tokens = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        self.usage.add(self.spec.model, um.prompt_token_count or 0, out_tokens)
        return json.loads(r.text)

    async def _call(self, items: list[Item]) -> list[Verdict]:
        system = REFERENCE_SYSTEM if any(it.reference is not None for it in items) else SYSTEM
        user = render(items)
        last = ""
        for attempt in range(4):
            try:
                async with self.sem:
                    if self.should_stop and self.should_stop():
                        raise Stop()
                    data = await (self._anthropic if self.spec.provider == "anthropic" else self._gemini)(system, user)
                got = {e["n"]: e for e in data.get("items", [])}
                out = []
                for n, it in enumerate(items, 1):
                    e = got.get(n)
                    if e is None:
                        out.append(Verdict(it.key, None, model=self.spec.label, error="missing from response"))
                    else:
                        out.append(Verdict(it.key, e["verdict"], (e.get("fix") or "").strip() if e["verdict"] == "error" else "",
                                           e.get("confidence"), e.get("reason", ""), self.spec.label))
                return out
            except Stop:
                raise
            except Exception as ex:                     # retried: rate limits, transient 5xx, malformed JSON
                last = f"{type(ex).__name__}: {str(ex)[:200]}"
                await asyncio.sleep(2 ** attempt + random.random())
        return [Verdict(it.key, None, model=self.spec.label, error=last) for it in items]

    async def judge(self, items: list[Item], on_done=None) -> list[Verdict]:
        """Judge items in groups of spec.batch (consecutive items, so a group is usually one passage). `on_done(verdicts)` per group."""
        groups = [items[i:i + self.spec.batch] for i in range(0, len(items), self.spec.batch)]

        async def run(g):
            v = await self._call(g)
            if on_done:
                on_done(v)
            return v

        results = await asyncio.gather(*(run(g) for g in groups))
        return [v for g in results for v in g]
