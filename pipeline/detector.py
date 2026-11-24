"""The Hebrew error detector (Sefaria/BEREL_3.0-hebrew-error-detector): an error probability for every word of a window.

Chunking and prediction follow ML-Workflows app/error_detection/steps/data.py `chunk_row` and evaluation.py `predict_words`
(not a package, so the inference half is re-implemented here): words are normalized, words that normalize to nothing are skipped,
whole words are packed into chunks of at most max_length pieces, and a word's score is the error-class probability at its first piece.
"""
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .text import model_word
from .windows import Window

MODEL_ID = os.environ.get("DETECTOR_MODEL", "Sefaria/BEREL_3.0-hebrew-error-detector")
CONFIG_FILE = "error_detection_config.json"


class Detector:
    def __init__(self, model_id: str = MODEL_ID, threads: Optional[int] = None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        path = model_id if Path(model_id).is_dir() else snapshot_download(model_id)
        torch.set_num_threads(threads or max(1, os.cpu_count() or 1))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(path).eval()   # CPU: MPS hung a Mac on BEREL (see wiki)
        cfg = Path(path) / CONFIG_FILE
        self.config = json.loads(cfg.read_text()) if cfg.exists() else {}
        self.max_length = int(self.config.get("max_length", 512))
        self.threshold = float(self.config.get("validation_threshold", 0.5))

    def chunks(self, window: Window) -> list[dict]:
        """Model inputs for one window: [{"input_ids", "first": positions of each word's first piece, "word_index": token indices}]."""
        words, index = [], []
        for i, tok in enumerate(window.tokens):
            w = model_word(tok)
            if w:
                words.append(w); index.append(i)
        if not words:
            return []
        tk = self.tokenizer
        pieces = tk(words, add_special_tokens=False)["input_ids"]
        budget = self.max_length - 2
        groups, cur, used = [], [], 0
        for w, ids in enumerate(pieces):
            ids = ids[:budget] or [tk.unk_token_id]
            if cur and used + len(ids) > budget:
                groups.append(cur); cur, used = [], 0
            cur.append((w, ids)); used += len(ids)
        if cur:
            groups.append(cur)
        out = []
        for g in groups:
            ids, first, widx = [tk.cls_token_id], [], []
            for w, p in g:
                first.append(len(ids)); widx.append(index[w]); ids += p
            ids.append(tk.sep_token_id)
            out.append({"input_ids": ids, "first": first, "word_index": widx})
        return out

    def score(self, windows: Iterable[Window], batch_size: int = 16,
              progress: Optional[Callable[[int], None]] = None) -> dict[int, list[float]]:
        """{window id: [score per token]} (tokens that normalize to nothing score 0). `progress(n)` is called after each batch with the
        number of windows finished in it."""
        torch = self.torch
        windows = list(windows)
        out = {w.id: [0.0] * len(w.tokens) for w in windows}
        pending: list[tuple[int, dict]] = []

        def run(batch):
            width = max(len(c["input_ids"]) for _, c in batch)
            ids = torch.full((len(batch), width), self.tokenizer.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), width), dtype=torch.long)
            for r, (_, c) in enumerate(batch):
                n = len(c["input_ids"])
                ids[r, :n] = torch.tensor(c["input_ids"]); att[r, :n] = 1
            with torch.inference_mode():
                probs = torch.softmax(self.model(input_ids=ids, attention_mask=att).logits.float(), -1)[..., 1].numpy()
            for r, (wid, c) in enumerate(batch):
                scores = probs[r, np.asarray(c["first"])]
                for i, s in zip(c["word_index"], scores):
                    out[wid][i] = float(s)

        done_windows = 0
        for w in windows:
            pending += [(w.id, c) for c in self.chunks(w)]
            done_windows += 1
            if len(pending) >= batch_size:
                run(pending); pending = []
                if progress:
                    progress(done_windows); done_windows = 0
        if pending:
            run(pending)
        if progress and done_windows:
            progress(done_windows)
        return out
