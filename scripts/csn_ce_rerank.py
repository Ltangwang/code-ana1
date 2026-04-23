"""CodeSearchNet: Cross-Encoder reranking over (query, code) pairs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import CrossEncoder


def load_csn_cross_encoder(
    model_name: str, device: Optional[str] = None
) -> CrossEncoder:
    if device is None or str(device).strip() == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_name, device=device)


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    model: CrossEncoder,
    max_code_chars: int = 2000,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Score each candidate with CE and return copies sorted by score descending."""
    if not candidates:
        return []
    pairs: List[List[str]] = []
    for c in candidates:
        code = c.get("code") or ""
        if len(code) > max_code_chars:
            code = code[:max_code_chars] + "\n... [TRUNCATED]"
        pairs.append([query, code])
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: -float(x[1]))
    out: List[Dict[str, Any]] = []
    for c, s in scored:
        d = dict(c)
        d["ce_score"] = float(s)
        out.append(d)
    return out
