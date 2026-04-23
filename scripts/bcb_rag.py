"""
BCB train-pair RAG: UniXcoder RoBERTa backbone CLS embeddings for similarity retrieval, cloud few-shot.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _bcb_train_records(
    base_dir: Path, max_pairs: int, seed: int, prefer_clone_ratio: float = 0.5
) -> List[Dict[str, Any]]:
    data_path = base_dir / "data.jsonl"
    train_path = base_dir / "train.txt"
    if not data_path.exists() or not train_path.exists():
        return []

    code_dict: Dict[Any, str] = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            idx = item.get("idx")
            if idx is not None:
                func = item.get("func", "")
                code_dict[str(idx)] = func
                code_dict[int(idx)] = func

    pairs: List[Tuple[int, int, int]] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pairs.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue

    if not pairs:
        return []

    rng = random.Random(seed)
    pos = [p for p in pairs if p[2] == 1]
    neg = [p for p in pairs if p[2] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), max(1, int(max_pairs * prefer_clone_ratio)))
    n_neg = min(len(neg), max(1, max_pairs - n_pos))
    chosen = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(chosen)
    chosen = chosen[:max_pairs]

    records: List[Dict[str, Any]] = []
    for i1, i2, lab in chosen:
        c1 = code_dict.get(i1, "")
        c2 = code_dict.get(i2, "")
        if not c1 or not c2:
            continue
        records.append(
            {
                "code1": c1,
                "code2": c2,
                "label": lab,
                "id1": i1,
                "id2": i2,
            }
        )
    return records


def _pair_text(c1: str, c2: str, cap: int = 1800) -> str:
    s = (c1 or "")[:cap] + "\n###SPLIT###\n" + (c2 or "")[:cap]
    return s[: cap * 2 + 20]


def _encode_single(
    orchestrator: Any, text: str, max_length: int
) -> np.ndarray:
    import torch

    model = orchestrator.code_encoder
    tok = orchestrator.code_tokenizer
    if model is None or tok is None:
        raise RuntimeError("UniXcoder not loaded on orchestrator")

    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    base = getattr(model, "roberta", None)
    if base is None:
        raise RuntimeError("Expected RoBERTa backbone on code_encoder")
    with torch.no_grad():
        out = base(**inputs)
        vec = out.last_hidden_state[:, 0, :].float().cpu().numpy().reshape(-1)
    return vec


def _cache_key(base_dir: Path, max_pairs: int, seed: int, max_length: int) -> str:
    train = base_dir / "train.txt"
    data = base_dir / "data.jsonl"
    h = hashlib.sha256()
    for p in (train, data):
        if p.exists():
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    h.update(f"{max_pairs}:{seed}:{max_length}".encode())
    return h.hexdigest()[:24]


class BCBRAGRetriever:
    """Train-pair embeddings + cosine Top-K retrieval."""

    def __init__(
        self,
        embeddings: np.ndarray,
        records: List[Dict[str, Any]],
    ):
        self.records = records
        x = embeddings.astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        self.embeddings = x / norms

    def query(
        self, orchestrator: Any, code1: str, code2: str, top_k: int, max_length: int
    ) -> List[Dict[str, Any]]:
        q = _encode_single(orchestrator, _pair_text(code1, code2), max_length)
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self.embeddings @ q
        k = min(top_k, len(self.records))
        idx = np.argsort(-sims)[:k]
        return [self.records[int(i)] for i in idx]

    @staticmethod
    def format_few_shot(examples: List[Dict[str, Any]], code_cap: int = 400) -> str:
        if not examples:
            return "(No retrieved examples — rely on definitions below.)"
        parts = []
        for i, ex in enumerate(examples, 1):
            lab = "CLONE (positive)" if ex.get("label") == 1 else "NOT clone (negative)"
            c1 = (ex.get("code1") or "")[:code_cap]
            c2 = (ex.get("code2") or "")[:code_cap]
            parts.append(
                f"#### Retrieved example {i} [{lab}]\n"
                f"```java\n{c1}\n```\n"
                f"```java\n{c2}\n```\n"
            )
        return "\n".join(parts)

    @classmethod
    def build_or_load(
        cls,
        orchestrator: Any,
        config: dict,
        base_dir: Path,
        seed: int,
    ) -> Optional["BCBRAGRetriever"]:
        rag_cfg = (config.get("clone_detection") or {}).get("rag") or {}
        if not rag_cfg.get("enabled", True):
            return None

        max_pairs = int(rag_cfg.get("max_index_pairs", 4000))
        encode_len = int(rag_cfg.get("encode_max_length", 256))
        cache_dir = Path(
            rag_cfg.get("cache_dir")
            or (config.get("models") or {}).get("root")
            or "."
        )
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(base_dir, max_pairs, seed, encode_len)
        cache_npz = cache_dir / f"bcb_rag_emb_{key}.npz"
        cache_meta = cache_dir / f"bcb_rag_emb_{key}.meta.json"

        records = _bcb_train_records(base_dir, max_pairs, seed)
        if not records:
            return None

        pair_sig = [
            [int(r["id1"]), int(r["id2"]), int(r["label"])] for r in records
        ]
        if cache_npz.exists() and cache_meta.exists():
            try:
                with open(cache_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    meta.get("pairs") == pair_sig
                    and int(meta.get("encode_max_length", 0)) == encode_len
                ):
                    data = np.load(cache_npz)
                    emb = data["embeddings"]
                    if len(emb) == len(records):
                        return cls(emb, records)
            except Exception:
                pass

        import torch

        print(
            f"BCB RAG: building UniXcoder embedding index ({len(records)} pairs), "
            f"encode_max_length={encode_len} ..."
        )
        embs: List[np.ndarray] = []
        batch_texts = [_pair_text(r["code1"], r["code2"]) for r in records]
        model = orchestrator.code_encoder
        tok = orchestrator.code_tokenizer
        device = next(model.parameters()).device
        base = getattr(model, "roberta", None)
        if base is None:
            return None

        bs = int(rag_cfg.get("index_batch_size", 16))
        for start in range(0, len(batch_texts), bs):
            chunk = batch_texts[start : start + bs]
            enc = tok(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=encode_len,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = base(**enc)
                cls_v = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            for row in cls_v:
                embs.append(row)

        mat = np.stack(embs, axis=0)
        np.savez_compressed(cache_npz, embeddings=mat)
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "key": key,
                    "pairs": pair_sig,
                    "encode_max_length": encode_len,
                },
                f,
            )
        return cls(mat, records)
