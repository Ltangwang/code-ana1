"""UniXcoder embeddings + cosine Top-K over CodeSearchNet rows."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.csn_data import load_csn_code_corpus, load_csn_dataset

def _encode_single(
    orchestrator: Any, text: str, max_length: int
) -> np.ndarray:
    import torch

    model = orchestrator.code_encoder
    tok = orchestrator.code_tokenizer
    if model is None or tok is None:
        raise RuntimeError("UniXcoder not loaded on orchestrator")

    # For UniXcoder, we should format the input as: <cls> text <sep>
    # The tokenizer handles this automatically when we pass a single string.
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    base = getattr(model, "roberta", model)
    
    with torch.no_grad():
        out = base(**inputs)
        # UniXcoder paper uses the representation of the [CLS] token
        # However, for code search, it's often better to use mean pooling over the sequence
        # Let's use the standard [CLS] token representation first, but normalize it
        vec = out.last_hidden_state[:, 0, :]
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        vec = vec.float().cpu().numpy().reshape(-1)
    return vec

def _cache_key(
    data_path: Path,
    max_samples: Optional[int],
    encode_len: int,
    corpus_mode: bool = False,
    *,
    model_tag: str = "",
) -> str:
    h = hashlib.sha256()
    if data_path.exists():
        st = data_path.stat()
        h.update(f"{data_path.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    ms = "all" if max_samples is None else str(max_samples)
    mode = "corpus" if corpus_mode else "paired"
    h.update(f"{ms}:{encode_len}:{mode}".encode())
    if model_tag:
        h.update(f"|model:{model_tag}".encode())
    return h.hexdigest()[:24]

class CSNRetriever:
    """CodeSearchNet Retriever: Embeds code and retrieves Top-K using cosine similarity."""

    def __init__(
        self,
        embeddings: np.ndarray,
        records: List[Dict[str, Any]],
    ):
        self.records = records
        x = embeddings.astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        self.embeddings = x / norms

    def search(
        self, orchestrator: Any, nl_query: str, top_k: int = 5, max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """Search for the most relevant code snippets given a natural language query."""
        q = _encode_single(orchestrator, nl_query, max_length)
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        
        sims = self.embeddings @ q
        k = min(top_k, len(self.records))
        idx = np.argsort(-sims)[:k]
        
        results = []
        for i in idx:
            record = self.records[int(i)].copy()
            record["similarity"] = float(sims[int(i)])
            results.append(record)
        return results

    @classmethod
    def build_or_load(
        cls,
        orchestrator: Any,
        data_path: Path,
        cache_dir: Path,
        max_samples: Optional[int] = 10000,
        encode_len: int = 512,
        batch_size: int = 32,
        corpus_mode: bool = False,
        cache_model_tag: str = "",
        strip_python_docstrings: bool = False,
    ) -> Optional["CSNRetriever"]:
        """Build or load the retriever index.

        If corpus_mode is True, load rows with non-empty code only (for codebase.jsonl).
        Otherwise require both NL and code (for train/test jsonl).
        """
        if not data_path.exists():
            print(f"Dataset not found: {data_path}")
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = (cache_model_tag or "").strip()
        tag_for_key = hashlib.sha256(tag.encode()).hexdigest()[:16] if tag else ""
        strip_tag = "stripdoc1" if strip_python_docstrings else ""
        key = _cache_key(
            data_path,
            max_samples,
            encode_len,
            corpus_mode,
            model_tag=f"{tag_for_key}|{strip_tag}" if strip_tag else tag_for_key,
        )
        cache_npz = cache_dir / f"csn_retriever_emb_{key}.npz"
        cache_meta = cache_dir / f"csn_retriever_emb_{key}.meta.json"

        ms_disp = "all" if max_samples is None else str(max_samples)
        mode_note = " (code corpus)" if corpus_mode else ""
        print(
            f"Loading CodeSearchNet dataset from {data_path}{mode_note} (max_samples={ms_disp})..."
        )
        if corpus_mode:
            records = load_csn_code_corpus(
                data_path,
                max_samples=max_samples,
                strip_python_docstrings=strip_python_docstrings,
            )
        else:
            records = load_csn_dataset(data_path, max_samples=max_samples)
        if not records:
            print("No records loaded.")
            return None

        # Check cache
        if cache_npz.exists() and cache_meta.exists():
            try:
                with open(cache_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("encode_max_length") == encode_len and meta.get("num_records") == len(records):
                    print(f"Loading cached embeddings from {cache_npz}...")
                    data = np.load(cache_npz)
                    emb = data["embeddings"]
                    if len(emb) == len(records):
                        return cls(emb, records)
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")

        import torch

        print(f"Building UniXcoder embeddings for {len(records)} code snippets (encode_max_length={encode_len})...")
        embs: List[np.ndarray] = []
        batch_texts = [r["code"] for r in records]
        
        model = orchestrator.code_encoder
        tok = orchestrator.code_tokenizer
        device = next(model.parameters()).device
        base = getattr(model, "roberta", model)

        for start in tqdm(range(0, len(batch_texts), batch_size), desc="Encoding Code"):
            chunk = batch_texts[start : start + batch_size]
            enc = tok(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=encode_len,
                padding="max_length",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = base(**enc)
                cls_v = out.last_hidden_state[:, 0, :]
                cls_v = torch.nn.functional.normalize(cls_v, p=2, dim=1)
                cls_v = cls_v.float().cpu().numpy()
            for row in cls_v:
                embs.append(row)

        mat = np.stack(embs, axis=0)
        
        print(f"Saving embeddings to {cache_npz}...")
        np.savez_compressed(cache_npz, embeddings=mat)
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "key": key,
                    "num_records": len(records),
                    "encode_max_length": encode_len,
                    "cache_model_tag": tag,
                    "strip_python_docstrings": strip_python_docstrings,
                },
                f,
            )
        return cls(mat, records)
