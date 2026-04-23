#!/usr/bin/env python3
"""
Fine-tune UniXcoder for CodeSearchNet retrieval (bi-encoder, in-batch contrastive).

Standalone script (not invoked from ``evaluate_code_search.py``). After training, set
``code_search`` / ``clone_detection`` ``model_path`` in config to ``--output-dir``, then run eval.

Data matches the evaluation stack: HuggingFace CodeSearchNet or GraphCodeBERT-filtered JSONL
(see ``scripts.csn_data.iter_csn_jsonl``).

Defaults (override with ``--train-jsonl`` / ``--valid-jsonl``):
  * Train: ``CodeSearchNet_Dataset/java/train.jsonl`` if present, else the java dir used for retrieval eval
  * Valid: ``validation.jsonl`` or ``valid.jsonl`` beside train, or ``--valid-split-ratio`` (typical for cleaned splits)

GraphCodeBERT-filtered java (avoids raw HF java by mistake):
  ``python scripts/train_unixcoder_csn_java_clean.py``

Examples:
  python scripts/train_unixcoder_csn.py --output-dir /path/to/unixcoder-csn-java
  python scripts/train_unixcoder_csn.py \\
    --train-jsonl /data/CodeSearchNet_Dataset/java/train.jsonl \\
    --valid-jsonl /data/CodeSearchNet_Dataset/java/valid.jsonl \\
    --output-dir /path/to/unixcoder-csn-java

Defaults: 5 epochs, lr 1e-5, grad accum 4, label smoothing 0.05, AMP, etc.
``--train-max-samples`` 0 reads the full train file. Writes ``csn_train_args.json`` under ``--output-dir``.

Concurrency: single GPU; ``--num-workers`` speeds IO. Launch multiple processes for different languages.
Multi-GPU DDP is not implemented.

Per-language wrappers:
  ``train_unixcoder_csn_python.py``, ``train_unixcoder_csn_go.py``, ``train_unixcoder_csn_javascript.py``,
  ``train_unixcoder_csn_php.py``, ``train_unixcoder_csn_ruby.py``
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.autodl_env import apply_autodl_data_disk_env

apply_autodl_data_disk_env()

from scripts.csn_data import iter_csn_jsonl
from shared.csn_python_code_strip import strip_python_code_docstrings as _strip_py_docs
from shared.csn_paths import (
    default_csn_java_dir,
    default_csn_java_dir_for_code_search,
    default_csn_validation_jsonl,
    default_hf_cache_for_training,
    default_unixcoder_csn_output_dir,
)


def _default_java_dir_for_train() -> Path:
    raw = default_csn_java_dir()
    if (raw / "train.jsonl").is_file():
        return raw
    return default_csn_java_dir_for_code_search()


def _resolve_hf_cache(cli: str | None) -> str | None:
    if cli:
        return cli
    try:
        import yaml

        cfg_path = ROOT / "config" / "settings.yaml"
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            hc = (y.get("models") or {}).get("huggingface_cache")
            if hc:
                return str(Path(str(hc)).expanduser())
    except Exception:
        pass
    return str(default_hf_cache_for_training())


def _apply_hf_cache(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    hub = p / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(p))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(p))


def load_samples(
    path: str | Path,
    max_samples: int | None = None,
    *,
    min_query_chars: int = 5,
    min_code_chars: int = 16,
    clean_query: bool = False,
    strip_python_code_docstrings: bool = False,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    limit = max_samples if (max_samples is not None and max_samples > 0) else None
    for i, item in enumerate(iter_csn_jsonl(path)):
        if limit is not None and i >= limit:
            break
        nl = (item.get("nl_query") or "").strip()
        code = (item.get("code") or "").strip()
        if strip_python_code_docstrings:
            code = _strip_py_docs(code)
        if clean_query:
            nl = " ".join(nl.split())
            code = " ".join(code.split())
        if (
            len(nl) >= min_query_chars
            and len(code) >= min_code_chars
        ):
            out.append({"query": nl, "code": code})
    return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_train_args_json(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "csn_train_args.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote training config snapshot: {p}")


class CSNDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.records[idx]


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _encode_cls(model: RobertaModel, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**batch)
    return out.last_hidden_state[:, 0, :]


def _collate(tokenizer: RobertaTokenizer, max_len: int):
    def fn(batch: List[Dict[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries = [x["query"] for x in batch]
        codes = [x["code"] for x in batch]
        q = tokenizer(
            queries,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        c = tokenizer(
            codes,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {"query_inputs": q, "code_inputs": c}

    return fn


@torch.no_grad()
def evaluate_mrr(
    model: RobertaModel,
    tokenizer: RobertaTokenizer,
    valid_ds: CSNDataset,
    batch_size: int,
    max_len: int,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> float:
    model.eval()
    loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate(tokenizer, max_len),
    )

    all_q: List[np.ndarray] = []
    all_c: List[np.ndarray] = []
    for batch in tqdm(loader, desc="Eval embed", leave=False):
        q_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
        c_inputs = {k: v.to(device) for k, v in batch["code_inputs"].items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            q_vec = _l2norm(_encode_cls(model, q_inputs)).cpu().numpy()
            c_vec = _l2norm(_encode_cls(model, c_inputs)).cpu().numpy()
        all_q.append(q_vec)
        all_c.append(c_vec)

    q_mat = np.concatenate(all_q, axis=0).astype(np.float32)
    c_mat = np.concatenate(all_c, axis=0).astype(np.float32)
    sims = q_mat @ c_mat.T

    rr_sum = 0.0
    n = sims.shape[0]
    for i in range(n):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1
        rr_sum += 1.0 / rank
    return rr_sum / max(1, n)


def main() -> None:
    default_java = _default_java_dir_for_train()
    default_train = default_java / "train.jsonl"
    default_valid = default_csn_validation_jsonl(default_java)

    parser = argparse.ArgumentParser(description="Fine-tune UniXcoder on CodeSearchNet")
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=None,
        help=f"Default: {default_train} (if it exists).",
    )
    parser.add_argument(
        "--valid-jsonl",
        type=str,
        default=None,
        help=f"Default: {default_valid} (if it exists). If --valid-split-ratio>0, split from train and ignore this.",
    )
    parser.add_argument(
        "--valid-split-ratio",
        type=float,
        default=0.0,
        help=(
            "Random dev fraction from already-loaded train-jsonl rows (seeded), e.g. 0.03=3%%. "
            "For GraphCodeBERT cleaned splits where valid/test may lack code; never split from test to avoid leakage. "
            "If >0, --valid-jsonl is not read."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_unixcoder_csn_output_dir()),
    )
    parser.add_argument("--base-model", type=str, default="microsoft/unixcoder-base")
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        help="Default: config settings models.huggingface_cache or data disk .cache/huggingface",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps; effective batch ≈ batch_size * this (same as existing csn_train_args.json).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="label_smoothing for symmetric InfoNCE cross_entropy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-query-chars", type=int, default=5)
    parser.add_argument("--min-code-chars", type=int, default=16)
    parser.add_argument(
        "--clean-query",
        action="store_true",
        help="Normalize whitespace on NL/code (off by default; matches existing snapshots).",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="Max lines to read from train-jsonl; <=0 means no limit (full file). Default 0.",
    )
    parser.add_argument(
        "--valid-max-samples",
        type=int,
        default=5000,
        help="When using --valid-jsonl only, cap lines read; --valid-split-ratio split is not truncated by this.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision (default matches csn_train_args: AMP on CUDA).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (>0 for parallel IO; 0 by default for reproducibility).",
    )
    parser.add_argument(
        "--strip-python-code-docstrings",
        action="store_true",
        help=(
            "Before training, strip first module/class/function docstring on code with AST, "
            "reducing query–code literal overlap; must match eval index strip settings."
        ),
    )
    parser.add_argument(
        "--no-strip-python-code-docstrings",
        action="store_true",
        help="If set together with --strip-python-code-docstrings, this wins and keeps docstrings.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_jsonl or default_train)
    valid_path = Path(args.valid_jsonl or default_valid)
    split_ratio = float(args.valid_split_ratio)
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training set not found: {train_path}\n"
            "Set CSN_OUTPUT_DIR / CSN_JAVA_DIR or pass --train-jsonl."
        )
    if split_ratio <= 0.0:
        if not valid_path.is_file():
            raise FileNotFoundError(
                f"Validation set not found: {valid_path}\n"
                "Pass --valid-jsonl, or use --valid-split-ratio to split validation from train."
            )
    elif split_ratio >= 1.0:
        raise ValueError("--valid-split-ratio must be a decimal in (0,1), e.g. 0.05")
    elif args.valid_jsonl:
        print(
            "Note: --valid-split-ratio>0 is set; --valid-jsonl is ignored; validation is split from train only."
        )

    _apply_hf_cache(_resolve_hf_cache(args.hf_cache))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(not args.no_amp and device.type == "cuda")
    _set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.base_model)
    model = RobertaModel.from_pretrained(args.base_model).to(device)

    train_max = int(args.train_max_samples)
    train_max_arg = train_max if train_max > 0 else None
    strip_docs = bool(args.strip_python_code_docstrings) and not bool(
        args.no_strip_python_code_docstrings
    )
    train_records = load_samples(
        str(train_path),
        train_max_arg,
        min_query_chars=args.min_query_chars,
        min_code_chars=args.min_code_chars,
        clean_query=args.clean_query,
        strip_python_code_docstrings=strip_docs,
    )
    if not train_records:
        raise RuntimeError(f"{train_path} has no valid (NL+code) rows; check JSONL format.")
    print(
        f"Loaded training pairs: {len(train_records)} "
        f"(train_max_samples={'full' if train_max_arg is None else train_max})"
        f"{' ; code-side docstring stripping on' if strip_docs else ''}"
    )

    if split_ratio > 0.0:
        if len(train_records) < 2:
            raise RuntimeError(
                "Fewer than 2 valid train rows; cannot split validation. Add data or use --valid-jsonl."
            )
        rng = np.random.RandomState(args.seed)
        order = rng.permutation(len(train_records))
        n_v = max(1, int(len(train_records) * split_ratio))
        n_v = min(n_v, len(train_records) - 1)
        # When splitting from train, use ratio only; do not cap with valid_max_samples
        valid_records = [train_records[i] for i in order[:n_v]]
        train_records = [train_records[i] for i in order[n_v:]]
        if not train_records or not valid_records:
            raise RuntimeError(
                "After train split, train or valid is empty. Add more data or lower --valid-split-ratio."
            )
        print(
            f"Random train split for validation: valid={len(valid_records)}, train={len(train_records)} "
            f"(ratio≈{split_ratio}, seed={args.seed})"
        )
    else:
        valid_records = load_samples(
            str(valid_path),
            args.valid_max_samples,
            min_query_chars=args.min_query_chars,
            min_code_chars=args.min_code_chars,
            clean_query=args.clean_query,
            strip_python_code_docstrings=strip_docs,
        )
        if not valid_records:
            raise RuntimeError(f"{valid_path} has no valid (NL+code) rows.")

    train_ds = CSNDataset(train_records)
    valid_ds = CSNDataset(valid_records)

    g = torch.Generator()
    g.manual_seed(args.seed)
    nw = max(0, int(args.num_workers))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate(tokenizer, args.max_length),
        drop_last=True,
        generator=g,
        num_workers=nw,
        persistent_workers=nw > 0,
    )

    accum = max(1, int(args.grad_accum_steps))
    n_batches = len(train_loader)
    steps_per_epoch = (n_batches + accum - 1) // accum
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_mrr = -1.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = int(args.batch_size) * accum
    snap: Dict[str, Any] = {
        "base_model": args.base_model,
        "train_jsonl": str(train_path.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": accum,
        "effective_batch": effective_batch,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "min_query_chars": args.min_query_chars,
        "min_code_chars": args.min_code_chars,
        "clean_query": args.clean_query,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_max_samples_raw": train_max,
        "train_pairs": train_max if train_max > 0 else "all",
        "strip_python_code_docstrings": strip_docs,
        "valid_pairs": args.valid_max_samples,
        "amp": use_amp,
    }
    if split_ratio > 0.0:
        snap["valid_split_ratio"] = split_ratio
        snap["valid_source"] = "train_random_split"
        snap["valid_jsonl"] = None
        snap["valid_pairs_actual"] = len(valid_records)
        snap["train_pairs_actual"] = len(train_records)
    else:
        snap["valid_jsonl"] = str(valid_path.resolve())
    _save_train_args_json(out_dir, snap)
    if n_batches % accum != 0:
        print(
            f"Note: batches per epoch {n_batches} is not divisible by grad_accum_steps={accum}; "
            "last accumulation group may have a small gradient scale skew; adjust sample count or batch_size."
        )

    global_sched_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        micro = 0
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            q_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
            c_inputs = {k: v.to(device) for k, v in batch["code_inputs"].items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                q_vec = _l2norm(_encode_cls(model, q_inputs))
                c_vec = _l2norm(_encode_cls(model, c_inputs))
                logits = (q_vec @ c_vec.T) / args.temperature
                labels = torch.arange(logits.size(0), device=device)
                ls = float(args.label_smoothing)
                loss = (
                    F.cross_entropy(logits, labels, label_smoothing=ls)
                    + F.cross_entropy(logits.T, labels, label_smoothing=ls)
                ) * 0.5
                loss = loss / accum

            scaler.scale(loss).backward()
            micro += 1
            running_loss += loss.item() * accum

            if micro % accum == 0 or step == n_batches:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                global_sched_step += 1
                sched.step()
                pbar.set_postfix(
                    loss=running_loss / step,
                    lr=sched.get_last_lr()[0],
                )

        mrr = evaluate_mrr(
            model,
            tokenizer,
            valid_ds,
            args.batch_size,
            args.max_length,
            device,
            use_amp=use_amp,
        )
        print(f"[epoch {epoch+1}] valid_mrr={mrr:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            print(f"Saved best model to: {out_dir}")

    print(f"Training finished. Best valid MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
