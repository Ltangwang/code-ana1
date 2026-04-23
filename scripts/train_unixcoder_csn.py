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
        help=f"默认: {default_train}（若存在）",
    )
    parser.add_argument(
        "--valid-jsonl",
        type=str,
        default=None,
        help=f"默认: {default_valid}（若存在）。若设置 --valid-split-ratio>0 则从 train 划分，忽略本项。",
    )
    parser.add_argument(
        "--valid-split-ratio",
        type=float,
        default=0.0,
        help=(
            "从 train-jsonl 已加载样本中按 seed 随机划出验证集比例，例如 0.03=3%%。"
            "用于 GraphCodeBERT 清洗版 valid/test 无代码正文的情形；勿从 test 划分以免泄漏。"
            ">0 时不再读取 --valid-jsonl。"
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
        help="默认: config settings models.huggingface_cache 或数据盘 .cache/huggingface",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="梯度累积步数；等效 batch ≈ batch_size * 本值（与既有 csn_train_args.json 一致）",
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
        help="对称 InfoNCE 上 cross_entropy 的 label_smoothing",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-query-chars", type=int, default=5)
    parser.add_argument("--min-code-chars", type=int, default=16)
    parser.add_argument(
        "--clean-query",
        action="store_true",
        help="对 NL/code 做空白归一化（默认关闭，与既有快照一致）",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="读取 train-jsonl 的最大行数；<=0 表示不限制（全量）。默认 0。",
    )
    parser.add_argument(
        "--valid-max-samples",
        type=int,
        default=5000,
        help="仅在使用 --valid-jsonl 时限制读取条数；--valid-split-ratio 划分不受此项截断。",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭 CUDA 混合精度（默认与 csn_train_args 一致：CUDA 上开启 AMP）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker 数（>0 时并发读盘；默认 0 便于复现）",
    )
    parser.add_argument(
        "--strip-python-code-docstrings",
        action="store_true",
        help=(
            "训练前从 code 侧用 AST 去掉模块/类/函数体首条 docstring，"
            "减弱「query 与 code 字面重合」；需与评测索引侧 strip 配置一致。"
        ),
    )
    parser.add_argument(
        "--no-strip-python-code-docstrings",
        action="store_true",
        help="与 --strip-python-code-docstrings 同时出现时本项优先，关闭去 docstring。",
    )
    args = parser.parse_args()

    train_path = Path(args.train_jsonl or default_train)
    valid_path = Path(args.valid_jsonl or default_valid)
    split_ratio = float(args.valid_split_ratio)
    if not train_path.is_file():
        raise FileNotFoundError(
            f"未找到训练集: {train_path}\n"
            "请设置 CSN_OUTPUT_DIR / CSN_JAVA_DIR 或传入 --train-jsonl。"
        )
    if split_ratio <= 0.0:
        if not valid_path.is_file():
            raise FileNotFoundError(
                f"未找到验证集: {valid_path}\n"
                "请传入 --valid-jsonl，或使用 --valid-split-ratio 从 train 中划分验证集。"
            )
    elif split_ratio >= 1.0:
        raise ValueError("--valid-split-ratio 须为 (0,1) 内小数，例如 0.05")
    elif args.valid_jsonl:
        print(
            "提示: 已设置 --valid-split-ratio>0，将忽略 --valid-jsonl，验证集仅从 train 划分。"
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
        raise RuntimeError(f"{train_path} 中无有效 (NL+code) 样本，请检查 JSONL 格式。")
    print(
        f"已加载训练样本: {len(train_records)} 对"
        f"（train_max_samples={'全量' if train_max_arg is None else train_max}）"
        f"{'；已启用 code 侧去 docstring' if strip_docs else ''}"
    )

    if split_ratio > 0.0:
        if len(train_records) < 2:
            raise RuntimeError(
                "train 有效样本不足 2 条，无法划分验证集；请增大数据或使用 --valid-jsonl。"
            )
        rng = np.random.RandomState(args.seed)
        order = rng.permutation(len(train_records))
        n_v = max(1, int(len(train_records) * split_ratio))
        n_v = min(n_v, len(train_records) - 1)
        # 从 train 划分时仅按比例，不用 valid_max_samples 截断（否则 10% 会被压成 5k）
        valid_records = [train_records[i] for i in order[:n_v]]
        train_records = [train_records[i] for i in order[n_v:]]
        if not train_records or not valid_records:
            raise RuntimeError(
                "从 train 划分 valid 后 train 或 valid 为空，请增大数据或减小 --valid-split-ratio。"
            )
        print(
            f"已从 train 随机划分验证集: valid={len(valid_records)}, train={len(train_records)} "
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
            raise RuntimeError(f"{valid_path} 中无有效 (NL+code) 样本。")

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
            f"注意: 每 epoch 批次数 {n_batches} 不能整除 grad_accum_steps={accum}，"
            "最后一个累积组梯度尺度可能略有偏差；可调样本数或 batch_size。"
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
