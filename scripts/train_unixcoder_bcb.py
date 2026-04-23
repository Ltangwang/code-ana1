#!/usr/bin/env python3
"""
Fine-tune microsoft/unixcoder-base as a binary clone classifier (BigCloneBench-style pairs).

独立脚本：evaluate_*.py 不会执行本训练；仅在需要更新 BCB 分类器时手动运行 export/train。

权重缓存：可通过 HF_HOME / HF_HUB_CACHE 或 config/settings.yaml 中 models.huggingface_cache 指定。

数据格式（JSONL 每行）：
  {"code1": "...", "code2": "...", "dfg1": "...", "dfg2": "...", "label": 0|1}

从 CodeXGLUE BCB 目录导出：
  python scripts/train_unixcoder_bcb.py export --bcb-root datasets/.../dataset
  python scripts/train_unixcoder_bcb.py train --train-jsonl data/bcb_train.jsonl --valid-jsonl data/bcb_valid.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.autodl_env import apply_autodl_data_disk_env

apply_autodl_data_disk_env()


def _resolve_hf_cache(cli: str | None) -> str | None:
    if cli:
        return cli
    try:
        import yaml

        cfg_path = _ROOT / "config" / "settings.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            hc = (y.get("models") or {}).get("huggingface_cache")
            if hc:
                return str(Path(str(hc)).expanduser())
    except Exception:
        pass
    return None


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


def load_pair_jsonl(data_path: str):
    from datasets import Dataset

    def jsonl_generator():
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield {
                        "code1": item.get("code1", ""),
                        "code2": item.get("code2", ""),
                        "dfg1": item.get("dfg1", ""),
                        "dfg2": item.get("dfg2", ""),
                        "label": int(item.get("label", 0)),
                    }
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    return Dataset.from_generator(jsonl_generator)


def tokenize_batch(examples, tokenizer, max_length: int):
    sep = getattr(tokenizer, "sep_token", "</s>")
    text1 = [c + f" {sep} " + d for c, d in zip(examples["code1"], examples["dfg1"])]
    text2 = [c + f" {sep} " + d for c, d in zip(examples["code2"], examples["dfg2"])]
    enc = tokenizer(
        text1,
        text2,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    enc["labels"] = examples["label"]
    return enc


def export_bcb_to_jsonl(
    bcb_root: Path,
    out_train: Path,
    out_valid: Path,
    out_test: Path | None = None,
) -> None:
    """Read data.jsonl + train.txt / valid.txt / test.txt → JSONL for training."""
    from edge.java_dfg_skeleton import extract_java_dfg_skeleton

    data_path = bcb_root / "data.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    code_dict: dict = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            idx = item.get("idx")
            func = item.get("func", "")
            if idx is not None:
                code_dict[str(idx)] = func
                code_dict[int(idx)] = func

    def write_pairs(txt_path: Path, dest: Path, max_pairs: int | None = None) -> int:
        if not txt_path.exists():
            return 0
        n = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "r", encoding="utf-8") as fin, open(
            dest, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if max_pairs is not None and n >= max_pairs:
                    break
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    i1, i2, lab = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                c1 = code_dict.get(i1, "")
                c2 = code_dict.get(i2, "")
                d1 = extract_java_dfg_skeleton(
                    c1, max_edges=32, prefer_tree_sitter=False
                )
                d2 = extract_java_dfg_skeleton(
                    c2, max_edges=32, prefer_tree_sitter=False
                )
                fout.write(
                    json.dumps(
                        {
                            "code1": c1,
                            "code2": c2,
                            "dfg1": d1,
                            "dfg2": d2,
                            "label": lab,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
                if n % 1000 == 0:
                    print(f"  Exported {n} pairs to {dest.name}...")
        return n

    nt = write_pairs(bcb_root / "train.txt", out_train, max_pairs=10000)
    nv = write_pairs(bcb_root / "valid.txt", out_valid, max_pairs=2000)
    print(f"Exported train: {nt} pairs -> {out_train}")
    print(f"Exported valid: {nv} pairs -> {out_valid}")
    if out_test:
        nte = write_pairs(bcb_root / "test.txt", out_test, max_pairs=2000)
        print(f"Exported test: {nte} pairs -> {out_test}")


def cmd_train(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score
    from transformers import (
        RobertaForSequenceClassification,
        RobertaTokenizer,
        Trainer,
        TrainingArguments,
    )

    cache = _resolve_hf_cache(args.hf_cache)
    _apply_hf_cache(cache)

    model_name = args.base_model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    train_ds = load_pair_jsonl(args.train_jsonl)
    eval_ds = load_pair_jsonl(args.valid_jsonl)

    def tok(ex):
        return tokenize_batch(ex, tokenizer, args.max_length)

    train_t = train_ds.map(
        tok,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenize train",
    )
    eval_t = eval_ds.map(
        tok,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenize eval",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float(accuracy_score(labels, preds))}

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=use_fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_t,
        eval_dataset=eval_t,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    best_dir = Path(args.final_model_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved best checkpoint to {best_dir}")


def main() -> None:
    default_cache = _resolve_hf_cache(None)

    p = argparse.ArgumentParser(description="UniXcoder BCB fine-tuning")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="BCB txt + data.jsonl -> train/valid JSONL")
    pe.add_argument(
        "--bcb-root",
        type=Path,
        default=_ROOT
        / "datasets/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset",
    )
    pe.add_argument(
        "--out-train",
        type=Path,
        default=_ROOT / "data/bcb_train.jsonl",
    )
    pe.add_argument(
        "--out-valid",
        type=Path,
        default=_ROOT / "data/bcb_valid.jsonl",
    )
    pe.add_argument("--out-test", type=Path, default=None)
    pe.set_defaults(
        func=lambda a: export_bcb_to_jsonl(
            a.bcb_root, a.out_train, a.out_valid, a.out_test
        )
    )

    pt = sub.add_parser("train", help="Fine-tune UniXcoder")
    pt.add_argument("--train-jsonl", type=str, required=True)
    pt.add_argument("--valid-jsonl", type=str, required=True)
    pt.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT / "unixcoder-bcb-finetuned"),
        help="HF Trainer checkpoints",
    )
    pt.add_argument(
        "--final-model-dir",
        type=str,
        default=str(_ROOT / "models/unixcoder-bcb-best"),
        help="最终保存的 model + tokenizer（可改为与 clone_detection.unixcoder.model_path 一致）",
    )
    pt.add_argument("--base-model", type=str, default="microsoft/unixcoder-base")
    pt.add_argument(
        "--hf-cache",
        type=str,
        default=default_cache,
        help="HF 缓存目录（默认读 config 或留空使用环境变量）",
    )
    pt.add_argument("--max-length", type=int, default=512)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--learning-rate", type=float, default=1e-5)
    pt.add_argument("--fp16", action="store_true", help="FP16 if CUDA available")
    pt.set_defaults(func=cmd_train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
