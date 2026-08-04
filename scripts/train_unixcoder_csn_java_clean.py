#!/usr/bin/env python3
"""
Java (GraphCodeBERT cleaned) CodeSearchNet finetune entry (UniXcoder bi-encoder; same hyperparams as train_unixcoder_csn.py).

Unlike plain train_unixcoder_csn.py, this entry **always** uses cleaned data under
``CodeSearchNet_clean_Dataset/java/train.jsonl`` and does not prefer raw ``CodeSearchNet_Dataset/java``,
so you do not pick uncleaned data when both trees exist.

- Default --output-dir: shared.csn_paths.default_unixcoder_csn_output_dir() (unixcoder-csn-java;
  should match clone_detection.unixcoder.model_path used by evaluate_code_search.py).
- Default train: ``<CSN_CLEAN_OUTPUT_DIR or default_csn_clean_dataset_root>/java/train.jsonl``.
- Default validation: random 3% from train (--valid-split-ratio 0.03, fixed seed), same as other cleaned languages;
  do not use test for train/val to avoid eval leakage.
- Does not enable Python-only code docstring stripping.

Usage:

  python scripts/train_unixcoder_csn_java_clean.py

  python scripts/train_unixcoder_csn_java_clean.py --train-jsonl /path/to/java/train.jsonl --output-dir /path/to/out
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import (
    default_csn_clean_dataset_root,
    default_unixcoder_csn_output_dir,
)


def _has_long_opt(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_defaults() -> None:
    argv = sys.argv[1:]
    inserts: list[str] = []

    if not _has_long_opt(argv, "--output-dir"):
        inserts.extend(
            [
                "--output-dir",
                str(default_unixcoder_csn_output_dir().resolve()),
            ]
        )

    java_clean = default_csn_clean_dataset_root() / "java"
    if java_clean.is_dir():
        if not _has_long_opt(argv, "--train-jsonl"):
            tr = java_clean / "train.jsonl"
            if tr.is_file():
                inserts.extend(["--train-jsonl", str(tr.resolve())])
    if not _has_long_opt(argv, "--valid-jsonl") and not _has_long_opt(
        argv, "--valid-split-ratio"
    ):
        inserts.extend(["--valid-split-ratio", "0.03"])

    sys.argv = [sys.argv[0]] + inserts + argv


def _argv_has_train_jsonl(argv: list[str]) -> bool:
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--train-jsonl":
            return True
        if a.startswith("--train-jsonl="):
            return True
        i += 1
    return False


if __name__ == "__main__":
    _inject_defaults()
    clean_train = default_csn_clean_dataset_root() / "java" / "train.jsonl"
    if not _argv_has_train_jsonl(sys.argv) and not clean_train.is_file():
        print(
            "Error: --train-jsonl not set and cleaned train file not found:\n"
            f"  {clean_train}\n"
            "Prepare CodeSearchNet_clean_Dataset/java/train.jsonl or pass --train-jsonl explicitly."
        )
        sys.exit(1)
    from scripts.train_unixcoder_csn import main

    main()
