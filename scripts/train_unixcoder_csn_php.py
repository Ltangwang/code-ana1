#!/usr/bin/env python3
"""
PHP cleaned CodeSearchNet finetune (UniXcoder bi-encoder; same hyperparams as train_unixcoder_csn.py).

- Default --output-dir: shared.csn_paths.default_unixcoder_csn_php_output_dir() (align with
  CODE_SEARCH_UNIXCODER_PHP_PATH from evaluate_code_search_php.py and code_search.unixcoder_model_path_php).
- Default train: CodeSearchNet_clean_Dataset/php/train.jsonl (full; --train-max-samples 0 = no cap).
- Default validation: 3% random from train (--valid-split-ratio 0.03, fixed seed),
  same as Go/JavaScript cleaned; do not use test for train/val to avoid eval leakage.
- Does not enable Python-only code docstring stripping.

Usage:

  python scripts/train_unixcoder_csn_php.py

  python scripts/train_unixcoder_csn_php.py --train-jsonl /path/to/php/train.jsonl --output-dir /path/to/out
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import (
    default_csn_clean_dataset_root,
    default_unixcoder_csn_php_output_dir,
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
                str(default_unixcoder_csn_php_output_dir().resolve()),
            ]
        )

    php_clean = default_csn_clean_dataset_root() / "php"
    if php_clean.is_dir():
        if not _has_long_opt(argv, "--train-jsonl"):
            tr = php_clean / "train.jsonl"
            if tr.is_file():
                inserts.extend(["--train-jsonl", str(tr.resolve())])

    if not _has_long_opt(argv, "--valid-jsonl") and not _has_long_opt(
        argv, "--valid-split-ratio"
    ):
        inserts.extend(["--valid-split-ratio", "0.03"])

    sys.argv = [sys.argv[0]] + inserts + argv


if __name__ == "__main__":
    _inject_defaults()
    from scripts.train_unixcoder_csn import main

    main()
