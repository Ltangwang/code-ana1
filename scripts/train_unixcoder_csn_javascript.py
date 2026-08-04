#!/usr/bin/env python3
"""
JavaScript cleaned CodeSearchNet finetune (UniXcoder bi-encoder; same hyperparams as train_unixcoder_csn.py).

- Default --output-dir: shared.csn_paths.default_unixcoder_csn_javascript_output_dir() (align with
  CODE_SEARCH_UNIXCODER_JAVASCRIPT_PATH and code_search.unixcoder_model_path_javascript).
- Default train: CodeSearchNet_clean_Dataset/javascript/train.jsonl (full; --train-max-samples 0 = no cap).
- Default validation: 3% random from train (--valid-split-ratio 0.03, fixed seed),
  same as Go/Python cleaned; do not use test for train/val to avoid eval leakage.
- Does not enable Python-only code docstring stripping.

Usage:

  python scripts/train_unixcoder_csn_javascript.py

  python scripts/train_unixcoder_csn_javascript.py --train-jsonl /path/to/javascript/train.jsonl --output-dir /path/to/out
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import (
    default_csn_clean_dataset_root,
    default_unixcoder_csn_javascript_output_dir,
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
                str(default_unixcoder_csn_javascript_output_dir().resolve()),
            ]
        )

    js_clean = default_csn_clean_dataset_root() / "javascript"
    if js_clean.is_dir():
        if not _has_long_opt(argv, "--train-jsonl"):
            tr = js_clean / "train.jsonl"
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
