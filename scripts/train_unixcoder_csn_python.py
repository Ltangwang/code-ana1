#!/usr/bin/env python3
"""
Entry point for Python cleaned CodeSearchNet finetuning (UniXcoder bi-encoder).

- Default --output-dir is unixcoder-csn-python, separate from Java unixcoder-csn-java to avoid overwriting.
- Default train: CodeSearchNet_clean_Dataset/python/train.jsonl (full; --train-max-samples default 0 = no cap).
- Default validation: random 3% from train (--valid-split-ratio 0.03, fixed seed) because cleaned valid/test
  often have no code body; do not split from test to avoid eval leakage.
- Default enables ``--strip-python-code-docstrings``: training code is AST-stripped to reduce literal overlap with query;
  keep ``code_search.strip_python_code_docstrings: true`` in config/settings.yaml so eval indexing matches.

You can still pass --valid-jsonl; if --valid-split-ratio>0, split takes precedence.

Usage:

  python scripts/train_unixcoder_csn_python.py

  # Explicitly disable docstring stripping (not recommended)
  python scripts/train_unixcoder_csn_python.py --no-strip-python-code-docstrings
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import default_csn_clean_dataset_root, default_unixcoder_csn_python_output_dir


def _has_long_opt(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_defaults() -> None:
    argv = sys.argv[1:]
    inserts: list[str] = []

    if not _has_long_opt(argv, "--output-dir"):
        inserts.extend(
            ["--output-dir", str(default_unixcoder_csn_python_output_dir().resolve())]
        )

    py_clean = default_csn_clean_dataset_root() / "python"
    if py_clean.is_dir():
        if not _has_long_opt(argv, "--train-jsonl"):
            tr = py_clean / "train.jsonl"
            if tr.is_file():
                inserts.extend(["--train-jsonl", str(tr.resolve())])

    if not _has_long_opt(argv, "--valid-jsonl") and not _has_long_opt(
        argv, "--valid-split-ratio"
    ):
        inserts.extend(["--valid-split-ratio", "0.03"])

    if not _has_long_opt(argv, "--no-strip-python-code-docstrings") and not _has_long_opt(
        argv, "--strip-python-code-docstrings"
    ):
        inserts.extend(["--strip-python-code-docstrings"])

    sys.argv = [sys.argv[0]] + inserts + argv


if __name__ == "__main__":
    _inject_defaults()
    from scripts.train_unixcoder_csn import main

    main()
