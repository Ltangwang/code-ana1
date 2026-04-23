#!/usr/bin/env python3
"""
Python 清洗版 CodeSearchNet 微调入口。

- 默认 --output-dir 为 unixcoder-csn-python，与 Java 的 unixcoder-csn-java 分离，避免覆盖。
- 默认 train：CodeSearchNet_clean_Dataset/python/train.jsonl（全量；--train-max-samples 默认 0=不限制）。
- 默认验证：从 train 中随机划出 3%（--valid-split-ratio 0.03，seed 固定），
  因清洗版 valid/test 无代码正文；勿从 test 划分以免评测泄漏。
- 默认开启 ``--strip-python-code-docstrings``：训练用 code 经 AST 去掉 docstring，减弱与 query 的字面重合；
  请在 config/settings.yaml 中保持 ``code_search.strip_python_code_docstrings: true`` 以便评测索引一致。

仍可用 --valid-jsonl 指定外部验证集；若同时设 --valid-split-ratio>0，以划分为准。

用法：

  python scripts/train_unixcoder_csn_python.py

  # 显式关闭去 docstring（不推荐）
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
