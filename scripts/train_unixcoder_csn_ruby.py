#!/usr/bin/env python3
"""
Ruby 清洗版 CodeSearchNet 微调入口（UniXcoder 双塔，超参与 train_unixcoder_csn.py 一致）。

- 默认 --output-dir 为 shared.csn_paths.default_unixcoder_csn_ruby_output_dir()（与
  evaluate_code_search_ruby.py 注入的 CODE_SEARCH_UNIXCODER_RUBY_PATH、以及 config 中
  code_search.unixcoder_model_path_ruby 应对齐为同一物理目录）。
- 默认 train：CodeSearchNet_clean_Dataset/ruby/train.jsonl（全量；--train-max-samples 默认 0=不限制）。
- 默认验证：从 train 中随机划出 3%（--valid-split-ratio 0.03，seed 固定），
  与 Go/PHP/JavaScript 清洗版一致。
- 防泄漏：不把 test.jsonl 用作训练或验证；评测集 test 仅用于 evaluate 脚本，勿并入 train。

用法：

  python scripts/train_unixcoder_csn_ruby.py

  python scripts/train_unixcoder_csn_ruby.py --train-jsonl /path/to/ruby/train.jsonl --output-dir /path/to/out
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import (
    default_csn_clean_dataset_root,
    default_unixcoder_csn_ruby_output_dir,
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
                str(default_unixcoder_csn_ruby_output_dir().resolve()),
            ]
        )

    ruby_clean = default_csn_clean_dataset_root() / "ruby"
    if ruby_clean.is_dir():
        if not _has_long_opt(argv, "--train-jsonl"):
            tr = ruby_clean / "train.jsonl"
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
