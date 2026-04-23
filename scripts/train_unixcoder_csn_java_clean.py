#!/usr/bin/env python3
"""
Java（GraphCodeBERT 清洗版）CodeSearchNet 微调入口（UniXcoder 双塔，超参与 train_unixcoder_csn.py 一致）。

与直接运行 train_unixcoder_csn.py 的区别：本入口**强制**使用清洗版数据根下的
``CodeSearchNet_clean_Dataset/java/train.jsonl``，不会优先落到原始 ``CodeSearchNet_Dataset/java``，
避免在同时存在两套数据时误用未清洗语料。

- 默认 --output-dir 为 shared.csn_paths.default_unixcoder_csn_output_dir()（unixcoder-csn-java；
  与 evaluate_code_search.py 使用的 clone_detection.unixcoder.model_path 应对齐为同一物理目录）。
- 默认 train：``<CSN_CLEAN_OUTPUT_DIR 或 default_csn_clean_dataset_root>/java/train.jsonl``。
- 默认验证：从 train 中随机划出 3%（--valid-split-ratio 0.03，seed 固定），与其它清洗版语言一致；
  勿用 test 做训练/验证以免评测泄漏。
- 不启用 Python 专用的 code 去 docstring。

用法：

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
            "错误: 未指定 --train-jsonl，且未找到清洗版训练文件:\n"
            f"  {clean_train}\n"
            "请准备 CodeSearchNet_clean_Dataset/java/train.jsonl，或显式传入 --train-jsonl。"
        )
        sys.exit(1)
    from scripts.train_unixcoder_csn import main

    main()
