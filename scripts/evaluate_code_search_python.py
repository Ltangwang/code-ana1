#!/usr/bin/env python3
"""
CodeSearchNet 清洗版 Python 评测入口。

双塔权重：优先 config/settings.yaml 的 code_search.unixcoder_model_path_python；
未配置或目录不存在时，用环境变量 CODE_SEARCH_UNIXCODER_PYTHON_PATH（本入口默认设为
unixcoder-csn-python 输出目录，与 Java 的 clone_detection.unixcoder.model_path 分离）。

本入口默认将边侧双塔召回、Success@K、Ollama/云候选池与 no_edge 云解救召回的 top-K
对齐为同一数值（见 _PYTHON_EVAL_TOP_K）；命令行显式传入 --top-k / --llm-pool-k /
--cloud-rescue-k 时仍以命令行为准。
"""

from __future__ import annotations

# 边侧 retrieve_k、评测 K、Ollama/云池子、cloud_rescue 的 max(retrieve_k,·) 对齐用
_PYTHON_EVAL_TOP_K = 5

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _has_long_opt(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_python_eval_k_defaults() -> None:
    """未在 CLI 指定时，将 top-k / 云边池子 / 云解救召回与边侧 K 对齐。"""
    argv = sys.argv[1:]
    k = str(_PYTHON_EVAL_TOP_K)
    inserts: list[str] = []
    if not _has_long_opt(argv, "--top-k"):
        inserts.extend(["--top-k", k])
    if not _has_long_opt(argv, "--llm-pool-k"):
        inserts.extend(["--llm-pool-k", k])
    if not _has_long_opt(argv, "--cloud-rescue-k"):
        inserts.extend(["--cloud-rescue-k", k])
    sys.argv = [sys.argv[0]] + inserts + argv


if __name__ == "__main__":
    from shared.csn_paths import default_unixcoder_csn_python_output_dir

    os.environ.setdefault(
        "CODE_SEARCH_UNIXCODER_PYTHON_PATH",
        str(default_unixcoder_csn_python_output_dir().resolve()),
    )
    _inject_python_eval_k_defaults()
    sys.argv = [sys.argv[0], "--language", "python"] + sys.argv[1:]
    from scripts.evaluate_code_search_non_java import main

    main()
