#!/usr/bin/env python3
"""
CodeSearchNet 清洗版 JavaScript 评测入口。

- 默认将双塔 retrieve_k、Success@K、Ollama/云候选池、no_edge 云解救的 top-K 对齐为同一数值
  （见 _JS_EVAL_TOP_K）；命令行显式传入 --top-k / --llm-pool-k / --cloud-rescue-k 时以命令行为准。
- 默认将权重目录设为 train_unixcoder_csn_javascript.py 的默认输出（CODE_SEARCH_UNIXCODER_JAVASCRIPT_PATH），
  与 config code_search.unixcoder_model_path_javascript 一致时可被 YAML 覆盖（以存在目录为准）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 边侧 retrieve_k、评测 K、Ollama/云池子、cloud_rescue 对齐（云边同一 K）
_JS_EVAL_TOP_K = 5

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _has_long_opt(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_js_eval_k_defaults() -> None:
    """未在 CLI 指定时，将 top-k / 云边池子 / 云解救召回与边侧 K 对齐为 5。"""
    argv = sys.argv[1:]
    k = str(_JS_EVAL_TOP_K)
    inserts: list[str] = []
    if not _has_long_opt(argv, "--top-k"):
        inserts.extend(["--top-k", k])
    if not _has_long_opt(argv, "--llm-pool-k"):
        inserts.extend(["--llm-pool-k", k])
    if not _has_long_opt(argv, "--cloud-rescue-k"):
        inserts.extend(["--cloud-rescue-k", k])
    sys.argv = [sys.argv[0]] + inserts + argv


if __name__ == "__main__":
    from shared.csn_paths import default_unixcoder_csn_javascript_output_dir

    os.environ.setdefault(
        "CODE_SEARCH_UNIXCODER_JAVASCRIPT_PATH",
        str(default_unixcoder_csn_javascript_output_dir().resolve()),
    )
    _inject_js_eval_k_defaults()
    sys.argv = [sys.argv[0], "--language", "javascript"] + sys.argv[1:]
    from scripts.evaluate_code_search_non_java import main

    main()
