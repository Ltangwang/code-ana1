#!/usr/bin/env python3
"""
CodeSearchNet (clean) Python evaluation entrypoint.

Bi-encoder weights: prefer config/settings.yaml code_search.unixcoder_model_path_python;
if missing or not a directory, use env CODE_SEARCH_UNIXCODER_PYTHON_PATH (this entry
defaults it to the unixcoder-csn-python output dir, separate from Java
clone_detection.unixcoder.model_path).

By default, align edge bi-encoder recall, Success@K, Ollama/cloud pool, and no_edge
cloud-rescue top-K (_PYTHON_EVAL_TOP_K). CLI --top-k / --llm-pool-k / --cloud-rescue-k
override when provided.
"""

from __future__ import annotations

# Default K for edge retrieve_k, eval K, Ollama/cloud pool, cloud_rescue alignment
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
    """If not set on CLI, inject top-k / pool / cloud-rescue K aligned to edge K."""
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
