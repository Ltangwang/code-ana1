#!/usr/bin/env python3
"""
CodeSearchNet (clean) Ruby evaluation entrypoint.

- By default, align bi-encoder retrieve_k, Success@K, Ollama/cloud candidate pool,
  and no_edge cloud-rescue top-K (_RUBY_EVAL_TOP_K). CLI overrides when
  --top-k / --llm-pool-k / --cloud-rescue-k are passed.
- Default checkpoint dir is train_unixcoder_csn_ruby.py output (CODE_SEARCH_UNIXCODER_RUBY_PATH);
  config code_search.unixcoder_model_path_ruby overrides when it is an existing directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Align edge retrieve_k, eval K, Ollama/cloud pool, cloud_rescue (same K edge/cloud)
_RUBY_EVAL_TOP_K = 10

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _has_long_opt(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_ruby_eval_k_defaults() -> None:
    """If not set on CLI, inject top-k / pool / cloud-rescue K aligned to edge K (10)."""
    argv = sys.argv[1:]
    k = str(_RUBY_EVAL_TOP_K)
    inserts: list[str] = []
    if not _has_long_opt(argv, "--top-k"):
        inserts.extend(["--top-k", k])
    if not _has_long_opt(argv, "--llm-pool-k"):
        inserts.extend(["--llm-pool-k", k])
    if not _has_long_opt(argv, "--cloud-rescue-k"):
        inserts.extend(["--cloud-rescue-k", k])
    sys.argv = [sys.argv[0]] + inserts + argv


if __name__ == "__main__":
    from shared.csn_paths import default_unixcoder_csn_ruby_output_dir

    os.environ.setdefault(
        "CODE_SEARCH_UNIXCODER_RUBY_PATH",
        str(default_unixcoder_csn_ruby_output_dir().resolve()),
    )
    _inject_ruby_eval_k_defaults()
    sys.argv = [sys.argv[0], "--language", "ruby"] + sys.argv[1:]
    from scripts.evaluate_code_search_non_java import main

    main()
