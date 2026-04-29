#!/usr/bin/env python3
"""Ruby CSN eval: delegates to ``evaluate_code_search_non_java`` with ``--language ruby``.

Defaults Top-K / pool / rescue to ``_RUBY_EVAL_TOP_K`` when those flags are omitted.
Checkpoint: ``CODE_SEARCH_UNIXCODER_RUBY_PATH`` or ``settings.yaml`` override.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RUBY_EVAL_TOP_K = 10

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _has_long_opt(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_ruby_eval_k_defaults() -> None:
    """Prepend K flags from ``_RUBY_EVAL_TOP_K`` when missing."""
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
