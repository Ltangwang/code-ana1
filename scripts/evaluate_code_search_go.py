#!/usr/bin/env python3
"""Go CSN eval → ``evaluate_code_search_non_java`` (--language go). K defaults: ``_GO_EVAL_TOP_K``. Weights: env + settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_GO_EVAL_TOP_K = 5

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _has_long_opt(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_go_eval_k_defaults() -> None:
    """Prepend K flags from ``_GO_EVAL_TOP_K`` when missing."""
    argv = sys.argv[1:]
    k = str(_GO_EVAL_TOP_K)
    inserts: list[str] = []
    if not _has_long_opt(argv, "--top-k"):
        inserts.extend(["--top-k", k])
    if not _has_long_opt(argv, "--llm-pool-k"):
        inserts.extend(["--llm-pool-k", k])
    if not _has_long_opt(argv, "--cloud-rescue-k"):
        inserts.extend(["--cloud-rescue-k", k])
    sys.argv = [sys.argv[0]] + inserts + argv


if __name__ == "__main__":
    from shared.csn_paths import default_unixcoder_csn_go_output_dir

    os.environ.setdefault(
        "CODE_SEARCH_UNIXCODER_GO_PATH",
        str(default_unixcoder_csn_go_output_dir().resolve()),
    )
    _inject_go_eval_k_defaults()
    sys.argv = [sys.argv[0], "--language", "go"] + sys.argv[1:]
    from scripts.evaluate_code_search_non_java import main

    main()
