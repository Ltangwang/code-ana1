"""
CodeSearchNet data dirs and default model/cache roots for eval and training.
- Raw HF export: default /root/autodl-fs/CodeSearchNet_Dataset (overridable via CSN_OUTPUT_DIR).
- GraphCodeBERT filtered: if CodeSearchNet_clean_Dataset/ exists at repo root it wins; else default
  /root/autodl-fs/CodeSearchNet_clean_Dataset (CSN_CLEAN_OUTPUT_DIR always overrides when set).
- Parent dir can be set with CSN_DATA_PARENT (default /root/autodl-fs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from shared.autodl_env import AUTODL_DATA_ROOT

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CSN_PARENT = Path("/root/autodl-fs")


def repo_root() -> Path:
    return _REPO_ROOT


def default_csn_data_parent() -> Path:
    """Default parent for raw / cleaned CodeSearchNet (when no per-dataset OUTPUT env is set)."""
    env = os.environ.get("CSN_DATA_PARENT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_CSN_PARENT.resolve()


def default_csn_dataset_root() -> Path:
    env = os.environ.get("CSN_OUTPUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (default_csn_data_parent() / "CodeSearchNet_Dataset").resolve()


def default_csn_java_dir() -> Path:
    return default_csn_dataset_root() / "java"


def default_csn_clean_dataset_root() -> Path:
    env = os.environ.get("CSN_CLEAN_OUTPUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    repo_clean = (_REPO_ROOT / "CodeSearchNet_clean_Dataset").resolve()
    if repo_clean.is_dir():
        return repo_clean
    return (default_csn_data_parent() / "CodeSearchNet_clean_Dataset").resolve()


def _csn_test_jsonl_has_loadable_query(java_dir: Path, max_lines: int = 3000) -> bool:
    """
    Same as scripts.csn_data.load_csn_dataset(require_code=False):
    at least one line with NL and url or code (cleaned test often has no code, url-only for codebase join).
    """
    p = java_dir / "test.jsonl"
    if not p.is_file():
        return False
    try:
        if p.stat().st_size == 0:
            return False
    except OSError:
        return False
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = item.get("func_code_string") or item.get("original_string") or ""
            nl = (
                item.get("func_documentation_string")
                or item.get("docstring")
                or ""
            )
            if not nl and item.get("docstring_tokens"):
                dt = item["docstring_tokens"]
                nl = " ".join(dt) if isinstance(dt, list) else str(dt)
            url = item.get("url") or ""
            if nl and (url or code):
                return True
    return False


def _autodl_sibling_java_dirs(*relative: str) -> list[Path]:
    """Try common repo dir names on the data disk to avoid empty reads from code-anal / code-ana1 / code-analyze mix-ups."""
    if not AUTODL_DATA_ROOT.is_dir():
        return []
    rel = Path(*relative)
    return [
        (AUTODL_DATA_ROOT / name / rel).resolve()
        for name in ("code-ana1", "code-anal", "code-analyze")
    ]


def default_csn_java_dir_for_code_search() -> Path:
    """
    Java data dir for code search eval: prefer cleaned (with test.jsonl), else raw.
    Override: CSN_JAVA_DIR.
    On AutoDL, also try same dataset dirs under AUTODL_DATA_ROOT for code-ana1, code-anal, code-analyze.
    If a dir has only test.jsonl that is empty or yields no query, skip and try the next
    (e.g. data in code-ana1 but code in code-anal).
    """
    env = os.environ.get("CSN_JAVA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    candidates = [
        default_csn_clean_dataset_root() / "java",
        default_csn_data_parent() / "CodeSearchNet_Dataset" / "java",
        _REPO_ROOT / "CodeSearchNet_Dataset" / "java",
        *_autodl_sibling_java_dirs("CodeSearchNet_clean_Dataset", "java"),
        *_autodl_sibling_java_dirs("CodeSearchNet_Dataset", "java"),
        Path(r"F:\Homework Graphics\code-analyze\CodeSearchNet_clean_Dataset\java"),
        Path(r"F:\Homework Graphics\code-anal\CodeSearchNet_clean_Dataset\java"),
        Path(r"F:\Homework Graphics\code-analyze\CodeSearchNet_Dataset\java"),
        Path(r"F:\Homework Graphics\code-anal\CodeSearchNet_Dataset\java"),
    ]
    for c in candidates:
        if c.is_dir() and _csn_test_jsonl_has_loadable_query(c):
            return c.resolve()
    return (default_csn_clean_dataset_root() / "java").resolve()


def default_csn_validation_jsonl(java_dir: Path | None = None) -> Path:
    """HF export usually uses validation.jsonl; if only valid.jsonl exists, use that."""
    base = java_dir if java_dir is not None else default_csn_java_dir()
    for name in ("validation.jsonl", "valid.jsonl"):
        p = base / name
        if p.is_file():
            return p
    return base / "validation.jsonl"


def default_eval_models_parent(config: dict | None) -> Path:
    """Parent for retrieval embedding caches (often subdirs like 1/)."""
    env = os.environ.get("CSN_EVAL_MODELS_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if AUTODL_DATA_ROOT.is_dir():
        return (AUTODL_DATA_ROOT / "models").resolve()
    if config:
        cfg_root = (config.get("models") or {}).get("root")
        if cfg_root:
            p = Path(str(cfg_root))
            if p.exists():
                return p
    return (_REPO_ROOT / "models").resolve()


def code_search_eval_results_dir(
    config: dict | None,
    cli_override: str | None = None,
) -> Path:
    """
    Where eval writes: results_code_search.json, CSV, LLM rerank disk cache.

    Precedence: CLI --results-dir > settings.yaml code_search_eval.results_output (if non-empty)
    > default code_search_eval/ next to embed cache (e.g. AutoDL data disk models/code_search_eval).

    If results_output is relative, it is relative to repo root (CODE-ANA1 / project root).
    """
    if cli_override and str(cli_override).strip():
        p = Path(str(cli_override).strip()).expanduser()
        if not p.is_absolute():
            p = _REPO_ROOT / p
        return p.resolve()

    raw = ""
    if config:
        cse = config.get("code_search_eval") or {}
        ro = cse.get("results_output")
        if ro is not None:
            raw = str(ro).strip()

    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = _REPO_ROOT / p
        return p.resolve()

    return default_eval_models_parent(config) / "code_search_eval"


def default_hf_cache_for_training() -> Path:
    """HF cache for training scripts (data disk preferred)."""
    h = os.environ.get("HF_HOME", "").strip()
    if h:
        return Path(h)
    if AUTODL_DATA_ROOT.is_dir():
        return AUTODL_DATA_ROOT / ".cache" / "huggingface"
    return _REPO_ROOT / "models" / "huggingface"


def default_unixcoder_csn_output_dir() -> Path:
    """Default Java CSN finetune output (do not share with Python)."""
    return default_eval_models_parent(None) / "unixcoder-csn-java"


def default_unixcoder_csn_python_output_dir() -> Path:
    """Default Python CSN finetune output; separate from Java to avoid clobbering unixcoder-csn-java."""
    return default_eval_models_parent(None) / "unixcoder-csn-python"


def default_unixcoder_csn_go_output_dir() -> Path:
    """Default Go (cleaned) CSN finetune output; separate from Java/Python."""
    return default_eval_models_parent(None) / "unixcoder-csn-go"


def default_unixcoder_csn_javascript_output_dir() -> Path:
    """Default JavaScript (cleaned) CSN finetune output; separate from other languages."""
    return default_eval_models_parent(None) / "unixcoder-csn-javascript"


def default_unixcoder_csn_php_output_dir() -> Path:
    """Default PHP (cleaned) CSN finetune output; separate from other languages."""
    return default_eval_models_parent(None) / "unixcoder-csn-php"


def default_unixcoder_csn_ruby_output_dir() -> Path:
    """Default Ruby (cleaned) CSN finetune output; separate from other languages."""
    return default_eval_models_parent(None) / "unixcoder-csn-ruby"
