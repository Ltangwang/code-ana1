"""
CodeSearchNet 数据目录与评测/训练用模型缓存根路径。
- 原始 HF 导出：默认 /root/autodl-fs/CodeSearchNet_Dataset（CSN_OUTPUT_DIR 可覆盖）。
- GraphCodeBERT 过滤版：若仓库根下存在 CodeSearchNet_clean_Dataset/ 则优先用之；否则默认
  /root/autodl-fs/CodeSearchNet_clean_Dataset（CSN_CLEAN_OUTPUT_DIR 始终优先覆盖）。
- 父目录可用 CSN_DATA_PARENT 统一覆盖（默认 /root/autodl-fs）。
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
    """CodeSearchNet 原始/清洗数据集的默认父目录（无单独 OUTPUT 环境变量时）。"""
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
    与 scripts.csn_data.load_csn_dataset(require_code=False) 一致：
    至少一行含 NL，且含 url 或 code（清洗版 test 常无 code，仅用 url 对齐 codebase）。
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
    """在数据盘下尝试常见仓库目录名，减轻 code-anal / code-ana1 / code-analyze 混用导致读空数据。"""
    if not AUTODL_DATA_ROOT.is_dir():
        return []
    rel = Path(*relative)
    return [
        (AUTODL_DATA_ROOT / name / rel).resolve()
        for name in ("code-ana1", "code-anal", "code-analyze")
    ]


def default_csn_java_dir_for_code_search() -> Path:
    """
    代码检索评估用 Java 数据目录：优先清洗版（含 test.jsonl），否则回退到原始目录。
    强制路径：环境变量 CSN_JAVA_DIR。
    在 AutoDL 上还会在 AUTODL_DATA_ROOT 下尝试 code-ana1、code-anal、code-analyze 中的同名数据集目录。
    若某目录仅有 test.jsonl 但为空或无法解析出查询，会跳过并尝试下一候选（例如数据在 code-ana1 而代码在 code-anal）。
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
    """HuggingFace 导出多为 validation.jsonl；若仅有 valid.jsonl 则使用之。"""
    base = java_dir if java_dir is not None else default_csn_java_dir()
    for name in ("validation.jsonl", "valid.jsonl"):
        p = base / name
        if p.is_file():
            return p
    return base / "validation.jsonl"


def default_eval_models_parent(config: dict | None) -> Path:
    """检索嵌入等缓存的父目录（其下常用子目录如 1/）。"""
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
    评测产物目录：results_code_search.json、CSV、LLM 重排磁盘缓存。

    优先级：CLI --results-dir > settings.yaml code_search_eval.results_output（非空）
    > 默认与嵌入缓存同逻辑下的 code_search_eval/（如 AutoDL 数据盘 models/code_search_eval）。

    results_output 为相对路径时相对于仓库根（CODE-ANA1 / 项目根）。
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
    """训练脚本 HF 缓存目录（数据盘优先）。"""
    h = os.environ.get("HF_HOME", "").strip()
    if h:
        return Path(h)
    if AUTODL_DATA_ROOT.is_dir():
        return AUTODL_DATA_ROOT / ".cache" / "huggingface"
    return _REPO_ROOT / "models" / "huggingface"


def default_unixcoder_csn_output_dir() -> Path:
    """Java CSN 微调默认输出目录（勿与 Python 共用）。"""
    return default_eval_models_parent(None) / "unixcoder-csn-java"


def default_unixcoder_csn_python_output_dir() -> Path:
    """Python CSN 微调默认输出目录；与 Java 分离，避免覆盖 unixcoder-csn-java。"""
    return default_eval_models_parent(None) / "unixcoder-csn-python"


def default_unixcoder_csn_go_output_dir() -> Path:
    """Go（清洗版）CSN 微调默认输出目录；与 Java/Python 分离。"""
    return default_eval_models_parent(None) / "unixcoder-csn-go"


def default_unixcoder_csn_javascript_output_dir() -> Path:
    """JavaScript（清洗版）CSN 微调默认输出目录；与其它语言分离。"""
    return default_eval_models_parent(None) / "unixcoder-csn-javascript"


def default_unixcoder_csn_php_output_dir() -> Path:
    """PHP（清洗版）CSN 微调默认输出目录；与其它语言分离。"""
    return default_eval_models_parent(None) / "unixcoder-csn-php"


def default_unixcoder_csn_ruby_output_dir() -> Path:
    """Ruby（清洗版）CSN 微调默认输出目录；与其它语言分离。"""
    return default_eval_models_parent(None) / "unixcoder-csn-ruby"
