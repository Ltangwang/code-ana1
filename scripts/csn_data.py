import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from shared.csn_python_code_strip import strip_python_code_docstrings as _strip_py_docs

def iter_csn_jsonl(file_path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    Iterate over a CodeSearchNet JSONL file and yield parsed dictionaries.
    支持两种 JSON 字段风格：
    - HuggingFace code_search_net：func_documentation_string、func_code_string、func_code_url
    - GraphCodeBERT / 清洗导出：docstring 或 docstring_tokens、original_string、url
    - func_name: The name of the function.
    - repository_name: The source repository.
    - func_code_url: The URL to the source code.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CodeSearchNet dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # HuggingFace / 原始 CodeSearchNet
                code = item.get("func_code_string") or item.get("original_string") or ""
                nl = (
                    item.get("func_documentation_string")
                    or item.get("docstring")
                    or ""
                )
                if not nl and item.get("docstring_tokens"):
                    dt = item["docstring_tokens"]
                    nl = " ".join(dt) if isinstance(dt, list) else str(dt)
                url = item.get("func_code_url") or item.get("url") or ""
                yield {
                    "nl_query": nl,
                    "code": code,
                    "func_name": item.get("func_name", ""),
                    "repository_name": item.get("repository_name", ""),
                    "url": url,
                    "language": item.get("language", "java"),
                }
            except json.JSONDecodeError:
                continue

def load_csn_dataset(
    file_path: str | Path,
    max_samples: int = None,
    require_code: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load up to max_samples from a CodeSearchNet JSONL file.
    require_code=True：保留 NL+code（训练/部分 HF test）。
    require_code=False：保留 NL 且（url 或 code），适用于 GraphCodeBERT 清洗 test（仅查询+url）。
    """
    data = []
    for i, item in enumerate(iter_csn_jsonl(file_path)):
        if max_samples is not None and i >= max_samples:
            break
        nl = item["nl_query"]
        code = item["code"]
        url = item.get("url") or ""
        if require_code:
            ok = bool(nl and code)
        else:
            ok = bool(nl and (url or code))
        if ok:
            data.append(item)
    return data


def load_csn_code_corpus(
    file_path: str | Path,
    max_samples: int | None = None,
    *,
    strip_python_docstrings: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load code snippets for a retrieval index (e.g. codebase.jsonl).
    Only non-empty code is required; NL may be empty.
    strip_python_docstrings：对 Python 片段用 AST 去掉函数/类/模块首条 docstring，与训练侧一致。
    """
    data = []
    for i, item in enumerate(iter_csn_jsonl(file_path)):
        if max_samples is not None and i >= max_samples:
            break
        code = item.get("code") or ""
        if not code:
            continue
        row = dict(item)
        if strip_python_docstrings:
            row["code"] = _strip_py_docs(code)
        if row.get("code"):
            data.append(row)
    return data
