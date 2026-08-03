#!/usr/bin/env python3
"""将 AdvTest / CoSQA 外部基准转换为 CSN 评测目录格式（test.jsonl + codebase.jsonl）。

生成后配合 scripts/evaluate_code_search_non_java.py 使用：

  CSN_LANG_DIR=/root/autodl-fs/dataset_eval/advtest \
    python scripts/evaluate_code_search_non_java.py --language advtest --top-k 10 --workers 12

  CSN_LANG_DIR=/root/autodl-fs/dataset_eval/cosqa \
    python scripts/evaluate_code_search_non_java.py --language cosqa --top-k 10 --workers 12

字段约定（scripts/csn_data.py 的 iter_csn_jsonl 可读取）：
  - nl_query  ← docstring
  - code      ← original_string
  - url       ← url（AdvTest 为 GitHub URL；CoSQA 为 cosqa://<retrieval_idx> 合成 id）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

ADVTEST_SRC = Path("/root/autodl-fs/dataset/AdvTest")
COSQA_SRC = Path("/root/autodl-fs/dataset/cosqa")
DEFAULT_OUT = Path("/root/autodl-fs/dataset_eval")


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_advtest(src: Path, out_dir: Path) -> Dict[str, int]:
    """AdvTest: query=docstring, GT 通过 url 匹配 test_code.jsonl。"""
    test_rows: list[Dict[str, Any]] = []
    with open(src / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            nl = (js.get("docstring") or "").strip()
            url = (js.get("url") or "").strip()
            code = js.get("function") or js.get("code") or ""
            if not nl or not url:
                continue
            test_rows.append(
                {"url": url, "docstring": nl, "original_string": code, "language": "python"}
            )

    code_rows: list[Dict[str, Any]] = []
    with open(src / "test_code.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            url = (js.get("url") or "").strip()
            code = js.get("function") or ""
            if not url or not code.strip():
                continue
            code_rows.append(
                {
                    "url": url,
                    "original_string": code,
                    "docstring": js.get("docstring") or "",
                    "language": "python",
                }
            )

    _write_jsonl(out_dir / "test.jsonl", test_rows)
    _write_jsonl(out_dir / "codebase.jsonl", code_rows)

    # GT 覆盖校验
    code_urls = {r["url"] for r in code_rows}
    hit = sum(1 for r in test_rows if r["url"] in code_urls)
    return {"test": len(test_rows), "codebase": len(code_rows), "gt_hit": hit}


def prepare_cosqa(src: Path, out_dir: Path) -> Dict[str, int]:
    """CoSQA: query=doc（web query），GT 通过 retrieval_idx 对齐 code_idx_map。"""
    with open(src / "code_idx_map.txt", encoding="utf-8") as f:
        code_idx: Dict[str, int] = json.load(f)
    idx_code = {v: k for k, v in code_idx.items()}

    code_rows = [
        {
            "url": f"cosqa://{idx}",
            "original_string": idx_code[idx],
            "language": "python",
        }
        for idx in sorted(idx_code)
    ]
    _write_jsonl(out_dir / "codebase.jsonl", code_rows)

    with open(src / "cosqa-retrieval-test-500.json", encoding="utf-8") as f:
        test_data = json.load(f)
    test_rows: list[Dict[str, Any]] = []
    gt_hit = 0
    for d in test_data:
        nl = (d.get("doc") or "").strip()
        ri = d.get("retrieval_idx")
        if not nl or ri is None or ri not in idx_code:
            continue
        url = f"cosqa://{ri}"
        if (idx_code[ri] or "").strip()[:60] == (d.get("code") or "").strip()[:60]:
            gt_hit += 1
        test_rows.append(
            {
                "url": url,
                "docstring": nl,
                "original_string": d.get("code") or "",
                "language": "python",
            }
        )
    _write_jsonl(out_dir / "test.jsonl", test_rows)
    return {"test": len(test_rows), "codebase": len(code_rows), "gt_hit": gt_hit}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--advtest-dir", type=str, default=str(ADVTEST_SRC))
    ap.add_argument("--cosqa-dir", type=str, default=str(COSQA_SRC))
    ap.add_argument("--skip-advtest", action="store_true")
    ap.add_argument("--skip-cosqa", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    if not args.skip_advtest:
        stats = prepare_advtest(Path(args.advtest_dir), out_root / "advtest")
        print(
            f"[AdvTest] test={stats['test']} codebase={stats['codebase']} "
            f"gt_hit={stats['gt_hit']} ({stats['gt_hit']/max(stats['test'],1)*100:.2f}%)"
            f" -> {out_root/'advtest'}"
        )
    if not args.skip_cosqa:
        stats = prepare_cosqa(Path(args.cosqa_dir), out_root / "cosqa")
        print(
            f"[CoSQA]   test={stats['test']} codebase={stats['codebase']} "
            f"gt_hit={stats['gt_hit']} ({stats['gt_hit']/max(stats['test'],1)*100:.2f}%)"
            f" -> {out_root/'cosqa'}"
        )


if __name__ == "__main__":
    main()
