#!/usr/bin/env python3
"""
检查 CodeSearchNet_clean_Dataset 下各语言目录是否齐全，并与 GraphCodeBERT 过滤版
「官方规模」参考 test 条数对比（来源：Microsoft GraphCodeBERT/codesearch 公开说明与常用引用；
若你本地预处理版本不同，以 CodeBERT 仓库 README 为准）。

用法：
  python scripts/check_csn_clean_completeness.py
  python scripts/check_csn_clean_completeness.py --root /root/autodl-fs/CodeSearchNet_clean_Dataset
  python scripts/check_csn_clean_completeness.py --lang python
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import default_csn_clean_dataset_root

# GraphCodeBERT 过滤后 test 集规模（示例数，非 jsonl 行数若有空行会略差）
# 参考：GraphCodeBERT Code Search 任务公开统计 / 社区复现表；java 与全量评测常见 10955 一致。
REFERENCE_TEST_EXAMPLES: dict[str, int] = {
    "go": 8122,
    "java": 10955,
    "javascript": 3291,
    "php": 14014,
    "python": 14918,
    "ruby": 1261,
}

REQUIRED_FILES_EVAL = ("test.jsonl",)
RECOMMENDED_FILES = ("codebase.jsonl", "train.jsonl", "valid.jsonl", "validation.jsonl")


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _check_lang(lang_dir: Path, lang: str) -> dict:
    ref = REFERENCE_TEST_EXAMPLES.get(lang)
    out: dict = {"language": lang, "path": str(lang_dir), "ok": True, "notes": []}
    if not lang_dir.is_dir():
        out["ok"] = False
        out["notes"].append("目录不存在")
        return out

    for name in REQUIRED_FILES_EVAL:
        p = lang_dir / name
        if not p.is_file():
            out["ok"] = False
            out["notes"].append(f"缺少必需文件: {name}")
        else:
            lines = _count_jsonl_lines(p)
            out[f"{name}_lines"] = lines
            if ref and name == "test.jsonl":
                diff = abs(lines - ref)
                tol = max(50, int(ref * 0.02))
                if diff > tol:
                    out["notes"].append(
                        f"test.jsonl 行数 {lines} 与参考值 {ref} 差异较大（容差±{tol}）；"
                        "可能未下全、预处理中断或版本不同。"
                    )
                    out["ok"] = False
                else:
                    out["notes"].append(
                        f"test.jsonl 行数 {lines} ≈ 参考 {ref}（在容差内）"
                    )

    for name in RECOMMENDED_FILES:
        if name in ("valid.jsonl", "validation.jsonl"):
            continue
        p = lang_dir / name
        out[f"has_{name}"] = p.is_file()
        if name == "codebase.jsonl" and not p.is_file():
            out["notes"].append("无 codebase.jsonl：全库检索评测将退化为仅用 test 建索引")

    has_valid = (lang_dir / "valid.jsonl").is_file() or (
        lang_dir / "validation.jsonl"
    ).is_file()
    out["has_valid_split"] = has_valid
    if not has_valid:
        out["notes"].append("无 valid.jsonl / validation.jsonl（训练/调参可选）")

    marker = lang_dir / "LANGUAGE_INFO.json"
    out["has_LANGUAGE_INFO"] = marker.is_file()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 GraphCodeBERT 清洗版 CSN 目录完整性")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="清洗数据根（默认 CSN_CLEAN_OUTPUT_DIR 或 default_csn_clean_dataset_root）",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="只检查一种语言子目录；默认检查 go/javascript/php/python/ruby/java",
    )
    args = parser.parse_args()
    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else default_csn_clean_dataset_root()
    )

    if args.lang:
        langs = [args.lang.strip().lower()]
    else:
        langs = ["go", "javascript", "php", "python", "ruby", "java"]

    print(f"清洗根目录: {root}\n")
    print(
        "参考 test 条数来源：GraphCodeBERT/CodeSearchNet 过滤版常用统计；"
        "若你使用不同 commit 的 preprocess，请以官方仓库为准。\n"
    )

    any_bad = False
    for lang in langs:
        report = _check_lang(root / lang, lang)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report.get("ok", False):
            any_bad = True
        print()

    if any_bad:
        sys.exit(1)
    print("检查完成：未发现硬性缺失（请仍留意 codebase 与 valid 的提示）。")


if __name__ == "__main__":
    main()
