#!/usr/bin/env python3
"""
清理 CodeSearchNet / GraphCodeBERT 准备阶段产生的「非 Java」缓存与中间目录，释放磁盘。
**不删除** `java/` 及其中的 jsonl。

默认处理两个根（均可通过环境变量改路径，与 csn_paths 一致）：

1) 清洗根（CSN_CLEAN_OUTPUT_DIR 或 仓库/CodeSearchNet_clean_Dataset/）
   - 顶层非 Java 语言目录：python, php, go, ruby, javascript, js
   - 若存在 dataset/，其下非 Java 子目录
   - _vendor/CodeBERT/GraphCodeBERT/codesearch/dataset/ 下非 Java 语言目录（php/final 等大文件常在此）
   - codesearch 目录内明显的非 Java 语料 .zip（不含 java.zip）

2) 可选：HF 导出根（--include-hf-export）
   - CodeSearchNet_Dataset/（或 CSN_OUTPUT_DIR）下非 Java 语言子目录

用法：
  python scripts/cleanup_csn_prep_residuals.py --dry-run   # 先看将删什么
  python scripts/cleanup_csn_prep_residuals.py               # 执行删除
  python scripts/cleanup_csn_prep_residuals.py --include-hf-export
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import default_csn_clean_dataset_root, default_csn_dataset_root

_NON_JAVA = frozenset({"python", "php", "go", "ruby", "javascript", "js"})

# codesearch 目录下常见语料包文件名（不含 java）
_NON_JAVA_ZIP = re.compile(
    r"(?:^|[-_/])(?:python|php|go|ruby|javascript)\.zip$",
    re.IGNORECASE,
)


def _is_non_java_lang_dir(name: str) -> bool:
    return name.lower() in _NON_JAVA


def _collect_clean_root_targets(clean_root: Path, remove_vendor: bool) -> list[Path]:
    found: list[Path] = []

    if not clean_root.is_dir():
        return found

    for child in clean_root.iterdir():
        if child.is_dir() and _is_non_java_lang_dir(child.name):
            found.append(child)

    staging = clean_root / "dataset"
    if staging.is_dir():
        for child in staging.iterdir():
            if child.is_dir() and _is_non_java_lang_dir(child.name):
                found.append(child)

    vendor_top = clean_root / "_vendor"
    if vendor_top.is_dir():
        if remove_vendor:
            found.append(vendor_top)
        else:
            vendor_dataset = (
                vendor_top
                / "CodeBERT"
                / "GraphCodeBERT"
                / "codesearch"
                / "dataset"
            )
            if vendor_dataset.is_dir():
                for child in vendor_dataset.iterdir():
                    if child.is_dir() and _is_non_java_lang_dir(child.name):
                        found.append(child)

            codesearch = vendor_top / "CodeBERT" / "GraphCodeBERT" / "codesearch"
            if codesearch.is_dir():
                for f in codesearch.iterdir():
                    if f.is_file() and f.suffix.lower() == ".zip":
                        if _NON_JAVA_ZIP.search(f.name):
                            found.append(f)

    return found


def _collect_hf_export_targets(hf_root: Path) -> list[Path]:
    found: list[Path] = []
    if not hf_root.is_dir():
        return found
    for child in hf_root.iterdir():
        if child.is_dir() and _is_non_java_lang_dir(child.name):
            found.append(child)
    return found


def _unique_paths(paths: list[Path]) -> list[Path]:
    by_key: dict[Path, Path] = {}
    for p in paths:
        try:
            k = p.resolve()
        except OSError:
            k = p
        by_key[k] = p
    return list(by_key.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="清理 CSN 准备阶段非 Java 缓存（保留 java/）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅列出将删除的路径，不执行删除",
    )
    parser.add_argument(
        "--clean-root",
        type=str,
        default=None,
        help="清洗数据根（默认 CSN_CLEAN_OUTPUT_DIR 或 仓库/CodeSearchNet_clean_Dataset）",
    )
    parser.add_argument(
        "--include-hf-export",
        action="store_true",
        help="同时删除 CodeSearchNet_Dataset/（或 CSN_OUTPUT_DIR）下非 Java 语言目录",
    )
    parser.add_argument(
        "--remove-vendor",
        action="store_true",
        help="额外删除整个 _vendor/（含 CodeBERT 克隆；java 成品若在 java/ 则不受影响）",
    )
    args = parser.parse_args()

    clean_root = (
        Path(args.clean_root).expanduser().resolve()
        if args.clean_root
        else default_csn_clean_dataset_root()
    )

    targets = _collect_clean_root_targets(clean_root, args.remove_vendor)

    if args.include_hf_export:
        targets.extend(_collect_hf_export_targets(default_csn_dataset_root()))

    targets = _unique_paths(targets)

    if not targets:
        print(
            f"未发现可清理的非 Java 项。\n"
            f"  清洗根: {clean_root}\n"
            f"  可加 --include-hf-export 或确认是否已有 _vendor/…/dataset/php 等目录。"
        )
        return

    print(f"清洗根: {clean_root}")
    print(f"将处理 {len(targets)} 个路径（不触碰 java/）：\n")
    for p in targets:
        print(f"{'[dry-run] ' if args.dry_run else ''}删除: {p}")
        if not args.dry_run:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                try:
                    p.unlink()
                except OSError as e:
                    print(f"  警告: 无法删除文件 {p}: {e}")

    if not args.dry_run:
        print("\n清理完成。java/ 目录未删除。")


if __name__ == "__main__":
    main()
