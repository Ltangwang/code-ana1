#!/usr/bin/env python3
"""
Remove non-Java prep artifacts for CodeSearchNet / GraphCodeBERT to free disk space.
**Does not** delete `java/` or its jsonl.

Two default roots (override via env, same as csn_paths):

1) Cleaned root (CSN_CLEAN_OUTPUT_DIR or repo/CodeSearchNet_clean_Dataset/)
   - Top-level non-Java language dirs: python, php, go, ruby, javascript, js
   - If `dataset/` exists, non-Java subdirs under it
   - Under _vendor/CodeBERT/GraphCodeBERT/codesearch/dataset/ non-Java lang dirs (e.g. php/final)
   - Obvious non-Java corpus .zip under codesearch (excluding java.zip)

2) Optional HF export (--include-hf-export)
   - Non-Java subdirs under CodeSearchNet_Dataset/ (or CSN_OUTPUT_DIR)

Usage:
  python scripts/cleanup_csn_prep_residuals.py --dry-run   # list what would be removed
  python scripts/cleanup_csn_prep_residuals.py               # delete
  python scripts/cleanup_csn_prep_residuals.py --include-hf-export
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import default_csn_clean_dataset_root, default_csn_dataset_root

_NON_JAVA = frozenset({"python", "php", "go", "ruby", "javascript", "js"})

# Common corpus zip names under codesearch (excluding java)
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
        description="Clean non-Java CSN prep caches (keeps java/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List paths only, do not delete",
    )
    parser.add_argument(
        "--clean-root",
        type=str,
        default=None,
        help="Cleaned data root (default CSN_CLEAN_OUTPUT_DIR or repo/CodeSearchNet_clean_Dataset)",
    )
    parser.add_argument(
        "--include-hf-export",
        action="store_true",
        help="Also remove non-Java dirs under CodeSearchNet_Dataset/ (or CSN_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--remove-vendor",
        action="store_true",
        help="Also remove entire _vendor/ (CodeBERT clone; java under java/ is unchanged)",
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
            f"Nothing to clean (non-Java).\n"
            f"  clean_root: {clean_root}\n"
            f"  try --include-hf-export or check for _vendor/.../dataset/php etc."
        )
        return

    print(f"clean_root: {clean_root}")
    print(f"Will process {len(targets)} path(s) (java/ untouched):\n")
    for p in targets:
        print(f"{'[dry-run] ' if args.dry_run else ''}remove: {p}")
        if not args.dry_run:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                try:
                    p.unlink()
                except OSError as e:
                    print(f"  warning: could not delete file {p}: {e}")

    if not args.dry_run:
        print("\nCleanup done. java/ was not removed.")


if __name__ == "__main__":
    main()
