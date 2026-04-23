#!/usr/bin/env python3
"""
Check CodeSearchNet_clean_Dataset language dirs for completeness vs GraphCodeBERT-style
reference test sizes (from Microsoft GraphCodeBERT/codesearch public notes;
if your preprocess differs, follow the CodeBERT repo README).

Usage:
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

# Reference test set sizes after GraphCodeBERT filtering (example counts; jsonl nonempty lines may differ slightly)
# Ref: GraphCodeBERT code search public stats; java full eval often 10955.
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
        out["notes"].append("directory missing")
        return out

    for name in REQUIRED_FILES_EVAL:
        p = lang_dir / name
        if not p.is_file():
            out["ok"] = False
            out["notes"].append(f"missing required file: {name}")
        else:
            lines = _count_jsonl_lines(p)
            out[f"{name}_lines"] = lines
            if ref and name == "test.jsonl":
                diff = abs(lines - ref)
                tol = max(50, int(ref * 0.02))
                if diff > tol:
                    out["notes"].append(
                        f"test.jsonl lines {lines} vs ref {ref} (tolerance ±{tol}); "
                        "download may be incomplete, preprocess interrupted, or version differs."
                    )
                    out["ok"] = False
                else:
                    out["notes"].append(
                        f"test.jsonl lines {lines} ≈ ref {ref} (within tolerance)"
                    )

    for name in RECOMMENDED_FILES:
        if name in ("valid.jsonl", "validation.jsonl"):
            continue
        p = lang_dir / name
        out[f"has_{name}"] = p.is_file()
        if name == "codebase.jsonl" and not p.is_file():
            out["notes"].append("no codebase.jsonl: full-corpus retrieval eval falls back to test-only index")

    has_valid = (lang_dir / "valid.jsonl").is_file() or (
        lang_dir / "validation.jsonl"
    ).is_file()
    out["has_valid_split"] = has_valid
    if not has_valid:
        out["notes"].append("no valid.jsonl / validation.jsonl (optional for train/tuning)")

    marker = lang_dir / "LANGUAGE_INFO.json"
    out["has_LANGUAGE_INFO"] = marker.is_file()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GraphCodeBERT-cleaned CSN directory completeness"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Cleaned data root (default CSN_CLEAN_OUTPUT_DIR or default_csn_clean_dataset_root)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Check one language subdir only; default all go/javascript/php/python/ruby/java",
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

    print(f"Cleaned root: {root}\n")
    print(
        "Reference test sizes: GraphCodeBERT / CodeSearchNet filtered common stats; "
        "if you use a different preprocess commit, follow the official repo.\n"
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
    print("Check passed: no hard failures (still review codebase/valid hints).")


if __name__ == "__main__":
    main()
