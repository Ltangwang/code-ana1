#!/usr/bin/env python3
"""
Compare local GraphCodeBERT-style cleaned data line counts to the official GitHub tables.

Reference (still in microsoft/CodeBERT as of 2024):
  https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/README.md
  Dev column usually maps to valid.jsonl (or validation.jsonl).

Usage:
  python scripts/check_csn_clean_vs_graphcodebert_official.py
  python scripts/check_csn_clean_vs_graphcodebert_official.py --root /path/to/CodeSearchNet_clean_Dataset
  python scripts/check_csn_clean_vs_graphcodebert_official.py --validate-json
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

# Matches official README "Data statistic about the cleaned dataset" (Training / Dev / Test / Candidates code)
_OFFICIAL: dict[str, dict[str, int]] = {
    "python": {
        "train": 251_820,
        "dev": 13_914,
        "test": 14_918,
        "codebase": 43_827,
    },
    "php": {
        "train": 241_241,
        "dev": 12_982,
        "test": 14_014,
        "codebase": 52_660,
    },
    "go": {
        "train": 167_288,
        "dev": 7_325,
        "test": 8_122,
        "codebase": 28_120,
    },
    "java": {
        "train": 164_923,
        "dev": 5_183,
        "test": 10_955,
        "codebase": 40_347,
    },
    "javascript": {
        "train": 58_025,
        "dev": 3_885,
        "test": 3_291,
        "codebase": 13_981,
    },
    "ruby": {
        "train": 24_927,
        "dev": 1_400,
        "test": 1_261,
        "codebase": 4_360,
    },
}

_LANG_ORDER = ("python", "php", "go", "java", "javascript", "ruby")


def _count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return -1
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _dev_path(lang_dir: Path) -> Path | None:
    v = lang_dir / "valid.jsonl"
    if v.is_file():
        return v
    v2 = lang_dir / "validation.jsonl"
    if v2.is_file():
        return v2
    return None


def _validate_jsonl(path: Path, max_lines: int | None = None) -> tuple[int, int]:
    """Returns (json_error_lines, scanned_nonempty_lines)."""
    bad = 0
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad += 1
            if max_lines is not None and n >= max_lines:
                break
    return bad, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check cleaned data vs microsoft/CodeBERT GraphCodeBERT/codesearch README sizes"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Cleaned data root (default CSN_CLEAN_OUTPUT_DIR or default_csn_clean_dataset_root)",
    )
    parser.add_argument(
        "--validate-json",
        action="store_true",
        help="json.loads every jsonl line (slow on large files; finds bad lines)",
    )
    args = parser.parse_args()
    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else default_csn_clean_dataset_root()
    )

    print(
        "Reference: microsoft/CodeBERT GraphCodeBERT/codesearch/README.md "
        "(cleaned dataset stats table)"
    )
    print(f"Local root: {root}\n")

    any_fail = False
    rows: list[tuple[str, str, str, str, str]] = []

    for lang in _LANG_ORDER:
        ref = _OFFICIAL[lang]
        d = root / lang
        if not d.is_dir():
            print(f"[{lang}] directory missing: {d}")
            any_fail = True
            rows.append((lang, "MISSING", "-", "-", "-"))
            continue

        train_n = _count_nonempty_lines(d / "train.jsonl")
        dev_p = _dev_path(d)
        dev_n = _count_nonempty_lines(dev_p) if dev_p else -1
        test_n = _count_nonempty_lines(d / "test.jsonl")
        cb_n = _count_nonempty_lines(d / "codebase.jsonl")

        def _cmp(name: str, got: int, want: int) -> str:
            nonlocal any_fail
            if got < 0:
                any_fail = True
                return f"{name}: FILE_MISSING"
            if got != want:
                any_fail = True
                return f"{name}: {got} != official {want} (diff {got - want:+d})"
            return f"{name}: {got} OK"

        parts = [
            _cmp("train", train_n, ref["train"]),
            _cmp("dev", dev_n, ref["dev"]),
            _cmp("test", test_n, ref["test"]),
            _cmp("codebase", cb_n, ref["codebase"]),
        ]
        ok_counts = (
            train_n == ref["train"]
            and dev_n == ref["dev"]
            and test_n == ref["test"]
            and cb_n == ref["codebase"]
            and train_n >= 0
            and dev_n >= 0
            and test_n >= 0
            and cb_n >= 0
        )
        status = "OK" if ok_counts else "FAIL"
        print(f"=== {lang} ({status}) ===")
        for p in parts:
            print(f"  {p}")
        if args.validate_json:
            for fname in ("train.jsonl", "valid.jsonl", "validation.jsonl", "test.jsonl", "codebase.jsonl"):
                fp = d / fname
                if not fp.is_file():
                    continue
                bad, scanned = _validate_jsonl(fp, max_lines=None)
                tag = "full" if scanned > 0 else "empty"
                if bad:
                    print(f"  JSON: {fname} errors={bad} (scanned {scanned} nonempty lines, {tag})")
                    any_fail = True
                else:
                    print(f"  JSON: {fname} OK ({scanned} nonempty lines)")
        print()
        rows.append(
            (
                lang,
                str(train_n),
                str(dev_n),
                str(test_n),
                str(cb_n),
            )
        )

    print("---- Summary (nonempty lines; dev=valid.jsonl or validation.jsonl) ----")
    print(f"{'lang':12} {'train':>10} {'dev':>10} {'test':>10} {'codebase':>10}")
    for lang, tr, de, te, cb in rows:
        print(f"{lang:12} {tr:>10} {de:>10} {te:>10} {cb:>10}")

    if any_fail:
        print(
            "\nConclusion: mismatch or missing files vs the official table; "
            "check preprocess version / download completeness."
        )
        sys.exit(1)
    print(
        "\nConclusion: train/dev/test/codebase nonempty line counts match the GitHub README table."
    )
    if not args.validate_json:
        print("(Did not run --validate-json; rerun with that flag for per-line JSON validation.)")


if __name__ == "__main__":
    main()
