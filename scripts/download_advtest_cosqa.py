#!/usr/bin/env python3
"""Download official AdvTest + CoSQA retrieval splits into /root/autodl-fs/dataset.

Sources (UniXcoder / CodeXGLUE):
  AdvTest: CodeXGLUE NL-code-search-Adv + HF google/code_x_glue_tc_nl_code_search_adv
  CoSQA:   Jun-jie-Huang/CoCLR data/search (retrieval JSON + code_idx_map)

Usage:
  python scripts/download_advtest_cosqa.py
  python scripts/download_advtest_cosqa.py --out-root /root/autodl-fs/dataset
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

CODEXGLUE_ADV_ZIP = (
    "https://github.com/microsoft/CodeXGLUE/raw/main/"
    "Text-Code/NL-code-search-Adv/dataset.zip"
)
COSQA_BASE = "https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search"
COSQA_FILES = [
    "code_idx_map.txt",
    "cosqa-retrieval-dev-500.json",
    "cosqa-retrieval-test-500.json",
    "cosqa-retrieval-train-19604.json",
]


def _wget(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    try:
        subprocess.check_call(
            ["wget", "-c", "--no-check-certificate", "-O", str(dest), url]
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        urlretrieve(url, dest)


def download_cosqa(out_root: Path) -> Path:
    d = out_root / "cosqa"
    d.mkdir(parents=True, exist_ok=True)
    for name in COSQA_FILES:
        _wget(f"{COSQA_BASE}/{name}", d / name)
    (d / "README.md").write_text(
        "# CoSQA retrieval split (UniXcoder)\n\n"
        "From https://github.com/Jun-jie-Huang/CoCLR\n"
        "Eval: test=cosqa-retrieval-test-500.json, codebase=code_idx_map.txt\n",
        encoding="utf-8",
    )
    return d


def download_advtest(out_root: Path) -> Path:
    """Prefer HF parquet export (complete code + Adv anonymization); keep CodeXGLUE zip meta."""
    from datasets import load_dataset

    d = out_root / "AdvTest"
    d.mkdir(parents=True, exist_ok=True)

    zip_path = d / "dataset.zip"
    if not (d / "preprocess.py").is_file():
        _wget(CODEXGLUE_ADV_ZIP, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(d)
        # zip contains top-level "dataset/"
        nested = d / "dataset"
        if nested.is_dir():
            for p in nested.iterdir():
                target = d / p.name
                if not target.exists():
                    p.rename(target)
        zip_path.unlink(missing_ok=True)

    print("Loading google/code_x_glue_tc_nl_code_search_adv ...")
    ds = load_dataset("google/code_x_glue_tc_nl_code_search_adv")

    def write_split(split: str, name: str) -> int:
        path = d / name
        n = 0
        with path.open("w", encoding="utf-8") as f:
            for i, row in enumerate(ds[split]):
                obj = dict(row)
                obj["idx"] = i
                if "function" not in obj and "code" in obj:
                    obj["function"] = obj["code"]
                f.write(json.dumps(obj) + "\n")
                n += 1
        print(f"  wrote {name}: {n}")
        return n

    write_split("train", "train.jsonl")
    write_split("validation", "valid.jsonl")
    write_split("test", "test.jsonl")

    (d / "README.md").write_text(
        "# AdvTest (CodeXGLUE NL-code-search-Adv)\n\n"
        "HF: google/code_x_glue_tc_nl_code_search_adv\n"
        "CodeXGLUE: Text-Code/NL-code-search-Adv\n"
        "Eval (UniXcoder): test.jsonl as queries and codebase.\n",
        encoding="utf-8",
    )
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=str,
        default="/root/autodl-fs/dataset",
        help="Parent directory for AdvTest/ and cosqa/",
    )
    ap.add_argument("--skip-advtest", action="store_true")
    ap.add_argument("--skip-cosqa", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_cosqa:
        print("=== CoSQA ===")
        print(download_cosqa(out))
    if not args.skip_advtest:
        print("=== AdvTest ===")
        print(download_advtest(out))
    print("Done.")


if __name__ == "__main__":
    main()
