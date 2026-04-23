#!/usr/bin/env python3
"""
Build GraphCodeBERT-filtered CodeSearchNet into CodeSearchNet_clean_Dataset/<lang>/.

By default **Java only**; `--all-languages` runs all six but still only copies java to output by default;
`--all-except-java` runs the **five non-Java** langs (go, javascript, php, python, ruby) and copies each to
`CodeSearchNet_clean_Dataset/{lang}/` with LANGUAGE_INFO.json.

Flow:
  1) Shallow-clone microsoft/CodeBERT into CodeSearchNet_clean_Dataset/_vendor/CodeBERT
  2) Unzip GraphCodeBERT/codesearch/dataset.zip
  3) Patch preprocess.py / run.sh as needed, then bash run.sh
  4) Copy dataset/<lang>/*.jsonl to output language folders
  5) Optional --remove-vendor to delete the clone and free space

If cloning fails, use --from-dataset-dir / --from-java-dir.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.csn_paths import default_csn_clean_dataset_root

CODEBERT_REPO = "https://github.com/microsoft/CodeBERT.git"

_NON_JAVA_LANG_DIRS = ("python", "php", "go", "ruby", "javascript", "js")

# Five non-Java langs from GraphCodeBERT preprocess output (dir names match common preprocess)
_LANGS_EXCEPT_JAVA = ("go", "javascript", "php", "python", "ruby")

_DISPLAY = {
    "go": "Go",
    "java": "Java",
    "javascript": "JavaScript",
    "php": "PHP",
    "python": "Python",
    "ruby": "Ruby",
}


def _copy_lang_jsonl(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = list(src_dir.glob("*.jsonl"))
    if not files:
        return 0
    for f in files:
        shutil.copy2(f, dst_dir / f.name)
    print(f"Copied {len(files)} file(s) -> {dst_dir}")
    return len(files)


def _write_clean_language_marker(dst_lang: Path, language_id: str) -> None:
    import json

    info = {
        "dataset": "CodeSearchNet_GraphCodeBERT_clean",
        "language_id": language_id,
        "display_name": _DISPLAY.get(language_id, language_id),
        "folder_name": language_id,
    }
    (dst_lang / "LANGUAGE_INFO.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _resolve_src_lang_dir(dataset_dir: Path, lang: str) -> Path | None:
    """Some versions use `js` instead of `javascript`."""
    p = dataset_dir / lang
    if p.is_dir():
        return p
    if lang == "javascript":
        alt = dataset_dir / "js"
        if alt.is_dir():
            return alt
    return None


def _copy_java_jsonl(src_java: Path, dst_java: Path) -> None:
    n = _copy_lang_jsonl(src_java, dst_java)
    if n == 0:
        raise FileNotFoundError(f"No jsonl under: {src_java}")


def _ensure_clone(vendor_codebert: Path) -> None:
    if (vendor_codebert / ".git").is_dir():
        print(f"Clone already exists: {vendor_codebert}")
        return
    vendor_codebert.parent.mkdir(parents=True, exist_ok=True)
    print(f"Shallow-cloning {CODEBERT_REPO} -> {vendor_codebert} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", CODEBERT_REPO, str(vendor_codebert)],
        check=True,
    )


def _unzip_dataset(codesearch: Path) -> Path:
    zpath = codesearch / "dataset.zip"
    if not zpath.is_file():
        raise FileNotFoundError(f"Missing {zpath} (incomplete clone?)")
    print(f"Unzipping {zpath} -> {codesearch} ...")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(codesearch)
    dataset_dir = codesearch / "dataset"
    if not (dataset_dir / "run.sh").is_file():
        raise FileNotFoundError(
            f"After unzip, missing {dataset_dir / 'run.sh'}; check dataset.zip layout."
        )
    return dataset_dir


def _patch_preprocess_java_only(preprocess_py: Path) -> None:
    """
    Shrink official preprocess.py language list to java only to avoid huge php/ruby/go dirs.
    First run backs up to preprocess.py.full_bak.
    """
    raw = preprocess_py.read_text(encoding="utf-8", errors="replace")
    backup = preprocess_py.with_name("preprocess.py.full_bak")
    if not backup.is_file():
        backup.write_text(raw, encoding="utf-8")

    lines = raw.splitlines(keepends=True)
    keys = (
        r"^\s*langs\s*=\s*\[",
        r"^\s*LANGS\s*=\s*\[",
        r"^\s*languages\s*=\s*\[",
    )
    out: list[str] = []
    done = False
    for line in lines:
        if not done and any(re.match(k, line) for k in keys):
            line = re.sub(r"\[[^\]]*\]", "['java']", line, count=1)
            done = True
        elif not done and re.match(r"^\s*for\s+lang\s+in\s*\[", line):
            line = re.sub(r"\[[^\]]*\]", "['java']", line, count=1)
            done = True
        out.append(line)

    new_text = "".join(out)
    if not done:
        # Single-line list, e.g. foo = ['python','java', ...]
        m = re.search(
            r"\[\'(?:python|java|javascript|go|php|ruby)\'"
            r"(?:\s*,\s*\'(?:python|java|javascript|go|php|ruby)\')*\s*\]",
            new_text,
        )
        if m:
            new_text = (
                new_text[: m.start()] + "['java']" + new_text[m.end() :]
            )
            done = True

    if not done:
        raise RuntimeError(
            f"Could not auto-detect language list in {preprocess_py}. "
            "Edit the file to java only, or use --all-languages."
        )

    preprocess_py.write_text(new_text, encoding="utf-8")
    print("Limited preprocess.py to Java only (backup: preprocess.py.full_bak)")


def _patch_preprocess_lang_list(preprocess_py: Path, langs: tuple[str, ...]) -> None:
    """Set first language list in preprocess to the given tuple (same backup as java-only)."""
    raw = preprocess_py.read_text(encoding="utf-8", errors="replace")
    backup = preprocess_py.with_name("preprocess.py.full_bak")
    if not backup.is_file():
        backup.write_text(raw, encoding="utf-8")

    list_literal = "[" + ", ".join(repr(x) for x in langs) + "]"
    lines = raw.splitlines(keepends=True)
    keys = (
        r"^\s*langs\s*=\s*\[",
        r"^\s*LANGS\s*=\s*\[",
        r"^\s*languages\s*=\s*\[",
    )
    out: list[str] = []
    done = False
    for line in lines:
        if not done and any(re.match(k, line) for k in keys):
            line = re.sub(r"\[[^\]]*\]", list_literal, line, count=1)
            done = True
        elif not done and re.match(r"^\s*for\s+lang\s+in\s*\[", line):
            line = re.sub(r"\[[^\]]*\]", list_literal, line, count=1)
            done = True
        out.append(line)

    new_text = "".join(out)
    if not done:
        m = re.search(
            r"\[\'(?:python|java|javascript|go|php|ruby)\'"
            r"(?:\s*,\s*\'(?:python|java|javascript|go|php|ruby)\')*\s*\]",
            new_text,
        )
        if m:
            new_text = new_text[: m.start()] + list_literal + new_text[m.end() :]
            done = True

    if not done:
        raise RuntimeError(
            f"Could not auto-detect language list in {preprocess_py}, set manually to {list_literal}"
        )

    preprocess_py.write_text(new_text, encoding="utf-8")
    print(f"Set preprocess.py languages to {list_literal} (backup: preprocess.py.full_bak)")


def _patch_run_sh_exclude_java(run_sh: Path) -> None:
    """Remove Java corpus zip download lines; keep go/python/php/ruby/javascript."""
    raw = run_sh.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"wget|curl", raw, re.I):
        return
    bak = run_sh.with_name("run.sh.full_bak")
    if not bak.is_file():
        bak.write_text(raw, encoding="utf-8")
    out_lines: list[str] = []
    for line in raw.splitlines(keepends=True):
        low = line.lower()
        if re.search(r"wget|curl", low) and ".zip" in low:
            # \bjava\b avoids javascript.zip; keep javascript lines
            if "javascript" not in low and re.search(r"\bjava\b", low):
                continue
        out_lines.append(line)
    run_sh.write_text("".join(out_lines), encoding="utf-8")
    print("Filtered Java .zip download lines in run.sh (backup: run.sh.full_bak)")


def _prune_java_under_dataset(dataset_dir: Path) -> None:
    for name in ("java",):
        p = dataset_dir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"Removed Java work dir: {p}")


def _prune_non_java_under_dataset(dataset_dir: Path) -> None:
    """Remove non-java language work dirs under dataset (leftover from failed runs)."""
    for name in _NON_JAVA_LANG_DIRS:
        p = dataset_dir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"Removed non-Java dir: {p}")


def _patch_run_sh_java_only(run_sh: Path) -> None:
    """
    Old run.sh may wget/curl per-lang zips; drop python/php/go/ruby/javascript download lines.
    No-op if run.sh has no download lines.
    """
    raw = run_sh.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"wget|curl", raw, re.I):
        return
    bak = run_sh.with_name("run.sh.full_bak")
    if not bak.is_file():
        bak.write_text(raw, encoding="utf-8")
    out_lines: list[str] = []
    for line in raw.splitlines(keepends=True):
        low = line.lower()
        if re.search(r"wget|curl", low) and ".zip" in low:
            if re.search(r"(?:javascript|python|php|go|ruby)\.zip", low):
                continue
        out_lines.append(line)
    run_sh.write_text("".join(out_lines), encoding="utf-8")
    print("Filtered non-Java .zip download lines in run.sh (backup: run.sh.full_bak)")


def _run_preprocess(dataset_dir: Path, codesearch: Path) -> None:
    print(f"Running preprocess (cwd={dataset_dir}), may take a while …")
    env = os.environ.copy()
    # So dataset scripts can import parser from parent codesearch
    env["PYTHONPATH"] = str(codesearch) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        ["bash", "run.sh"],
        cwd=str(dataset_dir),
        check=True,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare GraphCodeBERT-cleaned CodeSearchNet"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Cleaned data root (default CSN_CLEAN_OUTPUT_DIR or repo/CodeSearchNet_clean_Dataset)",
    )
    parser.add_argument(
        "--from-java-dir",
        type=str,
        default=None,
        help="Existing java dir (test.jsonl, codebase.jsonl, …); only copy to output",
    )
    parser.add_argument(
        "--from-dataset-dir",
        type=str,
        default=None,
        help="Post-run.sh dataset dir (should contain java/*.jsonl)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="After unzip, do not run run.sh (use when preprocess already done under dataset/java)",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Preprocess all six (disk heavy); default flow still only copies java to output",
    )
    parser.add_argument(
        "--all-except-java",
        action="store_true",
        help="Preprocess five non-Java langs and copy to go/javascript/php/python/ruby under output root",
    )
    parser.add_argument(
        "--remove-vendor",
        action="store_true",
        help="After success, remove _vendor/CodeBERT clone to save space",
    )
    args = parser.parse_args()

    if args.all_languages and args.all_except_java:
        parser.error("--all-languages and --all-except-java are mutually exclusive")
    if args.from_java_dir and args.all_except_java:
        parser.error("--from-java-dir is for Java only; do not use with --all-except-java")

    clean_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_csn_clean_dataset_root()
    )
    dst_java = clean_root / "java"

    if args.from_java_dir:
        src = Path(args.from_java_dir).expanduser().resolve()
        _copy_java_jsonl(src, dst_java)
        print("Done.")
        return

    if args.from_dataset_dir:
        base = Path(args.from_dataset_dir).expanduser().resolve()
        if args.all_except_java:
            for lid in _LANGS_EXCEPT_JAVA:
                src = _resolve_src_lang_dir(base, lid)
                if src is None:
                    print(f"Warning: {base}/{lid} not found, skip")
                    continue
                dst = clean_root / lid
                _copy_lang_jsonl(src, dst)
                _write_clean_language_marker(dst, lid)
            print("Done (--all-except-java + --from-dataset-dir).")
        else:
            src_java = base / "java"
            _copy_java_jsonl(src_java, dst_java)
            print("Done.")
        return

    vendor = clean_root / "_vendor" / "CodeBERT"
    codesearch = vendor / "GraphCodeBERT" / "codesearch"

    _ensure_clone(vendor)
    dataset_dir = _unzip_dataset(codesearch)

    if not args.skip_run:
        if args.all_except_java:
            rs = dataset_dir / "run.sh"
            if rs.is_file():
                _patch_run_sh_exclude_java(rs)
            pp = dataset_dir / "preprocess.py"
            if not pp.is_file():
                raise FileNotFoundError(f"Missing {pp}.")
            _prune_java_under_dataset(dataset_dir)
            _patch_preprocess_lang_list(pp, _LANGS_EXCEPT_JAVA)
            print("--all-except-java: will preprocess go/javascript/php/python/ruby.")
        elif not args.all_languages:
            rs = dataset_dir / "run.sh"
            if rs.is_file():
                _patch_run_sh_java_only(rs)
            pp = dataset_dir / "preprocess.py"
            if not pp.is_file():
                raise FileNotFoundError(f"Missing {pp}, cannot apply Java-only patch.")
            _prune_non_java_under_dataset(dataset_dir)
            _patch_preprocess_java_only(pp)
        else:
            print("--all-languages: will process all languages (disk heavy).")
        _run_preprocess(dataset_dir, codesearch)
    else:
        print("Skipped run.sh (--skip-run)")

    if args.all_except_java:
        any_ok = False
        for lid in _LANGS_EXCEPT_JAVA:
            src = _resolve_src_lang_dir(dataset_dir, lid)
            if src is None:
                print(f"Warning: preprocess output dir not found {dataset_dir}/{lid}, skip")
                continue
            dst = clean_root / lid
            if _copy_lang_jsonl(src, dst) > 0:
                any_ok = True
                _write_clean_language_marker(dst, lid)
            for name in ("test.jsonl", "codebase.jsonl"):
                if not (dst / name).is_file():
                    print(f"Warning: {lid} missing {name}")
        if not any_ok:
            raise FileNotFoundError(
                f"No jsonl copied; check {dataset_dir} for generated "
                f"{_LANGS_EXCEPT_JAVA}"
            )
        print(f"Done. Cleaned (non-Java) output root: {clean_root}")
    else:
        src_java = dataset_dir / "java"
        if not src_java.is_dir():
            raise FileNotFoundError(f"Missing {src_java}; check run.sh success.")
        _copy_java_jsonl(src_java, dst_java)

        for name in ("test.jsonl", "codebase.jsonl"):
            if not (dst_java / name).is_file():
                print(f"Warning: missing {name}; full-corpus eval may be unavailable.")

        print(f"Done. For eval use: {dst_java}")

    if args.remove_vendor and vendor.is_dir():
        shutil.rmtree(vendor, ignore_errors=True)
        print(f"Removed clone to free space: {vendor}")


if __name__ == "__main__":
    main()
