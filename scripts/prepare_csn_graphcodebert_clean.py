#!/usr/bin/env python3
"""
生成 GraphCodeBERT 过滤版 CodeSearchNet 到 CodeSearchNet_clean_Dataset/<语言>/。

默认 **仅处理 Java**；`--all-languages` 跑六语预处理但默认仍只复制 java 到输出；
`--all-except-java` 跑 **除 Java 外五语**（go, javascript, php, python, ruby），并分别复制到
`CodeSearchNet_clean_Dataset/{语言}/`，各目录含 LANGUAGE_INFO.json。

流程：
  1) 浅克隆 microsoft/CodeBERT 到 CodeSearchNet_clean_Dataset/_vendor/CodeBERT
  2) 解压 GraphCodeBERT/codesearch/dataset.zip
  3) 按模式补丁 preprocess.py / run.sh，再 bash run.sh
  4) 将 dataset/<lang>/*.jsonl 复制到输出根下对应语言文件夹
  5) 可选 --remove-vendor 删除克隆以释放空间

若网络无法克隆，可用 --from-dataset-dir / --from-java-dir。
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

# GraphCodeBERT 预处理输出的「除 Java 外」五语（目录名与常见 preprocess 一致）
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
    print(f"已复制 {len(files)} 个文件 -> {dst_dir}")
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
    """javascript 在部分版本里目录名为 js。"""
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
        raise FileNotFoundError(f"未找到 jsonl: {src_java}")


def _ensure_clone(vendor_codebert: Path) -> None:
    if (vendor_codebert / ".git").is_dir():
        print(f"已存在克隆: {vendor_codebert}")
        return
    vendor_codebert.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在浅克隆 {CODEBERT_REPO} -> {vendor_codebert} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", CODEBERT_REPO, str(vendor_codebert)],
        check=True,
    )


def _unzip_dataset(codesearch: Path) -> Path:
    zpath = codesearch / "dataset.zip"
    if not zpath.is_file():
        raise FileNotFoundError(f"缺少 {zpath}（克隆不完整？）")
    print(f"解压 {zpath} -> {codesearch} ...")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(codesearch)
    dataset_dir = codesearch / "dataset"
    if not (dataset_dir / "run.sh").is_file():
        raise FileNotFoundError(
            f"解压后未找到 {dataset_dir / 'run.sh'}，请检查 dataset.zip 结构。"
        )
    return dataset_dir


def _patch_preprocess_java_only(preprocess_py: Path) -> None:
    """
    将官方 preprocess.py 中「语言列表」缩成仅 java，避免生成 php/ruby/go 等目录占满磁盘。
    首次会备份为 preprocess.py.full_bak。
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
        # 单行列表，如 foo = ['python','java', ...]
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
            f"无法在 {preprocess_py} 中自动识别语言列表。"
            "请打开该文件手动改为仅 java，或使用 --all-languages。"
        )

    preprocess_py.write_text(new_text, encoding="utf-8")
    print("已限制 preprocess.py 为仅 Java（备份: preprocess.py.full_bak）")


def _patch_preprocess_lang_list(preprocess_py: Path, langs: tuple[str, ...]) -> None:
    """将 preprocess 中第一个语言列表改为给定元组（备份逻辑与 java-only 相同）。"""
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
            f"无法在 {preprocess_py} 中自动识别语言列表，请手动改为 {list_literal}"
        )

    preprocess_py.write_text(new_text, encoding="utf-8")
    print(f"已设置 preprocess.py 语言为 {list_literal}（备份: preprocess.py.full_bak）")


def _patch_run_sh_exclude_java(run_sh: Path) -> None:
    """去掉下载 Java 语料 zip 的行，保留 go/python/php/ruby/javascript 等。"""
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
            # 用 \bjava\b 避免误伤 javascript.zip；javascript 行保留
            if "javascript" not in low and re.search(r"\bjava\b", low):
                continue
        out_lines.append(line)
    run_sh.write_text("".join(out_lines), encoding="utf-8")
    print("已过滤 run.sh 中 Java 的 .zip 下载行（备份: run.sh.full_bak）")


def _prune_java_under_dataset(dataset_dir: Path) -> None:
    for name in ("java",):
        p = dataset_dir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"已删除 Java 工作目录: {p}")


def _prune_non_java_under_dataset(dataset_dir: Path) -> None:
    """删除 dataset 下除 java 外的语言工作目录（缓解之前失败 run 的残留）。"""
    for name in _NON_JAVA_LANG_DIRS:
        p = dataset_dir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"已删除非 Java 目录: {p}")


def _patch_run_sh_java_only(run_sh: Path) -> None:
    """
    旧版 run.sh 会 wget/curl 各语言 zip；去掉 python/php/go/ruby/javascript 的下载行。
    若 run.sh 无下载语句则跳过。
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
    print("已过滤 run.sh 中非 Java 的 .zip 下载行（备份: run.sh.full_bak）")


def _run_preprocess(dataset_dir: Path, codesearch: Path) -> None:
    print(f"运行预处理（cwd={dataset_dir}），可能较久 …")
    env = os.environ.copy()
    # 供 dataset 内脚本导入上层 codesearch 的 parser 等
    env["PYTHONPATH"] = str(codesearch) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        ["bash", "run.sh"],
        cwd=str(dataset_dir),
        check=True,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 GraphCodeBERT 清洗版 CodeSearchNet")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="清洗数据根目录（默认 CSN_CLEAN_OUTPUT_DIR 或 仓库/CodeSearchNet_clean_Dataset）",
    )
    parser.add_argument(
        "--from-java-dir",
        type=str,
        default=None,
        help="已生成好的 java 目录（含 test.jsonl、codebase.jsonl 等），仅复制到输出目录",
    )
    parser.add_argument(
        "--from-dataset-dir",
        type=str,
        default=None,
        help="已运行 run.sh 后的 dataset 目录（其下应有 java/*.jsonl）",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="解压后不执行 run.sh（仅当已手动预处理完成且文件已在 dataset/java 时配合默认流程使用）",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="预处理六语（磁盘占用大）；默认流程仍只复制 java 到输出（与历史行为一致）",
    )
    parser.add_argument(
        "--all-except-java",
        action="store_true",
        help="预处理除 Java 外五语，并复制到输出根下 go/javascript/php/python/ruby 各子目录",
    )
    parser.add_argument(
        "--remove-vendor",
        action="store_true",
        help="完成后删除 _vendor/CodeBERT 克隆（大幅省空间）",
    )
    args = parser.parse_args()

    if args.all_languages and args.all_except_java:
        parser.error("--all-languages 与 --all-except-java 不能同时使用")
    if args.from_java_dir and args.all_except_java:
        parser.error("--from-java-dir 仅用于 Java，不能与 --all-except-java 同时使用")

    clean_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_csn_clean_dataset_root()
    )
    dst_java = clean_root / "java"

    if args.from_java_dir:
        src = Path(args.from_java_dir).expanduser().resolve()
        _copy_java_jsonl(src, dst_java)
        print("完成。")
        return

    if args.from_dataset_dir:
        base = Path(args.from_dataset_dir).expanduser().resolve()
        if args.all_except_java:
            for lid in _LANGS_EXCEPT_JAVA:
                src = _resolve_src_lang_dir(base, lid)
                if src is None:
                    print(f"警告: 未找到 {base}/{lid}，跳过")
                    continue
                dst = clean_root / lid
                _copy_lang_jsonl(src, dst)
                _write_clean_language_marker(dst, lid)
            print("完成（--all-except-java + --from-dataset-dir）。")
        else:
            src_java = base / "java"
            _copy_java_jsonl(src_java, dst_java)
            print("完成。")
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
                raise FileNotFoundError(f"缺少 {pp}。")
            _prune_java_under_dataset(dataset_dir)
            _patch_preprocess_lang_list(pp, _LANGS_EXCEPT_JAVA)
            print("已启用 --all-except-java：将预处理 go/javascript/php/python/ruby。")
        elif not args.all_languages:
            rs = dataset_dir / "run.sh"
            if rs.is_file():
                _patch_run_sh_java_only(rs)
            pp = dataset_dir / "preprocess.py"
            if not pp.is_file():
                raise FileNotFoundError(f"缺少 {pp}，无法做 Java-only 补丁。")
            _prune_non_java_under_dataset(dataset_dir)
            _patch_preprocess_java_only(pp)
        else:
            print("已启用 --all-languages，将处理全部语言（磁盘占用大）。")
        _run_preprocess(dataset_dir, codesearch)
    else:
        print("已跳过 run.sh（--skip-run）")

    if args.all_except_java:
        any_ok = False
        for lid in _LANGS_EXCEPT_JAVA:
            src = _resolve_src_lang_dir(dataset_dir, lid)
            if src is None:
                print(f"警告: 未找到预处理输出目录 {dataset_dir}/{lid}，跳过")
                continue
            dst = clean_root / lid
            if _copy_lang_jsonl(src, dst) > 0:
                any_ok = True
                _write_clean_language_marker(dst, lid)
            for name in ("test.jsonl", "codebase.jsonl"):
                if not (dst / name).is_file():
                    print(f"警告: {lid} 缺少 {name}")
        if not any_ok:
            raise FileNotFoundError(
                f"未复制到任何语言 jsonl，请检查 {dataset_dir} 下是否生成 "
                f"{_LANGS_EXCEPT_JAVA}"
            )
        print(f"完成。清洗版（非 Java）输出根: {clean_root}")
    else:
        src_java = dataset_dir / "java"
        if not src_java.is_dir():
            raise FileNotFoundError(f"未找到 {src_java}，请检查 run.sh 是否成功。")
        _copy_java_jsonl(src_java, dst_java)

        for name in ("test.jsonl", "codebase.jsonl"):
            if not (dst_java / name).is_file():
                print(f"警告: 缺少 {name}，检索全库评估可能不可用。")

        print(f"完成。评测请使用: {dst_java}")

    if args.remove_vendor and vendor.is_dir():
        shutil.rmtree(vendor, ignore_errors=True)
        print(f"已删除克隆目录以释放空间: {vendor}")


if __name__ == "__main__":
    main()
