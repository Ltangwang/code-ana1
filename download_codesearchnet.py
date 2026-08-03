from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.autodl_env import apply_autodl_data_disk_env
from shared.csn_paths import default_csn_dataset_root

apply_autodl_data_disk_env()

from datasets import load_dataset

# 先写小 split，避免 SSH/会话中断时只有巨大的 train 在写、test 尚未落盘（云端评测找不到 test.jsonl）
_SPLIT_SAVE_ORDER = ("test", "validation", "train")

# Hugging Face `code_search_net` 各语言 config（与 datasets 库一致；不含聚合 config "all"）
_CSN_KNOWN_LANGS = frozenset({"go", "java", "javascript", "php", "python", "ruby"})

_CSN_DISPLAY_NAMES: dict[str, str] = {
    "go": "Go",
    "java": "Java",
    "javascript": "JavaScript",
    "php": "PHP",
    "python": "Python",
    "ruby": "Ruby",
}


def _ordered_split_names(keys: list[str]) -> list[str]:
    def sort_key(name: str) -> tuple[int, str]:
        try:
            return (_SPLIT_SAVE_ORDER.index(name), name)
        except ValueError:
            return (len(_SPLIT_SAVE_ORDER), name)

    return sorted(keys, key=sort_key)


def _atomic_write_jsonl_stream(path: Path, rows) -> None:
    """流式写入临时文件再 replace，避免大 train 集占满内存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".jsonl.tmp", prefix=path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _resolve_output_dir(output_dir: str) -> Path:
    """显式传入的相对路径：相对于仓库根目录（便于与本地目录结构一致）。"""
    p = Path(output_dir)
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _write_language_marker(lang_dir: Path, language_id: str, split_names: list[str]) -> None:
    """每个语言单独目录内标注语言信息，便于人工区分。"""
    info = {
        "dataset": "CodeSearchNet",
        "huggingface_id": "code_search_net",
        "language_id": language_id,
        "display_name": _CSN_DISPLAY_NAMES.get(
            language_id, language_id.replace("_", " ").title()
        ),
        "jsonl_splits": split_names,
        "folder_name": language_id,
    }
    marker = lang_dir / "LANGUAGE_INFO.json"
    marker.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def download_and_save(language: str = "java", output_dir: str | None = None) -> Path:
    apply_autodl_data_disk_env()
    out = _resolve_output_dir(output_dir) if output_dir else default_csn_dataset_root()

    if language not in _CSN_KNOWN_LANGS:
        raise ValueError(
            f"未知语言 {language!r}，可选: {sorted(_CSN_KNOWN_LANGS)}"
        )

    print(f"正在从 Hugging Face 下载 CodeSearchNet [{language}] 数据集...")
    print(f"输出根目录: {out}")
    ds = load_dataset("code_search_net", language)

    lang_dir = out / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    split_names = _ordered_split_names(list(ds.keys()))
    for split in split_names:
        output_file = lang_dir / f"{split}.jsonl"
        print(f"正在保存 {split} 集到 {output_file} ...")
        _atomic_write_jsonl_stream(output_file, ds[split])

    _write_language_marker(lang_dir, language, split_names)
    print(f"✅ [{language}] 已写入 {lang_dir}（含 LANGUAGE_INFO.json）")
    return out


def download_languages(
    languages: list[str], output_dir: str | None = None
) -> Path:
    """依次下载多个语言，每种语言一个子目录：{output_root}/{language_id}/"""
    apply_autodl_data_disk_env()
    root = _resolve_output_dir(output_dir) if output_dir else default_csn_dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    for i, lang in enumerate(languages):
        print(f"\n========== ({i+1}/{len(languages)}) 语言: {lang} ==========")
        download_and_save(lang, str(root))
    print(f"\n✅ 全部完成，共 {len(languages)} 种语言，根目录: {root}")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "下载 CodeSearchNet 为 jsonl；每种语言单独文件夹 "
            "`{输出根}/{语言}/`，内含 LANGUAGE_INFO.json 标注语言。"
        )
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="只下载一种语言（如 java）。若未指定且未使用下面多语言选项，则默认 java。",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="逗号分隔多种语言，如 go,python,ruby（与 HF config 名一致）",
    )
    parser.add_argument(
        "--all-except-java",
        action="store_true",
        help="下载除 Java 外全部语言（go, javascript, php, python, ruby），各占独立子目录",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出根目录；默认：CSN_OUTPUT_DIR 或 仓库下 CodeSearchNet_Dataset",
    )
    args = parser.parse_args()

    if args.all_except_java:
        langs = sorted(_CSN_KNOWN_LANGS - {"java"})
    elif args.languages:
        langs = [x.strip() for x in args.languages.split(",") if x.strip()]
        if not langs:
            parser.error("--languages 不能为空")
    elif args.language:
        langs = [args.language.strip()]
    else:
        langs = ["java"]

    invalid = [x for x in langs if x not in _CSN_KNOWN_LANGS]
    if invalid:
        parser.error(
            f"无效语言: {invalid}；可选: {sorted(_CSN_KNOWN_LANGS)}"
        )

    if len(langs) == 1:
        download_and_save(langs[0], args.output_dir)
    else:
        download_languages(langs, args.output_dir)
