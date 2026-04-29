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

# test/val before train so interrupted runs still have test.jsonl
_SPLIT_SAVE_ORDER = ("test", "validation", "train")

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
    """Tempfile + replace; avoids huge train in RAM."""
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
    """Relative paths from repo root."""
    p = Path(output_dir)
    if p.is_absolute():
        return p.resolve()
    return (_ROOT / p).resolve()


def _write_language_marker(lang_dir: Path, language_id: str, split_names: list[str]) -> None:
    """Write LANGUAGE_INFO.json under lang_dir."""
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
            f"Unknown language {language!r}; choose from: {sorted(_CSN_KNOWN_LANGS)}"
        )

    print(f"Downloading CodeSearchNet [{language}] from Hugging Face...")
    print(f"Output root: {out}")
    ds = load_dataset("code_search_net", language)

    lang_dir = out / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    split_names = _ordered_split_names(list(ds.keys()))
    for split in split_names:
        output_file = lang_dir / f"{split}.jsonl"
        print(f"Saving {split} split to {output_file} ...")
        _atomic_write_jsonl_stream(output_file, ds[split])

    _write_language_marker(lang_dir, language, split_names)
    print(f"OK [{language}] wrote to {lang_dir} (includes LANGUAGE_INFO.json)")
    return out


def download_languages(
    languages: list[str], output_dir: str | None = None
) -> Path:
    """Each lang under ``{root}/{lang}/``."""
    apply_autodl_data_disk_env()
    root = _resolve_output_dir(output_dir) if output_dir else default_csn_dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    for i, lang in enumerate(languages):
        print(f"\n========== ({i+1}/{len(languages)}) language: {lang} ==========")
        download_and_save(lang, str(root))
    print(f"\nOK all done, {len(languages)} language(s), root: {root}")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download CodeSearchNet as jsonl; each language in its own folder "
            "`{output_root}/{lang}/` with LANGUAGE_INFO.json."
        )
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Single language (e.g. java). If omitted and no multi-lang flags, default java.",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated languages, e.g. go,python,ruby (HF config names)",
    )
    parser.add_argument(
        "--all-except-java",
        action="store_true",
        help="All languages except Java (go, javascript, php, python, ruby), each in its own subdir",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output root; default: CSN_OUTPUT_DIR or CodeSearchNet_Dataset under repo",
    )
    args = parser.parse_args()

    if args.all_except_java:
        langs = sorted(_CSN_KNOWN_LANGS - {"java"})
    elif args.languages:
        langs = [x.strip() for x in args.languages.split(",") if x.strip()]
        if not langs:
            parser.error("--languages must not be empty")
    elif args.language:
        langs = [args.language.strip()]
    else:
        langs = ["java"]

    invalid = [x for x in langs if x not in _CSN_KNOWN_LANGS]
    if invalid:
        parser.error(
            f"Invalid language(s): {invalid}; choose from: {sorted(_CSN_KNOWN_LANGS)}"
        )

    if len(langs) == 1:
        download_and_save(langs[0], args.output_dir)
    else:
        download_languages(langs, args.output_dir)
