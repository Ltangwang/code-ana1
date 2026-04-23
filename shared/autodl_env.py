"""
AutoDL: point HuggingFace / PyTorch caches at the data disk.

Prefer /root/autodl-fs (large data volume), else /root/autodl-tmp.
Call apply_autodl_data_disk_env() before import transformers for best effect.
Set AUTODL_DATA_ROOT to override the data root.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_AUTODL_FS = Path("/root/autodl-fs")
_DEFAULT_AUTODL_TMP = Path("/root/autodl-tmp")


def _detect_data_root() -> Path | None:
    env = os.environ.get("AUTODL_DATA_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    if _DEFAULT_AUTODL_FS.is_dir():
        return _DEFAULT_AUTODL_FS.resolve()
    if _DEFAULT_AUTODL_TMP.is_dir():
        return _DEFAULT_AUTODL_TMP.resolve()
    return None


_root = _detect_data_root()
AUTODL_DATA_ROOT: Path = _root if _root is not None else _DEFAULT_AUTODL_FS


def apply_autodl_data_disk_env() -> None:
    root = _detect_data_root()
    if root is None:
        return

    hf_home = root / ".cache" / "huggingface"
    torch_home = root / ".cache" / "torch"
    pip_cache = root / ".cache" / "pip"
    tmp_dir = root / "tmp"

    for d in (hf_home, torch_home, pip_cache, tmp_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    if not os.environ.get("HF_HOME", "").strip():
        os.environ["HF_HOME"] = str(hf_home)
    if not os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip():
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    if not os.environ.get("TRANSFORMERS_CACHE", "").strip():
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    if not os.environ.get("TORCH_HOME", "").strip():
        os.environ["TORCH_HOME"] = str(torch_home)
    if not os.environ.get("XDG_CACHE_HOME", "").strip():
        os.environ["XDG_CACHE_HOME"] = str(root / ".cache")
    if tmp_dir.is_dir() and not os.environ.get("TMPDIR", "").strip():
        os.environ["TMPDIR"] = str(tmp_dir)
    if not os.environ.get("PIP_CACHE_DIR", "").strip():
        os.environ["PIP_CACHE_DIR"] = str(pip_cache)
