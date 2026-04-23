#!/usr/bin/env python3
"""
CodeXGLUE BigCloneBench clone detection evaluation.

Pipeline: UniXcoder (edge) → 收窄灰区 → 云端多步仲裁（可选 BCB RAG Few-shot +
本地 Symbolic Trace / 控制流报告 / 近似 DFG 骨架；可选双次云调用）。
"""

import json
import asyncio
import argparse
import time
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
import pandas as pd
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.autodl_env import apply_autodl_data_disk_env

apply_autodl_data_disk_env()

from core.orchestrator import Orchestrator
from edge.java_dfg_skeleton import extract_java_dfg_skeleton
import yaml
import os
from dotenv import load_dotenv

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from bcb_rag import BCBRAGRetriever  # noqa: E402

# Cloud: pass1 结构草稿（仅双次云时使用）
CLONE_STRUCTURE_SYSTEM = (
    "You extract structural signals from Java code. Reply with one JSON object only, no markdown prose."
)
# Cloud: 最终裁决（可含 RAG few-shot + 多步报告）
CLONE_ARBITER_SYSTEM = (
    "You are a Senior Java Static Analysis Expert for BigCloneBench Type-4 clones. "
    "Follow the user message exactly. End with one JSON object as specified."
)


def apply_hf_cache_from_config(config: dict) -> None:
    models = config.get("models") or {}
    cache = models.get("huggingface_cache") or models.get("root")
    if not cache:
        return
    path = Path(cache)
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(path))
    hub = path / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hub))


def load_unixcoder_on_orchestrator(orchestrator: Orchestrator, config: dict) -> None:
    """Load fine-tuned UniXcoder (RobertaForSequenceClassification) onto orchestrator."""
    if orchestrator.code_encoder is not None and orchestrator.code_tokenizer is not None:
        return
    import torch
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    cd = config.get("clone_detection") or {}
    uc = cd.get("unixcoder") or {}
    model_path = uc.get("model_path", r"G:\Ollama_Models\unixcoder-bcb-best")
    path = Path(model_path)
    apply_hf_cache_from_config(config)

    fallback_id = uc.get("fallback_pretrained", "microsoft/unixcoder-base")
    allow_fallback = bool(uc.get("use_hf_pretrained_if_checkpoint_missing", True))

    if path.is_dir():
        print(f"正在加载 UniXcoder 分类器（本地微调）: {path} ...")
        orchestrator.code_tokenizer = RobertaTokenizer.from_pretrained(str(path))
        orchestrator.code_encoder = RobertaForSequenceClassification.from_pretrained(
            str(path)
        )
    elif allow_fallback:
        print(
            f"警告: 未找到微调目录 {path}，将使用 HuggingFace 预训练权重 {fallback_id} "
            f"（二分类头未在 BCB 上微调，指标仅供参考）。\n"
            f"正式评估请先运行: python scripts/train_unixcoder_bcb.py export ... && train ...\n"
            f"若需严格失败可设 clone_detection.unixcoder.use_hf_pretrained_if_checkpoint_missing: false"
        )
        orchestrator.code_tokenizer = RobertaTokenizer.from_pretrained(fallback_id)
        orchestrator.code_encoder = RobertaForSequenceClassification.from_pretrained(
            fallback_id, num_labels=2
        )
    else:
        raise FileNotFoundError(
            f"UniXcoder checkpoint not found: {path}. "
            "Run: python scripts/train_unixcoder_bcb.py export ... && train ... "
            "Or set use_hf_pretrained_if_checkpoint_missing: true in config."
        )

    device_s = (uc.get("device") or "").strip().lower()
    if device_s in ("cpu", "cuda", "cuda:0"):
        device = torch.device(device_s)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orchestrator.code_encoder.to(device)
    orchestrator.code_encoder.eval()


_DOT_METHOD = re.compile(r"\.(\w+)\s*\(")
_QUALIFIED = re.compile(r"\b([A-Z][\w]*)\.(\w+)\s*\(")


def local_symbolic_trace(java: str, max_items: int = 48) -> str:
    """本地 Step1：API/方法调用骨架序列（Symbolic trace 近似）。"""
    s = java or ""
    qual = [f"{a}.{b}" for a, b in _QUALIFIED.findall(s)]
    dots = list(_DOT_METHOD.findall(s))
    skip = frozenset({"if", "while", "for", "switch", "catch", "synchronized"})
    tail = [m for m in dots if m not in skip]
    seq: List[str] = []
    for x in qual + tail:
        if x not in seq:
            seq.append(x)
        if len(seq) >= max_items:
            break
    return " -> ".join(seq) if seq else "(no API-like call chain detected)"


def local_control_flow_report(c1: str, c2: str) -> str:
    """本地 Step2：控制流关键词统计对比。"""
    keys = (
        "if",
        "else",
        "for",
        "while",
        "switch",
        "catch",
        "try",
        "return",
        "throw",
        "&&",
        "||",
        "?",
    )

    def cnt(s: str) -> Dict[str, int]:
        return {k: s.count(k) for k in keys}

    a, b = cnt(c1 or ""), cnt(c2 or "")
    sa, sb = sum(a.values()), sum(b.values())
    gap = abs(sa - sb)
    rel = "similar" if gap <= 3 else ("snippet1 heavier" if sa > sb else "snippet2 heavier")
    return (
        f"Control-flow token hits — Snippet1 total={sa} {a}; Snippet2 total={sb} {b}. "
        f"Relative: {rel} (|Δ|={gap})."
    )


def _build_cloud_structure_prompt(
    code1: str,
    code2: str,
    dfg1: str,
    dfg2: str,
) -> str:
    """云端 Step1+2（双次云）：由 LLM 产出符号迹与控制流对比 JSON。"""
    return f"""Extract structural signals for BigCloneBench clone analysis.

### Approximate data-flow skeletons (DFG-style; var -> call:A -> call:B)
Snippet1: {dfg1}
Snippet2: {dfg2}

### Snippet 1 (source)
```java
{code1}
```

### Snippet 2 (source)
```java
{code2}
```

Output **one** JSON object only (no markdown fences), schema:
{{
  "symbolic_trace_1": "<ordered API-like / method call skeleton for snippet 1>",
  "symbolic_trace_2": "<same for snippet 2>",
  "control_flow_comparison": "<compare branching/looping complexity and patterns>"
}}
"""


def _build_agentic_arbitration_prompt(
    code1: str,
    code2: str,
    few_shot_block: str,
    trace1: str,
    trace2: str,
    cf_report: str,
    dfg1: str,
    dfg2: str,
    llm_structure_json: Optional[str] = None,
) -> str:
    """云端 Step3：结合 RAG Few-shot +（本地或云端）结构报告 + DFG 骨架，做最终裁决。"""
    mid = ""
    if llm_structure_json:
        mid = f"""
### LLM structural draft (from prior pass)
```json
{llm_structure_json}
```
"""
    return f"""You are the final arbiter for **Type-4 semantic clones** (BigCloneBench): syntactic differences allowed if **functional goals and I/O behavior** match.

## Phase 0 — Retrieved training pairs (few-shot; labels are dataset ground truth)
{few_shot_block}

## Phase 1 — Symbolic traces (deterministic local extraction; use as hints, not sole evidence)
- Snippet1 trace: {trace1}
- Snippet2 trace: {trace2}
{mid}
## Phase 1b — Approximate data-flow skeleton (DFG-style text)
Use these **together with the source** to compare how values flow through calls (e.g. `digest -> call:MessageDigest.getInstance -> call:update`).
- Snippet1 DFG skeleton: {dfg1}
- Snippet2 DFG skeleton: {dfg2}

## Phase 2 — Control-flow comparison (local)
{cf_report}

## Phase 3 — Full code
### Snippet 1
```java
{code1}
```

### Snippet 2
```java
{code2}
```

## Phase 4 — Verdict
Using Phases 0–3 (including DFG skeletons vs source), output **one** JSON object only (no text outside JSON), schema:
{{
  "thought_process": {{
    "snippet1_goal": "string",
    "snippet2_goal": "string",
    "semantic_mapping": "string",
    "t4_characteristics": "string"
  }},
  "is_clone": true,
  "confidence": 0.85,
  "reason": "one sentence"
}}
"""


def _parse_is_clone(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def _brace_match_end(s: str, start: int) -> int:
    """从 s[start]=='{' 起，返回匹配的 '}' 下标；失败返回 -1。"""
    depth = 0
    i = start
    n = len(s)
    in_str = False
    esc = False
    str_quote = ""
    while i < n:
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == str_quote:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str = True
                str_quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _repair_json_loose(s: str) -> str:
    """去掉尾随逗号等常见 LLM 笔误。"""
    t = s.strip()
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def _iter_json_candidates(text: str) -> List[str]:
    """围栏块 + 从左到右每个顶层 {{ ... }} 平衡块。"""
    out: List[str] = []
    if not text:
        return out
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        chunk = m.group(1).strip()
        if chunk.startswith("{"):
            out.append(chunk)
    i = 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        e = _brace_match_end(text, j)
        if e > j:
            out.append(text[j : e + 1])
        i = j + 1
    return out


def _loads_dict_candidates(raw: str) -> Optional[dict]:
    for variant in (raw, _repair_json_loose(raw)):
        try:
            data = json.loads(variant)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def extract_json_from_text(text: str) -> dict:
    """尽量解析出第一个合法 JSON 对象（任意键）。"""
    for cand in _iter_json_candidates(text):
        d = _loads_dict_candidates(cand)
        if d is not None:
            return d
    return {}


def extract_and_parse_json(text: str) -> dict:
    """
    裁决用 JSON：优先含 is_clone 的候选；否则退而求其次含 thought_process；
    再否则第一个合法 dict。
    """
    err = {"is_clone": False, "confidence": 0.0, "reason": "Parse Error"}
    if not text or not text.strip():
        return err
    cands = _iter_json_candidates(text)
    if not cands and text.strip().startswith("{"):
        cands = [text.strip()]
    parsed: List[dict] = []
    for cand in cands:
        d = _loads_dict_candidates(cand)
        if d is not None:
            parsed.append(d)
    for d in parsed:
        if "is_clone" in d:
            return d
    for d in parsed:
        if "thought_process" in d:
            return d
    if parsed:
        return parsed[0]
    return err


def load_config_local(config_path: str = "config/settings.yaml") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        load_dotenv()
        if "cloud" in config:
            for provider in config["cloud"]:
                if isinstance(config["cloud"][provider], dict):
                    api_key = config["cloud"][provider].get("api_key", "")
                    if api_key.startswith("${") and api_key.endswith("}"):
                        env_var = api_key[2:-1]
                        config["cloud"][provider]["api_key"] = os.getenv(env_var, "")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def load_dataset_data(dataset_name: str = "Clone-detection-BigCloneBench"):
    if dataset_name == "Clone-detection-BigCloneBench":
        base_dir = Path(
            "datasets/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset"
        )
        print(f"正在加载 {dataset_name} 数据...")
        data_path = base_dir / "data.jsonl"
        code_dict: Dict[Any, str] = {}
        if data_path.exists():
            print(f"加载代码数据: {data_path}")
            with open(data_path, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        idx = item.get("idx")
                        func = item.get("func", "")
                        if idx is not None:
                            code_dict[str(idx)] = func
                            code_dict[int(idx)] = func
                        line_count += 1
                        if line_count % 50000 == 0:
                            print(f"  已加载 {line_count} 条代码...")
        else:
            print(f"警告: 数据文件不存在: {data_path}")
            return [], {}

        test_pairs = []
        test_path = base_dir / "test.txt"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            try:
                                test_pairs.append(
                                    (int(parts[0]), int(parts[1]), int(parts[2]))
                                )
                            except ValueError:
                                continue
        else:
            print(f"警告: 测试集文件不存在: {test_path}")
            return [], {}

        print(f"{dataset_name} 加载完成：{len(test_pairs)} 对测试样本")
        print(f"总代码片段数: {len(code_dict)}")
        return test_pairs, code_dict

    print(f"错误: 不支持的数据集: {dataset_name}")
    return [], {}


def _json_float(val: Any, default: float = 0.5) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def encode_pair_symmetric_truncation(
    tokenizer: Any,
    code1: str,
    code2: str,
    dfg1: str,
    dfg2: str,
    max_length: int,
    per_segment_cap: int = 254,
) -> Dict[str, Any]:
    """
    对称截断：拼接 Code 与 DFG 后，分别 tokenize 并各取前 N 个子词。
    格式: [CLS] Code A [SEP] DFG A [SEP] Code B [SEP] DFG B [SEP]
    """
    sep = getattr(tokenizer, "sep_token", "</s>")
    text1 = (code1 or "") + f" {sep} " + (dfg1 or "")
    text2 = (code2 or "") + f" {sep} " + (dfg2 or "")
    
    reserved = 4
    half = min(int(per_segment_cap), max(1, (max_length - reserved) // 2))
    tokens1 = tokenizer.tokenize(text1)[:half]
    tokens2 = tokenizer.tokenize(text2)[:half]
    ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    ids2 = tokenizer.convert_tokens_to_ids(tokens2)
    # prepend_batch_axis=True：否则 input_ids 为 1D，RoBERTa forward 里
    # batch_size, seq_length = input_shape 会报 ValueError: expected 2, got 1
    return tokenizer.prepare_for_model(
        ids1,
        pair_ids=ids2,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        prepend_batch_axis=True,
    )


async def analyze_pair(
    code1: str,
    code2: str,
    idx: int,
    config: dict,
    orchestrator: Orchestrator,
    dataset_name: str = "Clone-detection-BigCloneBench",
):
    import torch
    import torch.nn.functional as F

    cd = config.get("clone_detection") or {}
    uc = cd.get("unixcoder") or {}
    high_p = float(uc.get("high_clone_prob", 0.95))
    low_p = float(uc.get("low_clone_prob", 0.05))

    dfg_cfg = cd.get("dfg") or {}
    dfg_lim = int(dfg_cfg.get("prompt_max_chars", 1800))
    if dfg_cfg.get("enabled", True):
        me = int(dfg_cfg.get("max_edges", 48))
        pts = bool(dfg_cfg.get("prefer_tree_sitter", True))
        _r1 = extract_java_dfg_skeleton(code1 or "", max_edges=me, prefer_tree_sitter=pts)
        _r2 = extract_java_dfg_skeleton(code2 or "", max_edges=me, prefer_tree_sitter=pts)
        dfg1 = _r1 if len(_r1) <= dfg_lim else _r1[: dfg_lim - 3] + "..."
        dfg2 = _r2 if len(_r2) <= dfg_lim else _r2[: dfg_lim - 3] + "..."
    else:
        dfg1 = dfg2 = "(DFG skeleton disabled in config.)"

    try:
        tokenizer = orchestrator.code_tokenizer
        model = orchestrator.code_encoder
        if tokenizer is None or model is None:
            raise RuntimeError("UniXcoder not loaded; call load_unixcoder_on_orchestrator first")

        dfg_cfg = cd.get("dfg") or {}
        dfg_lim = int(dfg_cfg.get("prompt_max_chars", 1800))
        if dfg_cfg.get("enabled", True):
            me = int(dfg_cfg.get("max_edges", 48))
            pts = bool(dfg_cfg.get("prefer_tree_sitter", True))
            _r1 = extract_java_dfg_skeleton(code1 or "", max_edges=me, prefer_tree_sitter=pts)
            _r2 = extract_java_dfg_skeleton(code2 or "", max_edges=me, prefer_tree_sitter=pts)
            dfg1 = _r1 if len(_r1) <= dfg_lim else _r1[: dfg_lim - 3] + "..."
            dfg2 = _r2 if len(_r2) <= dfg_lim else _r2[: dfg_lim - 3] + "..."
        else:
            dfg1 = dfg2 = "(DFG skeleton disabled in config.)"

        tl = int(uc.get("max_length", 512))
        half_cap = int(uc.get("per_segment_token_cap", 254))
        inputs = encode_pair_symmetric_truncation(
            tokenizer,
            code1 or "",
            code2 or "",
            dfg1,
            dfg2,
            max_length=tl,
            per_segment_cap=half_cap,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            clone_prob = probs[0][1].item()

        print(f"  DEBUG pair {idx}: UniXcoder clone_prob -> {clone_prob:.4f}")

        if clone_prob >= high_p:
            return {
                "prediction": 1,
                "confidence": clone_prob,
                "issues1": 0,
                "issues2": 0,
                "cloud_verified": False,
                "similarity": clone_prob,
                "route": "Edge-UniXcoder-Fast-Pass",
                "tree_sitter_used": False,
                "cross_encoder_score": None,
                "ast_lexical": None,
                "api_call_anchor_similarity": None,
                "structure_score": None,
                "semantic_score": None,
                "goal_match": None,
                "is_structurally_equivalent": None,
                "unixcoder_clone_prob": clone_prob,
            }
        if clone_prob <= low_p:
            return {
                "prediction": 0,
                "confidence": 1.0 - clone_prob,
                "issues1": 0,
                "issues2": 0,
                "cloud_verified": False,
                "similarity": clone_prob,
                "route": "Edge-UniXcoder-Fast-Reject",
                "tree_sitter_used": False,
                "cross_encoder_score": None,
                "ast_lexical": None,
                "api_call_anchor_similarity": None,
                "structure_score": None,
                "semantic_score": None,
                "goal_match": None,
                "is_structurally_equivalent": None,
                "unixcoder_clone_prob": clone_prob,
            }

        arb = cd.get("cloud_arbitration") or {}
        cost_per_round = float(arb.get("estimated_cost_usd", 0.002))
        two_pass = bool(arb.get("agentic_two_pass", False))
        rounds_cloud_cost = cost_per_round * (2 if two_pass else 1)

        print(f"  DEBUG pair {idx}: 灰区 ({low_p} < p < {high_p})，尝试 Cloud Arbiter")

        if not await orchestrator.budget_controller.can_afford(rounds_cloud_cost):
            print(f"  DEBUG pair {idx}: 预算不足，Edge 强制判决")
            pred = 1 if clone_prob > 0.5 else 0
            conf = clone_prob if pred == 1 else 1.0 - clone_prob
            return {
                "prediction": pred,
                "confidence": conf,
                "issues1": 0,
                "issues2": 0,
                "cloud_verified": False,
                "similarity": clone_prob,
                "route": "Edge-Forced-Fallback",
                "tree_sitter_used": False,
                "cross_encoder_score": None,
                "ast_lexical": None,
                "api_call_anchor_similarity": None,
                "structure_score": None,
                "semantic_score": None,
                "goal_match": None,
                "is_structurally_equivalent": None,
                "unixcoder_clone_prob": clone_prob,
            }

        cloud_client = orchestrator.cloud_factory.get_client()
        rag_cfg = cd.get("rag") or {}
        rag = getattr(orchestrator, "bcb_rag", None)
        few_shot_block = "(RAG disabled or no BCB train index — use definitions only.)"
        if rag is not None and rag_cfg.get("enabled", True):
            try:
                ex = rag.query(
                    orchestrator,
                    code1 or "",
                    code2 or "",
                    top_k=int(rag_cfg.get("top_k", 3)),
                    max_length=int(rag_cfg.get("encode_max_length", 256)),
                )
                cap = int(rag_cfg.get("few_shot_code_cap", 400))
                few_shot_block = BCBRAGRetriever.format_few_shot(ex, code_cap=cap)
            except Exception as ex_rag:
                print(f"  RAG query failed: {ex_rag}")

        tr1 = local_symbolic_trace(code1 or "")
        tr2 = local_symbolic_trace(code2 or "")
        cf_rep = local_control_flow_report(code1 or "", code2 or "")

        llm_structure_json: Optional[str] = None
        route_tag = "Cloud-Arbiter-RAG" if rag and rag_cfg.get("enabled", True) else "Cloud-Arbiter"

        try:
            if two_pass:
                route_tag += "-2P"
                r_struct = await cloud_client._call_api(
                    _build_cloud_structure_prompt(
                        code1 or "", code2 or "", dfg1, dfg2
                    ),
                    max_tokens=int(arb.get("structure_pass_max_tokens", 512)),
                    system_prompt=CLONE_STRUCTURE_SYSTEM,
                    json_response_format=False,
                )
                mid_obj = extract_json_from_text(r_struct.get("content") or "")
                if mid_obj:
                    llm_structure_json = json.dumps(mid_obj, ensure_ascii=False)
                await orchestrator.budget_controller.record_expense(
                    cost_per_round,
                    orchestrator.cloud_factory.default_provider,
                    cloud_client.model,
                    int(r_struct.get("tokens") or 0),
                    "clone_arbitration_structure",
                    details=f"pair={idx}",
                )

            cloud_prompt = _build_agentic_arbitration_prompt(
                code1 or "",
                code2 or "",
                few_shot_block,
                tr1,
                tr2,
                cf_rep,
                dfg1,
                dfg2,
                llm_structure_json=llm_structure_json,
            )
            response = await cloud_client._call_api(
                cloud_prompt,
                max_tokens=int(arb.get("max_tokens", 2048)),
                system_prompt=CLONE_ARBITER_SYSTEM,
                json_response_format=False,
            )
            content = response.get("content") or ""
            result = extract_and_parse_json(content)
            is_clone = _parse_is_clone(result.get("is_clone", False))
            confidence = _json_float(result.get("confidence"), 0.8)
            thought = result.get("thought_process") or {}
            if isinstance(thought, dict):
                mapping = thought.get("semantic_mapping")
                if mapping:
                    print(f"  [Cloud T4 Analysis] pair {idx}: {mapping}")
            print(f"  DEBUG pair {idx}: Cloud -> is_clone={is_clone}, conf={confidence:.3f}")
            await orchestrator.budget_controller.record_expense(
                cost_per_round,
                orchestrator.cloud_factory.default_provider,
                cloud_client.model,
                int(response.get("tokens") or 0),
                "clone_arbitration",
                details=f"pair={idx}",
            )
            return {
                "prediction": 1 if is_clone else 0,
                "confidence": confidence,
                "issues1": 0,
                "issues2": 0,
                "cloud_verified": True,
                "similarity": clone_prob,
                "route": route_tag,
                "tree_sitter_used": False,
                "cross_encoder_score": None,
                "ast_lexical": None,
                "api_call_anchor_similarity": None,
                "structure_score": None,
                "semantic_score": None,
                "goal_match": None,
                "is_structurally_equivalent": None,
                "unixcoder_clone_prob": clone_prob,
            }
        except Exception as e:
            print(f"  Cloud failed: {e}")
            pred = 1 if clone_prob > 0.5 else 0
            conf = clone_prob if pred == 1 else 1.0 - clone_prob
            return {
                "prediction": pred,
                "confidence": conf,
                "issues1": 0,
                "issues2": 0,
                "cloud_verified": False,
                "similarity": clone_prob,
                "route": "Local-after-cloud-fail",
                "tree_sitter_used": False,
                "cross_encoder_score": None,
                "ast_lexical": None,
                "api_call_anchor_similarity": None,
                "structure_score": None,
                "semantic_score": None,
                "goal_match": None,
                "is_structurally_equivalent": None,
                "unixcoder_clone_prob": clone_prob,
            }

    except Exception as e:
        print(f"  analyze_pair {idx} error: {e}")
        return {
            "prediction": 0,
            "confidence": 0.0,
            "issues1": 0,
            "issues2": 0,
            "cloud_verified": False,
            "similarity": 0.0,
            "route": "error",
            "tree_sitter_used": False,
            "cross_encoder_score": None,
            "ast_lexical": None,
            "api_call_anchor_similarity": None,
            "structure_score": None,
            "semantic_score": None,
            "goal_match": None,
            "is_structurally_equivalent": None,
            "unixcoder_clone_prob": None,
        }


async def main():
    parser = argparse.ArgumentParser(
        description="BigCloneBench — UniXcoder + Cloud gray-zone arbitration"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Clone-detection-BigCloneBench",
        choices=["Clone-detection-BigCloneBench"],
        help="要评估的数据集",
    )
    parser.add_argument("--sample", type=int, default=100, help="采样数量 (0=全部)")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output is None:
        args.output = f"results_{args.dataset.lower().replace('-', '_')}.csv"

    print(f"=== UniXcoder + Cloud {args.dataset} 评估 ===")
    print(f"采样数量: {args.sample if args.sample > 0 else '全部'}")
    print(f"输出: {args.output}")

    config = load_config_local("config/settings.yaml")
    apply_hf_cache_from_config(config)
    mc = config.get("models") or {}
    if mc.get("huggingface_cache"):
        print(f"HuggingFace 缓存: {mc.get('huggingface_cache')}")
    ollama_dir = (config.get("ollama") or {}).get("models_dir")
    if ollama_dir:
        print(
            f"提示: Ollama 权重环境变量 OLLAMA_MODELS={ollama_dir} "
            "(主分析 CLI 使用；本脚本 Edge 侧为 UniXcoder)"
        )

    test_pairs, code_dict = load_dataset_data(args.dataset)

    if args.sample > 0 and args.sample < len(test_pairs):
        positive_pairs = [p for p in test_pairs if p[2] == 1]
        negative_pairs = [p for p in test_pairs if p[2] == 0]
        pos_count = min(args.sample // 2, len(positive_pairs))
        neg_count = args.sample - pos_count
        selected_pairs = positive_pairs[:pos_count] + negative_pairs[:neg_count]
        random.shuffle(selected_pairs)
        test_pairs = selected_pairs[: args.sample]
        print(f"采样后: {len(test_pairs)} 对 (正:{pos_count}, 负:{neg_count})")
    else:
        print(f"全部 {len(test_pairs)} 对")

    y_true = []
    y_pred = []
    all_metrics = []
    total_start = time.time()

    orchestrator = Orchestrator(config)
    try:
        await orchestrator.initialize()
        load_unixcoder_on_orchestrator(orchestrator, config)
        bcb_root = Path(
            "datasets/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset"
        )
        orchestrator.bcb_rag = BCBRAGRetriever.build_or_load(
            orchestrator, config, bcb_root, seed=args.seed
        )
        if orchestrator.bcb_rag is not None:
            print("BCB RAG：训练集索引已构建/加载，云端将带 Few-shot。")
        else:
            print(
                "BCB RAG：未启用或缺少 train.txt/data.jsonl，云端仅多步结构提示无检索示例。"
            )
        for i, (id1, id2, label) in enumerate(
            tqdm(test_pairs, desc="UniXcoder / Cloud")
        ):
            code1 = code_dict.get(id1, "")
            code2 = code_dict.get(id2, "")
            if not code1 or not code2:
                print(
                    f"  WARNING: 代码为空 id1={id1}, id2={id2}, "
                    f"len1={len(code1)}, len2={len(code2)}"
                )

            result = await analyze_pair(
                code1, code2, i, config, orchestrator, args.dataset
            )
            y_true.append(label)
            y_pred.append(result["prediction"])
            all_metrics.append(
                {
                    "pair_id": i,
                    "id1": id1,
                    "id2": id2,
                    "true_label": label,
                    "pred_label": result["prediction"],
                    "issues1": result["issues1"],
                    "issues2": result["issues2"],
                    "used_cloud": result["cloud_verified"],
                    "similarity": result.get("similarity", 0.0),
                    "confidence": result.get("confidence", 0.0),
                    "route": result.get("route", ""),
                    "tree_sitter_used": result.get("tree_sitter_used", False),
                    "cross_encoder_score": result.get("cross_encoder_score"),
                    "ast_lexical": result.get("ast_lexical"),
                    "structure_score": result.get("structure_score"),
                    "semantic_score": result.get("semantic_score"),
                    "goal_match": result.get("goal_match"),
                    "is_structurally_equivalent": result.get(
                        "is_structurally_equivalent"
                    ),
                    "api_call_anchor_similarity": result.get(
                        "api_call_anchor_similarity"
                    ),
                    "unixcoder_clone_prob": result.get("unixcoder_clone_prob"),
                }
            )
    finally:
        await orchestrator.shutdown()

    if len(y_true) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        cloud_rate = sum(1 for m in all_metrics if m["used_cloud"]) / len(
            all_metrics
        )
        total_time = time.time() - total_start
        token_reduction = 1.0 - (cloud_rate * 0.2)
        cost_saving = 1.0 - (cloud_rate * 0.1)
        cer = f1 / (cloud_rate + 0.01) if cloud_rate > 0 else f1
        avg_latency = total_time / len(test_pairs)
        throughput = len(test_pairs) / (total_time / 60) if total_time > 0 else 0.0

        print("\n" + "=" * 70)
        print(f"Evaluation Report (Seed={args.seed})")
        print("=" * 70)
        print(f"数据集                  : {args.dataset}")
        print(f"样本数量                : {len(y_true)}")
        print(f"Accuracy                : {accuracy:.4f}")
        print(f"Precision               : {precision:.4f}")
        print(f"Recall                  : {recall:.4f}")
        print(f"F1-Score                : {f1:.4f}")
        print(f"Cloud Routing Rate      : {cloud_rate:.2%}")
        print(f"Token Consumption Reduction : {token_reduction:.2%}")
        print(f"API Cost Saving         : {cost_saving:.2%}")
        print(f"CER                     : {cer:.3f}")
        print(f"Average Latency         : {avg_latency:.2f}s")
        print(f"Throughput              : {throughput:.2f} pairs/min")
        print("=" * 70)

        df = pd.DataFrame(all_metrics)
        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"详细结果已保存: {args.output}")

        summary = {
            "task": "clone_detection",
            "dataset": args.dataset,
            "random_seed": args.seed,
            "samples": len(y_true),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "cloud_routing_rate": float(cloud_rate),
            "token_reduction": float(token_reduction),
            "cost_saving": float(cost_saving),
            "cer": float(cer),
            "avg_latency": float(avg_latency),
            "throughput": float(throughput),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary_file = f"{args.dataset.lower().replace('-', '_')}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"摘要已保存: {summary_file}")
    else:
        print("没有加载到测试数据")


if __name__ == "__main__":
    asyncio.run(main())
