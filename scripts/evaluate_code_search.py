"""Java CodeSearchNet eval: UniXcoder Top-K, optional CE, Ollama on the pool, then cloud on escalation/rescue.

Train separately (train_unixcoder_csn*.py); this only loads checkpoints/HF weights.
CE off by default; pool size is min(llm_pool_k, retrieve_k). Cloud rerank/rescue per flags (e.g. --no-cloud-rescue-refine).
Metrics use K = --top-k; final rank prefers cloud > Ollama > edge.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from shared.autodl_env import apply_autodl_data_disk_env

apply_autodl_data_disk_env()

from core.orchestrator import Orchestrator
from shared.csn_paths import (
    _csn_test_jsonl_has_loadable_query,
    code_search_eval_results_dir,
    default_csn_clean_dataset_root,
    default_csn_java_dir_for_code_search,
    default_eval_models_parent,
)
from scripts.csn_data import load_csn_dataset
from scripts.csn_retriever import CSNRetriever
from scripts.csn_ce_rerank import load_csn_cross_encoder, rerank_candidates

_CODE_SEARCH_USE_CE = False


def _default_results_dir() -> Path:
    """evaluation_runs/ under cwd."""
    p = Path.cwd() / "evaluation_runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _next_results_code_search_path(out_dir: Path) -> Path:
    """results_code_search.json, results_code_search1.json, ..."""
    seen: List[int] = []
    for p in out_dir.glob("results_code_search*.json"):
        if p.name == "results_code_search.json":
            seen.append(0)
            continue
        m = re.fullmatch(r"results_code_search(\d+)\.json", p.name)
        if m:
            seen.append(int(m.group(1)))
    if not seen:
        return out_dir / "results_code_search.json"
    return out_dir / f"results_code_search{max(seen) + 1}.json"


CSN_RERANK_SYSTEM = """You are an expert Software Engineer and an intelligent Code Search Assistant.
Your task is to find the most relevant Java code snippet for a given natural language query.
The dataset is CodeSearchNet, where queries are typically the first sentence of a Javadoc comment.
You will be provided with a user's search query and a list of candidate code snippets retrieved by a search engine.

Evaluation Criteria:
1. Check if the method name, parameter types, and return type match the query's intent.
2. Focus on the code's control flow and core API calls to ensure it implements the requested functionality.
3. Select the ONE code snippet that best implements the functionality described in the query.
"""

def _build_rerank_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    """Full code in prompt (mind provider token limits)."""
    prompt = f"## User Search Query\n\"{query}\"\n\n## Candidate Code Snippets\n"
    for i, cand in enumerate(candidates):
        code = cand.get("code", "") or ""
        prompt += f"### Candidate {i}\n```java\n{code}\n```\n\n"
    
    prompt += """## Instructions
1. Output a <thinking> block where you briefly analyze how well each candidate matches the query based on method name, parameters, return type, and core logic.
2. Do NOT assume Candidate 0 is correct; the bi-encoder top-1 may be wrong. Compare ALL candidates independently and be willing to select a non-zero index if it better implements the query.
3. Output your final decision as a JSON object matching this schema:
{"best_candidate_index": <int>}
Where <int> is the index (0, 1, 2, ...) of the best matching candidate. If none match well, choose the one that is closest.
"""
    return prompt


CSN_NO_EDGE_REFINE_SYSTEM = """You are an expert at CodeSearchNet-style Java semantic code search.
The first retrieval pass (embedding search) failed to include the correct method in its shortlist.
Your job is to rewrite the user's natural-language query into ONE concise search string that a dense code retriever can match better: keep Java/API intent, method role, parameters, return type, and key verbs; you may add synonyms or decompose the Javadoc first sentence.
Output only valid JSON, no markdown fences."""

def _build_no_edge_refine_prompt(nl_query: str) -> str:
    return (
        "## Original query (often the first sentence of a method Javadoc)\n"
        f'"{nl_query}"\n\n'
        "## Task\n"
        "Produce a single refined search query string for a second embedding retrieval attempt.\n\n"
        "## Output format\n"
        'A single JSON object only, e.g. {"refined_search_query": "..."}\n'
    )


CSN_OLLAMA_SYSTEM = """You are an expert Java engineer evaluating CodeSearchNet-style retrieval.
The user query is almost always the first sentence of a method's Javadoc in this dataset.
Each candidate is a Java method body (or snippet) retrieved by embedding search and then filtered to this short list.
Your job: pick exactly ONE candidate index that best matches the Javadoc intent (method role, parameters, return type, and main control flow / API usage).
Be strict: prefer signatures and behavior described in the query over superficial token overlap.
Do NOT default to index 0; compare all candidates and choose the truly best one, even if it is not the first. If index 0 is genuinely best, you may choose it, but justify briefly in your thinking.
Set needs_escalation to true ONLY if you cannot decide between any candidates at all; otherwise choose the best one and set needs_escalation to false."""

def _build_ollama_rerank_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    lines = [
        "## Dataset context",
        "- CodeSearchNet Java split.",
        "- Query text ≈ first sentence of the gold method's Javadoc.",
        "",
        "## Natural language query",
        f'"{query}"',
        "",
        "## Candidates (indices 0..n-1 are fixed; answer must refer to these indices)",
    ]
    for i, cand in enumerate(candidates):
        code = cand.get("code", "") or ""
        lines.append(f"### Index {i}\n```java\n{code}\n```\n")
    n = len(candidates)
    lines.append(
        "## Instructions\n"
        "1) In a <thinking> block, compare each index: Javadoc intent vs method behavior, "
        "signature fit, and whether the code plausibly implements the described responsibility.\n"
        "2) Then output a single JSON object only (no extra text after it):\n"
        '{"best_candidate_index": <int>, "needs_escalation": <bool>}\n'
        f"best_candidate_index must be 0..{n-1}. "
        "Do NOT default to index 0; pick the candidate that best implements the query even if it is not first. "
        "Set needs_escalation to true ONLY if you cannot choose any candidate at all; "
        "otherwise choose the best index and set needs_escalation to false."
    )
    return "\n".join(lines)


def _ground_truth_index(candidates: List[Dict[str, Any]], ground_truth_url: str) -> int:
    for i, c in enumerate(candidates):
        if c.get("url") == ground_truth_url:
            return i
    return -1


def _llm_stage_rank(
    pool: List[Dict[str, Any]],
    best_idx: int,
    ground_truth_url: str,
    gt_rank_in_pool: int,
) -> int:
    """GT rank in pool after rerank; -1 if GT not in pool."""
    n = len(pool)
    if gt_rank_in_pool < 0 or gt_rank_in_pool >= n:
        return -1
    if 0 <= best_idx < n and (
        pool[best_idx].get("url") == ground_truth_url or best_idx == gt_rank_in_pool
    ):
        return 0
    return int(gt_rank_in_pool)


def _refined_search_query_from_parsed(parsed: dict) -> str:
    for k in ("refined_search_query", "refined_query", "search_query"):
        v = parsed.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _valid_best_candidate_index(parsed: dict, pool_len: int) -> Optional[int]:
    """best_candidate_index if int in range, else None."""
    if "best_candidate_index" not in parsed:
        return None
    v: Any = parsed["best_candidate_index"]
    if isinstance(v, bool):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if not isinstance(v, int):
        return None
    if v < 0 or v >= pool_len:
        return None
    return v


def _json_truthy(d: dict, *keys: str) -> bool:
    """True if any key is truthy (escalation-style flags)."""
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if v is True:
            return True
        if isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"):
            return True
        if isinstance(v, (int, float)) and v == 1 and not isinstance(v, bool):
            return True
    return False


def _ollama_requests_escalation(parsed: dict) -> bool:
    return _json_truthy(
        parsed, "needs_escalation", "uncertain", "needs_cloud", "request_cloud"
    )

def _brace_match_end(s: str, start: int) -> int:
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
    t = s.strip()
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t

def _iter_json_candidates(text: str) -> List[str]:
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
    for cand in _iter_json_candidates(text):
        d = _loads_dict_candidates(cand)
        if d is not None:
            return d
    return {}

def load_unixcoder_base(orchestrator: Orchestrator, config: dict) -> None:
    """Load UniXcoder tokenizer/model from settings or HF."""
    if orchestrator.code_encoder is not None and orchestrator.code_tokenizer is not None:
        return
    import torch
    from transformers import RobertaModel, RobertaTokenizer

    cd = config.get("clone_detection") or {}
    uc = cd.get("unixcoder") or {}
    fallback = uc.get("fallback_pretrained", "microsoft/unixcoder-base")
    model_name = str(fallback).strip() or "microsoft/unixcoder-base"
    raw_mp = (uc.get("model_path") or "").strip()
    print(f"[UniXcoder] config clone_detection.unixcoder.model_path: {raw_mp or '(not set)'}")
    if raw_mp:
        p = Path(raw_mp).expanduser()
        if p.exists():
            model_name = str(p.resolve())
            print(f"[UniXcoder] fine-tune weights loaded from (local dir): {model_name}")
        else:
            print(
                f"[UniXcoder] warning: model_path does not exist, using fallback_pretrained: {model_name}"
            )
    else:
        print(f"[UniXcoder] model_path not set, using fallback_pretrained: {model_name}")

    print(f"Loading UniXcoder Base Model: {model_name} ...")
    orchestrator.code_tokenizer = RobertaTokenizer.from_pretrained(model_name)
    orchestrator.code_encoder = RobertaModel.from_pretrained(model_name)

    device_s = (uc.get("device") or "").strip().lower()
    if device_s in ("cpu", "cuda", "cuda:0"):
        device = torch.device(device_s)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orchestrator.code_encoder.to(device)
    orchestrator.code_encoder.eval()

def _chunk_indexed_evenly(
    indexed: List[Tuple[int, Dict[str, Any]]], num_parts: int
) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Partition ``(global_idx, row)`` into ~equal chunks (≤ num_parts)."""
    n = len(indexed)
    if n == 0:
        return []
    num_parts = max(1, min(int(num_parts), n))
    base, rem = divmod(n, num_parts)
    out: List[List[Tuple[int, Dict[str, Any]]]] = []
    start = 0
    for i in range(num_parts):
        sz = base + (1 if i < rem else 0)
        out.append(indexed[start : start + sz])
        start += sz
    return out


def _empty_stages(
    query_idx: int,
    edge_hit: bool,
    edge_rank: int,
    cloud_fallback_reason: str = "none",
) -> Dict[str, Any]:
    return {
        "query_idx": query_idx,
        "edge_hit": edge_hit,
        "edge_rank": edge_rank,
        "ce_rank": -1,
        "ollama_rank": -1,
        "ollama_verified": False,
        "ollama_ok": False,
        "cloud_rank": -1,
        "cloud_verified": False,
        "cloud_fallback_reason": cloud_fallback_reason,
    }


def _attach_latency(
    result: Dict[str, Any], lat: Dict[str, float], t0: float
) -> Dict[str, Any]:
    """把分阶段 wall-clock（ms）写入单条 query 结果。"""
    out = dict(result)
    out["bi_encoder_ms"] = round(float(lat.get("bi_encoder_ms", 0.0)), 3)
    out["ollama_ms"] = round(float(lat.get("ollama_ms", 0.0)), 3)
    out["cloud_ms"] = round(float(lat.get("cloud_ms", 0.0)), 3)
    out["e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return out


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _latency_summary(values: List[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(vals),
        "mean_ms": round(sum(vals) / len(vals), 3),
        "p50_ms": round(_percentile(vals, 50), 3),
        "p95_ms": round(_percentile(vals, 95), 3),
    }


def _aggregate_latency_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总 bi-encoder / Ollama / cloud / e2e 延迟（仅统计实际发生该阶段的样本）。"""
    bi = [float(r["bi_encoder_ms"]) for r in results if float(r.get("bi_encoder_ms", 0) or 0) > 0]
    ol = [float(r["ollama_ms"]) for r in results if float(r.get("ollama_ms", 0) or 0) > 0]
    cl = [float(r["cloud_ms"]) for r in results if float(r.get("cloud_ms", 0) or 0) > 0]
    e2e = [float(r["e2e_ms"]) for r in results if float(r.get("e2e_ms", 0) or 0) > 0]
    return {
        "bi_encoder_latency": _latency_summary(bi),
        "ollama_latency": _latency_summary(ol),
        "cloud_latency": _latency_summary(cl),
        "e2e_latency": _latency_summary(e2e),
    }


async def _no_edge_cloud_rescue(
    idx: int,
    nl_query: str,
    ground_truth_url: str,
    orchestrator: Orchestrator,
    config: dict,
    pl: Dict[str, Any],
    query_max_length: int,
    search_lock: Optional[asyncio.Lock],
    lat: Optional[Dict[str, float]] = None,
    t0: Optional[float] = None,
) -> Dict[str, Any]:
    """
    首段 retrieve_k 内无 GT：视为第一次检索失败，由云端主导「重做」该 query（不调 Ollama/CE）。

    默认（cloud_rescue_refine=True）：
      1) 云端根据原 query 生成 refined_search_query（第二次检索用语）；
      2) 本地双塔仅用该字符串从全库召回 top cloud_rescue_k 候选（构成候选池）；
      3) 云端在候选池上输出 best_candidate_index（最终选定谁，不是沿用双塔顺序）。

    若关闭 refine（--no-cloud-rescue-refine）：跳过步骤 1，仍用原 query 召回候选池，仅步骤 3 由云重排。
    """
    if lat is None:
        lat = {"bi_encoder_ms": 0.0, "ollama_ms": 0.0, "cloud_ms": 0.0}
    if t0 is None:
        t0 = time.perf_counter()

    rescue_k = max(int(pl.get("cloud_rescue_k", 50)), int(pl.get("retrieve_k", 10)))
    use_refine = bool(pl.get("cloud_rescue_refine", True))

    arb = config.get("clone_detection", {}).get("cloud_arbitration", {})
    est_cloud_cost = float(arb.get("estimated_cost_usd", 0.002))
    rounds = 2 if use_refine else 1
    if not await orchestrator.budget_controller.can_afford(est_cloud_cost * rounds):
        return _attach_latency(
            {
                "query_idx": idx,
                "edge_hit": False,
                "edge_rank": -1,
                "ce_rank": -1,
                "ollama_rank": -1,
                "ollama_verified": False,
                "ollama_ok": False,
                "cloud_rank": -1,
                "cloud_verified": False,
                "cloud_fallback_reason": "budget_no_edge_rescue",
            },
            lat,
            t0,
        )

    cloud_client = orchestrator.cloud_factory.get_client()
    cloud_rank = -1
    cloud_verified = False
    cloud_fallback_reason = "no_edge_rescue_pending"

    search_query_for_pool = nl_query
    if use_refine:
        try:
            _tc = time.perf_counter()
            r0 = await cloud_client._call_api(
                _build_no_edge_refine_prompt(nl_query),
                system_prompt=CSN_NO_EDGE_REFINE_SYSTEM,
                max_tokens=512,
                json_response_format=False,
            )
            lat["cloud_ms"] += (time.perf_counter() - _tc) * 1000.0
            c0 = r0.get("content", "")
            p0 = extract_json_from_text(c0)
            rq = _refined_search_query_from_parsed(p0)
            if rq:
                search_query_for_pool = rq
            await orchestrator.budget_controller.record_expense(
                est_cloud_cost,
                orchestrator.cloud_factory.default_provider,
                cloud_client.model,
                int(r0.get("tokens") or 0),
                "code_search_no_edge_refine",
                details=f"query={idx}",
            )
        except Exception as e:
            print(f"  analyze_query {idx} no_edge 云端改写 query 调用失败，回退原 query: {e}")
            search_query_for_pool = nl_query

    def _run_rescue_pool() -> List[Dict[str, Any]]:
        if pl.get("no_bi_encoder"):
            cl = pl.get("corpus_list") or []
            return build_random_corpus_pool(
                cl,
                ground_truth_url,
                rescue_k,
                idx,
                seed=int(pl.get("random_pool_seed", 42)) + 79_199,
            )
        return orchestrator.csn_retriever.search(
            orchestrator,
            search_query_for_pool,
            top_k=rescue_k,
            max_length=query_max_length,
        )

    _tb = time.perf_counter()
    if search_lock is not None:
        async with search_lock:
            rescue_pool = await asyncio.to_thread(_run_rescue_pool)
    else:
        rescue_pool = await asyncio.to_thread(_run_rescue_pool)
    lat["bi_encoder_ms"] += (time.perf_counter() - _tb) * 1000.0

    gt_in_pool = _ground_truth_index(rescue_pool, ground_truth_url)

    try:
        _tc = time.perf_counter()
        response = await cloud_client._call_api(
            _build_rerank_prompt(nl_query, rescue_pool),
            system_prompt=CSN_RERANK_SYSTEM,
            max_tokens=1024,
            json_response_format=False,
        )
        lat["cloud_ms"] += (time.perf_counter() - _tc) * 1000.0
        content = response.get("content", "")
        parsed = extract_json_from_text(content)
        best_c = _valid_best_candidate_index(parsed, len(rescue_pool))
        if best_c is None:
            print(f"  analyze_query {idx}: no_edge 解救云返回无有效 best_candidate_index")
            cloud_fallback_reason = "no_edge_rescue_cloud_invalid_parse"
            cloud_verified = False
        else:
            cloud_rank = _llm_stage_rank(
                rescue_pool, best_c, ground_truth_url, gt_in_pool
            )
            cloud_verified = True
            cloud_fallback_reason = "cloud_success_no_edge_rescue"
            await orchestrator.budget_controller.record_expense(
                est_cloud_cost,
                orchestrator.cloud_factory.default_provider,
                cloud_client.model,
                int(response.get("tokens") or 0),
                "code_search_no_edge_rescue",
                details=f"query={idx}",
            )
    except Exception as e:
        print(f"  analyze_query {idx} no_edge cloud rescue error: {e}")
        cloud_verified = False
        cloud_rank = -1
        cloud_fallback_reason = "no_edge_rescue_cloud_api_error"

    return _attach_latency(
        {
            "query_idx": idx,
            "edge_hit": False,
            "edge_rank": -1,
            "ce_rank": -1,
            "ollama_rank": -1,
            "ollama_verified": False,
            "ollama_ok": False,
            "cloud_rank": cloud_rank,
            "cloud_verified": cloud_verified,
            "cloud_fallback_reason": cloud_fallback_reason,
        },
        lat,
        t0,
    )


async def analyze_query(
    query_item: Dict[str, Any],
    idx: int,
    config: dict,
    orchestrator: Orchestrator,
    pl: Dict[str, Any],
    query_max_length: int = 512,
    skip_cloud: bool = False,
    search_lock: Optional[asyncio.Lock] = None,
) -> Dict[str, Any]:
    retrieve_k = int(pl["retrieve_k"])
    llm_pool_k = int(pl["llm_pool_k"])
    use_ce = bool(pl["use_ce"])
    ce_model = pl.get("ce_model")
    ce_max_code_chars = int(pl["ce_max_code_chars"])
    ce_batch_size = int(pl["ce_batch_size"])
    ollama_deep_max_tokens = int(pl["ollama_deep_max_tokens"])
    ollama_deep_timeout = float(pl["ollama_deep_timeout"])

    t0 = time.perf_counter()
    lat: Dict[str, float] = {
        "bi_encoder_ms": 0.0,
        "ollama_ms": 0.0,
        "cloud_ms": 0.0,
    }

    def _fin(d: Dict[str, Any]) -> Dict[str, Any]:
        return _attach_latency(d, lat, t0)

    edge_hit = False
    edge_rank = -1
    try:
        nl_query = query_item["nl_query"]
        ground_truth_url = query_item["url"]

        def _sync_search_only() -> List[Dict[str, Any]]:
            return orchestrator.csn_retriever.search(
                orchestrator, nl_query, top_k=retrieve_k, max_length=query_max_length
            )

        def _random_topk() -> List[Dict[str, Any]]:
            import random as _rnd

            rng = _rnd.Random(idx * 1000003 + retrieve_k)
            recs = orchestrator.csn_retriever.records
            k = min(retrieve_k, len(recs))
            picks = rng.sample(range(len(recs)), k)
            return [dict(recs[i]) for i in picks]

        def _sync_bi_topk() -> List[Dict[str, Any]]:
            return orchestrator.csn_retriever.search(
                orchestrator, nl_query, top_k=retrieve_k, max_length=query_max_length
            )

        def _sync_ce_on(c_topk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if use_ce and ce_model is not None:
                return rerank_candidates(
                    nl_query,
                    c_topk,
                    ce_model,
                    max_code_chars=ce_max_code_chars,
                    batch_size=ce_batch_size,
                )
            return [dict(x) for x in c_topk]

        if skip_cloud:
            _tb = time.perf_counter()
            if pl.get("no_bi_encoder"):
                if search_lock is not None:
                    async with search_lock:
                        candidates_wide = await asyncio.to_thread(_random_topk)
                else:
                    candidates_wide = await asyncio.to_thread(_random_topk)
            else:
                if search_lock is not None:
                    async with search_lock:
                        candidates_wide = await asyncio.to_thread(_sync_search_only)
                else:
                    candidates_wide = await asyncio.to_thread(_sync_search_only)
            lat["bi_encoder_ms"] += (time.perf_counter() - _tb) * 1000.0
            edge_rank = _ground_truth_index(candidates_wide, ground_truth_url)
            edge_hit = edge_rank >= 0
            if not edge_hit:
                return _fin(_empty_stages(idx, False, -1, "no_edge_hit"))
            return _fin(_empty_stages(idx, True, edge_rank, "skip_cloud"))

        _tb = time.perf_counter()
        if pl.get("no_bi_encoder"):
            if search_lock is not None:
                async with search_lock:
                    candidates_wide = await asyncio.to_thread(_random_topk)
            else:
                candidates_wide = await asyncio.to_thread(_random_topk)
        else:
            if search_lock is not None:
                async with search_lock:
                    candidates_wide = await asyncio.to_thread(_sync_bi_topk)
            else:
                candidates_wide = await asyncio.to_thread(_sync_bi_topk)
        lat["bi_encoder_ms"] += (time.perf_counter() - _tb) * 1000.0

        edge_rank = _ground_truth_index(candidates_wide, ground_truth_url)
        edge_hit = edge_rank >= 0

        # 可部署信号：双塔短名单 top1 与 topK 分数差（不依赖 GT）
        _sims = [
            float(c.get("similarity", 0.0) or 0.0) for c in candidates_wide[:retrieve_k]
        ]
        bi_top1_score = _sims[0] if _sims else 0.0
        bi_topk_score = _sims[-1] if _sims else 0.0
        bi_score_margin = bi_top1_score - bi_topk_score
        score_margin_threshold = float(pl.get("score_margin_threshold", -1.0))
        low_margin_trigger = (
            score_margin_threshold >= 0.0
            and bi_score_margin < score_margin_threshold
        )

        if not edge_hit:
            if pl.get("bi_ce_only") or pl.get("bi_ollama_only"):
                return _empty_stages(idx, False, -1, "no_edge_hit")
            # 部署模式：不再用 GT 缺席作为上云触发；oracle_routing 才保留该 oracle 上界
            if not pl.get("enable_cloud_rescue", True) or not pl.get(
                "oracle_routing", False
            ):
                return _fin(_empty_stages(idx, False, -1, "no_edge_hit"))
            return await _no_edge_cloud_rescue(
                idx,
                nl_query,
                ground_truth_url,
                orchestrator,
                config,
                pl,
                query_max_length,
                search_lock,
                lat=lat,
                t0=t0,
            )

        if search_lock is not None:
            async with search_lock:
                ranked_full = await asyncio.to_thread(
                    lambda: _sync_ce_on(candidates_wide)
                )
        else:
            ranked_full = await asyncio.to_thread(lambda: _sync_ce_on(candidates_wide))

        pool = ranked_full[:llm_pool_k]
        ce_rank = _ground_truth_index(pool, ground_truth_url)

        if pl.get("bi_ce_only"):
            return _fin(
                {
                    "query_idx": idx,
                    "edge_hit": True,
                    "edge_rank": edge_rank,
                    "ce_rank": ce_rank,
                    "ollama_rank": -1,
                    "ollama_verified": False,
                    "ollama_ok": False,
                    "cloud_rank": -1,
                    "cloud_verified": False,
                    "cloud_fallback_reason": "bi_ce_only",
                }
            )

        ollama_rank = -1
        ollama_verified = False
        ollama_ok = False
        pre_cloud_trigger = "none"

        if pl.get("skip_ollama"):
            pre_cloud_trigger = "skip_ollama_direct_cloud"
        else:
            li = orchestrator.local_inference
            if li is None:
                raise RuntimeError("local_inference is not initialized")
            try:
                _to = time.perf_counter()
                ollama_text = await li.generate_text(
                    _build_ollama_rerank_prompt(nl_query, pool),
                    system=CSN_OLLAMA_SYSTEM,
                    max_tokens=ollama_deep_max_tokens,
                    timeout_sec=ollama_deep_timeout,
                )
                lat["ollama_ms"] += (time.perf_counter() - _to) * 1000.0
                if not (ollama_text and ollama_text.strip()):
                    print(f"  analyze_query {idx}: Ollama 空响应，触发云端重排")
                    pre_cloud_trigger = "ollama_empty_response"
                else:
                    op = extract_json_from_text(ollama_text)
                    wants_escalation = _ollama_requests_escalation(op)
                    best_o = _valid_best_candidate_index(op, len(pool))
                    if wants_escalation:
                        pre_cloud_trigger = "ollama_needs_escalation"
                        if best_o is not None:
                            ollama_rank = _llm_stage_rank(
                                pool, best_o, ground_truth_url, ce_rank
                            )
                            ollama_verified = True
                        print(
                            f"  analyze_query {idx}: Ollama needs_escalation=true，触发云端重排"
                        )
                    elif best_o is not None:
                        ollama_rank = _llm_stage_rank(
                            pool, best_o, ground_truth_url, ce_rank
                        )
                        ollama_verified = True
                        ollama_ok = True
                        pre_cloud_trigger = "none"
                    else:
                        pre_cloud_trigger = "ollama_invalid_index"
                        print(
                            f"  analyze_query {idx}: Ollama 无有效 best_candidate_index，触发云端重排"
                        )
            except Exception as e:
                print(f"  analyze_query {idx} Ollama error: {e}，触发云端重排")
                pre_cloud_trigger = "ollama_exception"

        if pl.get("bi_ollama_only"):
            _cfr = "bi_ollama_only" if ollama_ok else f"bi_ollama_only_{pre_cloud_trigger}"
            return _fin(
                {
                    "query_idx": idx,
                    "edge_hit": True,
                    "edge_rank": edge_rank,
                    "ce_rank": ce_rank,
                    "ollama_rank": ollama_rank,
                    "ollama_verified": ollama_verified,
                    "ollama_ok": ollama_ok,
                    "cloud_rank": -1,
                    "cloud_verified": False,
                    "cloud_fallback_reason": _cfr,
                    "bi_top1_score": bi_top1_score,
                    "bi_topk_score": bi_topk_score,
                    "bi_score_margin": bi_score_margin,
                }
            )

        cloud_rank = -1
        cloud_verified = False
        cloud_fallback_reason = "none"

        force_cloud = bool(pl.get("force_cloud", False))
        if low_margin_trigger and ollama_ok and not force_cloud:
            # 部署可观测 Condition B：短名单分数差过小，即使 Ollama 成功也上云复核
            pre_cloud_trigger = "low_score_margin"
            cloud_fallback_reason = "low_score_margin"
            cloud_client = orchestrator.cloud_factory.get_client()
            cloud_prompt = _build_rerank_prompt(nl_query, pool)
            arb = config.get("clone_detection", {}).get("cloud_arbitration", {})
            est_cloud_cost = float(arb.get("estimated_cost_usd", 0.002))

            if not await orchestrator.budget_controller.can_afford(est_cloud_cost):
                print(
                    f"  DEBUG query {idx}: Budget insufficient，无法发起云端重排"
                )
                cloud_fallback_reason = "budget_low_score_margin"
            else:
                try:
                    _tc = time.perf_counter()
                    response = await cloud_client._call_api(
                        cloud_prompt,
                        system_prompt=CSN_RERANK_SYSTEM,
                        max_tokens=1024,
                        json_response_format=False,
                    )
                    lat["cloud_ms"] += (time.perf_counter() - _tc) * 1000.0
                    content = response.get("content", "")
                    parsed = extract_json_from_text(content)
                    best_c = _valid_best_candidate_index(parsed, len(pool))
                    if best_c is None:
                        print(
                            f"  analyze_query {idx}: 云端返回无有效 best_candidate_index，不计费"
                        )
                        cloud_fallback_reason = "cloud_invalid_parse_low_margin"
                        cloud_verified = False
                    else:
                        cloud_rank = _llm_stage_rank(
                            pool, best_c, ground_truth_url, ce_rank
                        )
                        cloud_verified = True
                        cloud_fallback_reason = "cloud_success_low_margin"
                        from shared.prompts import estimate_cost
                        _est = estimate_cost(
                            cloud_prompt,
                            int(response.get("tokens") or 0),
                            cloud_client.model,
                        )
                        await orchestrator.budget_controller.record_expense(
                            _est,
                            orchestrator.cloud_factory.default_provider,
                            cloud_client.model,
                            int(response.get("tokens") or 0),
                            "code_search_rerank_low_margin",
                            details=f"query={idx}",
                        )
                except Exception as e:
                    print(f"  analyze_query {idx} cloud error: {e}")
                    cloud_fallback_reason = "cloud_api_error_low_margin"
        elif ollama_ok and not force_cloud:
            cloud_fallback_reason = "none"
        else:
            if force_cloud and ollama_ok:
                pre_cloud_trigger = "force_cloud_always"
            cloud_fallback_reason = pre_cloud_trigger
            cloud_client = orchestrator.cloud_factory.get_client()
            cloud_prompt = _build_rerank_prompt(nl_query, pool)
            arb = config.get("clone_detection", {}).get("cloud_arbitration", {})
            est_cloud_cost = float(arb.get("estimated_cost_usd", 0.002))

            if not await orchestrator.budget_controller.can_afford(est_cloud_cost):
                print(
                    f"  DEBUG query {idx}: Budget insufficient, cannot start cloud rerank"
                )
                cloud_fallback_reason = "budget_after_ollama_fail"
            else:
                try:
                    _tc = time.perf_counter()
                    response = await cloud_client._call_api(
                        cloud_prompt,
                        system_prompt=CSN_RERANK_SYSTEM,
                        max_tokens=1024,
                        json_response_format=False,
                    )
                    lat["cloud_ms"] += (time.perf_counter() - _tc) * 1000.0
                    content = response.get("content", "")
                    parsed = extract_json_from_text(content)
                    best_c = _valid_best_candidate_index(parsed, len(pool))
                    if best_c is None:
                        print(
                            f"  analyze_query {idx}: 云端返回无有效 best_candidate_index，不计费"
                        )
                        cloud_fallback_reason = "cloud_invalid_parse"
                        cloud_verified = False
                    else:
                        cloud_rank = _llm_stage_rank(
                            pool, best_c, ground_truth_url, ce_rank
                        )
                        cloud_verified = True
                        cloud_fallback_reason = "cloud_success_after_fallback"
                        from shared.prompts import estimate_cost
                        _est = estimate_cost(
                            cloud_prompt,
                            int(response.get("tokens") or 0),
                            cloud_client.model,
                        )
                        await orchestrator.budget_controller.record_expense(
                            _est,
                            orchestrator.cloud_factory.default_provider,
                            cloud_client.model,
                            int(response.get("tokens") or 0),
                            "code_search_rerank",
                            details=f"query={idx}",
                        )
                except Exception as e:
                    print(f"  analyze_query {idx} cloud error: {e}")
                    cloud_fallback_reason = "cloud_api_error"

        return _fin(
            {
                "query_idx": idx,
                "edge_hit": True,
                "edge_rank": edge_rank,
                "ce_rank": ce_rank,
                "ollama_rank": ollama_rank,
                "ollama_verified": ollama_verified,
                "ollama_ok": ollama_ok,
                "cloud_rank": cloud_rank,
                "cloud_verified": cloud_verified,
                "cloud_fallback_reason": cloud_fallback_reason,
                "bi_top1_score": bi_top1_score,
                "bi_topk_score": bi_topk_score,
                "bi_score_margin": bi_score_margin,
            }
        )

    except Exception as e:
        print(f"  analyze_query {idx} error: {e}")
        return _fin(
            {
                "query_idx": idx,
                "edge_hit": edge_hit,
                "edge_rank": edge_rank,
                "ce_rank": -1,
                "ollama_rank": -1,
                "ollama_verified": False,
                "ollama_ok": False,
                "cloud_rank": -1,
                "cloud_verified": False,
                "cloud_fallback_reason": "pipeline_exception",
            }
        )


def _pipeline_final_rank_for_metrics(
    r: Dict[str, Any], *, skip_cloud: bool, bi_ce_only: bool = False
) -> int:
    """Final 0-based GT rank matching edge–cloud combined MRR; -1 if the pipeline has no valid output."""
    if skip_cloud:
        if r.get("edge_hit") and int(r.get("edge_rank", -1)) >= 0:
            return int(r["edge_rank"])
        return -1
    if bi_ce_only:
        if not r.get("edge_hit"):
            return -1
        cr = int(r.get("ce_rank", -1))
        return cr if cr >= 0 else -1
    edge_rank = int(r.get("edge_rank", -1)) if r.get("edge_hit") else -1

    if r.get("cloud_verified") and int(r.get("cloud_rank", -1)) >= 0:
        return int(r["cloud_rank"])

    if r.get("ollama_verified") and int(r.get("ollama_rank", -1)) >= 0:
        return int(r["ollama_rank"])

    if r.get("ollama_ok") and int(r.get("ollama_rank", -1)) >= 0:
        return int(r["ollama_rank"])

    return edge_rank if edge_rank >= 0 else -1


async def run_evaluation(args: argparse.Namespace, config: dict):
    orchestrator = Orchestrator(config)
    try:
        await orchestrator.initialize()
        load_unixcoder_base(orchestrator, config)
        cd0 = config.get("clone_detection") or {}
        uc0 = cd0.get("unixcoder") or {}
        embed_model_tag = uc0.get("fallback_pretrained", "microsoft/unixcoder-base")
        _mp = (uc0.get("model_path") or "").strip()
        if _mp:
            _p = Path(_mp).expanduser()
            if _p.exists():
                embed_model_tag = str(_p.resolve())

        clean_root = default_csn_clean_dataset_root()
        clean_java = (clean_root / "java").resolve()
        env_java = os.environ.get("CSN_JAVA_DIR", "").strip()
        if env_java:
            dataset_dir = Path(env_java).expanduser().resolve()
        elif clean_java.is_dir() and _csn_test_jsonl_has_loadable_query(clean_java):
            dataset_dir = clean_java
        else:
            dataset_dir = default_csn_java_dir_for_code_search()

        is_clean = False
        try:
            dataset_dir.resolve().relative_to(clean_root.resolve())
            is_clean = True
        except ValueError:
            pass
        if is_clean:
            print(f"Dataset directory (GraphCodeBERT clean): {dataset_dir}")
        else:
            print(
                f"Dataset directory (java under non-clean root or explicit CSN_JAVA_DIR): {dataset_dir}\n"
                f"  Prefer for eval: {clean_java}\n"
                f"  For protocol/paper alignment, prepare GraphCodeBERT-style cleaned java under {clean_java}.\n"
                f"  Or unset CSN_JAVA_DIR to use the clean path when test.jsonl is loadable."
            )
        test_path = dataset_dir / "test.jsonl"
        codebase_path = dataset_dir / "codebase.jsonl"

        cfg_root = Path(str(config.get("models", {}).get("root", "G:/Ollama_Models")))
        if cfg_root.exists():
            cache_dir = cfg_root / "1"
        else:
            cache_dir = default_eval_models_parent(config) / "1"
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Full-corpus index: GraphCodeBERT-style codebase.jsonl; fallback to test.jsonl only if missing.
        index_max = None if int(args.index_size) <= 0 else int(args.index_size)
        if codebase_path.is_file():
            index_path = codebase_path
            corpus_mode = True
            print(f"Indexer: full codebase {index_path.name} (corpus_mode=True).")
        else:
            index_path = test_path
            corpus_mode = False
            print(
                f"Indexer: {index_path.name} only (codebase.jsonl not found under {dataset_dir})."
            )

        orchestrator.csn_retriever = CSNRetriever.build_or_load(
            orchestrator,
            data_path=index_path,
            cache_dir=cache_dir,
            max_samples=index_max,
            encode_len=args.encode_len,
            batch_size=32,
            corpus_mode=corpus_mode,
            cache_model_tag=embed_model_tag or "",
        )
        
        if orchestrator.csn_retriever is None:
            print("Failed to initialize CSN Retriever.")
            return
            
        # Load test queries
        print(f"Loading test queries from {test_path}...")
        sample_max = None if int(args.sample) <= 0 else int(args.sample)
        test_queries = load_csn_dataset(
            test_path, max_samples=sample_max, require_code=False
        )
        print(f"Loaded {len(test_queries)} test queries.")
        if not test_queries:
            print(
                "Error: zero test queries. Check:\n"
                f"  - test.jsonl exists and is non-empty: {test_path}\n"
                "  - project/data path: this repo may look under code-ana1 / code-anal / code-analyze on the data disk\n"
                "  - clean test.jsonl may be NL + url only (no code); ensure url is non-empty and the file is JSONL\n"
                "  - CSN_JAVA_DIR points to a java directory with a valid test.jsonl"
            )
            return

        cs = config.get("code_search") or {}
        retrieve_k = max(1, int(args.top_k))
        llm_pool_k = int(cs.get("llm_pool_k", 10))
        if getattr(args, "llm_pool_k", None) is not None:
            llm_pool_k = int(args.llm_pool_k)
        llm_pool_k = min(llm_pool_k, retrieve_k)
        use_ce = bool(
            _CODE_SEARCH_USE_CE or bool(getattr(args, "use_ce", False))
        )
        ce_model_name = str(cs.get("ce_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        if getattr(args, "ce_model", None):
            ce_model_name = str(args.ce_model)
        ce_max_code_chars = int(cs.get("ce_max_code_chars", 2000))
        ce_batch_size = int(cs.get("ce_batch_size", 16))
        ollama_deep_max_tokens = int(cs.get("ollama_deep_max_tokens", 4096))
        ollama_deep_timeout = float(cs.get("ollama_deep_timeout", 120))
        cloud_rescue_k = int(cs.get("cloud_rescue_k", 50))
        if getattr(args, "cloud_rescue_k", None) is not None:
            cloud_rescue_k = int(args.cloud_rescue_k)
        cloud_rescue_refine = bool(cs.get("cloud_rescue_refine", True))
        if getattr(args, "no_cloud_rescue_refine", False):
            cloud_rescue_refine = False
        score_margin_threshold = float(cs.get("score_margin_threshold", -1.0))
        if getattr(args, "score_margin_threshold", None) is not None:
            score_margin_threshold = float(args.score_margin_threshold)
        oracle_routing = bool(getattr(args, "oracle_routing", False))

        ce_model = None
        skip_cloud = bool(getattr(args, "skip_cloud", False))
        bi_ce_only = bool(getattr(args, "bi_ce_only", False))
        bi_ollama_only = bool(getattr(args, "bi_ollama_only", False))
        if bi_ce_only and skip_cloud:
            print("Error: --bi-ce-only and --skip-cloud cannot be used together.")
            return
        if bi_ollama_only and skip_cloud:
            print("Error: --bi-ollama-only and --skip-cloud cannot be used together.")
            return
        if bi_ollama_only and bi_ce_only:
            print("Error: --bi-ollama-only and --bi-ce-only cannot be used together.")
            return
        if bi_ce_only:
            use_ce = True
            if llm_pool_k != retrieve_k:
                print(
                    f"[--bi-ce-only] CE pool aligned with bi-encoder Top-K: llm_pool_k "
                    f"{llm_pool_k} -> {retrieve_k} (= --top-k)."
                )
            llm_pool_k = retrieve_k
        if bi_ollama_only:
            if getattr(args, "use_ce", False):
                print("Note: --bi-ollama-only conflicts with --use-ce; running without CE.")
            use_ce = False
            if llm_pool_k != retrieve_k:
                print(
                    f"[--bi-ollama-only] Ollama pool aligned with bi-encoder Top-K: llm_pool_k "
                    f"{llm_pool_k} -> {retrieve_k} (= --top-k)."
                )
            llm_pool_k = retrieve_k

        if bi_ce_only:
            print(f"Loading Cross-Encoder (--bi-ce-only): {ce_model_name} ...")
            ce_model = load_csn_cross_encoder(ce_model_name)
        elif bi_ollama_only:
            print("[--bi-ollama-only] Skipping Cross-Encoder load.")
        elif not skip_cloud and use_ce:
            print(f"Loading Cross-Encoder: {ce_model_name} ...")
            ce_model = load_csn_cross_encoder(ce_model_name)
        elif not skip_cloud and not use_ce:
            print(
                "Cross-Encoder is off (script has _CODE_SEARCH_USE_CE=False); "
                "LLM pool is the bi-encoder order truncated."
            )

        if not skip_cloud and not bi_ce_only:  # bi-ollama-only needs Ollama
            ok, omsg = await orchestrator.local_inference.health_check()
            if not ok:
                print(f"Ollama unavailable: {omsg}")
                raise SystemExit(1)
            li = orchestrator.local_inference
            print(
                f"[Ollama] connected {li.base_url.rstrip('/')}, model {li.model_name}"
            )

        force_cloud = bool(getattr(args, "force_cloud", False))
        if bi_ollama_only and force_cloud:
            print("Note: --force-cloud is ignored under --bi-ollama-only.")
        pl: Dict[str, Any] = {
            "retrieve_k": retrieve_k,
            "llm_pool_k": llm_pool_k,
            "use_ce": use_ce,
            "ce_model": ce_model,
            "ce_max_code_chars": ce_max_code_chars,
            "ce_batch_size": ce_batch_size,
            "ollama_deep_max_tokens": ollama_deep_max_tokens,
            "ollama_deep_timeout": ollama_deep_timeout,
            "force_cloud": False if bi_ollama_only else force_cloud,
            "cloud_rescue_k": cloud_rescue_k,
            "cloud_rescue_refine": cloud_rescue_refine,
            "enable_cloud_rescue": (
                False
                if bi_ollama_only
                else (not bool(getattr(args, "no_cloud_rescue", False)))
            ),
            "bi_ce_only": bi_ce_only,
            "bi_ollama_only": bi_ollama_only,
            "skip_ollama": skip_ollama,
            "no_bi_encoder": no_bi_encoder,
            "score_margin_threshold": score_margin_threshold,
            "oracle_routing": oracle_routing,
        }
        if skip_cloud:
            print(
                f"Pipeline: bi-encoder retrieve_k={retrieve_k}, "
                f"eval Success@{args.top_k} on bi list only (--skip-cloud)"
            )
        elif bi_ce_only:
            print(
                f"Pipeline (--bi-ce-only): bi-encoder retrieve_k={retrieve_k}, "
                f"CE rerank pool={llm_pool_k}, eval Success@{args.top_k} uses CE pool rank; "
                f"no Ollama/cloud; no_edge_hit counts as failure"
            )
        elif bi_ollama_only:
            print(
                f"Pipeline (--bi-ollama-only): bi-encoder retrieve_k={retrieve_k}, "
                f"Ollama pool={llm_pool_k}, eval Success@{args.top_k} uses Ollama output rank; "
                f"no CE, no cloud calls; no_edge_hit counts as failure"
            )
        else:
            cloud_note = (
                "; also force cloud on every query after Ollama success (--force-cloud)"
                if force_cloud
                else "; cloud on failure or needs_escalation"
            )
            if getattr(args, "no_cloud_rescue", False):
                rescue_txt = "no_cloud_rescue: no cloud rescue for no_edge_hit"
            elif cloud_rescue_refine:
                rescue_txt = (
                    f"no_edge_hit -> cloud refines search query, bi-encoder recalls top-{cloud_rescue_k}, "
                    "cloud picks in pool"
                )
            else:
                rescue_txt = (
                    f"no_edge_hit -> original query, bi-encoder top-{cloud_rescue_k}, "
                    "cloud picks in pool (--no-cloud-rescue-refine)"
                )
            ce_txt = "CE -> " if use_ce else "no CE -> "
            print(
                f"Pipeline: retrieve_k={retrieve_k} (= --top-k), "
                f"Success@{args.top_k}, "
                f"{ce_txt}Ollama{cloud_note} (pool={llm_pool_k}=min(config, retrieve_k)); "
                f"{rescue_txt}"
            )

        workers = max(1, int(getattr(args, "workers", 1)))
        indexed = list(enumerate(test_queries))
        partitions = _chunk_indexed_evenly(indexed, workers)
        # When using multiple partitions, serialize bi-encoder+CE (GPU) to avoid concurrent coroutines holding the model.
        search_lock: Optional[asyncio.Lock] = (
            asyncio.Lock() if len(partitions) > 1 else None
        )

        async def _run_partition(
            part: List[Tuple[int, Dict[str, Any]]],
        ) -> List[Dict[str, Any]]:
            local: List[Dict[str, Any]] = []
            for i, query_item in part:
                res = await analyze_query(
                    query_item,
                    i,
                    config,
                    orchestrator,
                    pl,
                    query_max_length=args.query_max_len,
                    skip_cloud=skip_cloud,
                    search_lock=search_lock,
                )
                local.append(res)
            return local

        with tqdm(total=len(test_queries), desc="Evaluating Code Search") as pbar:
            async def _run_partition_with_progress(
                part: List[Tuple[int, Dict[str, Any]]],
            ) -> List[Dict[str, Any]]:
                out = await _run_partition(part)
                pbar.update(len(out))
                return out

            nested = await asyncio.gather(
                *[_run_partition_with_progress(p) for p in partitions]
            )
        results: List[Dict[str, Any]] = []
        for sub in nested:
            results.extend(sub)
        results.sort(key=lambda r: r["query_idx"])
            
        # Calculate metrics (Edge@K: is GT in the top K of retrieve_k bi-encoder results?)
        eval_k = int(args.top_k)
        edge_mrr = 0.0
        ce_mrr = 0.0
        ollama_mrr = 0.0
        cloud_mrr = 0.0
        edge_success_at_k = 0
        ce_success_at_k = 0
        ollama_success_at_k = 0
        cloud_success_at_k = 0
        edge_cloud_combined_mrr_sum = 0.0

        for r in results:
            if r["edge_hit"]:
                if r["edge_rank"] >= 0 and r["edge_rank"] < eval_k:
                    edge_success_at_k += 1
                edge_mrr += 1.0 / (r["edge_rank"] + 1)

            if r.get("ce_rank", -1) >= 0:
                if 0 <= r["ce_rank"] < eval_k:
                    ce_success_at_k += 1
                ce_mrr += 1.0 / (r["ce_rank"] + 1)

            if not skip_cloud and not bi_ce_only:
                orank = r.get("ollama_rank", -1)
                if orank >= 0 and orank < eval_k:
                    ollama_success_at_k += 1
                if orank >= 0:
                    ollama_mrr += 1.0 / (orank + 1)

                crank = r.get("cloud_rank", -1)
                if crank >= 0 and crank < eval_k:
                    cloud_success_at_k += 1
                if crank >= 0:
                    cloud_mrr += 1.0 / (crank + 1)

            fr = _pipeline_final_rank_for_metrics(
                r, skip_cloud=skip_cloud, bi_ce_only=bi_ce_only
            )
            if fr == 0:
                edge_cloud_combined_mrr_sum += 1.0
            elif fr > 0:
                edge_cloud_combined_mrr_sum += 1.0 / (fr + 1)

        n = len(results)
        edge_cloud_combined_success_at_k = sum(
            1
            for r in results
            if 0
            <= _pipeline_final_rank_for_metrics(
                r, skip_cloud=skip_cloud, bi_ce_only=bi_ce_only
            )
            < eval_k
        )
        if n > 0:
            edge_mrr /= n
            ce_mrr /= n
            ollama_mrr /= n
            cloud_mrr /= n
        edge_cloud_combined_mrr = (
            edge_cloud_combined_mrr_sum / n if n else 0.0
        )

        def _pct(a: int, d: int) -> str:
            return f"{a / d * 100:.2f}%" if d else "n/a"

        print("\n=== Code Search Evaluation Results ===")
        print(f"Total Queries: {n}")
        print(
            f"Edge Success@{eval_k} (within first {eval_k} of {retrieve_k} bi-encoder): "
            f"{edge_success_at_k}/{n} ({_pct(edge_success_at_k, n)})"
        )
        print(f"Edge MRR (over {retrieve_k}): {edge_mrr:.4f}")
        if not skip_cloud:
            if use_ce:
                print(
                    f"CE Success@{eval_k} (in pool {llm_pool_k}): "
                    f"{ce_success_at_k}/{n} ({_pct(ce_success_at_k, n)})"
                )
                print(f"CE MRR: {ce_mrr:.4f}")
            if not bi_ce_only:
                print(
                    f"Ollama Success@{eval_k}: {ollama_success_at_k}/{n} ({_pct(ollama_success_at_k, n)})"
                )
                print(f"Ollama MRR: {ollama_mrr:.4f}")
            if not bi_ce_only and not bi_ollama_only:
                print(
                    f"Cloud Success@{eval_k}: {cloud_success_at_k}/{n} ({_pct(cloud_success_at_k, n)})"
                )
                print(f"Cloud MRR: {cloud_mrr:.4f}")

        cfr_counter: Counter[str] = Counter(
            str(r.get("cloud_fallback_reason", "none")) for r in results
        )
        ollama_ok_rate = (
            sum(1 for r in results if r.get("ollama_ok")) / n if n else 0.0
        )
        cloud_call_rate = (
            sum(1 for r in results if r.get("cloud_verified")) / n if n else 0.0
        )
        # Cloud call rate: cloud API was actually invoked (incl. parse errors or API errors, excluding budget block)
        _CLOUD_API_INVOKED = frozenset(
            {
                "cloud_success_after_fallback",
                "cloud_invalid_parse",
                "cloud_api_error",
                "cloud_success_no_edge_rescue",
                "no_edge_rescue_cloud_invalid_parse",
                "no_edge_rescue_cloud_api_error",
                "cloud_success_low_margin",
                "cloud_invalid_parse_low_margin",
                "cloud_api_error_low_margin",
            }
        )
        cloud_invocation_count = sum(
            1
            for r in results
            if str(r.get("cloud_fallback_reason", "")) in _CLOUD_API_INVOKED
        )
        cloud_invocation_rate = cloud_invocation_count / n if n else 0.0
        if not skip_cloud and not bi_ce_only:
            print(f"Ollama OK rate (no cloud needed): {ollama_ok_rate*100:.2f}%")
            if not bi_ollama_only:
                print(f"Cloud call success rate (verified): {cloud_call_rate*100:.2f}%")
            print("cloud_fallback_reason breakdown:", dict(cfr_counter))

        print("\n--- Edge–cloud combined summary ---")
        print(
            f"Edge Success@{eval_k} (GT in top {eval_k} of {retrieve_k} bi-encoder results): "
            f"{edge_success_at_k}/{n} ({_pct(edge_success_at_k, n)})"
        )
        _final_label = (
            "bi-encoder only"
            if skip_cloud
            else (
                "bi-encoder+CE (--bi-ce-only)"
                if bi_ce_only
                else ("bi-encoder+Ollama (--bi-ollama-only)" if bi_ollama_only else "full pipeline")
            )
        )
        _decision_txt = (
            "CE pool rank"
            if bi_ce_only
            else ("Ollama output rank" if bi_ollama_only else "cloud > Ollama")
        )
        _success_note = (
            "same as CE pool rank"
            if bi_ce_only
            else (
                "same as Ollama rank (no cloud)"
                if bi_ollama_only
                else "final rank matches MRR: cloud > Ollama > edge"
            )
        )
        print(
            f"Edge–cloud combined MRR (final decision: {_decision_txt}; "
            f"{_final_label}, denominator = total queries n): "
            f"{edge_cloud_combined_mrr:.4f}"
        )
        print(
            f"Edge–cloud combined Success@{eval_k} ({_success_note}; "
            f"0<=rank<{eval_k} counts as success; retrieve_k={retrieve_k}): "
            f"{edge_cloud_combined_success_at_k}/{n} "
            f"({_pct(edge_cloud_combined_success_at_k, n)})"
        )
        if skip_cloud:
            print("Cloud API invocation rate: 0/{} (0.00%) [--skip-cloud]".format(n))
        elif bi_ce_only:
            print("Cloud API invocation rate: 0/{} (0.00%) [--bi-ce-only]".format(n))
        elif bi_ollama_only:
            print("Cloud API invocation rate: 0/{} (0.00%) [--bi-ollama-only]".format(n))
        else:
            print(
                f"Cloud API invocation rate: {cloud_invocation_count}/{n} "
                f"({cloud_invocation_rate*100:.2f}%)"
            )

        # Save results
        metrics_out: Dict[str, Any] = {
            "total_queries": n,
            "retrieve_k": retrieve_k,
            "eval_top_k": eval_k,
            "llm_pool_k": llm_pool_k,
            "bi_ce_only": bi_ce_only,
            "bi_ollama_only": bi_ollama_only,
            "skip_ollama": skip_ollama,
            "no_bi_encoder": no_bi_encoder,
            "edge_success_at_k": edge_success_at_k,
            "edge_mrr": edge_mrr,
            "edge_cloud_combined_mrr": edge_cloud_combined_mrr,
            "edge_cloud_combined_success_at_k": edge_cloud_combined_success_at_k,
            "ollama_ok_rate": ollama_ok_rate,
            "cloud_call_rate": cloud_call_rate,
            "cloud_invocation_count": cloud_invocation_count,
            "cloud_invocation_rate": cloud_invocation_rate,
            "cloud_fallback_breakdown": dict(cfr_counter),
        }
        # 本次评测累计云花费（按 token 估算）
        try:
            _bs = await orchestrator.get_budget_status()
            metrics_out["cloud_estimated_cost_usd_total"] = round(_bs.used_budget, 6)
            metrics_out["cloud_estimated_cost_usd_per_query"] = (
                round(_bs.used_budget / n, 6) if n else 0.0
            )
        except Exception:
            pass

        # 分阶段延迟：mean / P50 / P95（ms）
        latency_metrics = _aggregate_latency_metrics(results)
        metrics_out.update(latency_metrics)
        print("\n--- Latency (ms) ---")
        for stage_key, label in [
            ("bi_encoder_latency", "Bi-encoder"),
            ("ollama_latency", "Ollama/SLM"),
            ("cloud_latency", "Cloud"),
            ("e2e_latency", "End-to-end"),
        ]:
            s = latency_metrics.get(stage_key, {})
            print(
                f"{label}: n={s.get('count', 0)}, "
                f"mean={s.get('mean_ms', 0):.1f}, "
                f"P50={s.get('p50_ms', 0):.1f}, "
                f"P95={s.get('p95_ms', 0):.1f}"
            )

        if eval_k == 1:
            metrics_out["success_at_1"] = edge_success_at_k
            if not skip_cloud and not bi_ce_only:
                metrics_out["ollama_success_at_1"] = ollama_success_at_k
        if not skip_cloud:
            metrics_out.update(
                {
                    "ce_success_at_k": ce_success_at_k,
                    "ce_mrr": ce_mrr,
                    "ollama_success_at_k": ollama_success_at_k,
                    "ollama_mrr": ollama_mrr,
                    "cloud_success_at_k": cloud_success_at_k,
                    "cloud_mrr": cloud_mrr,
                }
            )
        out_dir = code_search_eval_results_dir(
            config, getattr(args, "results_dir", None)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = _next_results_code_search_path(out_dir)
        metrics_out["results_path"] = str(results_path.resolve())
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics_out, "details": results}, f, indent=2)
        print(f"\n结果已写入: {results_path.resolve()}")
            
    finally:
        await orchestrator.shutdown()

def _load_config_with_env(config_path: str = "config/settings.yaml") -> dict:
    """YAML from disk; substitute ${VAR} in cloud.*.api_key."""
    import os

    import yaml
    from dotenv import load_dotenv

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    load_dotenv()
    cloud = config.get("cloud")
    if isinstance(cloud, dict):
        for provider in cloud:
            prov = cloud[provider]
            if not isinstance(prov, dict):
                continue
            api_key = prov.get("api_key", "")
            if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                prov["api_key"] = os.getenv(env_var, "")
    return config


_JAVA_EVAL_TOP_K = 1


def _has_long_opt_java(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_java_eval_k_defaults() -> None:
    """Prepend --top-k/--llm-pool-k/--cloud-rescue-k from _JAVA_EVAL_TOP_K if absent."""
    argv = sys.argv[1:]
    k = str(_JAVA_EVAL_TOP_K)
    inserts: list[str] = []
    if not _has_long_opt_java(argv, "--top-k"):
        inserts.extend(["--top-k", k])
    if not _has_long_opt_java(argv, "--llm-pool-k"):
        inserts.extend(["--llm-pool-k", k])
    if not _has_long_opt_java(argv, "--cloud-rescue-k"):
        inserts.extend(["--cloud-rescue-k", k])
    sys.argv = [sys.argv[0]] + inserts + argv


def main():
    _inject_java_eval_k_defaults()
    parser = argparse.ArgumentParser(description="Evaluate Code Search with Edge-Cloud")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Number of test queries to eval; <=0 means full test.jsonl (default 0 = all)",
    )
    parser.add_argument(
        "--index-size",
        type=int,
        default=0,
        help="Max index pool size; <=0 means all (default: full codebase.jsonl, else full test.jsonl if missing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="K for Success@K: bi-encoder retrieve_k, Ollama/cloud pool, all Success metrics (default 1; main also injects alignment with cloud_rescue when omitted)",
    )
    parser.add_argument(
        "--llm-pool-k",
        type=int,
        default=None,
        help="Ollama/cloud candidate pool cap (actual pool = min(this, retrieve_k), retrieve_k = --top-k)",
    )
    parser.add_argument(
        "--bi-ollama-only",
        action="store_true",
        dest="bi_ollama_only",
        help="Bi-encoder + Ollama only: no CE, no cloud (not even after Ollama failure); "
        "mutually exclusive with --skip-cloud / --bi-ce-only / --use-ce; "
        "sets llm_pool_k = --top-k so the Ollama pool matches bi-encoder Top-K",
    )
    parser.add_argument(
        "--bi-ce-only",
        action="store_true",
        dest="bi_ce_only",
        help="Bi-encoder + CE only: final metrics use CE pool rank, no Ollama/cloud; "
        "mutually exclusive with --skip-cloud; sets llm_pool_k = --top-k to align CE with bi-encoder Top-K (--use-ce optional)",
    )
    parser.add_argument(
        "--use-ce",
        action="store_true",
        help="Rerank bi-encoder Top-K with Cross-Encoder before Ollama/cloud (for bi-encoder only use --skip-cloud, not this)",
    )
    parser.add_argument(
        "--no-ce",
        action="store_true",
        help="(Reserved placeholder; CE off by default unless --use-ce)",
    )
    parser.add_argument(
        "--ce-model",
        type=str,
        default=None,
        help="Override Cross-Encoder model name from config",
    )
    parser.add_argument("--encode-len", type=int, default=512, help="Code embedding max length")
    parser.add_argument("--query-max-len", type=int, default=512, help="Query encoding max length")
    parser.add_argument(
        "--skip-cloud",
        action="store_true",
        help="Bi-encoder retrieval only, no cloud rerank",
    )
    parser.add_argument(
        "--force-cloud",
        action="store_true",
        help=(
            "After edge_hit and Ollama, always call cloud rerank (higher cloud share; "
            "vs default 'skip cloud when Ollama succeeds'; for cost/ablation studies"
        ),
    )
    parser.add_argument(
        "--cloud-rescue-k",
        type=int,
        default=None,
        help=(
            "On no_edge_hit: bi-encoder similarity top-K pool, cloud reranks on that set "
            "(default K from config code_search.cloud_rescue_k, else 50)"
        ),
    )
    parser.add_argument(
        "--no-cloud-rescue",
        action="store_true",
        help="On no_edge_hit, skip bi-encoder pool + cloud rescue (record failure only; for ablation)",
    )
    parser.add_argument(
        "--no-cloud-rescue-refine",
        action="store_true",
        help=(
            "no_edge_hit rescue: skip cloud query refinement; use original query for bi-encoder top-K, "
            "cloud only picks in pool (saves one cloud call; refine on by default, see config code_search.cloud_rescue_refine)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel partitions: samples split evenly, partitions run concurrently, "
        "in-order within partition (asyncio in one loop; avoids threads vs shared Orchestrator)",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Ablation: skip the Ollama stage; go to cloud routing directly after CE shortlist",
    )
    parser.add_argument(
        "--no-bi-encoder",
        action="store_true",
        help="Ablation: replace the bi-encoder shortlist with a random pool (no edge retrieval)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "评测 JSON 输出目录（默认：/root/autodl-fs/code_search_eval，"
            "或 config code_search_eval.results_output）"
        ),
    )
    parser.add_argument(
        "--score-margin-threshold",
        type=float,
        default=None,
        help="部署可观测 Condition B：短名单 top1 与 topK 分数差低于该阈值则上云复核（默认取 config code_search.score_margin_threshold）",
    )
    parser.add_argument(
        "--oracle-routing",
        action="store_true",
        help="Oracle upper bound（仅评测对照用）：允许用 GT 缺席作为上云触发；主实验不要开",
    )
    args = parser.parse_args()

    config = _load_config_with_env("config/settings.yaml")
    asyncio.run(run_evaluation(args, config))

if __name__ == "__main__":
    main()
