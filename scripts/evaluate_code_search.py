"""
Evaluate Code Search using an Edge-Cloud Architecture.

Fine-tuning (UniXcoder on CSN/BCB) is **not** invoked here: run ``scripts/train_unixcoder_csn.py`` or
``scripts/train_unixcoder_bcb.py`` manually. This script only loads weights from config (or HF
base checkpoints) and never starts training.

Edge: UniXcoder bi-encoder over the full CodeSearchNet corpus (``codebase.jsonl`` when present);
``retrieve_k = --top-k`` (same K as Success@K).
Local: Cross-Encoder is off by default; the Ollama/cloud pool is ``min(configured llm_pool_k, retrieve_k)``.
Cloud: If Ollama fails (empty/invalid) or sets ``needs_escalation``/``uncertain``; cloud success is
``cloud_success_after_fallback``. If the bi-encoder misses GT within ``retrieve_k`` (``no_edge_hit``),
a cloud-led rescue runs by default: (1) ``refined_search_query`` from the cloud, (2) bi-encoder
re-ranks with that string, top-``cloud_rescue_k`` pool, (3) cloud returns ``best_candidate_index``.
``--no-cloud-rescue-refine`` skips (1) and reuses the original query for (2). Malformed cloud JSON
may skip billing where implemented.
Metrics: Edge/Ollama/CE/Cloud Success@K all use K = ``--top-k``; combined edge-cloud Success@K and MRR
use the final pipeline rank (cloud > Ollama > edge as applicable).
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Add project root to sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from shared.autodl_env import apply_autodl_data_disk_env

apply_autodl_data_disk_env()

from core.orchestrator import Orchestrator
from shared.csn_paths import (
    _csn_test_jsonl_has_loadable_query,
    default_csn_clean_dataset_root,
    default_csn_java_dir_for_code_search,
    default_eval_models_parent,
)
from scripts.csn_data import load_csn_dataset
from scripts.csn_retriever import CSNRetriever
from scripts.csn_ce_rerank import load_csn_cross_encoder, rerank_candidates

_CODE_SEARCH_USE_CE = False


def _default_results_dir() -> Path:
    """Project-relative folder for run outputs (keeps repository root uncluttered)."""
    p = Path.cwd() / "evaluation_runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _next_results_code_search_path(out_dir: Path) -> Path:
    """Monotonic names under out_dir: first ``results_code_search.json``, then ``results_code_search1.json``, etc."""
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


# --- Cloud Reranking Prompt ---

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
    """Cloud rerank: full candidate code in the prompt (not truncated); very long inputs may hit provider token limits."""
    prompt = f"## User Search Query\n\"{query}\"\n\n## Candidate Code Snippets\n"
    for i, cand in enumerate(candidates):
        code = cand.get("code", "") or ""
        prompt += f"### Candidate {i}\n```java\n{code}\n```\n\n"
    
    prompt += """## Instructions
1. Output a <thinking> block where you briefly analyze how well each candidate matches the query based on method name, parameters, return type, and core logic.
2. Output your final decision as a JSON object matching this schema:
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


# --- Ollama deep rerank (CodeSearchNet Java) ---

CSN_OLLAMA_SYSTEM = """You are an expert Java engineer evaluating CodeSearchNet-style retrieval.
The user query is almost always the first sentence of a method's Javadoc in this dataset.
Each candidate is a Java method body (or snippet) retrieved by embedding search and then filtered to this short list.
Your job: pick exactly ONE candidate index that best matches the Javadoc intent (method role, parameters, return type, and main control flow / API usage).
Be strict: prefer signatures and behavior described in the query over superficial token overlap.
If you are not confident (candidates too close, contradictory, or query-code mismatch), set needs_escalation to true so a stronger cloud model can rerank; you may still provide your best_guess index."""

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
        "Set needs_escalation to true if you want a cloud rerank (uncertain or low confidence); "
        "false if you are confident in your choice."
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
    """0-based rank of GT in the pool. If GT is not in the pool, return -1; do not fake rank 1 (see same function in non_java)."""
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
    """Return a valid index from JSON if present; otherwise treat Ollama/model output as unusable."""
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
    """If needs_escalation / uncertain (etc.) is true, request cloud re-check."""
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

# --- JSON Parsing ---

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

# --- Evaluation Logic ---

def load_unixcoder_base(orchestrator: Orchestrator, config: dict) -> None:
    """Load UniXcoder base model for feature extraction."""
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
    """Split a list of (global index, sample) into num_parts chunks; at most as many parts as samples."""
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


async def _no_edge_cloud_rescue(
    idx: int,
    nl_query: str,
    ground_truth_url: str,
    orchestrator: Orchestrator,
    config: dict,
    pl: Dict[str, Any],
    query_max_length: int,
    search_lock: Optional[asyncio.Lock],
) -> Dict[str, Any]:
    """
    If GT is not in the first retrieve_k results, treat the first retrieval as failed; the cloud leads a "redo" of the query (no Ollama/CE).

    Default (cloud_rescue_refine=True):
      1) Cloud produces refined_search_query from the original query (text for the second retrieval);
      2) Local bi-encoder uses only that string to recall top cloud_rescue_k candidates from the full index (the pool);
      3) Cloud outputs best_candidate_index on that pool (final pick, not the bi-encoder order).

    If refine is off (--no-cloud-rescue-refine): skip step 1, still build the pool with the original query; only step 3 is cloud rerank.
    """
    rescue_k = max(int(pl.get("cloud_rescue_k", 50)), int(pl.get("retrieve_k", 10)))
    use_refine = bool(pl.get("cloud_rescue_refine", True))

    arb = config.get("clone_detection", {}).get("cloud_arbitration", {})
    est_cloud_cost = float(arb.get("estimated_cost_usd", 0.002))
    rounds = 2 if use_refine else 1
    if not await orchestrator.budget_controller.can_afford(est_cloud_cost * rounds):
        return {
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
        }

    cloud_client = orchestrator.cloud_factory.get_client()
    cloud_rank = -1
    cloud_verified = False
    cloud_fallback_reason = "no_edge_rescue_pending"

    search_query_for_pool = nl_query
    if use_refine:
        try:
            r0 = await cloud_client._call_api(
                _build_no_edge_refine_prompt(nl_query),
                system_prompt=CSN_NO_EDGE_REFINE_SYSTEM,
                max_tokens=512,
                json_response_format=False,
            )
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
            print(f"  analyze_query {idx} no_edge cloud query-refine call failed, using original query: {e}")
            search_query_for_pool = nl_query

    def _run_rescue_pool() -> List[Dict[str, Any]]:
        return orchestrator.csn_retriever.search(
            orchestrator,
            search_query_for_pool,
            top_k=rescue_k,
            max_length=query_max_length,
        )

    if search_lock is not None:
        async with search_lock:
            rescue_pool = await asyncio.to_thread(_run_rescue_pool)
    else:
        rescue_pool = await asyncio.to_thread(_run_rescue_pool)

    gt_in_pool = _ground_truth_index(rescue_pool, ground_truth_url)

    try:
        response = await cloud_client._call_api(
            _build_rerank_prompt(nl_query, rescue_pool),
            system_prompt=CSN_RERANK_SYSTEM,
            max_tokens=1024,
            json_response_format=False,
        )
        content = response.get("content", "")
        parsed = extract_json_from_text(content)
        best_c = _valid_best_candidate_index(parsed, len(rescue_pool))
        if best_c is None:
            print(f"  analyze_query {idx}: no_edge rescue cloud returned no valid best_candidate_index")
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

    return {
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
    }


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

    edge_hit = False
    edge_rank = -1
    try:
        nl_query = query_item["nl_query"]
        ground_truth_url = query_item["url"]

        def _sync_search_only() -> List[Dict[str, Any]]:
            return orchestrator.csn_retriever.search(
                orchestrator, nl_query, top_k=retrieve_k, max_length=query_max_length
            )

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
            if search_lock is not None:
                async with search_lock:
                    candidates_wide = await asyncio.to_thread(_sync_search_only)
            else:
                candidates_wide = await asyncio.to_thread(_sync_search_only)
            edge_rank = _ground_truth_index(candidates_wide, ground_truth_url)
            edge_hit = edge_rank >= 0
            if not edge_hit:
                return _empty_stages(idx, False, -1, "no_edge_hit")
            return _empty_stages(idx, True, edge_rank, "skip_cloud")

        if search_lock is not None:
            async with search_lock:
                candidates_wide = await asyncio.to_thread(_sync_bi_topk)
        else:
            candidates_wide = await asyncio.to_thread(_sync_bi_topk)

        edge_rank = _ground_truth_index(candidates_wide, ground_truth_url)
        edge_hit = edge_rank >= 0

        if not edge_hit:
            if pl.get("bi_ce_only") or pl.get("bi_ollama_only"):
                return _empty_stages(idx, False, -1, "no_edge_hit")
            if not pl.get("enable_cloud_rescue", True):
                return _empty_stages(idx, False, -1, "no_edge_hit")
            return await _no_edge_cloud_rescue(
                idx,
                nl_query,
                ground_truth_url,
                orchestrator,
                config,
                pl,
                query_max_length,
                search_lock,
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
            return {
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

        li = orchestrator.local_inference
        if li is None:
            raise RuntimeError("local_inference is not initialized")

        ollama_rank = -1
        ollama_verified = False
        ollama_ok = False
        pre_cloud_trigger = "none"

        try:
            ollama_text = await li.generate_text(
                _build_ollama_rerank_prompt(nl_query, pool),
                system=CSN_OLLAMA_SYSTEM,
                max_tokens=ollama_deep_max_tokens,
                timeout_sec=ollama_deep_timeout,
            )
            if not (ollama_text and ollama_text.strip()):
                print(f"  analyze_query {idx}: Ollama empty response, triggering cloud rerank")
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
                        f"  analyze_query {idx}: Ollama needs_escalation=true, triggering cloud rerank"
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
                        f"  analyze_query {idx}: Ollama has no valid best_candidate_index, triggering cloud rerank"
                    )
        except Exception as e:
            print(f"  analyze_query {idx} Ollama error: {e}, triggering cloud rerank")
            pre_cloud_trigger = "ollama_exception"

        if pl.get("bi_ollama_only"):
            _cfr = "bi_ollama_only" if ollama_ok else f"bi_ollama_only_{pre_cloud_trigger}"
            return {
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
            }

        cloud_rank = -1
        cloud_verified = False
        cloud_fallback_reason = "none"

        force_cloud = bool(pl.get("force_cloud", False))
        if ollama_ok and not force_cloud:
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
                    response = await cloud_client._call_api(
                        cloud_prompt,
                        system_prompt=CSN_RERANK_SYSTEM,
                        max_tokens=1024,
                        json_response_format=False,
                    )
                    content = response.get("content", "")
                    parsed = extract_json_from_text(content)
                    best_c = _valid_best_candidate_index(parsed, len(pool))
                    if best_c is None:
                        print(
                            f"  analyze_query {idx}: cloud returned no valid best_candidate_index, no charge"
                        )
                        cloud_fallback_reason = "cloud_invalid_parse"
                        cloud_verified = False
                    else:
                        cloud_rank = _llm_stage_rank(
                            pool, best_c, ground_truth_url, ce_rank
                        )
                        cloud_verified = True
                        cloud_fallback_reason = "cloud_success_after_fallback"
                        await orchestrator.budget_controller.record_expense(
                            est_cloud_cost,
                            orchestrator.cloud_factory.default_provider,
                            cloud_client.model,
                            int(response.get("tokens") or 0),
                            "code_search_rerank",
                            details=f"query={idx}",
                        )
                except Exception as e:
                    print(f"  analyze_query {idx} cloud error: {e}")
                    cloud_fallback_reason = "cloud_api_error"

        return {
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
        }

    except Exception as e:
        print(f"  analyze_query {idx} error: {e}")
        return {
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
    if r.get("cloud_verified"):
        return int(r.get("cloud_rank", -1))
    if r.get("ollama_verified") and int(r.get("ollama_rank", -1)) >= 0:
        return int(r["ollama_rank"])
    if r.get("ollama_ok") and int(r.get("ollama_rank", -1)) >= 0:
        return int(r["ollama_rank"])
    return -1


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
                f"  For protocol/paper alignment, use {clean_java} or run: python scripts/prepare_csn_graphcodebert_clean.py\n"
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
        results_path = _next_results_code_search_path(_default_results_dir())
        metrics_out["results_path"] = str(results_path.resolve())
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics_out, "details": results}, f, indent=2)
        print(f"\nWrote results to: {results_path.resolve()}")
            
    finally:
        await orchestrator.shutdown()

def _load_config_with_env(config_path: str = "config/settings.yaml") -> dict:
    """Same as main.load_config: load YAML and expand ${ENV} placeholders in cloud.*.api_key."""
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


# When not set on the CLI, align --top-k / --llm-pool-k / --cloud-rescue-k to the same K (one K for edge and cloud)
_JAVA_EVAL_TOP_K = 1


def _has_long_opt_java(argv: list[str], name: str) -> bool:
    if name in argv:
        return True
    return any(a.startswith(name + "=") for a in argv)


def _inject_java_eval_k_defaults() -> None:
    """When not set on the CLI, align top-k, cloud/edge pool, and cloud-rescue K to 1 for Java eval."""
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
    args = parser.parse_args()

    config = _load_config_with_env("config/settings.yaml")
    asyncio.run(run_evaluation(args, config))

if __name__ == "__main__":
    main()
