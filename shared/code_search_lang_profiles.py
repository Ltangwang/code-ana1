"""
Per-language config for non-Java CodeSearchNet eval (used with evaluate_code_search_non_java.py).
lang_id matches subdirs under CodeSearchNet_clean_Dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeSearchLangProfile:
    """Prompt and code-fence settings for one language."""

    lang_id: str
    code_fence: str
    display_name: str
    rerank_system: str
    no_edge_refine_system: str
    ollama_system: str
    refine_query_label: str
    ollama_dataset_context: str


def _p(
    lang_id: str,
    code_fence: str,
    display_name: str,
    doc_phrase: str,
    engineer: str,
    method_unit: str,
) -> CodeSearchLangProfile:
    """Build structurally similar prompts for each language (aligned with Java eval, terms only differ)."""
    rerank_system = f"""You are an expert Software Engineer and an intelligent Code Search Assistant.
Your task is to find the most relevant {display_name} code snippet for a given natural language query.
The dataset is CodeSearchNet, where queries are typically {doc_phrase}.
You will be provided with a user's search query and a list of candidate code snippets retrieved by a search engine.

Evaluation Criteria:
1. Check if the function/method name, parameter types, and return type (or equivalent) match the query's intent.
2. Focus on the code's control flow and core API calls to ensure it implements the requested functionality.
3. Select the ONE code snippet that best implements the functionality described in the query.
"""

    no_edge_refine = f"""You are an expert at CodeSearchNet-style {display_name} semantic code search.
The first retrieval pass (embedding search) failed to include the correct {method_unit} in its shortlist.
Your job is to rewrite the user's natural-language query into ONE concise search string that a dense code retriever can match better: keep {display_name}/API intent, {method_unit} role, parameters, return values, and key verbs; you may add synonyms or decompose {doc_phrase}.
Output only valid JSON, no markdown fences."""

    ollama_sys = f"""You are an expert {engineer} evaluating CodeSearchNet-style retrieval.
The user query is almost always {doc_phrase} in this dataset.
Each candidate is a {display_name} {method_unit} (or snippet) retrieved by embedding search and then filtered to this short list.
Your job: pick exactly ONE candidate index that best matches the documentation intent ({method_unit} role, parameters, return values, and main control flow / API usage).
Be strict: prefer signatures and behavior described in the query over superficial token overlap.
Do NOT default to index 0; compare all candidates and choose the truly best one, even if it is not the first. If index 0 is genuinely best, you may choose it, but justify briefly in your thinking.
Set needs_escalation to true ONLY if you cannot decide between any candidates at all; otherwise choose the best one and set needs_escalation to false."""

    refine_label = f"## Original query ({doc_phrase})\n"

    ds_ctx = f"- CodeSearchNet {display_name} split.\n- Query text ≈ {doc_phrase}."

    return CodeSearchLangProfile(
        lang_id=lang_id,
        code_fence=code_fence,
        display_name=display_name,
        rerank_system=rerank_system,
        no_edge_refine_system=no_edge_refine,
        ollama_system=ollama_sys,
        refine_query_label=refine_label,
        ollama_dataset_context=ds_ctx,
    )


LANGUAGE_PROFILES: dict[str, CodeSearchLangProfile] = {
    "go": _p(
        "go",
        "go",
        "Go",
        "the first sentence of a Go documentation comment (// or /* */ before the function)",
        "Go",
        "function",
    ),
    "javascript": _p(
        "javascript",
        "javascript",
        "JavaScript",
        "the first sentence of a JSDoc-style comment for the function",
        "JavaScript",
        "function",
    ),
    "php": _p(
        "php",
        "php",
        "PHP",
        "the first sentence of a PHPDoc block for the function",
        "PHP",
        "function",
    ),
    "python": _p(
        "python",
        "python",
        "Python",
        "the first sentence of the function's docstring",
        "Python",
        "function",
    ),
    "ruby": _p(
        "ruby",
        "ruby",
        "Ruby",
        "the first sentence of documentation above the method (e.g. RDoc-style)",
        "Ruby",
        "method",
    ),
    # --- 外部基准（Python 函数；复用 python 微调双塔） ---
    "advtest": CodeSearchLangProfile(
        lang_id="advtest",
        code_fence="python",
        display_name="Python",
        rerank_system="""You are an expert Software Engineer and an intelligent Code Search Assistant.
Your task is to find the most relevant Python code snippet for a given natural language query.
The dataset is AdvTest (CodeXGLUE NL-code-search-Adv), where queries are function docstrings and
candidate code has normalized identifier names (e.g. functions renamed to Func, arguments to arg_0).
You will be provided with a user's search query and a list of candidate code snippets retrieved by a search engine.

Evaluation Criteria:
1. Judge relevance by functionality and control flow, not by identifier names (they are normalized).
2. Check whether the code's core API calls and logic implement the behavior described in the query.
3. Select the ONE code snippet that best implements the functionality described in the query.
""",
        no_edge_refine_system="""You are an expert at AdvTest-style Python semantic code search.
The first retrieval pass (embedding search) failed to include the correct function in its shortlist.
Your job is to rewrite the user's natural-language query into ONE concise search string that a dense code retriever can match better: keep Python/API intent, function role, parameters, return values, and key verbs; you may add synonyms or decompose the docstring.
Output only valid JSON, no markdown fences.""",
        ollama_system="""You are an expert Python engineer evaluating AdvTest-style retrieval.
The user query is the docstring of the target function; candidate code has normalized identifiers (Func, arg_0, ...), so judge by behavior, not names.
Each candidate is a Python function (or snippet) retrieved by embedding search and then filtered to this short list.
Your job: pick exactly ONE candidate index that best matches the docstring intent (function role, parameters, return values, and main control flow / API usage).
Do NOT default to index 0; compare all candidates and choose the truly best one, even if it is not the first. If index 0 is genuinely best, you may choose it, but justify briefly in your thinking.
Set needs_escalation to true ONLY if you cannot decide between any candidates at all; otherwise choose the best one and set needs_escalation to false.""",
        refine_query_label="## Original query (function docstring)\n",
        ollama_dataset_context=(
            "- AdvTest (CodeXGLUE NL-code-search-Adv), Python functions.\n"
            "- Query text = the target function's docstring; identifiers in code are normalized (Func/arg_0)."
        ),
    ),
    "cosqa": CodeSearchLangProfile(
        lang_id="cosqa",
        code_fence="python",
        display_name="Python",
        rerank_system="""You are an expert Software Engineer and an intelligent Code Search Assistant.
Your task is to find the most relevant Python code snippet for a given natural language query.
The dataset is CoSQA, where queries are real web search queries about Python programming (Stack Overflow style, often short and keyword-like).
You will be provided with a user's search query and a list of candidate code snippets retrieved by a search engine.

Evaluation Criteria:
1. Check if the code accomplishes the task asked in the web query (functionality over wording).
2. Focus on the code's control flow and core API calls to ensure it implements the requested functionality.
3. Select the ONE code snippet that best implements the functionality described in the query.
""",
        no_edge_refine_system="""You are an expert at CoSQA-style Python semantic code search.
The first retrieval pass (embedding search) failed to include the correct function in its shortlist.
Your job is to rewrite the user's web-style natural-language query into ONE concise search string that a dense code retriever can match better: keep Python/API intent, the requested operation, input/output types, and key verbs; you may add synonyms.
Output only valid JSON, no markdown fences.""",
        ollama_system="""You are an expert Python engineer evaluating CoSQA-style retrieval.
The user query is a real web search query about Python programming (short, keyword-like, e.g. from Stack Overflow search).
Each candidate is a Python function (or snippet) retrieved by embedding search and then filtered to this short list.
Your job: pick exactly ONE candidate index that best accomplishes the task described in the query (requested operation, inputs/outputs, and main API usage).
Do NOT default to index 0; compare all candidates and choose the truly best one, even if it is not the first. If index 0 is genuinely best, you may choose it, but justify briefly in your thinking.
Set needs_escalation to true ONLY if you cannot decide between any candidates at all; otherwise choose the best one and set needs_escalation to false.""",
        refine_query_label="## Original query (web search query)\n",
        ollama_dataset_context=(
            "- CoSQA benchmark, Python functions.\n"
            "- Query text = a real web search query (short, keyword-like) about a Python programming task."
        ),
    ),
}

NON_JAVA_LANG_IDS = frozenset(LANGUAGE_PROFILES.keys())
