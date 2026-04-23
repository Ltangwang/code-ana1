"""
CodeSearchNet 非 Java 评测用语言配置（与 evaluate_code_search_non_java.py 配套）。
lang_id 与 CodeSearchNet_clean_Dataset 下子目录名一致。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeSearchLangProfile:
    """一种语言的 prompt / 代码围栏配置。"""

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
    """生成各语言同质结构的 prompt（与 Java 版 evaluate 对齐，仅术语替换）。"""
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
If you are not confident (candidates too close, contradictory, or query-code mismatch), set needs_escalation to true so a stronger cloud model can rerank; you may still provide your best_guess index."""

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
}

NON_JAVA_LANG_IDS = frozenset(LANGUAGE_PROFILES.keys())
