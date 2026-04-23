"""
Approximate DFG-style skeleton for Java method snippets: serialize RHS call chains as text, e.g.
digest -> call:MessageDigest.getInstance -> call:update
(Full program DFG needs SSA/pointer analysis; this augments clone-detection prompts.)
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional


def _strip_java_comments(code: str) -> str:
    if not code:
        return ""
    s = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def _rhs_to_call_chain(rhs: str) -> str:
    """Extract call chain call:Qualifier.name or call:name in occurrence order from RHS."""
    rhs = (rhs or "").strip()
    if not rhs:
        return ""
    parts: List[str] = []
    for m in re.finditer(
        r"(?:([\w]+)\s*\.\s*)?([\w]+)\s*\(",
        rhs,
    ):
        qual, name = m.group(1), m.group(2)
        if name in ("if", "while", "for", "switch", "catch", "new"):
            continue
        if qual:
            parts.append(f"call:{qual}.{name}")
        else:
            parts.append(f"call:{name}")
    if parts:
        return " -> ".join(parts)
    # No calls: scalar / variable ref
    v = re.sub(r"\s+", " ", rhs)[:48]
    return f"val:{v}" if v else ""


def _dfg_edges_regex(java: str, max_edges: int) -> List[str]:
    s = _strip_java_comments(java)
    edges: List[str] = []

    def add(lhs: str, rhs: str) -> None:
        if len(edges) >= max_edges:
            return
        lhs = (lhs or "").strip()
        chain = _rhs_to_call_chain(rhs)
        if lhs and chain:
            edges.append(f"{lhs} -> {chain}")

    # Local decl: roughly Type name = rhs; (avoid over-matching)
    for m in re.finditer(
        r"(?:^|[;{}])\s*[\w.<>\[\],\s]+\s+(\w+)\s*=\s*([^;]+);",
        s,
        re.MULTILINE,
    ):
        add(m.group(1), m.group(2))

    # Simple assign: name = rhs; (exclude ==, <=, etc.)
    for m in re.finditer(
        r"(?:^|[;{}])\s*(\w+)\s*=\s*([^;]+);",
        s,
        re.MULTILINE,
    ):
        rhs = m.group(2)
        if re.match(r"^\s*=", rhs):
            continue
        add(m.group(1), rhs)

    for m in re.finditer(r"\breturn\s+([^;]+);", s):
        chain = _rhs_to_call_chain(m.group(1))
        if chain:
            edges.append(f"return -> {chain}")
        if len(edges) >= max_edges:
            break

    # Call statement: receiver.method( ... );
    for m in re.finditer(r"(?:^|[;{}])\s*([\w]+)\.(\w+)\s*\(", s, re.MULTILINE):
        if len(edges) >= max_edges:
            break
        recv, meth = m.group(1), m.group(2)
        if meth in ("class", "if", "while", "for", "switch", "catch", "new"):
            continue
        edge = f"{recv} -> call:{meth}"
        if not edges or edges[-1] != edge:
            edges.append(edge)

    return edges[:max_edges]


def _walk_tree_sitter_dfg(source: str, max_edges: int) -> Optional[List[str]]:
    try:
        import tree_sitter_java as tsjava  # type: ignore
        from tree_sitter import Language
    except ImportError:
        return None
    try:
        from tree_sitter import Parser
    except ImportError:
        return None

    try:
        lang = Language(tsjava.language())
        parser = Parser(lang)
    except Exception:
        return None

    data = source.encode("utf-8")
    tree = parser.parse(data)
    edges: List[str] = []

    def text(n) -> str:
        return data[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")

    def visit(node) -> None:
        if len(edges) >= max_edges:
            return
        t = node.type
        if t == "assignment_expression":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None and right is not None:
                lhs = text(left).strip()
                rhs = text(right)
                chain = _rhs_to_call_chain(rhs)
                if lhs and chain:
                    edges.append(f"{lhs} -> {chain}")
        elif t == "variable_declarator":
            name = node.child_by_field_name("name")
            val = node.child_by_field_name("value")
            if name is not None and val is not None:
                lhs = text(name).strip()
                chain = _rhs_to_call_chain(text(val))
                if lhs and chain:
                    edges.append(f"{lhs} -> {chain}")
        elif t == "return_statement":
            for ch in node.children:
                if ch.type not in ("return", ";"):
                    chain = _rhs_to_call_chain(text(ch))
                    if chain:
                        edges.append(f"return -> {chain}")
                    break
        for ch in node.children:
            visit(ch)

    try:
        visit(tree.root_node)
    except Exception:
        return None
    return edges[:max_edges] if edges else None


def extract_java_dfg_skeleton(
    java: str,
    max_edges: int = 48,
    prefer_tree_sitter: bool = True,
) -> str:
    """
    Serialize approximate data-flow edges to one line / short text for LLM vs source.

    Args:
        java: Java method or snippet source
        max_edges: max edges to keep
        prefer_tree_sitter: prefer tree-sitter-java edges when available
    """
    if not (java or "").strip():
        return "(empty snippet)"

    edges: List[str] = []
    if prefer_tree_sitter:
        ts_edges = _walk_tree_sitter_dfg(java, max_edges)
        if ts_edges:
            edges = ts_edges

    if not edges:
        edges = _dfg_edges_regex(java, max_edges)

    if not edges:
        return "(no data-flow edges inferred)"

    return " | ".join(edges)
