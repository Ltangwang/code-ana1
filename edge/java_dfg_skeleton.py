"""
Java 方法片段的近似 DFG 骨架：将赋值/返回等右侧的调用链序列化为文本，
例如：digest -> call:MessageDigest.getInstance -> call:update
（完整程序级 DFG 需 SSA/指针分析；此处服务于克隆检测提示词增强。）
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
    """从右侧表达式抽取按出现顺序的调用链 call:Qualifier.name 或 call:name。"""
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
    # 无调用：标量 / 变量引用
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

    # 局部声明：大致 Type name = rhs;（避免匹配过宽）
    for m in re.finditer(
        r"(?:^|[;{}])\s*[\w.<>\[\],\s]+\s+(\w+)\s*=\s*([^;]+);",
        s,
        re.MULTILINE,
    ):
        add(m.group(1), m.group(2))

    # 简单赋值：name = rhs;（排除 ==、<= 等）
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

    # 调用语句：receiver.method( ... );
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
    将近似数据流边序列化为单行/短文本，供 LLM 与源码对照。

    Args:
        java: Java 方法或片段源码
        max_edges: 最多保留的边数
        prefer_tree_sitter: 若 tree-sitter-java 可用且解析出边则优先使用
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
