"""
从 Python 源码字符串中去掉「块级」文档串，降低 NL(docstring) 与 code 侧字面重合。

仅用于训练/检索索引侧对 `code` 字段的预处理；query 仍使用 nl_query。
对无法解析的片段原样返回。
"""

from __future__ import annotations

import ast
from typing import List


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr):
        return False
    v = stmt.value
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return True
    return isinstance(v, ast.Str)


def _strip_leading_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
    if body and _is_docstring_expr(body[0]):
        return body[1:]
    return body


class _DocstringStripper(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = _strip_leading_docstring(node.body)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = _strip_leading_docstring(node.body)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.body = _strip_leading_docstring(node.body)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = _strip_leading_docstring(node.body)
        return self.generic_visit(node)


def strip_python_code_docstrings(code: str) -> str:
    """
    解析为 AST 后移除 module / class / function / async def 体首条 docstring，再 ast.unparse。
    解析失败或 unparse 过短则返回原文。
    """
    raw = (code or "").strip()
    if len(raw) < 8:
        return code
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return code
    try:
        new_tree = ast.fix_missing_locations(_DocstringStripper().visit(tree))
        out = ast.unparse(new_tree)
    except (AttributeError, TypeError, ValueError):
        return code
    out = out.strip()
    if len(out) < max(16, len(raw) // 10):
        return code
    return out
