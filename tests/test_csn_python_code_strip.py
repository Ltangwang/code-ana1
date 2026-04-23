"""shared.csn_python_code_strip 单测。"""

from shared.csn_python_code_strip import strip_python_code_docstrings


def test_strip_function_docstring():
    src = '''def f(a, b):
    """adds two numbers"""
    return a + b
'''
    out = strip_python_code_docstrings(src)
    assert "adds two numbers" not in out
    assert "return a + b" in out


def test_strip_preserves_when_parse_fails():
    raw = "not valid python {{{"
    assert strip_python_code_docstrings(raw) == raw
