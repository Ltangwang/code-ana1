"""Tests for AST analyzer."""

import pytest
from edge.ast_analyzer import ASTAnalyzer, extract_minimal_context
from shared.schemas import CodeLanguage


def test_extract_minimal_context():
    """Test minimal context extraction."""
    code = """
def func1():
    pass

def func2():
    x = 1
    y = 2
    return x + y

def func3():
    pass
"""
    # Extract around line 6 (y = 2)
    context = extract_minimal_context(code, (6, 6), max_context_lines=2)
    assert "func2" in context
    assert "x = 1" in context
    assert "y = 2" in context
    assert "return x + y" in context


def test_language_detection():
    """Test language detection from file extension."""
    analyzer = ASTAnalyzer()
    
    assert analyzer._detect_language("test.py") == CodeLanguage.PYTHON
    assert analyzer._detect_language("test.java") == CodeLanguage.JAVA
    assert analyzer._detect_language("test.js") == CodeLanguage.JAVASCRIPT
    assert analyzer._detect_language("test.cpp") == CodeLanguage.CPP


def test_function_extraction_python():
    """Test Python function extraction."""
    code = """
def simple_func():
    return 42

def complex_func(a, b):
    if a > b:
        return a
    else:
        return b
"""
    
    analyzer = ASTAnalyzer()
    functions = analyzer._extract_functions_regex(code, CodeLanguage.PYTHON)
    
    assert len(functions) >= 2
    assert any("simple_func" in f['name'] for f in functions)
    assert any("complex_func" in f['name'] for f in functions)


def test_complexity_factors():
    """Test complexity factor calculation."""
    # Simple code
    simple_code = """
def simple():
    return 42
"""
    
    # Complex code
    complex_code = """
def complex(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                if i > 10:
                    return i
    return -1
"""
    
    analyzer = ASTAnalyzer()
    
    simple_factors = analyzer._calculate_complexity_factors(simple_code, CodeLanguage.PYTHON)
    complex_factors = analyzer._calculate_complexity_factors(complex_code, CodeLanguage.PYTHON)
    
    # Complex code should have higher cyclomatic complexity
    assert complex_factors['cyclomatic_complexity'] > simple_factors['cyclomatic_complexity']
    
    # Complex code should have deeper nesting
    assert complex_factors['nesting_depth'] > simple_factors['nesting_depth']


def test_hotspot_scoring():
    """Test hotspot score calculation."""
    analyzer = ASTAnalyzer()
    
    # High complexity factors should yield high score
    high_complexity = {
        'cyclomatic_complexity': 0.9,
        'nesting_depth': 0.8,
        'function_length': 0.7,
        'exception_handling_missing': 1.0,
        'null_checks_missing': 0.8,
        'resource_management': 0.9
    }
    
    # Low complexity factors should yield low score
    low_complexity = {
        'cyclomatic_complexity': 0.1,
        'nesting_depth': 0.1,
        'function_length': 0.2,
        'exception_handling_missing': 0.0,
        'null_checks_missing': 0.0,
        'resource_management': 0.0
    }
    
    high_score = analyzer._calculate_hotspot_score(high_complexity)
    low_score = analyzer._calculate_hotspot_score(low_complexity)
    
    assert high_score > low_score
    assert 0 <= high_score <= 1
    assert 0 <= low_score <= 1


def test_missing_exception_handling_detection():
    """Test detection of missing exception handling."""
    code_with_error = """
def read_file(filename):
    f = open(filename, 'r')
    content = f.read()
    return content
"""
    
    code_safe = """
def read_file(filename):
    try:
        f = open(filename, 'r')
        content = f.read()
        return content
    except Exception as e:
        return None
"""
    
    analyzer = ASTAnalyzer()
    
    factors_unsafe = analyzer._calculate_complexity_factors(code_with_error, CodeLanguage.PYTHON)
    factors_safe = analyzer._calculate_complexity_factors(code_safe, CodeLanguage.PYTHON)
    
    assert factors_unsafe['exception_handling_missing'] > factors_safe['exception_handling_missing']

