# Testing Guide

## Testing Philosophy

- **Unit Tests**: Test individual components in isolation
- **Mock External Services**: Never call real Ollama/OpenAI in tests
- **Fast Tests**: Full suite should run in < 10 seconds
- **Deterministic**: Tests should pass consistently

## Test Structure

### Naming Convention

```
tests/
├── test_<module_name>.py
│   ├── test_<function_name>()
│   ├── test_<function_name>_<scenario>()
│   └── test_<class_name>_<method>()
```

Example:
```python
# tests/test_strategy_manager.py
def test_should_upload_low_confidence()
def test_should_upload_critical_severity()
def test_filter_drafts_respects_max_uploads()
```

### Test File Organization

```python
"""Tests for StrategyManager."""

import pytest
from core.strategy_manager import StrategyManager
from shared.schemas import AnalysisDraft, BudgetStatus, ...

# Fixtures
@pytest.fixture
def strategy_manager():
    """Create strategy manager instance."""
    return StrategyManager(config={...})

@pytest.fixture
def sample_draft():
    """Create sample analysis draft."""
    return AnalysisDraft(...)

# Tests
def test_basic_functionality(strategy_manager):
    """Test basic upload decision."""
    assert ...

def test_edge_case(strategy_manager, sample_draft):
    """Test edge case handling."""
    assert ...
```

## Pytest Configuration

See `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
asyncio_mode = auto  # Auto-detect async tests
addopts = -v --tb=short
markers =
    asyncio: async test
    integration: integration test
    slow: slow running test
```

## Writing Unit Tests

### Sync Tests

```python
def test_hotspot_scoring():
    """Test hotspot score calculation."""
    analyzer = ASTAnalyzer()
    
    factors = {
        'cyclomatic_complexity': 0.8,
        'nesting_depth': 0.6
    }
    
    score = analyzer._calculate_hotspot_score(factors)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # High complexity should yield high score
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_budget_tracking():
    """Test budget expense tracking."""
    controller = BudgetController(
        total_budget=100.0,
        db_path=":memory:"  # In-memory database
    )
    await controller.initialize()
    
    await controller.record_expense(5.0, "openai", "gpt-4")
    
    status = await controller.get_status()
    assert status.used_budget == 5.0
    assert status.remaining_budget == 95.0
```

### Using Fixtures

```python
@pytest.fixture
async def budget_controller():
    """Create budget controller with in-memory DB."""
    controller = BudgetController(
        total_budget=100.0,
        daily_budget=10.0,
        db_path=":memory:"
    )
    await controller.initialize()
    yield controller
    # Cleanup (if needed)

@pytest.mark.asyncio
async def test_can_afford(budget_controller):
    """Test affordability check."""
    assert await budget_controller.can_afford(50.0) is True
    assert await budget_controller.can_afford(150.0) is False
```

### Parametrized Tests

```python
@pytest.mark.parametrize("remaining_pct,expected_threshold", [
    (0.5, 0.6),   # Normal
    (0.3, 0.48),  # Warning
    (0.15, 0.3),  # Low
    (0.05, 0.2),  # Critical
])
def test_threshold_adjustment(remaining_pct, expected_threshold):
    """Test threshold adjusts based on budget."""
    budget = BudgetStatus(total_budget=100.0, used_budget=100.0*(1-remaining_pct))
    strategy = StrategyManager(config={...})
    
    threshold = strategy._get_effective_threshold(budget)
    assert threshold == expected_threshold
```

## Mocking External Services

### Mocking Ollama

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_local_inference():
    """Test Ollama inference with mocked API."""
    mock_response = {
        'response': '{"has_issue": true, "confidence": 0.8, ...}'
    }
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value=mock_response
        )
        
        inference = OllamaInference(config)
        async with inference:
            draft = await inference.analyze_fragment(fragment)
        
        assert draft.confidence.score == 0.8
```

### Mocking Cloud APIs

```python
from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_cloud_verification():
    """Test cloud verification with mocked OpenAI."""
    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(
            content='{"is_real_bug": true, "verification_confidence": 0.9}'
        ))
    ]
    fake_response.usage = MagicMock(total_tokens=1500)
    
    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=fake_response
        )
        mock_openai.return_value = mock_client
        
        client = CloudClient(config, provider="openai")
        result = await client.verify(draft)
        
        assert result.verified is True
        assert result.tokens_used == 1500
```

## Test Data Factories

Create reusable test data:

```python
# tests/conftest.py (shared fixtures)

@pytest.fixture
def sample_code_fragment():
    """Standard code fragment for testing."""
    return CodeFragment(
        file_path="test.py",
        start_line=10,
        end_line=20,
        content="def test(): pass",
        language=CodeLanguage.PYTHON
    )

@pytest.fixture
def high_confidence_draft(sample_code_fragment):
    """Draft with high confidence."""
    return AnalysisDraft(
        fragment=sample_code_fragment,
        issue_type=IssueType.BUG,
        severity=Severity.MEDIUM,
        description="Potential bug",
        suggested_fixes=["Fix 1"],
        confidence=ConfidenceScore(score=0.9, reasoning="Clear issue"),
        model_name="test-model"
    )

@pytest.fixture
def low_confidence_draft(sample_code_fragment):
    """Draft with low confidence."""
    draft = AnalysisDraft(...)
    draft.confidence.score = 0.3
    return draft
```

## Integration Tests

Mark with `@pytest.mark.integration`:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete analysis workflow (requires Ollama running)."""
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    results = await orchestrator.analyze_file("examples/sample_code/buggy_python.py")
    
    assert len(results) > 0
    assert all(r.final_confidence > 0 for r in results)
    
    await orchestrator.shutdown()
```

Run integration tests separately:
```bash
pytest -m integration  # Run only integration tests
pytest -m "not integration"  # Skip integration tests
```

## Testing Best Practices

### 1. One Assert Per Concept

```python
# Good
def test_budget_initialization():
    """Test budget initializes correctly."""
    assert status.total_budget == 100.0

def test_budget_starts_at_zero():
    """Test used budget starts at zero."""
    assert status.used_budget == 0.0

# Bad
def test_budget():
    assert status.total_budget == 100.0
    assert status.used_budget == 0.0
    assert status.remaining == 100.0
    # Too many assertions!
```

### 2. Descriptive Test Names

```python
# Good
def test_should_upload_when_confidence_below_threshold()
def test_should_not_upload_when_budget_exhausted()
def test_critical_severity_always_uploads()

# Bad
def test_upload()
def test_strategy()
def test_case_1()
```

### 3. Arrange-Act-Assert Pattern

```python
def test_expense_tracking():
    # Arrange
    controller = BudgetController(total_budget=100.0)
    
    # Act
    await controller.record_expense(25.0, "openai", "gpt-4")
    
    # Assert
    status = await controller.get_status()
    assert status.used_budget == 25.0
```

### 4. Test Edge Cases

```python
def test_empty_input():
    """Test behavior with empty input."""
    result = analyzer.analyze_code("", "test.py", CodeLanguage.PYTHON)
    assert result == []

def test_very_large_input():
    """Test behavior with large input."""
    large_code = "x = 1\n" * 10000
    result = analyzer.analyze_code(large_code, "test.py", CodeLanguage.PYTHON)
    assert isinstance(result, list)

def test_malformed_json_response():
    """Test handling of malformed API response."""
    mock_response = {'response': 'not json'}
    draft = inference._parse_response(mock_response, fragment)
    assert draft.confidence.score < 0.5  # Should have low confidence
```

### 5. Use Fixtures for Setup

```python
@pytest.fixture
def good_budget():
    """Budget with sufficient funds."""
    return BudgetStatus(total_budget=100.0, used_budget=20.0)

@pytest.fixture
def low_budget():
    """Budget with limited funds."""
    return BudgetStatus(total_budget=100.0, used_budget=85.0)

def test_normal_strategy(good_budget):
    """Test strategy with good budget."""
    # Use good_budget fixture
    
def test_conservative_strategy(low_budget):
    """Test strategy with low budget."""
    # Use low_budget fixture
```

## Coverage Requirements

Aim for:
- **Unit Tests**: 80%+ coverage
- **Integration Tests**: Cover critical paths
- **Edge Cases**: Test boundary conditions

Check coverage:
```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html
```

Focus coverage on:
- `core/` - Critical business logic
- `edge/` - Analysis algorithms
- `shared/schemas.py` - Data validation

Skip coverage for:
- `main.py` - CLI (tested manually)
- `tests/` - Test code itself

## Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_strategy_manager.py

# Specific test
pytest tests/test_strategy_manager.py::test_should_upload_low_confidence

# With verbose output
pytest -v

# With coverage
pytest --cov

# Skip slow tests
pytest -m "not slow"

# Fail fast (stop on first failure)
pytest -x
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Examples Reference

See existing tests:
- `tests/test_ast_analyzer.py` - Sync testing patterns
- `tests/test_strategy_manager.py` - Fixtures and parametrization
- `tests/test_budget_controller.py` - Async testing and database mocking

---

**Remember**: Good tests are fast, isolated, deterministic, and readable. When in doubt, write more tests rather than fewer.

