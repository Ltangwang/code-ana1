# Coding Standards & Best Practices

## Python Style Guide

### General Rules

1. **PEP 8 Compliance**: Follow PEP 8 with these exceptions:
   - Line length: 100 characters (not 79)
   - Use `black` for auto-formatting

2. **Type Hints**: Required for all public functions
   ```python
   # Good
   async def analyze(self, code: str, language: CodeLanguage) -> AnalysisDraft:
       pass
   
   # Bad
   async def analyze(self, code, language):
       pass
   ```

3. **Docstrings**: Use Google style for classes and public methods
   ```python
   def calculate_score(factors: Dict[str, float]) -> float:
       """Calculate hotspot score from complexity factors.
       
       Args:
           factors: Dict mapping factor names to values (0.0-1.0)
       
       Returns:
           Weighted score between 0.0 and 1.0
       
       Example:
           >>> calculate_score({'complexity': 0.8, 'nesting': 0.6})
           0.7
       """
   ```

### Async/Await Patterns

**Rule**: ALL I/O operations must be async

```python
# Good - Async HTTP
async def call_api(self, url: str) -> Dict:
    async with self.session.get(url) as response:
        return await response.json()

# Bad - Blocking
def call_api(self, url: str) -> Dict:
    return requests.get(url).json()  # Blocks event loop!
```

**Rule**: Use `async with` for resource management

```python
# Good
async with OllamaInference(config) as inference:
    result = await inference.analyze(code)

# Implement __aenter__ and __aexit__
async def __aenter__(self):
    self._session = aiohttp.ClientSession()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self._session:
        await self._session.close()
```

**Rule**: Use `asyncio.gather` for parallel operations

```python
# Good - Parallel
results = await asyncio.gather(
    *[analyze_one(f) for f in fragments]
)

# Bad - Sequential
results = []
for f in fragments:
    results.append(await analyze_one(f))  # Slow!
```

**Rule**: Use `asyncio.Semaphore` for concurrency control

```python
# Limit to 3 concurrent cloud calls
sem = asyncio.Semaphore(3)

async def verify_one(draft):
    async with sem:
        return await cloud_client.verify(draft)
```

### Pydantic Models

**Rule**: Use Pydantic for all data that crosses module boundaries

```python
from pydantic import BaseModel, Field, validator

class CodeFragment(BaseModel):
    file_path: str = Field(..., description="Source file path")
    start_line: int = Field(..., ge=1)  # >= 1
    end_line: int = Field(..., ge=1)
    content: str
    
    @validator('end_line')
    def validate_range(cls, v, values):
        if 'start_line' in values and v < values['start_line']:
            raise ValueError('end_line must be >= start_line')
        return v
    
    class Config:
        frozen = False  # Allow mutation if needed
```

**Rule**: Use `Field()` for documentation and validation

```python
class ConfidenceScore(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    reasoning: str = Field(..., min_length=10)
```

### Error Handling

**Rule**: Catch specific exceptions, not bare `except`

```python
# Good
try:
    result = await api_call()
except aiohttp.ClientError as e:
    logger.error("api_error", error=str(e))
    return fallback_result
except asyncio.TimeoutError:
    logger.error("api_timeout")
    return None

# Bad
try:
    result = await api_call()
except:  # Don't do this!
    return None
```

**Rule**: Use `@retry` decorator for transient failures

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10)
)
async def call_cloud_api(self, prompt: str):
    # Automatically retries on exception
    # Waits 1s, 2s, 4s, ... up to 10s
    return await self._make_request(prompt)
```

**Rule**: Always log errors with context

```python
try:
    result = await process(data)
except Exception as e:
    logger.error(
        "processing_failed",
        error_type=type(e).__name__,
        error_message=str(e),
        data_id=data.id,
        stack_trace=traceback.format_exc()
    )
    raise  # Re-raise after logging
```

### Logging Standards

**Rule**: Use structured logging (structlog)

```python
import structlog
logger = structlog.get_logger(__name__)

# Good - Structured
logger.info(
    "upload_decision",
    fragment="test.py:42",
    should_upload=True,
    confidence=0.35,
    reason="Low confidence"
)

# Bad - String formatting
logger.info(f"Uploading test.py:42 (confidence: 0.35)")
```

**Rule**: Use standard event names (see `shared/logger.py`)

```python
# Standard events
"analysis_start"
"analysis_complete"
"local_score"
"upload_decision"
"cloud_verification"
"budget_update"
"error"
```

**Rule**: Include relevant context in every log

```python
# Always include:
# - What happened (event name)
# - Where (file/function)
# - Why (reason/error)
# - Metrics (latency, cost, etc.)

logger.info(
    "cloud_verification",
    draft_id="test.py:42",
    verified=True,
    latency_ms=1234,
    tokens=1500,
    cost=0.015
)
```

### Configuration Management

**Rule**: Use YAML for configuration, `.env` for secrets

```python
# config/settings.yaml
ollama:
  model_name: "codellama:7b"

# .env
OPENAI_API_KEY=sk-xxx
```

**Rule**: Load config once at startup

```python
# main.py
def load_config(path: str = "config/settings.yaml") -> Dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    load_dotenv()
    config = substitute_env_vars(config)  # ${VAR_NAME}
    
    return config
```

**Rule**: Pass config down, don't use globals

```python
# Good
class Analyzer:
    def __init__(self, config: Dict[str, Any]):
        self.threshold = config['threshold']

# Bad
GLOBAL_CONFIG = load_config()  # Don't do this

class Analyzer:
    def __init__(self):
        self.threshold = GLOBAL_CONFIG['threshold']
```

### Testing Standards

**Rule**: Use pytest with async support

```python
import pytest

@pytest.mark.asyncio
async def test_analyze_fragment():
    inference = OllamaInference(config)
    result = await inference.analyze_fragment(fragment)
    assert result.confidence.score > 0.0
```

**Rule**: Use fixtures for common setup

```python
@pytest.fixture
async def budget_controller():
    controller = BudgetController(
        total_budget=100.0,
        db_path=":memory:"  # In-memory for tests
    )
    await controller.initialize()
    yield controller
    # Cleanup if needed

@pytest.mark.asyncio
async def test_budget(budget_controller):
    await budget_controller.record_expense(5.0, "openai", "gpt-4")
    status = await budget_controller.get_status()
    assert status.used_budget == 5.0
```

**Rule**: Mock external APIs

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_cloud_call():
    with patch('cloud.client.AsyncOpenAI') as mock_client:
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=fake_response
        )
        
        client = CloudClient(config)
        result = await client.verify(draft)
        assert result.verified == True
```

### Code Organization

**Rule**: One class per file (unless tightly coupled)

```python
# Good
# edge/ast_analyzer.py
class ASTAnalyzer:
    pass

@dataclass
class Hotspot:  # OK - tightly coupled to ASTAnalyzer
    pass

# Bad - unrelated classes
class ASTAnalyzer:
    pass

class BudgetController:  # Should be in separate file
    pass
```

**Rule**: Group imports (stdlib, third-party, local)

```python
# Standard library
import asyncio
from datetime import datetime
from typing import List, Optional

# Third-party
import aiohttp
from pydantic import BaseModel
import structlog

# Local
from shared.schemas import CodeFragment, AnalysisDraft
from edge.confidence_scorer import ConfidenceScorer
```

**Rule**: Use `__all__` in `__init__.py`

```python
# edge/__init__.py
from .ast_analyzer import ASTAnalyzer, Hotspot
from .local_inference import OllamaInference

__all__ = [
    "ASTAnalyzer",
    "Hotspot",
    "OllamaInference",
]
```

### Performance Guidelines

**Rule**: Use generators for large sequences

```python
# Good - Memory efficient
def extract_functions(lines: List[str]):
    for i, line in enumerate(lines):
        if line.startswith('def '):
            yield extract_function_at(lines, i)

# Bad - Loads everything
def extract_functions(lines: List[str]):
    return [extract_function_at(lines, i) 
            for i, line in enumerate(lines) 
            if line.startswith('def ')]
```

**Rule**: Batch operations when possible

```python
# Good
drafts = await inference.analyze_batch(fragments, batch_size=5)

# Bad
drafts = [await inference.analyze_fragment(f) for f in fragments]
```

**Rule**: Cache expensive computations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_language_parser(language: CodeLanguage):
    # Expensive operation
    return Language.build_library(...)
```

### Security Guidelines

**Rule**: Never log secrets

```python
# Good
logger.info("api_call", provider="openai", model="gpt-4")

# Bad
logger.info("api_call", api_key=config['api_key'])  # Don't log keys!
```

**Rule**: Validate all external input

```python
# Good
class CodeFragment(BaseModel):
    file_path: str = Field(..., max_length=500)
    content: str = Field(..., max_length=100000)  # Limit size

# Bad
def analyze(file_path: str):
    with open(file_path) as f:  # No validation!
        return process(f.read())
```

**Rule**: Use parameterized queries (when using SQL)

```python
# Good
await db.execute(
    "INSERT INTO budget_usage (cost, provider) VALUES (?, ?)",
    (cost, provider)
)

# Bad
await db.execute(
    f"INSERT INTO budget_usage VALUES ({cost}, '{provider}')"
)  # SQL injection risk!
```

## Code Review Checklist

Before committing, verify:

- [ ] Type hints on all public functions
- [ ] Docstrings on classes and public methods
- [ ] All I/O is async
- [ ] Error handling with specific exceptions
- [ ] Structured logging with standard event names
- [ ] Tests for new functionality
- [ ] No secrets in code
- [ ] Config changes documented
- [ ] `pytest` passes
- [ ] `black` formatting applied
- [ ] `ruff` linter passes

## Common Mistakes to Avoid

1. **Blocking I/O in async code**
   ```python
   # Wrong
   async def analyze(self):
       time.sleep(1)  # Blocks!
   
   # Right
   async def analyze(self):
       await asyncio.sleep(1)
   ```

2. **Forgetting to await**
   ```python
   # Wrong
   result = process_async()  # Returns coroutine, not result!
   
   # Right
   result = await process_async()
   ```

3. **Catching too broadly**
   ```python
   # Wrong
   except Exception:  # Too broad
   
   # Right
   except (ValueError, KeyError) as e:
   ```

4. **Not validating config**
   ```python
   # Wrong
   threshold = config['threshold']  # May not exist
   
   # Right
   threshold = config.get('threshold', 0.6)  # Default value
   ```

5. **Modifying config at runtime**
   ```python
   # Wrong
   config['threshold'] = 0.5  # Mutates shared config
   
   # Right
   local_threshold = config['threshold']  # Copy value
   ```

## Tools

```bash
# Format code
black .

# Lint
ruff check .
ruff check --fix .

# Type check
mypy . --ignore-missing-imports

# Test
pytest
pytest --cov  # With coverage

# All at once
make format lint test
```

---

**Reference**: Follow patterns in existing code. When in doubt, check how similar functionality is implemented elsewhere in the codebase.

