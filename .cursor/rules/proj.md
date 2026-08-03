# Edge-Cloud Code Analysis - Main Project Rules

## Project Overview

This is an **Edge-Cloud Collaborative Code Analysis** system that uses:
- **Edge (Local)**: Ollama small models (1.3B-7B) for initial analysis
- **Cloud (Remote)**: GPT-4/Claude for verification of uncertain cases
- **Strategy**: Budget-aware decision making on what to upload

**Core Concept**: Inspired by Speculative Decoding - local model drafts, cloud model verifies.

## Quick Reference

| Component | File | Purpose |
|-----------|------|---------|
| Entry Point | `main.py` | CLI interface |
| Orchestrator | `core/orchestrator.py` | Main workflow coordinator |
| Edge Analysis | `edge/ast_analyzer.py`, `edge/local_inference.py` | Local code analysis |
| Cloud Verification | `cloud/client.py` | Cloud API calls |
| Strategy | `core/strategy_manager.py` | Upload decision logic |
| Budget | `core/budget_controller.py` | Cost tracking |
| Data Models | `shared/schemas.py` | All Pydantic models |

## Architecture Pattern

```
Input → AST Hotspots → Local LLM → Confidence Score → Strategy Decision
                                                              ↓
                                                    [High] → Direct Output
                                                    [Low]  → Cloud Verify → Final Result
                                                              ↑
                                                        Budget Monitor
```

## Key Implementation Rules

### 1. Data Flow (Always Follow This Order)

```python
# 1. Parse and identify hotspots
hotspots = ast_analyzer.analyze_file(file_path)

# 2. Local inference with confidence
drafts = await local_inference.analyze_batch(fragments)

# 3. Strategy decides upload
to_verify = strategy_manager.filter_drafts(drafts, budget_status)

# 4. Cloud verification (parallel)
verifications = await cloud_client.verify_batch(to_verify)

# 5. Combine results
results = combine_results(drafts, verifications)
```

### 2. Confidence Scoring (Must Include)

Every `AnalysisDraft` MUST have:
```python
confidence = ConfidenceScore(
    score=0.0-1.0,  # Required
    reasoning="Why this score",  # Required
    factors={  # Optional but recommended
        'model_score': 0.7,
        'code_complexity': 0.2
    }
)
```

### 3. Budget-Aware Decisions

Upload threshold adjusts based on budget:
```python
# Reference: core/strategy_manager.py line ~80
if budget.remaining_percent < 0.2:
    threshold = 0.3  # More strict
else:
    threshold = 0.6  # Normal
```

Always check budget before cloud calls:
```python
if not await budget_controller.can_afford(estimated_cost):
    return local_result_only
```

### 4. Async Everything

All I/O operations MUST be async:
```python
# Good
async def analyze(self, code: str) -> Result:
    async with self.session.post(...) as resp:
        return await resp.json()

# Bad - Blocks event loop
def analyze(self, code: str) -> Result:
    resp = requests.post(...)
    return resp.json()
```

### 5. Structured Logging

Use these exact log keys:
```python
logger.info(
    "upload_decision",
    fragment=location,
    should_upload=bool,
    reason=str,
    confidence=float,
    budget_remaining_pct=float
)
```

See `shared/logger.py` for all standard log functions.

## Module Interaction Contract

### Edge → Core
```python
# edge/local_inference.py provides
async def analyze_fragment(fragment: CodeFragment) -> AnalysisDraft
```

### Core → Cloud
```python
# cloud/client.py provides
async def verify(draft: AnalysisDraft, mode: str) -> VerificationResult
```

### Core → Strategy
```python
# core/strategy_manager.py provides
def should_upload(draft: AnalysisDraft, budget: BudgetStatus) -> UploadDecision
```

## Configuration Hierarchy

1. **Environment Variables** (`.env`) - Highest priority
   - API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
   
2. **YAML Config** (`config/settings.yaml`) - Default values
   - Models, thresholds, budgets
   
3. **CLI Arguments** - Override for single run
   - `--max-cloud-calls`, `--output`

## Error Handling Standards

```python
# Always wrap external calls
try:
    result = await external_api_call()
except aiohttp.ClientError as e:
    logger.error("api_call_failed", error=str(e))
    return fallback_result
except Exception as e:
    logger.error("unexpected_error", error=str(e))
    raise  # Re-raise unexpected errors
```

## Testing Requirements

Each module should have:
- Unit tests in `tests/test_<module>.py`
- Async tests use `@pytest.mark.asyncio`
- Mock external APIs (don't call real Ollama/OpenAI in tests)

Example:
```python
@pytest.mark.asyncio
async def test_budget_tracking(budget_controller):
    await budget_controller.record_expense(5.0, "openai", "gpt-4")
    status = await budget_controller.get_status()
    assert status.used_budget == 5.0
```

## Common Patterns

### Adding a New Analysis Rule

1. **Edge side** (`edge/ast_analyzer.py`):
```python
def _detect_new_pattern(self, code: str) -> float:
    # Return 0.0-1.0 score
    has_issue = "dangerous_pattern" in code
    return 0.8 if has_issue else 0.0
```

2. **Update complexity factors**:
```python
factors['new_pattern'] = self._detect_new_pattern(code)
```

3. **Cloud side** (`shared/prompts.py`):
```python
# Add to verification prompt
"Check for: ... and dangerous_pattern usage"
```

### Adding a New Provider

1. Update `config/settings.yaml`:
```yaml
cloud:
  new_provider:
    api_key: "${NEW_PROVIDER_KEY}"
    model: "model-name"
    base_url: "https://api.newprovider.com/v1"
```

2. Use existing `cloud/client.py` (OpenAI-compatible)
3. Or extend `CloudClient` class if needed

### Adjusting Strategy

Edit `config/thresholds.yaml`:
```yaml
upload:
  max_confidence_for_upload: 0.5  # Change from 0.6
```

## Performance Guidelines

- **Batch Size**: Default 5 for local, 3 for cloud
- **Timeouts**: 30s for Ollama, 60s for cloud
- **Max Concurrent**: 3 cloud calls (configurable)

Increase for faster processing:
```yaml
strategy:
  max_concurrent_cloud_calls: 10
performance:
  local_batch_size: 20
```

## File References

For detailed information, see:
- **Architecture**: `.cursor/rules/architecture.md`
- **Coding Standards**: `.cursor/rules/coding-standards.md`
- **Module Details**: `.cursor/rules/modules.md`
- **Testing Guide**: `.cursor/rules/testing.md`
- **Usage Examples**: `.cursor/rules/usage-guide.md`

## Quick Commands

```bash
# Analyze
python main.py analyze --file test.py

# Check budget
python main.py budget-status

# Health check
python main.py health-check

# Run tests
pytest
```

## When Making Changes

1. **Adding features**: Follow the data flow order (AST → Local → Strategy → Cloud)
2. **Changing thresholds**: Update `config/thresholds.yaml` first
3. **New prompts**: Add to `shared/prompts.py` with format methods
4. **New schemas**: Add to `shared/schemas.py` as Pydantic models
5. **Always**: Update tests and check `pytest` passes

---

**Last Updated**: Based on current implementation (v0.1.0)
**Core Philosophy**: Local first, cloud when needed, budget always in mind.
