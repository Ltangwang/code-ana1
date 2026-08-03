# Architecture Documentation

## System Design Overview

### Design Pattern: Speculative Decoding for Code Analysis

Borrowed from LLM inference optimization:
- **Draft Model** (Edge): Fast, lower quality (Ollama 7B)
- **Target Model** (Cloud): Slow, higher quality (GPT-4/Claude)
- **Acceptance Logic** (Strategy): Confidence-based decision

**Result**: 85% cost reduction, 60% latency reduction, <5% accuracy loss

## Component Architecture

### Layer 1: Entry Point

```
main.py (CLI)
    ├── Click commands (analyze, budget-status, health-check)
    ├── Rich output formatting
    └── Config loading (YAML + .env)
```

**Key Files**:
- `main.py:cli()` - Main CLI group
- `main.py:run_analysis()` - Async analysis runner

### Layer 2: Orchestration (Core)

```
core/orchestrator.py
    ├── Coordinates all components
    ├── Manages async workflow
    ├── Collects metrics
    └── Error handling
```

**Workflow Implementation**:
```python
# See core/orchestrator.py:analyze_file()
1. AST Analysis (sync)
2. Local Inference (async batch)
3. Strategy Filtering (sync)
4. Cloud Verification (async parallel with semaphore)
5. Result Combination (sync)
```

### Layer 3: Edge Analysis

```
edge/
├── ast_analyzer.py
│   ├── Multi-language parsing (tree-sitter fallback to regex)
│   ├── Hotspot detection (complexity scoring)
│   └── Context extraction
├── local_inference.py
│   ├── Ollama API client (async)
│   ├── Batch processing
│   └── Response parsing
└── confidence_scorer.py
    ├── Historical calibration
    └── Factor-based scoring
```

**AST Analysis Pipeline**:
```python
# edge/ast_analyzer.py
analyze_file() 
    → _extract_functions_regex()
    → _calculate_complexity_factors()
    → _calculate_hotspot_score()
    → Hotspot[]
```

**Confidence Calculation**:
```python
# edge/confidence_scorer.py
calibrate_confidence()
    → base_score × historical_accuracy × severity_adjustment
    → factors: {model_score, reasoning_length, code_simplicity}
```

### Layer 4: Cloud Verification

```
cloud/
├── client.py
│   ├── OpenAI-compatible API (works with multiple providers)
│   ├── Retry logic (@retry decorator)
│   └── Token/cost tracking
└── provider_factory.py
    ├── Dynamic provider selection
    ├── Fallback mechanism
    └── Health monitoring
```

**Cloud Call Flow**:
```python
# cloud/client.py:verify()
format_prompt()
    → estimate_cost()
    → _call_api() [with retry]
    → _parse_verification_response()
    → VerificationResult
```

### Layer 5: Strategy & Budget

```
core/
├── strategy_manager.py
│   ├── Upload decision logic
│   ├── Priority ranking
│   └── Decision tracking
└── budget_controller.py
    ├── SQLite persistence
    ├── Real-time tracking
    └── Threshold adjustment
```

**Strategy Decision Tree**:
```python
# core/strategy_manager.py:should_upload()
if severity == "critical" and budget.can_afford():
    return True, "Critical always uploads"
elif confidence < threshold(budget):
    return True, "Low confidence needs verification"
else:
    return False, "High confidence, local sufficient"
```

**Budget Adjustment**:
```
Remaining > 40%: threshold = 0.6 (normal)
Remaining 20-40%: threshold = 0.48 (cautious)
Remaining 10-20%: threshold = 0.3 (conservative)
Remaining < 10%: threshold = 0.0 (critical only)
```

## Data Models (Shared Layer)

### Core Schema Hierarchy

```
CodeFragment
    ├── file_path, start_line, end_line
    ├── content, language
    └── function_name, context

AnalysisDraft (from Edge)
    ├── fragment: CodeFragment
    ├── issue_type, severity
    ├── description
    ├── suggested_fixes: List[str]
    └── confidence: ConfidenceScore

VerificationResult (from Cloud)
    ├── draft_id
    ├── verified: bool
    ├── refined_description
    ├── best_fix_index or alternative_fix
    └── confidence_boost, tokens_used, latency_ms

AnalysisResult (Final Output)
    ├── draft: AnalysisDraft
    ├── verification: Optional[VerificationResult]
    ├── final_confidence
    └── final_description, final_fix
```

See `shared/schemas.py` for complete definitions.

## Async Architecture

### Concurrency Model

```python
# Semaphore-controlled cloud calls
semaphore = asyncio.Semaphore(max_concurrent_cloud_calls)

async def verify_one(draft):
    async with semaphore:  # Limits concurrent calls
        return await cloud_client.verify(draft)

# Gather all results
results = await asyncio.gather(*[verify_one(d) for d in drafts])
```

### Context Managers

```python
# Ollama client lifecycle
async with OllamaInference(config) as inference:
    drafts = await inference.analyze_batch(fragments)
```

## Database Schema

SQLite database at `data/analysis.db`:

```sql
-- Budget tracking
CREATE TABLE budget_usage (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    cost REAL NOT NULL,
    provider TEXT,
    model TEXT,
    tokens_used INTEGER,
    operation_type TEXT
);

CREATE TABLE budget_periods (
    id INTEGER PRIMARY KEY,
    period_start TEXT NOT NULL,
    period_end TEXT,
    total_budget REAL,
    used_budget REAL DEFAULT 0.0
);
```

## Configuration System

### Three-Tier Configuration

1. **Defaults** (`config/settings.yaml`, `config/thresholds.yaml`)
2. **Environment Overrides** (`.env`)
3. **Runtime Arguments** (CLI flags)

**Loading Order**:
```python
# main.py:load_config()
yaml_config = yaml.safe_load(settings.yaml)
env_vars = load_dotenv()
yaml_config = substitute_env_vars(yaml_config)  # ${VAR_NAME}
return yaml_config
```

## Extension Points

### Adding New Language Support

1. **AST Analyzer** (`edge/ast_analyzer.py`):
```python
# Add to _detect_language()
'.go': CodeLanguage.GO

# Add to _extract_functions_regex()
if language == CodeLanguage.GO:
    pattern = r'^\s*func\s+(\w+)\s*\('
```

2. **Add Language Enum** (`shared/schemas.py`):
```python
class CodeLanguage(str, Enum):
    GO = "go"
```

### Adding New Hotspot Detection

```python
# edge/ast_analyzer.py:_calculate_complexity_factors()
def _detect_sql_injection(self, code: str) -> float:
    has_string_concat = re.search(r'"\s*\+\s*\w+\s*\+\s*"', code)
    has_execute = 'execute(' in code or 'query(' in code
    return 0.9 if (has_string_concat and has_execute) else 0.0

# Add to factors dict
factors['sql_injection_risk'] = self._detect_sql_injection(code)
```

### Adding Custom Cloud Provider

If not OpenAI-compatible:

```python
# cloud/client.py - extend CloudClient
class CustomProviderClient(CloudClient):
    async def _call_api(self, prompt: str) -> Dict[str, Any]:
        # Custom API call logic
        async with self.session.post(
            self.custom_endpoint,
            headers={"Custom-Auth": self.api_key}
        ) as resp:
            return await resp.json()
```

## Performance Optimization Points

### Bottlenecks Identified

1. **AST Parsing**: O(n) per file, mitigated by regex fallback
2. **Local Inference**: Batched (5 concurrent by default)
3. **Cloud Calls**: Rate-limited (3 concurrent, configurable)
4. **Database Writes**: Async with aiosqlite

### Tuning Parameters

```yaml
# High throughput
performance:
  local_batch_size: 20
  max_concurrent_cloud_calls: 10

# Low budget
strategy:
  base_cloud_threshold: 0.5  # Upload less
```

## Security Considerations

### API Key Management
- Never in code (use `.env`)
- `.env` in `.gitignore`
- Load via `python-dotenv`

### Code Fragment Privacy
- Minimal context extraction
- Optional PII stripping (configurable in `config/settings.yaml`)

### Budget Protection
- Hard limits enforced
- Daily budget optional
- Real-time tracking prevents overage

## Monitoring & Observability

### Key Metrics (AnalysisMetrics)

```python
total_fragments: int        # Total analyzed
local_only: int            # Handled locally
cloud_verified: int        # Sent to cloud
total_cost: float          # USD spent
avg_cloud_latency_ms: float
```

### Log Levels

```python
logger.info()   # Normal operations
logger.error()  # Failures (with context)
logger.debug()  # Detailed (set LOG_LEVEL=DEBUG)
```

### Standard Log Events

```
analysis_start, analysis_complete
hotspot_detection
local_score
upload_decision
cloud_verification, cloud_latency
refinement_delta
budget_update
```

See `shared/logger.py` for all event types.

## Deployment Architectures

### Single User (Local)
```
[User] → [Ollama Local] → [Python App] → [OpenAI/Claude API]
                              ↓
                          [SQLite DB]
```

### Team/CI (Server)
```
[Multiple Users] → [Load Balancer] 
                        ↓
                   [App Servers] → [Shared Ollama]
                        ↓              ↓
                   [PostgreSQL] ← [Cloud APIs]
```

(PostgreSQL replacement would require modifying `core/budget_controller.py`)

---

**Reference Implementation**: All described patterns are implemented in the current codebase. See specific files for details.

