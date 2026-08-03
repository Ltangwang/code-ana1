# Module-by-Module Reference

## Edge Module (`edge/`)

### ast_analyzer.py

**Purpose**: Parse code and identify suspicious "hotspots"

**Key Classes**:
```python
@dataclass
class Hotspot:
    fragment: CodeFragment
    hotspot_score: float  # 0.0-1.0
    complexity_factors: Dict[str, float]
    reason: str

class ASTAnalyzer:
    def analyze_file(file_path, language) -> List[Hotspot]
    def analyze_code(code, file_path, language) -> List[Hotspot]
```

**Complexity Factors** (see line ~180):
- `cyclomatic_complexity`: Decision points (if/for/while) / 10
- `nesting_depth`: Max brace/indent depth / 6
- `function_length`: Line count / 100
- `exception_handling_missing`: 1.0 if open/read without try
- `null_checks_missing`: 0.7 if access without null check
- `resource_management`: 0.8 if open without close

**Hotspot Score Formula**:
```python
score = Σ(factor_value × weight)
weights = {
    'cyclomatic_complexity': 0.3,
    'nesting_depth': 0.2,
    'function_length': 0.15,
    'exception_handling_missing': 0.15,
    'null_checks_missing': 0.1,
    'resource_management': 0.1
}
```

**Usage Example**:
```python
analyzer = ASTAnalyzer(config={'complexity': {...}})
hotspots = analyzer.analyze_file('test.py', CodeLanguage.PYTHON)
for h in hotspots:
    print(f"{h.fragment.get_location()}: {h.hotspot_score:.2f}")
```

**Extension**: Add detection for new patterns in `_calculate_complexity_factors()`

---

### local_inference.py

**Purpose**: Call Ollama API for local code analysis

**Key Class**:
```python
class OllamaInference:
    async def analyze_fragment(fragment, n_fixes=3) -> AnalysisDraft
    async def analyze_batch(fragments, batch_size=5) -> List[AnalysisDraft]
```

**Configuration** (from `config/settings.yaml`):
```yaml
ollama:
  base_url: "http://localhost:11434"
  model_name: "codellama:7b"
  timeout: 30
  temperature: 0.1
  max_tokens: 512
```

**API Call** (line ~92):
```python
payload = {
    "model": self.model_name,
    "prompt": formatted_prompt,
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 512}
}
response = await session.post(f"{base_url}/api/generate", json=payload)
```

**Response Parsing** (line ~130):
- Extracts JSON from model output
- Parses: `has_issue`, `issue_type`, `severity`, `description`, `suggested_fixes`, `confidence`
- Falls back to low-confidence draft on parse error

**Confidence Calculation** (line ~232):
```python
final_score = base_score * 0.8  # 80% model weight
final_score += min(reasoning_length / 50, 1.0) * 0.1  # 10% reasoning
final_score += max(0, 1.0 - code_lines / 100) * 0.1  # 10% simplicity
```

**Usage**:
```python
async with OllamaInference(config) as inference:
    draft = await inference.analyze_fragment(fragment)
    print(f"Confidence: {draft.confidence.score}")
```

---

### confidence_scorer.py

**Purpose**: Calibrate confidence scores using historical data

**Key Classes**:
```python
@dataclass
class HistoricalAccuracy:
    total_predictions: int
    correct_predictions: int
    false_positives: int
    false_negatives: int
    
    @property
    def accuracy(self) -> float

class ConfidenceScorer:
    def calibrate_confidence(draft, historical_data) -> ConfidenceScore
    def record_feedback(draft, was_correct: bool)
    def should_trust(draft, threshold=0.7) -> bool
```

**Calibration Formula** (line ~45):
```python
calibrated = base_score * (0.7 + 0.3 * historical_accuracy)
calibrated *= (0.8 + 0.2 * issue_type_accuracy)
calibrated *= severity_penalty[severity]  # 0.9 for critical, 1.1 for info
```

**Tracking**:
```python
scorer = ConfidenceScorer()
scorer.record_feedback(draft, was_correct=True)
stats = scorer.get_model_stats("codellama:7b")
# Use stats.accuracy for future calibration
```

---

## Cloud Module (`cloud/`)

### client.py

**Purpose**: Unified client for cloud LLM APIs (OpenAI-compatible)

**Key Class**:
```python
class CloudClient:
    async def verify(draft, mode="verification") -> VerificationResult
    async def select_best_fix(fragment, description, candidates) -> (int, str)
    def get_metrics() -> Dict[str, Any]
```

**Modes**:
- `"verification"`: Quick check, uses `CLOUD_VERIFICATION_PROMPT`
- `"refinement"`: Deep analysis, uses `CLOUD_REFINEMENT_PROMPT`

**Retry Logic** (line ~38):
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def verify(self, draft: AnalysisDraft, mode: str):
    # 3 attempts with exponential backoff
```

**API Call** (line ~129):
```python
response = await self.client.chat.completions.create(
    model=self.model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    temperature=0.1,
    max_tokens=500
)
```

**Cost Tracking** (line ~103):
```python
estimated_cost = estimate_cost(prompt, expected_tokens, model)
await budget_controller.record_expense(
    cost=estimated_cost,
    provider=self.provider,
    model=self.model,
    tokens_used=actual_tokens
)
```

---

### provider_factory.py

**Purpose**: Manage multiple cloud providers with fallback

**Key Class**:
```python
class ProviderFactory:
    def get_client(provider=None, fallback=True) -> CloudClient
    async def check_health(provider) -> bool
    async def check_all_health() -> Dict[str, bool]
    def get_available_providers() -> List[str]
```

**Fallback Order** (line ~62):
```python
fallback_order = [
    CloudProvider.OPENAI,
    CloudProvider.ANTHROPIC,
    CloudProvider.CUSTOM
]
```

**Usage**:
```python
factory = ProviderFactory(config['cloud'])
client = factory.get_client()  # Uses default
# If default fails, automatically tries next in fallback_order

# Manual switch
factory.switch_default_provider("anthropic")
```

---

## Core Module (`core/`)

### orchestrator.py

**Purpose**: Main workflow coordinator

**Key Class**:
```python
class Orchestrator:
    async def initialize()
    async def analyze_file(file_path, language) -> List[AnalysisResult]
    async def analyze_directory(dir_path, pattern) -> Dict[str, List[AnalysisResult]]
    async def get_metrics() -> AnalysisMetrics
    async def get_budget_status() -> BudgetStatus
```

**Main Workflow** (line ~85, `analyze_file()`):
```python
1. hotspots = ast_analyzer.analyze_file(file_path)
2. drafts = await local_inference.analyze_batch(fragments)
3. to_verify, decisions = strategy_manager.filter_drafts(drafts, budget)
4. verifications = await _verify_in_cloud(to_verify)  # Parallel
5. results = _combine_results(drafts, verifications)
```

**Concurrency Control** (line ~183):
```python
semaphore = asyncio.Semaphore(max_concurrent_cloud_calls)

async def verify_one(draft):
    async with semaphore:
        return await cloud_client.verify(draft)
```

**Initialization**:
```python
orchestrator = Orchestrator(config)
await orchestrator.initialize()  # Sets up budget DB, local inference
```

---

### strategy_manager.py

**Purpose**: Decide which drafts need cloud verification

**Key Class**:
```python
class StrategyManager:
    def should_upload(draft, budget_status) -> UploadDecision
    def filter_drafts(drafts, budget, max_uploads) -> (List[Draft], List[Decision])
    def get_decision_stats() -> Dict[str, Any]
```

**Decision Logic** (line ~47):
```python
# Rule 1: Always upload critical (if budget allows)
if severity == "critical" and budget.can_afford(0.01):
    return True, "Critical severity always requires verification"

# Rule 2: Low confidence below threshold
elif confidence < effective_threshold(budget):
    return True, "Low confidence below threshold"

# Rule 3: High confidence - no upload
else:
    return False, "High confidence - local analysis sufficient"

# Rule 4: Budget emergency override
if budget.remaining_percent < 0.1 and severity != "critical":
    return False, "Budget critical - only uploading critical issues"
```

**Threshold Adjustment** (line ~152):
```python
def _get_effective_threshold(budget: BudgetStatus) -> float:
    remaining = budget.remaining_percent
    if remaining < 0.1:    return 0.2   # Emergency
    elif remaining < 0.2:  return 0.3   # Low
    elif remaining < 0.4:  return 0.48  # Warning
    else:                  return 0.6   # Normal
```

**Priority Ranking** (line ~121):
```python
# Higher score = higher priority for upload
score = severity_priority[severity]  # critical:5, high:4, medium:3, low:2, info:1
score += (1.0 - confidence) * 3      # Lower confidence = higher priority
```

---

### budget_controller.py

**Purpose**: Track and manage API budget

**Key Class**:
```python
class BudgetController:
    async def initialize()
    async def record_expense(cost, provider, model, tokens_used, ...)
    async def get_status() -> BudgetStatus
    async def can_afford(estimated_cost) -> bool
    async def get_daily_usage() -> float
    async def get_usage_history(days=7) -> List[Dict]
    async def get_usage_by_provider() -> Dict[str, float]
```

**Database Schema** (line ~56):
```sql
CREATE TABLE budget_usage (
    timestamp TEXT,
    cost REAL,
    provider TEXT,
    model TEXT,
    tokens_used INTEGER,
    operation_type TEXT
);
```

**Recording Expense** (line ~135):
```python
await budget_controller.record_expense(
    cost=0.015,
    provider="openai",
    model="gpt-4-turbo",
    tokens_used=1500,
    operation_type="verification"
)
```

**Alert Trigger** (line ~174):
```python
if budget.remaining_percent < 0.2 and not self._alert_triggered:
    print(f"⚠️  LOW BUDGET ALERT: Only {remaining_percent:.1%} remaining!")
    self._alert_triggered = True
```

---

## Shared Module (`shared/`)

### schemas.py

**Purpose**: All Pydantic data models

**Key Models**:

```python
# Input
class CodeFragment(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: CodeLanguage
    function_name: Optional[str]

# Edge output
class AnalysisDraft(BaseModel):
    fragment: CodeFragment
    issue_type: IssueType
    severity: Severity
    description: str
    suggested_fixes: List[str]
    confidence: ConfidenceScore
    model_name: str

# Cloud output
class VerificationResult(BaseModel):
    draft_id: str
    verified: bool
    refined_description: Optional[str]
    best_fix_index: Optional[int]
    alternative_fix: Optional[str]
    confidence_boost: float
    cloud_model: str
    tokens_used: int
    latency_ms: float

# Final output
class AnalysisResult(BaseModel):
    draft: AnalysisDraft
    verification: Optional[VerificationResult]
    final_confidence: float
    final_description: str
    final_fix: Optional[str]
```

**Enums**:
- `CodeLanguage`: PYTHON, JAVA, JAVASCRIPT, TYPESCRIPT, CPP, C
- `IssueType`: BUG, SECURITY, PERFORMANCE, CODE_QUALITY, LOGIC_ERROR
- `Severity`: CRITICAL, HIGH, MEDIUM, LOW, INFO

---

### prompts.py

**Purpose**: LLM prompt templates and cost estimation

**Key Templates**:

1. **LOCAL_BUG_DETECTION_PROMPT** (line ~15): For Ollama models
   - Input: `{language}`, `{location}`, `{function_name}`, `{code}`
   - Output: JSON with issue details and confidence

2. **CLOUD_VERIFICATION_PROMPT** (line ~38): Quick verification
   - Input: Code + detected issue + candidate fixes
   - Output: JSON with verification result

3. **CLOUD_REFINEMENT_PROMPT** (line ~68): Deep analysis
   - Input: Code + low-confidence analysis
   - Output: JSON with refined analysis

**Usage**:
```python
prompt = PromptTemplates.format_local_prompt(fragment)
prompt = PromptTemplates.format_verification_prompt(draft)
prompt = PromptTemplates.format_refinement_prompt(draft)
```

**Cost Estimation** (line ~192):
```python
estimated_cost = estimate_cost(
    prompt="...",
    expected_response_tokens=300,
    model="gpt-4-turbo"
)
# Uses pricing table at line ~161
```

---

### logger.py

**Purpose**: Structured logging configuration

**Key Functions**:
```python
configure_logging(level="INFO", format="structured", output_dir="logs")
get_logger(name=__name__)

# Convenience functions
log_local_score(logger, location, score, reasoning)
log_upload_decision(logger, location, should_upload, reason, confidence, budget)
log_cloud_latency(logger, provider, model, latency_ms, tokens)
log_refinement_delta(logger, location, original_conf, final_conf, boost)
log_budget_update(logger, operation, cost, remaining, remaining_pct)
```

**Output Example**:
```json
{
  "event": "upload_decision",
  "fragment": "test.py:42-55",
  "upload": true,
  "reason": "Low confidence (0.35) below threshold",
  "confidence": 0.35,
  "budget_remaining_pct": 0.45,
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

---

## Main Entry (`main.py`)

**CLI Commands**:

```bash
python main.py analyze --file <path>
python main.py analyze --dir <path> --pattern "**/*.py"
python main.py budget-status
python main.py health-check
```

**Key Functions**:
- `cli()`: Click command group
- `run_analysis()`: Async wrapper for analysis
- `display_results()`: Rich table output
- `display_metrics()`: Show stats
- `export_results()`: Save to JSON

**Adding New Command**:
```python
@cli.command()
@click.option('--option', help='Description')
@click.pass_context
def new_command(ctx, option):
    """Command description."""
    config = ctx.obj['config']
    asyncio.run(async_handler(config, option))
```

---

**Quick Module Reference**:
- Parsing: `edge/ast_analyzer.py`
- Local AI: `edge/local_inference.py`
- Cloud AI: `cloud/client.py`
- Decisions: `core/strategy_manager.py`
- Money: `core/budget_controller.py`
- Glue: `core/orchestrator.py`

