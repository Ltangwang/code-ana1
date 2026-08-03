# Usage Guide & Common Scenarios

## Quick Start

### Basic Commands

```bash
# Single file analysis
python main.py analyze --file path/to/file.py

# Directory analysis  
python main.py analyze --dir src/

# With file pattern
python main.py analyze --dir src/ --pattern "**/*.java"

# Limit cloud calls
python main.py analyze --dir src/ --max-cloud-calls 10

# Export results
python main.py analyze --file test.py --output results.json

# Check budget
python main.py budget-status

# Health check providers
python main.py health-check
```

## Common Workflows

### Workflow 1: Analyzing a New Project

```bash
# Step 1: Start with small sample
python main.py analyze --file src/main.py

# Step 2: Check budget usage
python main.py budget-status

# Step 3: Adjust settings if needed
# Edit config/settings.yaml

# Step 4: Analyze full project
python main.py analyze --dir src/ --output analysis.json
```

### Workflow 2: CI/CD Integration

```bash
#!/bin/bash
# ci-analyze.sh

# Analyze only changed files
git diff --name-only HEAD~1 | grep '\.py$' > changed_files.txt

while read file; do
    python main.py analyze --file "$file" --max-cloud-calls 2
done < changed_files.txt

# Check for critical issues
if grep -q '"severity": "critical"' results.json; then
    echo "Critical issues found!"
    exit 1
fi
```

### Workflow 3: Incremental Analysis

```python
# analyze_incremental.py
import subprocess
import json
from pathlib import Path

def analyze_new_files(since_commit="HEAD~1"):
    """Analyze only files changed since commit."""
    # Get changed files
    result = subprocess.run(
        ["git", "diff", "--name-only", since_commit],
        capture_output=True, text=True
    )
    files = [f for f in result.stdout.split('\n') if f.endswith('.py')]
    
    # Analyze each
    all_results = {}
    for file in files:
        subprocess.run(["python", "main.py", "analyze", "--file", file])
    
    return all_results
```

## Configuration Scenarios

### Scenario 1: Cost-Optimized Setup

**Goal**: Minimize cloud costs

```yaml
# config/settings.yaml
strategy:
  base_cloud_threshold: 0.5  # Upload less (only confidence < 0.5)
  max_concurrent_cloud_calls: 2

cloud:
  openai:
    model: "gpt-3.5-turbo"  # 10x cheaper than GPT-4

budget:
  total_budget: 5.0  # Lower budget
```

**Expected**: ~5-10% cloud upload rate, $0.003/file

### Scenario 2: Quality-Optimized Setup

**Goal**: Maximum accuracy

```yaml
# config/settings.yaml
strategy:
  base_cloud_threshold: 0.7  # Upload more (confidence < 0.7)
  max_concurrent_cloud_calls: 5

cloud:
  openai:
    model: "gpt-4-turbo-preview"  # Best model

budget:
  total_budget: 50.0  # Higher budget
```

**Expected**: ~30-40% cloud upload rate, $0.015/file

### Scenario 3: Speed-Optimized Setup

**Goal**: Fastest analysis

```yaml
# config/settings.yaml
performance:
  local_batch_size: 20  # More parallel local calls
  max_concurrent_cloud_calls: 10  # More parallel cloud calls

strategy:
  base_cloud_threshold: 0.6  # Normal

ollama:
  model_name: "deepseek-coder:1.3b"  # Smaller, faster model
```

**Expected**: 2-3x faster, slight accuracy trade-off

### Scenario 4: Budget-Constrained Setup

**Goal**: Operate under strict budget

```yaml
# config/settings.yaml
budget:
  total_budget: 2.0
  daily_budget: 0.5  # Strict daily limit

strategy:
  base_cloud_threshold: 0.4  # Very selective
  always_upload_severity: ["critical"]  # Only critical
```

System will auto-adjust threshold as budget depletes.

## Tuning Parameters

### Confidence Threshold

Controls when to upload to cloud:

```yaml
# config/settings.yaml
strategy:
  base_cloud_threshold: 0.6
```

| Value | Upload Rate | Cost | Accuracy |
|-------|-------------|------|----------|
| 0.4 | ~10% | Low | Good |
| 0.6 | ~20% | Medium | Better |
| 0.8 | ~40% | High | Best |

**Rule of Thumb**: Start with 0.6, adjust based on budget/quality needs.

### Batch Sizes

Controls throughput:

```yaml
performance:
  local_batch_size: 5    # Ollama concurrent calls
  max_concurrent_cloud_calls: 3  # Cloud concurrent calls
```

| Local Batch | Cloud Batch | Speed | Memory |
|-------------|-------------|-------|--------|
| 5 | 3 | Normal | Low |
| 10 | 5 | Fast | Medium |
| 20 | 10 | Fastest | High |

**Constraint**: Don't exceed Ollama server capacity (typically 10-20)

### Model Selection

```yaml
ollama:
  model_name: "codellama:7b"  # Default

# Alternatives:
# "deepseek-coder:1.3b"  - Fastest, less accurate
# "deepseek-coder:6.7b"  - Balanced
# "codellama:13b"        - Best accuracy, slower
# "qwen2.5-coder:7b"     - Good balance
```

```yaml
cloud:
  openai:
    model: "gpt-4-turbo-preview"  # Default

# Alternatives:
# "gpt-3.5-turbo"       - 10x cheaper, good quality
# "gpt-4"               - Highest quality, expensive
# "gpt-4-turbo"         - Fast GPT-4, good value
```

## Output Formats

### Terminal Output

```
Found 3 potential issues:

test.py
╭──────┬──────────┬──────┬────────────────┬────────────┬──────────╮
│ Line │ Severity │ Type │ Description    │ Confidence │ Verified │
├──────┼──────────┼──────┼────────────────┼────────────┼──────────┤
│ 5    │ high     │ bug  │ Division by 0  │ 95%        │ ✓ Yes    │
│ 10   │ medium   │ bug  │ Index OOB      │ 85%        │ Local    │
╰──────┴──────────┴──────┴────────────────┴────────────┴──────────╯
```

### JSON Output

```json
{
  "timestamp": "2024-01-15T10:30:45",
  "files": {
    "test.py": [
      {
        "location": "test.py:5-8",
        "severity": "high",
        "type": "bug",
        "description": "Division by zero without check",
        "confidence": 0.95,
        "suggested_fix": "Add zero check: if b != 0: return a / b",
        "was_verified": true
      }
    ]
  }
}
```

### Programmatic Access

```python
from core.orchestrator import Orchestrator
import asyncio

async def main():
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    results = await orchestrator.analyze_file("test.py")
    
    for result in results:
        print(f"Issue at {result.location}")
        print(f"Confidence: {result.final_confidence}")
        print(f"Fix: {result.final_fix}")
    
    await orchestrator.shutdown()

asyncio.run(main())
```

## Troubleshooting

### Problem: Ollama Connection Failed

**Symptoms**:
```
Error: Ollama API error: Connection refused
```

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull required model
ollama pull codellama:7b
```

### Problem: High Cloud Costs

**Symptoms**: Budget depleting quickly

**Diagnosis**:
```bash
python main.py budget-status
# Check "Usage by Provider"
```

**Solutions**:
1. Increase threshold: `base_cloud_threshold: 0.5` → `0.4`
2. Use cheaper model: `gpt-4` → `gpt-3.5-turbo`
3. Lower daily budget: `daily_budget: 1.0`

### Problem: Low Accuracy

**Symptoms**: Many false positives/negatives

**Solutions**:
1. Decrease threshold: `base_cloud_threshold: 0.6` → `0.7`
2. Use better cloud model: `gpt-3.5-turbo` → `gpt-4-turbo`
3. Try different local model: `codellama:7b` → `deepseek-coder:6.7b`

### Problem: Slow Analysis

**Symptoms**: Taking too long per file

**Diagnosis**: Check which stage is slow
```bash
# Look for high latencies in logs
grep "latency" logs/analysis.log
```

**Solutions**:
1. Increase batch sizes
2. Use smaller local model: `codellama:13b` → `codellama:7b`
3. Reduce cloud calls: Increase threshold

### Problem: API Rate Limits

**Symptoms**:
```
Error: Rate limit exceeded
```

**Solution**:
```yaml
# config/settings.yaml
performance:
  cloud_rate_limit_per_minute: 10  # Reduce from 20
```

### Problem: Database Locked

**Symptoms**:
```
Error: database is locked
```

**Solution**:
```bash
# Only one process should access DB
# Kill other instances
ps aux | grep "python main.py"
kill <pid>
```

## Advanced Usage

### Custom Analysis Rules

Add to `edge/ast_analyzer.py`:

```python
def _detect_sql_injection(self, code: str) -> float:
    """Detect potential SQL injection."""
    has_concat = re.search(r'"\s*\+\s*\w+', code)
    has_execute = any(x in code for x in ['execute(', '.query('])
    return 0.9 if (has_concat and has_execute) else 0.0

# In _calculate_complexity_factors():
factors['sql_injection'] = self._detect_sql_injection(code)
```

Update weights in `config/settings.yaml`:
```yaml
ast:
  hotspot_weights:
    sql_injection: 0.15
```

### Custom Cloud Prompts

Edit `shared/prompts.py`:

```python
CUSTOM_VERIFICATION_PROMPT = """
You are a security expert. Review this code for SQL injection.

Code:
{code}

Detected Issue:
{description}

Respond in JSON:
{{
  "is_vulnerable": true/false,
  "severity": "critical/high/medium/low",
  "exploit_scenario": "...",
  "recommended_fix": "..."
}}
"""
```

### Integration with Other Tools

```python
# integrate_with_sonarqube.py
from core.orchestrator import Orchestrator
import requests

async def analyze_and_report(file_path):
    # Analyze with our tool
    orchestrator = Orchestrator(config)
    results = await orchestrator.analyze_file(file_path)
    
    # Convert to SonarQube format
    issues = convert_to_sonar_format(results)
    
    # Upload to SonarQube
    requests.post(
        "https://sonarqube.example.com/api/issues/import",
        json=issues
    )
```

## Best Practices

1. **Start Small**: Test on small subset before full project
2. **Monitor Budget**: Check `budget-status` regularly
3. **Tune Iteratively**: Adjust thresholds based on results
4. **Use Output Files**: Save results for comparison
5. **Track Metrics**: Monitor upload rate and costs
6. **Version Control Config**: Commit `config/settings.yaml` changes
7. **Document Changes**: Note why thresholds were adjusted

## Performance Benchmarks

Typical performance on standard hardware:

| Project Size | Files | Local Time | Cloud Calls | Total Time | Cost |
|--------------|-------|------------|-------------|------------|------|
| Small | 10 | 15s | 2 | 20s | $0.02 |
| Medium | 100 | 2min | 18 | 3min | $0.18 |
| Large | 1000 | 20min | 150 | 30min | $1.50 |

**Hardware**: M1 Mac, Ollama with codellama:7b, GPT-4-turbo

---

**For More**: See `README.md` for features, `architecture.md` for design details.

