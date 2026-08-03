# Deployment & Operations Guide

## Local Development Setup

### Prerequisites

- Python 3.9+
- Ollama (https://ollama.ai)
- Git

### Installation Steps

```bash
# 1. Clone repository
git clone <repo-url>
cd code-analyze

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup Ollama
ollama serve  # In separate terminal
ollama pull codellama:7b

# 5. Configure environment
cp config/.env.example .env
# Edit .env and add API keys

# 6. Test installation
python main.py health-check
python main.py analyze --file examples/sample_code/buggy_python.py
```

### Verify Installation

```bash
# Check all components
ollama list  # Should show codellama:7b
python main.py health-check  # Should show providers healthy
pytest  # Should pass all tests
```

## Configuration Management

### Environment Variables

Create `.env` file:
```bash
# Required
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Optional
OLLAMA_BASE_URL=http://localhost:11434
DATABASE_PATH=data/analysis.db
LOG_LEVEL=INFO
TOTAL_BUDGET=10.0
DAILY_BUDGET=2.0
```

### Configuration Files

```yaml
# config/settings.yaml - Main config
ollama:
  base_url: "${OLLAMA_BASE_URL}"
  model_name: "codellama:7b"
  timeout: 30

cloud:
  default_provider: "openai"
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-turbo-preview"

budget:
  total_budget: 10.0
  daily_budget: 2.0

strategy:
  base_cloud_threshold: 0.6
```

```yaml
# config/thresholds.yaml - Tuning parameters
confidence:
  high: 0.7
  medium: 0.4
  low: 0.0

complexity:
  cyclomatic:
    high: 20
    critical: 30
```

### Configuration Priority

1. CLI arguments (highest)
2. Environment variables
3. YAML config files (lowest)

## Production Deployment

### Single-Server Deployment

```
[Nginx] → [Gunicorn + FastAPI Wrapper] → [Python App] → [Ollama]
                                              ↓
                                         [PostgreSQL]
```

#### Step 1: Wrap with API

Create `api.py`:
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from core.orchestrator import Orchestrator

app = FastAPI()
orchestrator = None

@app.on_event("startup")
async def startup():
    global orchestrator
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()

@app.post("/analyze")
async def analyze(file_path: str, background_tasks: BackgroundTasks):
    results = await orchestrator.analyze_file(file_path)
    return {"results": [r.dict() for r in results]}

@app.get("/budget")
async def budget():
    status = await orchestrator.get_budget_status()
    return status.dict()
```

#### Step 2: Deploy with Gunicorn

```bash
# Install additional deps
pip install fastapi uvicorn gunicorn

# Run with Gunicorn
gunicorn api:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

#### Step 3: Nginx Configuration

```nginx
server {
    listen 80;
    server_name analysis.example.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Run application
CMD ["python", "main.py", "analyze", "--dir", "/input"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    command: serve
  
  analyzer:
    build: .
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./input:/input
      - ./output:/output
    command: >
      python main.py analyze 
      --dir /input 
      --output /output/results.json

volumes:
  ollama-data:
```

#### Usage

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f analyzer

# Stop
docker-compose down
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: code-analyzer
  template:
    metadata:
      labels:
        app: code-analyzer
    spec:
      containers:
      - name: analyzer
        image: code-analyzer:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        - name: OLLAMA_BASE_URL
          value: "http://ollama-service:11434"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
spec:
  selector:
    app: ollama
  ports:
  - port: 11434
```

## Database Management

### SQLite (Default)

Location: `data/analysis.db`

```bash
# Backup
cp data/analysis.db data/analysis.db.backup

# View data
sqlite3 data/analysis.db "SELECT * FROM budget_usage ORDER BY timestamp DESC LIMIT 10"

# Reset budget
sqlite3 data/analysis.db "DELETE FROM budget_usage"
```

### PostgreSQL (Production)

Modify `core/budget_controller.py`:

```python
# Replace aiosqlite with asyncpg
import asyncpg

class BudgetController:
    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            host='localhost',
            database='code_analysis',
            user='analyzer',
            password='...'
        )
```

Schema migration:
```sql
-- PostgreSQL version
CREATE TABLE budget_usage (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    cost DECIMAL(10, 6) NOT NULL,
    provider VARCHAR(50),
    model VARCHAR(100),
    tokens_used INTEGER,
    operation_type VARCHAR(50)
);

CREATE INDEX idx_budget_timestamp ON budget_usage(timestamp);
CREATE INDEX idx_budget_provider ON budget_usage(provider);
```

## Monitoring

### Health Checks

```bash
# Application health
python main.py health-check

# Ollama health
curl http://localhost:11434/api/tags

# Database health
sqlite3 data/analysis.db "SELECT 1"
```

### Metrics Collection

Use Prometheus + Grafana:

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

analysis_counter = Counter('analyses_total', 'Total analyses')
cloud_calls = Counter('cloud_calls_total', 'Cloud API calls')
latency = Histogram('analysis_latency_seconds', 'Analysis latency')
budget_remaining = Gauge('budget_remaining_dollars', 'Remaining budget')

# In orchestrator
analysis_counter.inc()
cloud_calls.inc()
with latency.time():
    results = await analyze()
budget_remaining.set(status.remaining_budget)
```

### Log Aggregation

Using ELK Stack:

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/logs/*.log
  json.keys_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

### Alerting

```yaml
# alertmanager.yml
route:
  receiver: 'team-email'
  
receivers:
- name: 'team-email'
  email_configs:
  - to: 'team@example.com'

# Alerts
groups:
- name: code-analyzer
  rules:
  - alert: HighBudgetUsage
    expr: budget_remaining_dollars < 2.0
    annotations:
      summary: "Budget critically low"
```

## Backup & Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
cp data/analysis.db backups/analysis_${DATE}.db

# Backup config
tar -czf backups/config_${DATE}.tar.gz config/ .env

# Backup logs
tar -czf backups/logs_${DATE}.tar.gz logs/

# Keep only last 30 days
find backups/ -mtime +30 -delete
```

### Recovery

```bash
# Restore database
cp backups/analysis_20240115.db data/analysis.db

# Restore config
tar -xzf backups/config_20240115.tar.gz

# Verify
python main.py budget-status
```

## Performance Tuning

### System Resources

Recommended:
- **CPU**: 4+ cores (for parallel processing)
- **Memory**: 8GB+ (Ollama needs 4-8GB depending on model)
- **Disk**: SSD recommended for database

### Ollama Optimization

```bash
# Set concurrent requests
export OLLAMA_NUM_PARALLEL=10

# Set GPU memory (if using GPU)
export OLLAMA_GPU_LAYERS=35
```

### Python Optimization

```python
# Use PyPy for 2-5x speedup
pypy3 -m pip install -r requirements.txt
pypy3 main.py analyze --dir src/
```

## Security Best Practices

### 1. API Key Management

- Never commit `.env` to git
- Use secrets manager in production (AWS Secrets Manager, HashiCorp Vault)
- Rotate keys regularly

```python
# Use secrets manager
import boto3

def get_api_key():
    client = boto3.client('secretsmanager')
    return client.get_secret_value(SecretId='openai-key')['SecretString']
```

### 2. Network Security

```yaml
# docker-compose with network isolation
services:
  analyzer:
    networks:
      - internal
  ollama:
    networks:
      - internal
networks:
  internal:
    internal: true
```

### 3. Input Validation

Already implemented in `shared/schemas.py`:
```python
class CodeFragment(BaseModel):
    file_path: str = Field(..., max_length=500)
    content: str = Field(..., max_length=100000)
```

### 4. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze():
    ...
```

## Troubleshooting Production Issues

### High Memory Usage

```bash
# Monitor memory
top -p $(pgrep -f "python main.py")

# Reduce batch sizes
# config/settings.yaml
performance:
  local_batch_size: 3
  max_concurrent_cloud_calls: 2
```

### Database Locks

```bash
# Check for multiple processes
ps aux | grep "python main.py"

# Ensure only one writer
# Or switch to PostgreSQL for concurrent access
```

### Ollama Crashes

```bash
# Check Ollama logs
journalctl -u ollama -f

# Restart Ollama
systemctl restart ollama

# Reduce model size
ollama pull codellama:7b  # Instead of :13b
```

## Maintenance

### Regular Tasks

**Daily**:
- Check budget status
- Review error logs
- Monitor disk space

**Weekly**:
- Backup database
- Review cloud costs
- Update dependencies

**Monthly**:
- Rotate logs
- Update models
- Review and tune thresholds

### Updates

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Update Ollama models
ollama pull codellama:7b

# Run tests
pytest
```

---

**For Production Support**: Document your specific deployment and keep runbooks updated.

