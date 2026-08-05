#!/usr/bin/env bash
# Always-cloud rerank for 5 non-Java languages: bi-encoder Top-K -> cloud (no Ollama).
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-12}"
TOP_K="${TOP_K:-10}"

for lang in ruby javascript go php python; do
  echo "========== ALWAYS-CLOUD: ${lang} =========="
  python3 scripts/evaluate_code_search_non_java.py \
    --language "${lang}" \
    --top-k "${TOP_K}" \
    --workers "${WORKERS}" \
    --skip-ollama
  echo "========== DONE: ${lang} =========="
done
