#!/usr/bin/env bash
# Oracle upper bound for 5 non-Java languages (Top-K=10, --oracle-routing).
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-12}"
TOP_K="${TOP_K:-10}"
MARGIN="${MARGIN:-0.08}"

for lang in ruby javascript go php python; do
  echo "========== ORACLE: ${lang} =========="
  python3 scripts/evaluate_code_search_non_java.py \
    --language "${lang}" \
    --top-k "${TOP_K}" \
    --workers "${WORKERS}" \
    --score-margin-threshold "${MARGIN}" \
    --oracle-routing
  echo "========== DONE: ${lang} =========="
done
