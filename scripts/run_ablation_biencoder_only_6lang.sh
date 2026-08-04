#!/usr/bin/env bash
# Ablation: bi-encoder (UniXcoder) only, no Ollama / no cloud API.
# Six languages as in CSN: Java + python/go/javascript/php/ruby (separate processes, separate result files).
#
# Usage:
#   bash scripts/run_ablation_biencoder_only_6lang.sh
# Optional env:
#   EXTRA_ARGS="--sample 100"   # first N queries only, for debugging; omit for full runs
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOP_K="${TOP_K:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

COMMON=(--skip-cloud --top-k "$TOP_K" $EXTRA_ARGS)

echo "=== [1/6] Java: evaluate_code_search.py ${COMMON[*]} ==="
python scripts/evaluate_code_search.py "${COMMON[@]}"

for lang in python go javascript php ruby; do
  echo "=== [lang] $lang: evaluate_code_search_non_java.py --language $lang ${COMMON[*]} ==="
  python scripts/evaluate_code_search_non_java.py --language "$lang" "${COMMON[@]}"
done

echo "All done. See metrics in each run's results_code_search*.json outputs."
