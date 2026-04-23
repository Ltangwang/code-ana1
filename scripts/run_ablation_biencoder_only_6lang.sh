#!/usr/bin/env bash
# 消融：纯双塔（UniXcoder），无 Ollama / 无云端 API。
# 六种语言与 CSN 一致：Java + python/go/javascript/php/ruby（各跑独立进程，结果文件分开）。
#
# 用法：
#   bash scripts/run_ablation_biencoder_only_6lang.sh
# 可选环境变量：
#   EXTRA_ARGS="--sample 100"   # 仅测前 N 条，调试用；全量请不要设
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

echo "全部完成。请在各次运行输出的 results_code_search*.json 中查看 metrics。"
