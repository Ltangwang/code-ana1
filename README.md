# code-ana1: Bi-encoder retrieval + Ollama + cloud code search pipeline

This repository implements **CodeSearchNet-style code retrieval**: a **bi-encoder** does dense recall over the full corpus, **Ollama** reranks and gates in the candidate pool, and a **cloud LLM** can rewrite queries, expand search, and rerank as a fallback when triggers or budget allow—trading off latency, cost, and quality.

**Repository:** <https://github.com/ltangwang/code-ana1>

A more detailed flow diagram is in `figures/code_search_pipeline.mmd` (view with [mermaid.live](https://mermaid.live) or a VS Code Mermaid extension).

## Core pipeline

Typical eval entry: `scripts/evaluate_code_search.py` (non-Java and per-language variants: `scripts/evaluate_code_search_non_java.py` and language-specific scripts).

1. **Bi-encoder retrieval:** Encode queries and the codebase with UniXcoder (or the configured encoder), build an index or load vector cache, and **Top-K** similarity rank each query.
2. **(Optional) Cross-encoder rerank:** Rerank Top-K to shrink the pool before the LLM (can be disabled or use bi-encoder + Ollama only, e.g. `--bi-ollama-only` / `--bi-ce-only`).
3. **Ollama:** Produce structured output on the pool (e.g. `best_candidate_index`, `needs_escalation`) as the **edge** reranker and gate.
4. **Cloud:** If GT is outside bi-encoder Top-K, trigger **query rewrite** / wider search, or cloud API **rerank/parse**; subject to config and budget.

Orchestration reuses **Ollama session**, **multi-cloud factory**, and **budget** in `core/orchestrator.py`; retrieval and bi-encoder logic live under **`scripts/`** and **`shared/`**.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) (local reranking)
- API keys for cloud branches when used (see `.env` and `settings.yaml`)
- GPU/VRAM depends on model and scale for bi-encoder and Cross-encoder

## Install

```powershell
git clone https://github.com/ltangwang/code-ana1.git
cd code-ana1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux / macOS: `source .venv/bin/activate`.

## Configuration

- **`config/settings.yaml`:** Create locally or copy from backup; **ignored by `.gitignore` by default** so machine paths and secrets are not committed.
- **`config/thresholds.yaml`:** Thresholds, may ship with the repo.
- **`.env`:** Cloud `API_KEY`, etc. (do not commit).

Eval scripts read `ollama`, `cloud`, `budget`, and related keys from `settings.yaml`; bi-encoder paths, cache dirs, and retrieval Top-K follow script args and config—see each script’s `--help`.

## Running code search eval

Language, data paths, Top-K, and whether to skip cloud depend on your local `settings` and datasets, for example:

```powershell
python scripts/evaluate_code_search.py --help
python scripts/evaluate_code_search_non_java.py --help
```

Typical flow: **prepare preprocessed CodeSearchNet** → **load or train UniXcoder retrieval checkpoint** → **run eval scripts** → results may go to `evaluation_runs/` (gitignored by default).

Dataset download: repo root `download_codesearchnet.py`; large data and vector caches are usually in `.gitignore`.

## Project layout (summary)

```
code-ana1/
├── scripts/
│   ├── evaluate_code_search.py      # Code search eval (bi-encoder + CE + Ollama + cloud)
│   ├── evaluate_code_search_non_java.py
│   ├── evaluate_code_search_*.py    # Per-language eval entrypoints
│   ├── csn_retriever.py / csn_data.py
│   ├── csn_ce_rerank.py             # Cross-encoder rerank
│   └── train_unixcoder_csn*.py      # Bi-encoder / retrieval training
├── core/                            # Orchestration and budget (shared with eval)
├── cloud/                           # Cloud provider clients
├── edge/                            # Local inference (Ollama), etc.
├── shared/                          # CSN paths, language profiles, schemas, etc.
├── figures/code_search_pipeline.mmd # Pipeline diagram
├── examples/code_search_smoke/      # Tiny Ruby JSONL fixture (format + smoke test; see its README)
├── config/
└── requirements.txt
```

## Other scripts

| Area | Scripts (summary) |
|------|---------------------|
| Clone detection | `evaluate_clone_detection.py` |
| BCB train / RAG | `train_unixcoder_bcb.py`, `bcb_rag.py` |

## Tests

```powershell
pytest
```

## License

MIT License

## Note

This repo is for **research and experiments**. Do not push `.env`, real keys, or `config/settings.yaml` with internal paths to a public remote. Bi-encoder caches (e.g. `csn_retriever_emb_*`), `evaluation_runs/`, etc. are ignored by default to avoid large files in git.
