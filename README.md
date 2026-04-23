# code-ana1: Bi-encoder retrieval + Ollama + cloud code search

This repository implements **CodeSearchNet-style code retrieval**: a **bi-encoder** recalls candidates from the full corpus, **Ollama** reranks and gates on the candidate pool, and a **cloud LLM** can rewrite queries, widen search, and rerank when the pipeline escalates—subject to config and budget.

**Repository:** [github.com/ltangwang/code-ana1](https://github.com/ltangwang/code-ana1)

Flow diagram: `figures/code_search_pipeline.mmd` ([mermaid.live](https://mermaid.live) or a VS Code Mermaid extension).

---

## End-to-end pipeline

Typical entrypoints: `scripts/evaluate_code_search.py` (Java) and `scripts/evaluate_code_search_non_java.py` / per-language wrappers (e.g. `evaluate_code_search_ruby.py`).

1. **Bi-encoder:** Encode queries and code with UniXcoder (or a fine-tuned checkpoint), build or load a vector index, **Top-K** dense retrieval (this repo standardizes **K = 10** in docs and smoke defaults unless you override flags).
2. **(Optional) Cross-encoder:** Rerank the bi-encoder Top-K before the LLM stage (`--use-ce`, or ablations like `--bi-ce-only`).
3. **Ollama (edge):** Must be **running** locally. The model in `config/settings.yaml` under `ollama` (e.g. `base_url`, `model_name`) is used for structured rerank JSON (`best_candidate_index`, `needs_escalation`, etc.).
4. **Cloud:** Configure at least one provider in `settings.yaml` / `.env` (`cloud.*.api_key` with `${ENV}` placeholders). Used for query refinement, rescue retrieval, and cloud rerank when the pipeline does not stop at Ollama.

Supporting code: `core/orchestrator.py` (Ollama session, cloud factory, budget), `scripts/`, `shared/`.

---

## Prerequisites

| Component | What you need |
|-----------|----------------|
| **Python** | 3.9+ |
| **Ollama** | Install from [ollama.com](https://ollama.com/), run the daemon, `ollama pull` the model name that matches `settings.yaml` → `ollama.model_name`. |
| **Cloud LLM** | API keys in `.env` and referenced from `settings.yaml` (e.g. OpenAI-compatible or DashScope). Required for full pipeline (not for `--skip-cloud` smoke). |
| **Hardware** | GPU recommended for UniXcoder / Cross-encoder; CPU possible but slower. |
| **Config** | Create local `config/settings.yaml` (gitignored by default). See `config/thresholds.yaml` for optional reference values. |

---

## Install

```powershell
git clone https://github.com/ltangwang/code-ana1.git
cd code-ana1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS: `source .venv/bin/activate`.

---

## Configuration (minimal)

- **`config/settings.yaml`:** `ollama` block, `cloud` providers, `budget`, `code_search` (paths, `cloud_rescue_k`, etc.). Do not commit secrets.
- **`.env`:** e.g. `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, as referenced by your YAML.

Eval scripts substitute `${VAR}` in `cloud.*.api_key` from the environment.

---

## Running evaluation (K = 10)

**Convention:** use **`--top-k 10 --llm-pool-k 10 --cloud-rescue-k 10`** so bi-encoder recall, LLM pool, and cloud rescue share the same K unless you intentionally ablate.

**Full pipeline (intended): bi-encoder + Ollama + cloud** — Ruby smoke corpus, **no** `--skip-cloud`:

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py --sample 3 --top-k 10 --llm-pool-k 10 --cloud-rescue-k 10 --index-size 20 --pretrained-base-only
```

Before running:

1. Start **Ollama** and ensure `ollama.model_name` is pulled.
2. Fill **cloud** keys for the providers your `settings.yaml` uses.
3. For real benchmarks, point `CSN_LANG_DIR` (or language dirs under `CodeSearchNet_clean_Dataset`) at full **GraphCodeBERT-clean** splits—not only the smoke fixture.

**Middle tier — bi-encoder + Ollama (`--bi-ollama-only`):** Same retrieval and Ollama as the full pipeline, but **no cloud** calls. Handy when Ollama is available but cloud keys are not.

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py --sample 3 --bi-ollama-only --top-k 10 --llm-pool-k 10 --cloud-rescue-k 10 --index-size 20 --pretrained-base-only
```

**Optional — bi-encoder only (`--skip-cloud`):** Skips Ollama and cloud. Use for **retrieval-only** smoke (e.g. CI). Not a substitute for the full **bi-encoder + Ollama + cloud** run.

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py --sample 3 --skip-cloud --top-k 10 --llm-pool-k 10 --cloud-rescue-k 10 --index-size 20 --pretrained-base-only
```

See also: `examples/code_search_smoke/README.sample.md` (conceptual pipeline + Linux/PowerShell for all three modes).

---

## Project layout (summary)

```
code-ana1/
├── scripts/                    # Eval, training, retriever, CE rerank
├── core/                       # Orchestration + budget
├── cloud/                      # Provider clients
├── edge/                       # Ollama local inference
├── shared/                     # CSN paths, language profiles, schemas
├── figures/code_search_pipeline.mmd
├── examples/code_search_smoke/ # Tiny Ruby JSONL + README.sample.md
├── config/
└── requirements.txt
```

### Where the smoke mini-dataset lives (including on GitHub)

After you **clone** the repo or open it on **GitHub**, the synthetic Ruby JSONL used in the commands above is under:

**`examples/code_search_smoke/ruby/`** — files `codebase.jsonl` (index) and `test.jsonl` (queries).  
**How to use this fixture:** see **`examples/code_search_smoke/README.sample.md`** (sample documentation).

---

## Tests

```powershell
pytest
```

## License

MIT License

## Note

Research / experiments only. Do not push `.env`, real keys, or machine-specific `settings.yaml` to a public remote. Caches (`csn_retriever_emb_*`), `evaluation_runs/`, etc. are gitignored by default.
