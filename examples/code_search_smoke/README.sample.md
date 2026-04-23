# Code search sample fixture (Ruby)

## Where this document and data live

| What | Path (from repository root) |
|------|------------------------------|
| **This file** | `examples/code_search_smoke/README.sample.md` |
| **Ruby JSONL (index)** | `examples/code_search_smoke/ruby/codebase.jsonl` |
| **Ruby JSONL (queries)** | `examples/code_search_smoke/ruby/test.jsonl` |

After you clone the repo or browse it on GitHub, open the **`examples/code_search_smoke`** folder: **`README.sample.md`** (this file) sits next to **`ruby/`**. It is the only documentation for this sample.

---

**Why Ruby?** In CodeSearchNet (Husain et al., 2019), **Ruby has the smallest documented-function count** among the six languages (Table 1, ~57k vs larger languages). This folder ships **synthetic JSONL** in GraphCodeBERT-clean style (`docstring`, `original_string`, `url`) to exercise parsers and paths **without the full corpus**.

**Not for publishing benchmark scores.** Use official CodeSearchNet / cleaned splits for MRR and Success@K.

---

## Pipeline this sample follows (conceptual order)

End-to-end design is **three layers** on top of the same Top-K pool (here **K = 10**):

| Stage | What runs | Role |
|-------|-----------|------|
| **1. Dual-tower (bi-encoder)** | UniXcoder encodes query + code | Dense retrieval over `codebase.jsonl` → **Top-K** candidates for each query in `test.jsonl`. |
| **2. Ollama (edge)** | Local LLM | Reranks / interprets the **LLM pool** (aligned with Top-K), returns structured JSON (e.g. best index, whether to escalate). |
| **3. Cloud** | Remote LLM APIs | Used when the pipeline **escalates**: e.g. Ollama empty/invalid, `needs_escalation`, or **no-edge-hit rescue** (refined query + re-retrieve + cloud pick), per `settings.yaml` and budget. |

**Optional cross-encoder:** Add `--use-ce` on the full pipeline to rerank the bi-encoder Top-K **before** Ollama. The smoke commands below leave CE **off** for simplicity.

---

## How to run the smoke (three CLI modes)

Run from the **repository root**. Set `CSN_LANG_DIR` to **`examples/code_search_smoke/ruby`**.

**K convention:** keep **`--top-k 10 --llm-pool-k 10 --cloud-rescue-k 10`** so retrieval, Ollama/cloud pool, and cloud rescue share the same K (the Ruby wrapper also injects these defaults if you omit them).

**Flags are mutually exclusive:** do not combine `--skip-cloud` with `--bi-ollama-only`.

### Mode 3 — **Full pipeline: bi-encoder + Ollama + cloud** (recommended)

This is the **intended** system: dual-tower recall → Ollama on the pool → cloud when escalation or rescue requires it.

**Prerequisites:** Ollama running + model pulled (`ollama.model_name` in `config/settings.yaml`); cloud keys in `.env` / `settings.yaml`; local `config/settings.yaml`.

**Linux / macOS**

```bash
export CSN_LANG_DIR="$(pwd)/examples/code_search_smoke/ruby"
python scripts/evaluate_code_search_ruby.py \
  --sample 3 \
  --top-k 10 \
  --llm-pool-k 10 \
  --cloud-rescue-k 10 \
  --index-size 20 \
  --pretrained-base-only
```

**Windows (PowerShell)**

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py `
  --sample 3 `
  --top-k 10 `
  --llm-pool-k 10 `
  --cloud-rescue-k 10 `
  --index-size 20 `
  --pretrained-base-only
```

First-time `--pretrained-base-only` downloads UniXcoder **base** weights from Hugging Face.

### Mode 2 — **Bi-encoder + Ollama** (no cloud API calls)

Use **`--bi-ollama-only`**: same dual-tower retrieval and Ollama stage as above, but **no cloud** (no rescue/refine/rerank via remote APIs). Useful when you have Ollama but **no** cloud keys, or you want to isolate edge LLM behavior.

**Prerequisites:** Ollama + `settings.yaml` as for Mode 3; cloud keys **not** required.

**Linux / macOS**

```bash
export CSN_LANG_DIR="$(pwd)/examples/code_search_smoke/ruby"
python scripts/evaluate_code_search_ruby.py \
  --sample 3 \
  --bi-ollama-only \
  --top-k 10 \
  --llm-pool-k 10 \
  --cloud-rescue-k 10 \
  --index-size 20 \
  --pretrained-base-only
```

**Windows (PowerShell)**

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py `
  --sample 3 `
  --bi-ollama-only `
  --top-k 10 `
  --llm-pool-k 10 `
  --cloud-rescue-k 10 `
  --index-size 20 `
  --pretrained-base-only
```

### Mode 1 — **Bi-encoder only** (`--skip-cloud`)

**Neither Ollama nor cloud** run. Metrics reflect **retrieval-only** Success@K on the bi-encoder ranked list. Use for **CI**, quick index smoke, or when you have **no** Ollama and **no** cloud.

```bash
export CSN_LANG_DIR="$(pwd)/examples/code_search_smoke/ruby"
python scripts/evaluate_code_search_ruby.py \
  --sample 3 \
  --skip-cloud \
  --top-k 10 \
  --llm-pool-k 10 \
  --cloud-rescue-k 10 \
  --index-size 20 \
  --pretrained-base-only
```

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py `
  --sample 3 `
  --skip-cloud `
  --top-k 10 `
  --llm-pool-k 10 `
  --cloud-rescue-k 10 `
  --index-size 20 `
  --pretrained-base-only
```

---

## Layout

```
examples/code_search_smoke/
├── README.sample.md  # this file (sample documentation)
└── ruby/
    ├── codebase.jsonl   # retrieval index (10 synthetic methods)
    └── test.jsonl       # 3 queries (docstring + gold url)
```

## License / provenance

All snippets and docstrings are **synthetic** for this repository and may be redistributed with the project.
