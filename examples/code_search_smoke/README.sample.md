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

## Intended run: **bi-encoder + Ollama + cloud** (**K = 10**)

The **default, end-to-end** path this repo is built for is:

**dense bi-encoder (dual-tower) retrieval** → optional cross-encoder → **Ollama** on the candidate pool → **cloud** when the pipeline escalates (query rewrite, rescue retrieval, cloud rerank), as configured.

The commands below run that **full** stack (do **not** pass `--skip-cloud`). A separate subsection further down explains **bi-encoder-only** runs and **why** they exist—they are **not** a substitute for the full pipeline.

### Before you run

1. **Ollama:** Install and start the service. `ollama pull` the model name that matches `config/settings.yaml` → `ollama.model_name`.
2. **Cloud:** Set API keys in `.env` and wire `cloud.*` in `settings.yaml` (e.g. `${OPENAI_API_KEY}`).
3. **Config:** Local `config/settings.yaml` must exist (see repository root `README.md`).
4. **K:** Use **`--top-k 10 --llm-pool-k 10 --cloud-rescue-k 10`** so retrieval, LLM pool, and cloud rescue share **K = 10** (edge/cloud aligned).

### Commands

Run from the **repository root**; point `CSN_LANG_DIR` at the `ruby` subfolder under **this** directory (`examples/code_search_smoke/ruby`).

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

First-time `--pretrained-base-only` will download UniXcoder base weights from Hugging Face.

### Optional: bi-encoder only (`--skip-cloud`)

**Why this mode exists:** With `--skip-cloud`, the script **does not** call Ollama or cloud providers. That is useful when you only want to **verify retrieval** (indexing, Top-K, metrics shape), or you **lack** a running Ollama daemon / **cloud API keys**, or you are in **CI** and want a fast, deterministic smoke without external services.

**It is not the main story:** The designed pipeline is **bi-encoder + Ollama + cloud**. Use `--skip-cloud` only as a **shortcut or debug** path; for a real end-to-end check, use the commands above **without** `--skip-cloud`.

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
