# Code search smoke fixture (Ruby)

## Where this README and data live

| What | Path (from repository root) |
|------|------------------------------|
| **This file** | `examples/code_search_smoke/README.md` |
| **Ruby JSONL (index)** | `examples/code_search_smoke/ruby/codebase.jsonl` |
| **Ruby JSONL (queries)** | `examples/code_search_smoke/ruby/test.jsonl` |

After you clone the repo or browse it on GitHub, open the **`examples/code_search_smoke`** folder: the **`README.md`** next to `ruby/` is this document. There is no other README for this sample.

---

**Why Ruby?** In CodeSearchNet (Husain et al., 2019), **Ruby has the smallest documented-function count** among the six languages (Table 1, ~57k vs larger languages). This folder ships **synthetic JSONL** in GraphCodeBERT-clean style (`docstring`, `original_string`, `url`) to exercise parsers and paths **without the full corpus**.

**Not for publishing benchmark scores.** Use official CodeSearchNet / cleaned splits for MRR and Success@K.

---

## Full pipeline (Ollama + cloud, **K = 10**)

This matches the intended **end-to-end** run: bi-encoder → (optional CE) → **Ollama** → **cloud** when escalated.

### Before you run

1. **Ollama:** Install and start the service. `ollama pull <model>` for the model named in `config/settings.yaml` → `ollama.model_name`.
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

### Bi-encoder only (no Ollama, no cloud)

For a quick shape check:

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
├── README.md        # this file
└── ruby/
    ├── codebase.jsonl   # retrieval index (10 synthetic methods)
    └── test.jsonl       # 3 queries (docstring + gold url)
```

## License / provenance

All snippets and docstrings are **synthetic** for this repository and may be redistributed with the project.
