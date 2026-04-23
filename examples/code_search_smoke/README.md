# Code search smoke fixture (Ruby)

**Why Ruby?** In the CodeSearchNet paper (Husain et al., 2019), **Ruby has the smallest slice of functions with documentation** among the six languages (Table 1: ~57k documented functions vs. larger counts for Go, PHP, Python, Java, JavaScript). This repo ships a **tiny synthetic JSONL** in the same shape as GraphCodeBERT-clean style exports (`docstring`, `original_string`, `url`) so you can verify paths and parsers **without downloading the full corpus**.

**Not for reporting benchmark numbers.** Use the official CodeSearchNet / cleaned splits for real MRR and Success@K.

## Layout

```
examples/code_search_smoke/ruby/
├── codebase.jsonl   # small retrieval index (4 methods)
└── test.jsonl       # 2 queries (NL + gold url; code optional)
```

## Run (bi-encoder only, no Ollama / no cloud)

From the repository root, point `CSN_LANG_DIR` at the `ruby` folder (absolute path recommended).

**Linux / macOS**

```bash
export CSN_LANG_DIR="$(pwd)/examples/code_search_smoke/ruby"
python scripts/evaluate_code_search_ruby.py \
  --sample 2 \
  --skip-cloud \
  --top-k 4 \
  --llm-pool-k 4 \
  --cloud-rescue-k 4 \
  --index-size 10 \
  --pretrained-base-only
```

**Windows (PowerShell)**

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py `
  --sample 2 `
  --skip-cloud `
  --top-k 4 `
  --llm-pool-k 4 `
  --cloud-rescue-k 4 `
  --index-size 10 `
  --pretrained-base-only
```

You still need a local `config/settings.yaml` (see repository README), GPU/CPU RAM for UniXcoder, and Hugging Face cache access for `--pretrained-base-only` the first time.

## License / provenance

All code snippets and docstrings in this fixture are **synthetic** (written for this repo) so they can be redistributed with the project.
