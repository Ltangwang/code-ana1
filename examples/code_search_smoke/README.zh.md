# 代码检索冒烟数据（Ruby）

**为何 Ruby？** CodeSearchNet 论文 Table 1 中，六种语言里 **Ruby 有文档的函数数最少**（约 5.7 万）。本目录提供 **合成 JSONL**（`docstring`、`original_string`、`url`），用于在不下载完整语料时检查路径与解析。

**不能用于正式论文指标。** 真实 MRR / Success@K 请用官方或清洗全量数据。

**对外英文说明（GitHub）：** [`README.md`](README.md)。

---

## 完整流程（Ollama + 云端，**K = 10**）

与主仓库一致：**双塔 →（可选 CE）→ Ollama → 云端**。

### 运行前

1. **启动 Ollama**，并 `ollama pull` 与 `settings.yaml` 中 `ollama.model_name` 一致的模型。  
2. **配置云端**：`.env` + `settings.yaml` 里 `cloud` 与 `${API_KEY}` 占位。  
3. 本地存在 **`config/settings.yaml`**。  
4. 命令行统一：**`--top-k 10 --llm-pool-k 10 --cloud-rescue-k 10`**。

### 命令（全链路）

仓库根目录，`CSN_LANG_DIR` 指向本目录下的 `ruby`（建议绝对路径）。

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

**Windows（PowerShell）**

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

首次 `--pretrained-base-only` 会从 Hugging Face 拉取 UniXcoder 基座。

### 仅双塔（`--skip-cloud`）

不启 Ollama、不走云，只做索引与双塔指标冒烟：

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

## 目录

```
examples/code_search_smoke/ruby/
├── codebase.jsonl   # 10 条合成方法（索引）
└── test.jsonl       # 3 条查询（docstring + 金标 url）
```

## 许可与来源

全部为 **合成数据**，可随本仓库分发。
