# 代码检索冒烟数据（Ruby）

**为什么选 Ruby？** 在 CodeSearchNet 论文（Husain et al., 2019）中，六种语言里 **Ruby 带文档的函数数量最少**（Table 1：约 5.7 万条有文档函数；Go、PHP、Python、Java、JavaScript 都更大）。本仓库提供体积极小的 **合成 JSONL**，字段形态与 GraphCodeBERT 清洗导出一致（`docstring`、`original_string`、`url`），便于在不下载完整语料的情况下 **检查路径与解析是否正常**。

**不能用来报告正式榜单指标。** 真实的 MRR、Success@K 请使用官方 CodeSearchNet / 清洗划分数据。

## 目录结构

```
examples/code_search_smoke/ruby/
├── codebase.jsonl   # 小型检索索引（4 个方法）
└── test.jsonl       # 2 条查询（自然语言 + 金标 url；可无代码正文）
```

## 运行方式（仅双塔，不开 Ollama / 云端）

在仓库根目录，将 `CSN_LANG_DIR` 指向本目录下的 `ruby` 文件夹（**建议使用绝对路径**）。

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

**Windows（PowerShell）**

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

你仍需要本地的 `config/settings.yaml`（见仓库根目录 README）、运行 UniXcoder 所需的 GPU/CPU 内存，以及首次使用 `--pretrained-base-only` 时访问 Hugging Face 缓存以下载基座模型。

## 许可与来源

本目录内所有代码片段与 docstring 均为 **合成数据**（为本仓库编写），可随项目一并分发。
