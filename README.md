# code-ana1：双塔检索 + Ollama + 云端 的代码检索流水线

本仓库实现 **CodeSearchNet 风格的代码检索**：**双塔（Bi-encoder）** 在全库上做稠密召回，**Ollama** 在候选池上重排与门控，在触发条件与预算允许时由 **云端大模型** 改写查询、扩大检索或重排，以在延迟、成本与效果之间折中。

**仓库地址：** <https://github.com/ltangwang/code-ana1>

更详细的流程图见 `figures/code_search_pipeline.mmd`（可用 [mermaid.live](https://mermaid.live) 或 VS Code Mermaid 插件查看）。

## 核心流水线

典型评测入口：`scripts/evaluate_code_search.py`（非 Java 与各语言入口见 `scripts/evaluate_code_search_non_java.py` 及按语言划分的脚本）。

1. **双塔检索：** 用 UniXcoder（或配置的编码器）对查询与代码库编码，建索引或加载向量缓存，对每条查询做 **Top-K** 相似度排序。
2. **（可选）Cross-Encoder 重排：** 在 Top-K 上进一步排序以缩小进入 LLM 的候选池（可关闭，或仅用双塔 + Ollama，例如 `--bi-ollama-only` / `--bi-ce-only`）。
3. **Ollama：** 在候选池上输出结构化结果（如 `best_candidate_index`、`needs_escalation`），作为 **边侧** 重排与门控。
4. **云端：** 若 GT 不在双塔 Top-K 内，可触发 **查询改写** / 更大范围检索，或由云端 API **重排/解析**；受配置与预算约束。

编排层在 `core/orchestrator.py` 中复用 **Ollama 会话**、**多云工厂** 与 **预算**；双塔与检索逻辑主要在 **`scripts/`** 与 **`shared/`**。

## 环境要求

- Python 3.9+
- [Ollama](https://ollama.com/)（本地重排）
- 使用云端分支时需配置 API Key（见 `.env` 与 `settings.yaml`）
- 双塔与 Cross-Encoder 的 GPU/显存依模型与规模而定

## 安装

```powershell
git clone https://github.com/ltangwang/code-ana1.git
cd code-ana1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS：`source .venv/bin/activate`。

## 配置

- **`config/settings.yaml`：** 在本地创建或从备份复制；**默认被 `.gitignore` 忽略**，避免提交本机路径与密钥。
- **`config/thresholds.yaml`：** 阈值等，可随仓库提供。
- **`.env`：** 云端 `API_KEY` 等（勿提交）。

评测脚本从 `settings.yaml` 读取 `ollama`、`cloud`、`budget` 及相关项；双塔路径、缓存目录、检索 Top-K 以脚本参数与配置为准，见各脚本 `--help`。

## 运行代码检索评测

语言、数据路径、Top-K、是否跳过云端等取决于本地 `settings` 与数据集，例如：

```powershell
python scripts/evaluate_code_search.py --help
python scripts/evaluate_code_search_non_java.py --help
```

常见流程：**准备预处理后的 CodeSearchNet** → **加载或训练 UniXcoder 检索 checkpoint** → **运行评测脚本** → 结果可写入 `evaluation_runs/`（默认被 git 忽略）。

数据集下载：仓库根目录 `download_codesearchnet.py`；大体量数据与向量缓存通常在 `.gitignore` 中。

## 项目结构（摘要）

```
code-ana1/
├── scripts/
│   ├── evaluate_code_search.py      # 检索评测（双塔 + CE + Ollama + 云）
│   ├── evaluate_code_search_non_java.py
│   ├── evaluate_code_search_*.py    # 按语言的评测入口
│   ├── csn_retriever.py / csn_data.py
│   ├── csn_ce_rerank.py             # Cross-Encoder 重排
│   └── train_unixcoder_csn*.py      # 双塔 / 检索训练
├── core/                            # 编排与预算（评测复用）
├── cloud/                           # 云厂商客户端
├── edge/                            # 本地推理（Ollama）等
├── shared/                          # CSN 路径、语言 profile、schema 等
├── figures/code_search_pipeline.mmd # 流水线示意图
├── examples/code_search_smoke/      # 极小 Ruby JSONL 样例（格式 + 冒烟；见其 README）
├── config/
└── requirements.txt
```

## 其它脚本

| 方向 | 脚本（概要） |
|------|----------------|
| 克隆检测 | `evaluate_clone_detection.py` |
| BCB 训练 / RAG | `train_unixcoder_bcb.py`、`bcb_rag.py` |

## 测试

```powershell
pytest
```

## 许可证

MIT License

## 说明

本仓库用于 **研究与实验**。请勿将 `.env`、真实密钥或含内网路径的 `config/settings.yaml` 推送到公开远程。双塔缓存（如 `csn_retriever_emb_*`）、`evaluation_runs/` 等默认已忽略，避免大文件进入 git。
