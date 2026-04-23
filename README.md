# code-ana1：双塔检索 + Ollama + 云端的代码检索流水线

本仓库围绕 **CodeSearchNet 风格代码检索** 搭建：**双塔（Bi-encoder）** 在全库上做向量召回，**Ollama** 在候选池上做本地重排与决策，在触发条件或预算允许时由**云端大模型**完成查询改写、扩展检索与最终重排等兜底步骤，从而在延迟、成本与效果之间折中。

**仓库**：<https://github.com/ltangwang/code-ana1>

更细的流程图见：`figures/code_search_pipeline.mmd`（可用 [mermaid.live](https://mermaid.live) 或 VS Code Mermaid 插件查看）。

## 核心流水线

典型评测入口：`scripts/evaluate_code_search.py`（多语言变体见 `scripts/evaluate_code_search_non_java.py` 及各语言脚本）。

1. **双塔检索**：使用 UniXcoder（或配置的代码编码器）对查询与代码库编码，建索引/加载向量缓存，对每条查询做 **Top-K 相似度排序**。
2. **（可选）Cross-Encoder 精排**：在 Top-K 上进一步排序，缩小进入 LLM 的候选池（可通过参数关闭或仅用双塔 + Ollama，如 `--bi-ollama-only` / `--bi-ce-only` 等）。
3. **Ollama**：对候选池生成结构化结果（如 `best_candidate_index`、`needs_escalation`），作为**边缘侧**的主要重排与门控。
4. **云端**：在 GT 未落入双塔 Top-K 时触发**改写检索**、在更大候选集上再搜、或由云端 API 做**重排/解析**；受配置与预算约束。

编排上复用 `core/orchestrator.py` 中的 **Ollama 会话**、**多云工厂**与**预算/策略**；检索与双塔逻辑在 **`scripts/`** 与 **`shared/`** 中实现。

## 环境要求

- Python 3.9+
- [Ollama](https://ollama.com/)（本地重排）
- 运行带云端分支的评测时需配置对应 API Key（见 `.env` 与 `settings.yaml`）
- GPU/显存：双塔与 Cross-Encoder 依模型与规模而定

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

- **`config/settings.yaml`**：需在本地创建或从备份复制；**默认被 `.gitignore` 忽略**，避免提交本机模型路径与密钥。
- **`config/thresholds.yaml`**：阈值等，可随仓库提供。
- **`.env`**：云端 `API_KEY` 等（勿提交）。

评测脚本从 `settings.yaml` 读取 `ollama`、`cloud`、`budget`、`strategy` 等；双塔模型路径、缓存目录、检索 Top-K 以脚本参数与配置为准，详见各脚本 `--help`。

## 运行代码检索评测（示例）

具体语言、数据路径、Top-K、是否跳过云端等以你本地 `settings` 与数据集为准，例如：

```powershell
python scripts/evaluate_code_search.py --help
python scripts/evaluate_code_search_non_java.py --help
```

常用思路：**准备 CodeSearchNet 预处理数据** → **加载或训练 UniXcoder 检索 checkpoint** → **运行评测脚本** → 结果可写入 `evaluation_runs/`（默认被 git 忽略）。

数据集下载可参考仓库根目录 `download_codesearchnet.py`；大型数据与向量缓存目录通常在 `.gitignore` 中。

## 项目结构（摘要）

```
code-ana1/
├── scripts/
│   ├── evaluate_code_search.py      # 检索评测（双塔 + CE + Ollama + 云）
│   ├── evaluate_code_search_non_java.py
│   ├── evaluate_code_search_*.py    # 按语言拆分的评测入口
│   ├── csn_retriever.py / csn_data.py
│   ├── csn_ce_rerank.py             # Cross-Encoder 精排
│   └── train_unixcoder_csn*.py      # 双塔/检索模型训练相关
├── core/                            # 编排、预算、策略（评测复用）
├── cloud/                           # 云厂商客户端
├── edge/                            # 本地推理（Ollama）等
├── shared/                          # CSN 路径、语言 profile、schema 等
├── figures/code_search_pipeline.mmd # 检索流水线示意图
├── config/
└── requirements.txt
```

## 其他脚本

| 方向 | 脚本（概要） |
|------|----------------|
| 克隆检测评估 | `evaluate_clone_detection.py` |
| 缺陷检测相关 | `evaluate_defect_detection.py` |
| BCB 训练/检索 | `train_unixcoder_bcb.py`、`bcb_rag.py` |

## 测试

```powershell
pytest
```

## 许可证

MIT License

## 说明

本仓库用于**研究与实验**。请勿将 `.env`、含真实密钥或内网路径的 `config/settings.yaml` 推送到公开仓库。双塔向量缓存（如 `csn_retriever_emb_*`）、`evaluation_runs/` 等默认已忽略，避免大文件入库。
