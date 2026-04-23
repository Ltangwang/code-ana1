# code-ana1：双塔检索 + Ollama + 云端 代码检索

本仓库实现 **CodeSearchNet 风格代码检索**：**双塔** 在全库稠密召回，**Ollama** 在候选池上重排与门控，需要时由 **云端大模型** 改写查询、扩大检索或重排（受配置与预算约束）。

**仓库：** <https://github.com/ltangwang/code-ana1>

流程图：`figures/code_search_pipeline.mmd`（可用 [mermaid.live](https://mermaid.live) 或 VS Code Mermaid 插件）。

**对外（GitHub）主文档为英文：** 见根目录 [`README.md`](README.md)。

---

## 完整流程（你要跑通的全链路）

典型入口：`scripts/evaluate_code_search.py`（Java）、`scripts/evaluate_code_search_non_java.py` 及各语言封装（如 `evaluate_code_search_ruby.py`）。

1. **双塔：** 用 UniXcoder（或微调 checkpoint）对查询与代码编码，建索引或加载向量缓存，做 **Top-K** 相似度检索。文档与冒烟样例中 **统一取 K = 10**（除非你在命令行刻意改参数）。
2. **（可选）Cross-encoder：** 在双塔 Top-K 上再精排（如 `--use-ce` 或消融 `--bi-ce-only`）。
3. **Ollama（边侧）：** 须 **先启动** 本机 Ollama 服务，并 `ollama pull` 与 `config/settings.yaml` 里 `ollama.model_name` 一致的模型；用于输出结构化重排结果等。
4. **云端：** 在 `settings.yaml` / `.env` 中配置至少一家云厂商及 API Key；在 Ollama 不足以结案或需解救检索时，走云端改写 / 重排等分支。

实现上：`core/orchestrator.py` 负责 Ollama 会话、多云客户端工厂与预算；检索与双塔逻辑在 `scripts/`、`shared/`。

---

## 环境前提

| 组件 | 要求 |
|------|------|
| Python | 3.9+ |
| Ollama | 安装并启动服务，模型与 `settings.yaml` 中 `ollama` 一致 |
| 云端 | `.env` + `settings.yaml` 中 `cloud` 与密钥占位符 |
| 硬件 | 建议 GPU；CPU 可跑但较慢 |
| 配置 | 本地 `config/settings.yaml`（默认不入库） |

---

## 安装

```powershell
git clone https://github.com/ltangwang/code-ana1.git
cd code-ana1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS：`source .venv/bin/activate`。

---

## 评测时 K 固定为 10（推荐写法）

为与文档一致，建议使用：

**`--top-k 10 --llm-pool-k 10 --cloud-rescue-k 10`**

Ruby 冒烟数据示例（全链路，需 Ollama + 云端已配置）：

```powershell
$env:CSN_LANG_DIR = "$PWD\examples\code_search_smoke\ruby"
python scripts/evaluate_code_search_ruby.py --sample 3 --top-k 10 --llm-pool-k 10 --cloud-rescue-k 10 --index-size 20 --pretrained-base-only
```

仅双塔、不连 Ollama/云（快速形状检查）：

```powershell
python scripts/evaluate_code_search_ruby.py --sample 3 --skip-cloud --top-k 10 --llm-pool-k 10 --cloud-rescue-k 10 --index-size 20 --pretrained-base-only
```

更细的冒烟目录说明（含中英文分支）：`examples/code_search_smoke/README.zh.md`。

---

## 项目结构（摘要）

```
code-ana1/
├── scripts/
├── core/
├── cloud/
├── edge/
├── shared/
├── figures/code_search_pipeline.mmd
├── examples/code_search_smoke/
├── config/
└── requirements.txt
```

## 其它脚本

| 方向 | 脚本 |
|------|------|
| 克隆检测 | `evaluate_clone_detection.py` |
| BCB 训练 / RAG | `train_unixcoder_bcb.py`、`bcb_rag.py` |

## 测试

```powershell
pytest
```

## 许可证

MIT License

## 说明

研究用途。勿将 `.env`、真实密钥或带内网路径的 `settings.yaml` 推到公开仓库。`csn_retriever_emb_*`、`evaluation_runs/` 等默认已忽略。
