# Edge-Cloud 协作代码分析原型

在边缘端使用本地小模型（如 Ollama）做快速草稿分析，按需将低置信度或关键片段提交云端大模型校验，并结合预算控制降低 API 成本。适用于缺陷检测、多语言 AST 热点抽取等研究与教学场景。

## 核心能力

- **边缘优先**：本地 Ollama 推理，可配置模型与超时。
- **稀疏上云**：按置信度、严重级别与预算决定是否调用云端。
- **成本感知**：内存内预算跟踪，低预算时可收紧上云策略（见 `config/settings.yaml`）。
- **多语言**：Python、Java、JavaScript、C/C++（tree-sitter）。
- **异步编排**：`core/orchestrator.py` 协调 AST → 本地推理 → 策略 → 云端。
- **多云**：支持 OpenAI、Anthropic、阿里云 DashScope（OpenAI 兼容接口）及自定义兼容端点。

## 架构概览

```
代码输入 → AST 热点 → 本地 Ollama → 置信度/策略 → [高置信] 直接输出
                                              ↘ [需验证] 云端 LLM → 合并结果 → 预算更新
```

## 环境要求

- Python 3.9+
- [Ollama](https://ollama.com/)（本地推理）
- 至少配置一个云端提供商的 API Key（见下文环境变量）

## 安装

```powershell
git clone <你的仓库地址>
cd code-analyze
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

在项目根目录创建 `.env`（该文件已被 `.gitignore` 忽略，勿提交），按需填入密钥，例如：

```env
# 与 config/settings.yaml 中 ${变量名} 对应
DASHSCOPE_API_KEY=sk-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```

`main.py` 会从 `.env` 读取环境变量，并将 `config/settings.yaml` 里 `cloud.*.api_key` 的 `${变量名}` 替换为对应环境变量值。本机专属路径等可直接改本地 `settings.yaml` 或使用不跟踪的副本（勿将含密钥的文件提交到 Git）。

`.gitignore` 中的 `config/secrets.yaml` 供你自行扩展「从单独 YAML 读密钥」时使用；当前默认流程以 `.env` 为准。

## Ollama

安装 Ollama 后拉取与 `config/settings.yaml` 中 `ollama.model_name` 一致的模型，例如：

```powershell
ollama pull qwen2.5-coder:7b
```

大块模型与 Hugging Face 缓存路径可在 `config/settings.yaml` 的 `models` 段配置；也可通过环境变量 `OLLAMA_MODELS` 指定 Ollama 模型目录（与官方说明一致）。

## 配置说明

主配置：`config/settings.yaml`。

- **ollama**：`base_url`、`model_name`、`timeout` 等。
- **cloud**：`default_provider` 及各厂商的 `api_key`（支持 `${ENV_VAR}` 占位）、`model`、`base_url`。
- **budget**：总预算、日预算、告警比例等（当前为进程内统计，无数据库持久化）。
- **strategy**：上云置信度阈值、最大并发云端请求数等。
- **ast**：圈复杂度、嵌套深度、克隆检测相似度权重等。
- **clone_detection**：UniXcoder 路径、DFG 骨架、BCB RAG 等与克隆/云端仲裁相关项。

另见 `config/thresholds.yaml`，可与 `settings.yaml` 中的 AST、策略项对照调参（以运行时实际加载的 `settings.yaml` 为准）。

## 命令行用法

```powershell
# 指定配置文件（可选，默认 config/settings.yaml）
python main.py --config config/settings.yaml analyze --file examples/sample_code/buggy_python.py

# 扫描目录
python main.py analyze --dir . --pattern "**/*.py"

# 限制云端调用规模（会写入 strategy 相关覆盖）
python main.py analyze --file examples/sample_code/buggy_python.py --max-cloud-calls 5

# 导出 JSON
python main.py analyze --file examples/sample_code/buggy_python.py --output results.json

# 预算状态（内存）
python main.py budget-status

# 检查已配置的云厂商连通性
python main.py health-check
```

语言选项：`--language` 支持 `python` / `java` / `javascript` / `cpp`。

安装本项目包后也可使用入口：`code-analyze`（见 `setup.py` 的 `console_scripts`）。

## 项目结构

```
code-analyze/
├── main.py                 # CLI 入口
├── core/                   # 编排、策略、预算
├── edge/                   # AST、本地推理、置信度、Java DFG 骨架等
├── cloud/                  # 云厂商客户端与工厂
├── shared/                 # 模式、提示词、日志
├── config/
│   ├── settings.yaml       # 主配置（可提交）
│   └── thresholds.yaml     # 阈值等
├── scripts/                # 评估与训练辅助脚本（见下）
├── tests/
├── examples/sample_code/   # 示例缺陷代码
├── download_codesearchnet.py   # 下载 CodeSearchNet 到本地目录（默认不入库）
└── requirements.txt
```

## 评估与实验脚本（`scripts/`）

以下脚本依赖 `transformers`、`datasets` 等，已在 `requirements.txt` 中列出。具体参数请使用 `python scripts/<name>.py --help` 查看。

| 脚本 | 用途（概要） |
|------|----------------|
| `evaluate_clone_detection.py` | 克隆检测评估（UniXcoder 等） |
| `evaluate_code_search.py` | 代码检索 / Code Search 评估 |
| `evaluate_defect_detection.py` | 缺陷检测相关评估 |
| `train_unixcoder_bcb.py` | BigCloneBench 上 UniXcoder 训练流程 |
| `train_unixcoder_csn.py` | CodeSearchNet 相关训练 |
| `csn_retriever.py` / `csn_data.py` | CodeSearchNet 检索与数据辅助 |
| `bcb_rag.py` | BCB RAG 索引与 Few-shot 相关 |

**数据集**：`CodeSearchNet_Dataset/` 默认被 `.gitignore` 排除。需要时在仓库根目录执行：

```powershell
python download_codesearchnet.py
```

评估产生的 `results_*.json`、`results_*.csv` 等默认也被忽略，避免误提交大文件。

## 运行测试

```powershell
pytest
pytest tests/test_strategy_manager.py
pytest --cov=. --cov-report=html
```

## 日志

使用 `structlog`（JSON 等），运行时可关注上云决策、云端延迟、预算更新等字段，便于对照 `config/settings.yaml` 调试策略。

## 设计理念（简述）

类比推测解码：本地模型快速给出草稿与置信度，云端在必要时做验证或细化；接受/拒绝由策略与预算共同决定，以在成本与质量之间折中。

## 许可证

MIT License

## 说明

本仓库为**原型 / 研究向**实现：生产环境使用前请自行做安全审计、密钥管理与充分测试。不要将 `.env` 或含 API Key / 私钥的 YAML 推送到公开远程仓库。
