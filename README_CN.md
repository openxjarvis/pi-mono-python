# pi-mono-python

[pi-mono](../pi-mono) TypeScript monorepo 的 Python 移植版本 — 与 TypeScript 相同的 `packages/` 树，代码、逻辑、算法和文件夹结构对齐。

**对齐版本：** pi-mono TypeScript **v0.84.3**（2026-08-24）。模块清单见 [`ALIGNMENT.md`](ALIGNMENT.md)（以及 `docs/ALIGNMENT_REPORT_2026-08.md`）。

| TypeScript | Python | 说明 |
|---|---|---|
| `@mariozechner/pi-ai` | `pi_ai` | 统一的 LLM 流式层（Google、Anthropic、OpenAI、Bedrock 等） |
| `@mariozechner/pi-agent-core` | `pi_agent` | 代理循环、工具执行、状态管理 |
| `@mariozechner/pi-coding-agent` | `pi_coding_agent` | 编码代理 CLI，包含文件工具：read、write、edit、bash、powershell（Windows）、grep、find、ls |
| `@mariozechner/pi-tui` | `pi_tui` | 终端 UI 库，具备差异化渲染引擎 |
| `@earendil-works/pi-telemetry` | `pi_telemetry` | 供应商无关的遥测契约与内存记录器 |
| `@earendil-works/pi-protocol` | `pi_protocol` | 长度前缀 CBOR 会话协议 |
| `@earendil-works/pi-client` | `pi_client` | 传输无关的远程会话客户端 |
| `@earendil-works/pi-server` | `pi_server` | 实验性会话服务端 + Unix 监听器 |
| `@earendil-works/pi-session-backend-sqlite-node` | `pi_session_backend_sqlite` | SQLite 会话后端（`sqlite3` 对应 `node:sqlite`） |
| `@earendil-works/pi-evals` | `pi_evals` | 编码代理评测脚手架（pytest 对应 vitest-evals） |

---

## 安装

### 前置要求

- **Python 3.11+** — 检查版本：`python3 --version`
- **[uv](https://docs.astral.sh/uv/)** — 快速的 Python 包管理器

如果没有安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 克隆和安装

```bash
git clone https://github.com/openxjarvis/pi-mono-python.git
cd pi-mono-python

# 一步安装所有四个包及其依赖
uv sync
```

---

## 快速开始

### 1. 配置 API 密钥

在项目根目录创建 `.env` 文件：

```bash
# Google Gemini（推荐默认）
GEMINI_API_KEY=your_key_here

# 可选 — 根据需要添加其他提供商
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=        # GEMINI_API_KEY 的替代方案
AWS_ACCESS_KEY_ID=     # 用于 AWS Bedrock
AWS_SECRET_ACCESS_KEY=
```

> **重要：** `.env` 会在运行时自动加载。**永远不要提交到 git。**

### 2. 启动交互式 TUI

```bash
uv run --package pi-coding-agent pi
```

这会打开功能完整的终端 UI，你可以在其中与编码代理聊天。

**键盘快捷键：**

| 按键 | 操作 |
|-----|--------|
| `Enter` | 发送消息 |
| `Shift+Enter` | 在输入中换行 |
| `/` | 斜杠命令补全 |
| `@` | 文件路径补全 |
| `Ctrl+P` | 切换到下一个模型 |
| `Ctrl+C` / `Esc` | 退出 |

### 3. 尝试一个简单的任务

在终端中输入：

```
创建一个 Python 函数来计算斐波那契数列
```

代理会编写代码并将其保存到你的当前目录。

---

## 常见用法

### 单个提示（非交互式）

用于脚本或快速任务：

```bash
uv run --package pi-coding-agent pi --print "用 Python 写一个快速排序"
```

代理的响应会打印到标准输出并退出。

### 切换模型

```bash
# 使用特定模型
uv run --package pi-coding-agent pi --model gemini-2.5-pro-preview

# 使用提供商 + 模型名称
uv run --package pi-coding-agent pi --provider google --model gemini-2.0-flash

# 列出所有可用模型
uv run --package pi-coding-agent pi --list-models
```

### 恢复之前的会话

```bash
# 继续最近的会话
uv run --package pi-coding-agent pi --continue

# 从之前的会话列表中选择
uv run --package pi-coding-agent pi --resume
```

### TUI 中的斜杠命令

在交互式 TUI 中输入 `/` 查看可用命令：

| 命令 | 说明 |
|---------|-------------|
| `/model <name>` | 切换到不同的模型 |
| `/thinking <level>` | 设置思考详细程度：`minimal` · `low` · `medium` · `high` · `xhigh` |
| `/compact` | 压缩对话上下文以节省 token |
| `/session` | 显示会话统计（已使用的 token、成本估算） |
| `/tools` | 列出代理可用的所有工具 |

### 完整的 CLI 帮助

```bash
uv run --package pi-coding-agent pi --help
```

---

## 运行测试

### 所有测试

```bash
uv run pytest
```

### 按包测试

```bash
uv run pytest packages/tui/tests/          # TUI 组件
uv run pytest packages/ai/tests/           # AI 提供商
uv run pytest packages/agent/tests/        # 代理核心
uv run pytest packages/coding-agent/tests/ # CLI + 编码代理
```

### 实时 API 测试（需要 `GEMINI_API_KEY`）

```bash
uv run pytest packages/ai/tests/ --live -v

# 或通过环境变量
LIVE_TESTS=1 uv run pytest packages/ai/tests/ -v
```

> 默认情况下，所有测试都针对 mock 运行 — 不需要 API 密钥，不消耗配额。

---

## 测试状态

| 包 | 测试数 | 状态 |
|---------|-------|--------|
| `pi_tui` + `pi_agent` | — | ✅ 通过 |
| `pi_ai` | 163 | ✅ 通过（7 个跳过 = 仅实时测试） |
| `pi_coding_agent` | 305+ | ✅ 通过 |
| **总计** | **625** | **✅ 全部通过**（7 个跳过） |

---

## 项目结构

```
pi-mono-python/
├── .env                          ← API 密钥（永远不要提交）
├── pyproject.toml                ← uv 工作区根
├── conftest.py                   ← 全局 pytest 配置（.env 加载器）
└── packages/
    ├── ai/                       ← LLM 提供商层
    │   └── src/pi_ai/
    │       ├── providers/        ← google.py、openai.py、anthropic.py 等
    │       ├── stream.py         ← 统一的 stream_simple() / complete_simple()
    │       └── utils/            ← 溢出检测、JSON 解析等
    ├── agent/                    ← 核心代理循环
    │   └── src/pi_agent/
    │       ├── agent.py          ← 主运行循环
    │       ├── tools/            ← 工具注册和执行
    │       └── session.py        ← 会话状态
    ├── coding-agent/             ← CLI 入口点和扩展
    │   └── src/pi_coding_agent/
    │       ├── cli.py            ← `pi` 命令
    │       ├── core/             ← AgentSession、系统提示、工具
    │       └── modes/interactive/← TUI 交互模式
    └── tui/                      ← 终端 UI 库
        └── src/pi_tui/
            ├── components/       ← Editor、SelectList、Markdown 等
            ├── tui.py            ← 差异化渲染引擎
            └── keys.py           ← Kitty 键盘协议解析器
```

---

## TypeScript → Python 映射

| TypeScript | Python |
|---|---|
| `interface X {}` | `class X(BaseModel):` 或 `@dataclass` |
| `type X = A \| B` | `X = Union[A, B]` |
| `async function f()` | `async def f()` |
| `AsyncIterable<T>` | `AsyncGenerator[T, None]` |
| `AbortSignal` | `asyncio.Event`（取消令牌） |
| `EventEmitter` | `dict[str, list[Callable]]` |
| TypeBox schema | `pydantic.BaseModel` |
| `vitest` | `pytest` + `pytest-asyncio` |

---

## 常见问题

| 问题 | 解决方案 |
|---------|----------|
| `uv: command not found` | 运行安装脚本：`curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `GEMINI_API_KEY not set` | 将你的密钥添加到 `.env` |
| `ModuleNotFoundError: pi_tui` | 使用 `uv run --package pi-coding-agent pi` 而不是直接使用 `python` |
| TUI 显示乱码字符 | 确保你的终端支持 UTF-8（iTerm2、Warp 或任何现代终端） |
| 测试被跳过 | 添加 `--live` 来运行真实的 API 测试 |
| `400 thought_signature` 错误 | 升级到最新版本 — 此问题已在 google 提供商中修复 |

---

## 使用场景

### 作为独立的编码助手

直接使用 `pi` CLI 进行代码生成、文件编辑、bash 命令执行：

```bash
# 交互式对话
uv run --package pi-coding-agent pi

# 快速任务
uv run --package pi-coding-agent pi --print "分析当前目录的代码复杂度"
```

### 作为 openclaw-python 的依赖

[openclaw-python](https://github.com/openxjarvis/openclaw-python) 使用这些包构建完整的 AI 网关系统，支持：

- Telegram、飞书等多渠道接入
- Web UI 管理界面
- 定时任务调度
- 子代理系统
- 权限管理

如果你需要完整的 AI 助手网关，请查看 openclaw-python 项目。

---

## 相关项目

- **pi-mono TypeScript** — [github.com/badlogic/pi-mono](https://github.com/badlogic/pi-mono) — 上游参考实现
- **openclaw-python** — [github.com/openxjarvis/openclaw-python](https://github.com/openxjarvis/openclaw-python) — 使用这些包构建的完整 AI 网关

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
