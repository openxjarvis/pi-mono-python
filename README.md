# pi-mono-python

Python mirror of the [pi-mono](../pi-mono) TypeScript monorepo.

Mirrors four packages with aligned code, logic, and folder structure:

| TypeScript | Python | Description |
|---|---|---|
| `@mariozechner/pi-ai` | `pi_ai` | Unified LLM API (Anthropic, OpenAI, Google, Bedrock) |
| `@mariozechner/pi-agent-core` | `pi_agent` | Agent loop, state management, tool execution |
| `@mariozechner/pi-coding-agent` | `pi_coding_agent` | Coding agent with tools: read, write, edit, bash, grep, find, ls |
| `@mariozechner/pi-tui` | `pi_tui` | Terminal UI library with differential rendering engine |

---

## 前置条件 / Prerequisites

| 工具 | 最低版本 |
|------|---------|
| Python | 3.11+ |
| `uv` | 最新版 |

```bash
# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 快速开始 / Quick Start

```bash
# 1. 进入项目目录
cd pi-mono-python

# 2. 安装所有依赖（一次性安装全部 4 个子包）
uv sync

# 3. 配置 API Key（编辑 .env 文件）
# GEMINI_API_KEY=your_key_here

# 4. 启动 pi（交互 TUI 模式）```

---

## 配置 API Key / Configure API Key

编辑根目录的 `.env` 文件：

```env
# Google Gemini（默认 provider）
GEMINI_API_KEY=your_key_here

# 其他可选
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

`.env` 在运行时自动加载，**不要提交到 git**。

---

## 运行 `pi` CLI

### 交互模式（默认，启动 TUI）

```bash
uv run --package pi-coding-agent pi
```

进入全功能终端 UI：

| 快捷键 | 功能 |
|--------|------|
| `Enter` | 发送消息 |
| `Shift+Enter` | 输入框换行 |
| `/` | 触发 slash command 补全 |
| `@` | 文件路径补全 |
| `Ctrl+C` / `Esc` | 退出 |

### 非交互（一次性提问）

```bash
uv run --package pi-coding-agent pi --print "用 Python 写一个快速排序"
```

### 指定模型

```bash
# 使用 Gemini
uv run --package pi-coding-agent pi --model gemini-2.5-pro-preview

# 明确指定 provider 和 model
uv run --package pi-coding-agent pi --provider google --model gemini-2.0-flash
```

### 继续上次会话

```bash
uv run --package pi-coding-agent pi --continue
```

### 选择历史会话恢复

```bash
uv run --package pi-coding-agent pi --resume
```

### 查看所有可用模型

```bash
uv run --package pi-coding-agent pi --list-models
```

### 查看完整帮助

```bash
uv run --package pi-coding-agent pi --help
```

---

## 运行测试 / Run Tests

### 全部测试（530 个）

```bash
uv run pytest
```

### 按包单独测试

```bash
# TUI 组件（135 个）
uv run --package pi-tui pytest packages/tui/tests/ -v

# AI 提供商
uv run pytest packages/ai/tests/ -v

# Agent 核心
uv run pytest packages/agent/tests/ -v

# Coding Agent（245 个）
uv run pytest packages/coding-agent/tests/ -v
```

### 真实 Gemini API 测试（Live）

```bash
# 确保 .env 中 GEMINI_API_KEY 已填写
uv run pytest packages/ai/tests/ -v --live

# 或通过环境变量
LIVE_TESTS=1 uv run pytest packages/ai/tests/ -v
```

> 不加 `--live` 的测试全部使用 mock，不消耗 API 额度。

---

## 项目结构 / Project Structure

```
pi-mono-python/
├── .env                              ← API Keys（勿提交 git）
├── pyproject.toml                    ← uv workspace 根配置
├── conftest.py                       ← 全局测试配置（加载 .env）
├── packages/
│   ├── ai/                           ← LLM 提供商层
│   │   └── src/pi_ai/
│   │       ├── providers/            ← google.py, openai.py, anthropic.py...
│   │       └── stream.py             ← 统一流式输出
│   ├── agent/                        ← 核心 Agent 逻辑
│   │   └── src/pi_agent/
│   │       ├── agent.py              ← Agent 主循环
│   │       ├── tools/                ← 工具执行层
│   │       └── session.py            ← 会话管理
│   ├── coding-agent/                 ← CLI 入口 & 扩展系统
│   │   └── src/pi_coding_agent/
│   │       ├── cli.py                ← `pi` 命令入口
│   │       ├── modes/interactive/    ← TUI 交互模式
│   │       └── core/tools/           ← read, write, edit, bash, grep...
│   └── tui/                          ← 终端 UI 库
│       └── src/pi_tui/
│           ├── components/           ← Editor, SelectList, Markdown...
│           ├── tui.py                ← 差异渲染引擎
│           └── keys.py               ← Kitty 键盘协议解析
```

---

## TypeScript → Python 映射

| TypeScript | Python |
|---|---|
| `interface X {}` | `class X(BaseModel):` 或 `@dataclass` |
| `type X = A \| B` | `X = Union[A, B]` |
| `async function f()` | `async def f()` |
| `AsyncIterable<T>` | `AsyncGenerator[T, None]` |
| `AbortSignal` | `asyncio.Event`（取消令牌）|
| `EventEmitter` | `dict[str, list[Callable]]` |
| TypeBox schema | `pydantic.BaseModel` |
| `vitest` | `pytest` + `pytest-asyncio` |

---

## 常见问题 / FAQ

| 问题 | 解决方案 |
|------|---------|
| `uv: command not found` | 运行安装脚本：`curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `GEMINI_API_KEY not set` | 在 `.env` 文件中填入 key |
| `ModuleNotFoundError: pi_tui` | 使用 `uv run --package pi-coding-agent pi` 而非直接 `python` |
| TUI 显示乱码 | 确保终端支持 UTF-8，推荐 iTerm2 或 Warp |
| 测试被跳过（skipped） | 加 `--live` 运行真实 API 测试 |

---

## 测试状态 / Test Status

| 包 | 测试数 | 状态 |
|----|--------|------|
| `pi_tui` | 135 | ✅ passed |
| `pi_ai` + `pi_agent` | 150 | ✅ passed（7 skipped = live only）|
| `pi_coding_agent` | 245 | ✅ passed |
| **合计** | **530** | **✅ 全部通过** |
