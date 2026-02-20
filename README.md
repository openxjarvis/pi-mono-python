# pi-mono-python

Python port of the [pi-mono](../pi-mono) TypeScript monorepo — four packages with aligned code, logic, algorithms, and folder structure.

| TypeScript | Python | Description |
|---|---|---|
| `@mariozechner/pi-ai` | `pi_ai` | Unified LLM streaming layer (Google, Anthropic, OpenAI, Bedrock, …) |
| `@mariozechner/pi-agent-core` | `pi_agent` | Agent loop, tool execution, state management |
| `@mariozechner/pi-coding-agent` | `pi_coding_agent` | Coding agent CLI with file tools: read, write, edit, bash, grep, find, ls |
| `@mariozechner/pi-tui` | `pi_tui` | Terminal UI library with differential rendering engine |

---

## Requirements

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| [`uv`](https://docs.astral.sh/uv/) | latest |

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation

```bash
git clone https://github.com/openxjarvis/pi-mono-python.git
cd pi-mono-python

# Install all four packages and their dependencies in one step
uv sync
```

---

## Configuration

Create or edit `.env` in the project root:

```env
# Google Gemini (default provider)
GEMINI_API_KEY=your_key_here

# Optional — add whichever providers you need
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=        # alternative to GEMINI_API_KEY
AWS_ACCESS_KEY_ID=     # for Bedrock
AWS_SECRET_ACCESS_KEY=
```

> `.env` is loaded automatically at runtime. **Never commit it to git.**

---

## Usage

### Interactive TUI (default)

```bash
uv run --package pi-coding-agent pi
```

Launches the full-featured terminal UI:

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line in input |
| `/` | Slash command completion |
| `@` | File path completion |
| `Ctrl+P` | Cycle to next model |
| `Ctrl+C` / `Esc` | Quit |

**Slash commands** (type `/` to see completions):

| Command | Description |
|---------|-------------|
| `/model <name>` | Switch model by name |
| `/thinking <level>` | Set thinking level: `minimal` · `low` · `medium` · `high` · `xhigh` |
| `/compact` | Compact conversation context |
| `/session` | Show session statistics (tokens, cost) |
| `/tools` | List active tools |

---

### Non-interactive (single prompt)

```bash
uv run --package pi-coding-agent pi --print "Write a quicksort in Python"
```

Prints the agent's response to stdout and exits — useful for scripting.

---

### Specify a model

```bash
# By model ID
uv run --package pi-coding-agent pi --model gemini-2.5-pro-preview

# By provider + model
uv run --package pi-coding-agent pi --provider google --model gemini-2.0-flash

# List all available models
uv run --package pi-coding-agent pi --list-models
```

---

### Session management

```bash
# Resume the most recent session
uv run --package pi-coding-agent pi --continue

# Pick a previous session from a list
uv run --package pi-coding-agent pi --resume
```

---

### Full CLI reference

```bash
uv run --package pi-coding-agent pi --help
```

---

## Running Tests

### All tests

```bash
uv run pytest
```

### Per-package

```bash
uv run pytest packages/tui/tests/          # TUI components
uv run pytest packages/ai/tests/           # AI providers
uv run pytest packages/agent/tests/        # Agent core
uv run pytest packages/coding-agent/tests/ # CLI + coding agent
```

### Live API tests (requires `GEMINI_API_KEY`)

```bash
uv run pytest packages/ai/tests/ --live -v

# Or via environment variable
LIVE_TESTS=1 uv run pytest packages/ai/tests/ -v
```

> All tests run against mocks by default — no API key required, no quota consumed.

---

## Test Status

| Package | Tests | Status |
|---------|-------|--------|
| `pi_tui` | 135 | ✅ passed |
| `pi_ai` + `pi_agent` | 156 | ✅ passed (7 skipped = live-only) |
| `pi_coding_agent` | 287 | ✅ passed |
| **Total** | **578** | **✅ all passing** |

---

## Project Structure

```
pi-mono-python/
├── .env                          ← API keys (never commit)
├── pyproject.toml                ← uv workspace root
├── conftest.py                   ← global pytest config (.env loader)
└── packages/
    ├── ai/                       ← LLM provider layer
    │   └── src/pi_ai/
    │       ├── providers/        ← google.py, openai.py, anthropic.py, …
    │       ├── stream.py         ← unified stream_simple() / complete_simple()
    │       └── utils/            ← overflow detection, JSON parse, …
    ├── agent/                    ← core agent loop
    │   └── src/pi_agent/
    │       ├── agent.py          ← main run loop
    │       ├── tools/            ← tool registry & execution
    │       └── session.py        ← session state
    ├── coding-agent/             ← CLI entry point & extensions
    │   └── src/pi_coding_agent/
    │       ├── cli.py            ← `pi` command
    │       ├── core/             ← AgentSession, system prompt, tools
    │       └── modes/interactive/← TUI interactive mode
    └── tui/                      ← terminal UI library
        └── src/pi_tui/
            ├── components/       ← Editor, SelectList, Markdown, …
            ├── tui.py            ← differential rendering engine
            └── keys.py           ← Kitty keyboard protocol parser
```

---

## TypeScript → Python Mapping

| TypeScript | Python |
|---|---|
| `interface X {}` | `class X(BaseModel):` or `@dataclass` |
| `type X = A \| B` | `X = Union[A, B]` |
| `async function f()` | `async def f()` |
| `AsyncIterable<T>` | `AsyncGenerator[T, None]` |
| `AbortSignal` | `asyncio.Event` (cancellation token) |
| `EventEmitter` | `dict[str, list[Callable]]` |
| TypeBox schema | `pydantic.BaseModel` |
| `vitest` | `pytest` + `pytest-asyncio` |

---

## FAQ

| Problem | Solution |
|---------|----------|
| `uv: command not found` | Run the install script: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `GEMINI_API_KEY not set` | Add your key to `.env` |
| `ModuleNotFoundError: pi_tui` | Use `uv run --package pi-coding-agent pi` instead of `python` directly |
| TUI shows garbled characters | Ensure your terminal supports UTF-8 (iTerm2, Warp, or any modern terminal) |
| Tests are skipped | Add `--live` to run real API tests |
| `400 thought_signature` error | Upgrade to the latest version — this is fixed in the google provider |
