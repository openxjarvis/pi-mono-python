# Alignment report — pi-mono TypeScript v0.84.3

**Date:** 2026-08-25  
**Target:** [pi-mono](../pi-mono) TypeScript **v0.84.3**  
**Python:** same `packages/` tree as TypeScript (`ai`, `agent`, `coding-agent`, `tui`, `telemetry`, `protocol`, `client`, `server`, `session-backends`, `evals`)

## Correspondence contract

- Same folder tree: `foo/bar.ts` → `foo/bar.py` (or `foo/bar/__init__.py`)
- Same public API and behavior; Python naming is `snake_case` + `asyncio.Event`
- Node-only bits become Python equivalents (`socket.AF_UNIX`, `httpx` pool, `ctypes` modifiers, thread image worker, `sqlite3`)
- Stubs / leftover `stubs.py` registries do not count as done

## Inventory (portable `src/*.ts`, tests excluded)

| Package | TS files | Python twins | Missing |
|---|---:|---:|---:|
| `ai` | 176 | 176 | 0 |
| `agent` | 50 | 50 | 0 |
| `coding-agent` | 202 | 202 | 0 |
| `tui` | 40 | 40 | 0 |
| `telemetry` | 6 | 6 | 0 |
| `protocol` | 8 | 8 | 0 |
| `client` | 10 | 10 | 0 |
| `server` | 17 | 17 | 0 |
| `session-backends/sqlite-node` | 18 | 18 | 0 |
| `evals` | 8 | 8 | 0 |

Every portable TypeScript `src/` file in `packages/` has a Python twin.

## Python equivalents (Node-only)

| TypeScript | Python |
|---|---|
| `harness/env/nodejs.ts` | `harness/env/python.py` (`ExecutionEnv` on the local FS / subprocess) |
| `cli/experimental` unix socket | `cli_sub/experimental/server.py` (`socket.AF_UNIX` JSONL) |
| `core/http-dispatcher.ts` (undici) | `core/http_dispatcher.py` (`httpx` connection pool) |
| `native-modifiers.ts` (`.node` addon) | `pi_tui/native_modifiers.py` (`ctypes` Carbon / Win32) |
| `utils/photon.ts` (WASM) | `utils/photon.py` (Pillow) |
| `utils/clipboard-native.ts` | `utils/clipboard_native.py` (OS clipboard CLIs) |
| `utils/image-resize-worker.ts` | `utils/image_resize_worker.py` (thread pool) |
| `node:sqlite` / `sqlite-node` | `sqlite3` in `packages/session-backends/sqlite-node` |
| `vitest-evals` | `pi_evals.pytest_evals` |
| Unix `node:net` sockets | `asyncio.open_unix_connection` / `start_unix_server` |

## Intentionally out of scope

- Bun SEA / `bun/*` register hooks, npm installer
- Compiling Node `.node` binaries (ctypes / pure-Python instead)
- `mom` / `pods` / `web-ui` (not present in current TypeScript `packages/`)
- `utils/windows-self-update.ts` — quarantine installer; Python ships `windows_self_update.py` as an explicit non-implementation (`WINDOWS_SELF_UPDATE_SUPPORTED = False`)
- coding-agent `examples/` sample extensions (not `src/`)

## Notes on thin twins

- `pi_ai/api/*` and some `cli/*` files are re-export shims onto the Python-idiomatic modules (`providers/openai_completions.py`, `cli_sub/…`). The behavior lives in those modules, not a second copy.
- Interactive mode is `InteractiveMode` (`create_interactive_tui`, main-screen / alt-screen, slash commands, selectors). Non-TTY falls back to readline.
- Deepened this round (behavior, not shims): `client/transcript.py`, `client/remote_session.py`, `package_manager_cli.py`, harness `session/testing/conformance.py`, `armin` / `model_selector` / `scoped_models_selector` / `config_selector`, `tree_selector` (ASCII flatten/filter/viewport + `on_select` → `navigate_tree`), `session_selector` (scope/sort/rename/delete). Slash commands now include `/export` `/copy` `/reload` `/logout` `/share` `/debug`.
- `pi_tui` includes the alternate-screen stack: `layout`, `tui_main_screen`, `tui_alt_screen`, `scroll_view`, `alt_screen_search`, Kitty crop APIs, LaTeX matrices / stacked fractions.

## How to re-check

```bash
uv run pytest
```
