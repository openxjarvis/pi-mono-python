# pi-mono-python Parity Review (against pi-mono)

This document tracks parity status and remaining gaps between:

- TypeScript source: `pi-mono`
- Python source: `pi-mono-python`

Scope priority followed:

1. Fix interactive TUI "input-only/no-assistant-render" issue
2. Strict `packages/tui` parity pass
3. High-level parity backlog for `agent`, `ai`, `coding-agent`

## 1) TUI no-response root cause and fix

### Reproduced symptom

- In interactive mode (`uv run --package pi-coding-agent pi`), user messages rendered (`You: ...`), but assistant output was often missing.

### Event trace used

Observed event sequence for `你好`:

- `agent_start`
- `turn_start`
- `message_start` (user)
- `message_end` (user)
- `message_start` (assistant)
- `message_update:text_start`
- `message_update:text_delta` ...
- `message_update:text_end`
- `message_end` (assistant)
- `turn_end`
- `agent_end`

### Root behavioral gap

Python interactive handler depended mainly on `text_delta` updates, while TS interactive UI updates from full assistant message lifecycle/state. If a provider emits sparse/no delta patterns, assistant text can be missed.

### Implemented fix

- File: `packages/coding-agent/src/pi_coding_agent/modes/interactive/tui.py`
- Updated event lifecycle handling to align with TS semantics:
  - `message_start`: initialize assistant stream row
  - `message_update`: prefer full assistant message snapshot rendering
  - `message_end`: finalize with full snapshot fallback even without deltas
  - `agent_end`: completion signal
  - `turn_end`: surface errors

Also hardened scheduler semantics:

- File: `packages/tui/src/pi_tui/tui.py`
  - `start()`: capture running loop with `get_running_loop()` fallback
  - `request_render()`: deterministic scheduling (`call_soon` on same loop, `call_soon_threadsafe` cross-thread)
  - guarded synchronous fallback with logging

## 2) `packages/tui` strict parity pass

### File-level inventory

TS `packages/tui/src` and Python `packages/tui/src/pi_tui` are structurally aligned (core modules/components present in both).

### Confirmed aligned areas

- Differential render pipeline in `tui` core
- Overlay layout/composition
- Cursor marker extraction and hardware cursor placement
- Key parsing / stdin buffering modules present
- Component set parity (`editor`, `text`, `markdown`, `loader`, `select-list`, etc.)

### Important behavioral notes

- TS uses `process.nextTick` render scheduling; Python uses asyncio loop scheduling. This is an expected language-level adaptation, now hardened for deterministic behavior.
- Python keeps defensive logging around render fallback and exceptions.

## 3) Added/updated tests

### New regression coverage

- `packages/coding-agent/tests/test_e2e.py`
  - `test_tui_initial_messages_render_without_text_delta`
    - verifies assistant output renders even when only `message_start/message_end` snapshots are emitted (no `text_delta`)

- `packages/tui/tests/test_tui.py`
  - `test_start_captures_running_event_loop`
  - `test_request_render_schedules_render_tick_on_running_loop`

### Existing focused tests rerun

- TUI and coding-agent TUI-specific tests pass after changes.

## 4) Remaining parity gaps (prioritized backlog)

These are the key differences still remaining versus TS and should be addressed in follow-up waves.

### P0 (highest)

1. `coding-agent` interactive mode feature surface
   - TS interactive mode is much richer (status/footer/tool panels, more command flows, extension-integrated UI behavior).
   - Python interactive mode remains simplified.

### P1

2. Extension/UI integration parity
   - TS has deeper extension-driven interactive UI integration and richer command/widget handling.
   - Python extension integration is present but not fully behavior-equivalent.

3. Provider behavior parity in `ai`
   - Optional provider pathways need strict behavior verification under live runs (event timing, stop reasons, errors, retries).

### P2

4. RPC and CLI edge-case parity
   - Core commands are largely present, but edge behavior and diagnostics should be matched line-by-line against TS command paths.

5. Test depth parity
   - Increase parity-specific integration tests around interactive event ordering, tool-call rendering flows, and provider-specific streaming differences.

## 5) Acceptance status for this wave

- Fixed: interactive "no assistant render" regression path
- Fixed: render scheduling hardening in TUI loop integration
- Added: regression tests for no-delta assistant rendering and scheduler behavior
- Delivered: strict parity review + prioritized remaining backlog

