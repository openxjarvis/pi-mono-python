"""
Interactive component stubs aligned to TypeScript component naming.

These classes are placeholders to preserve structure/API names while the
full Textual implementation is incrementally completed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComponentStub:
    name: str


_NAMES = [
    "assistant_message",
    "bash_execution",
    "bordered_loader",
    "branch_summary_message",
    "compaction_summary_message",
    "config_selector",
    "countdown_timer",
    "custom_editor",
    "custom_message",
    "diff",
    "dynamic_border",
    "extension_editor",
    "extension_input",
    "extension_selector",
    "footer",
    "keybinding_hints",
    "login_dialog",
    "model_selector",
    "oauth_selector",
    "scoped_models_selector",
    "session_selector",
    "settings_selector",
    "show_images_selector",
    "theme_selector",
    "thinking_selector",
    "tool_execution",
    "tree_selector",
    "user_message",
    "user_message_selector",
    "visual_truncate",
]

STUB_COMPONENTS: dict[str, ComponentStub] = {name: ComponentStub(name=name) for name in _NAMES}
