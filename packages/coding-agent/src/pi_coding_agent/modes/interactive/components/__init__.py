"""Interactive mode UI components — mirrors TypeScript component namespace."""

from .component import Component
from .assistant_message import AssistantMessageComponent
from .user_message import UserMessageComponent
from .tool_execution import ToolExecutionComponent
from .footer import FooterComponent, format_cwd_for_footer, format_tokens
from .model_selector import ModelSelectorComponent
from .thinking_selector import ThinkingSelectorComponent
from .settings_selector import SettingsSelectorComponent
from .session_selector import SessionSelectorComponent
from .login_dialog import LoginDialogComponent
from .trust_selector import TrustSelectorComponent
from .first_time_setup import FirstTimeSetupComponent, FirstTimeSetupResult
from .countdown_timer import CountdownTimer
from .bash_execution import BashExecutionComponent
from .bordered_loader import BorderedLoaderComponent
from .branch_summary_message import BranchSummaryMessageComponent
from .compaction_summary_message import CompactionSummaryMessageComponent
from .config_selector import ConfigSelectorComponent
from .custom_editor import CustomEditorComponent
from .custom_message import CustomMessageComponent
from .diff import DiffComponent, render_diff
from .dynamic_border import DynamicBorder
from .extension_editor import ExtensionEditorComponent
from .extension_input import ExtensionInputComponent
from .extension_selector import ExtensionSelectorComponent
from .keybinding_hints import KeybindingHintsComponent
from .oauth_selector import OAuthSelectorComponent
from .scoped_models_selector import ScopedModelsSelectorComponent
from .show_images_selector import ShowImagesSelectorComponent
from .theme_selector import ThemeSelectorComponent
from .tree_selector import TreeSelectorComponent
from .user_message_selector import UserMessageSelectorComponent
from .visual_truncate import VisualTruncateComponent, VisualTruncateResult, truncate_to_visual_lines

__all__ = [
    "Component",
    "AssistantMessageComponent",
    "UserMessageComponent",
    "ToolExecutionComponent",
    "FooterComponent",
    "format_cwd_for_footer",
    "format_tokens",
    "ModelSelectorComponent",
    "ThinkingSelectorComponent",
    "SettingsSelectorComponent",
    "SessionSelectorComponent",
    "LoginDialogComponent",
    "TrustSelectorComponent",
    "FirstTimeSetupComponent",
    "FirstTimeSetupResult",
    "CountdownTimer",
    "BashExecutionComponent",
    "BorderedLoaderComponent",
    "BranchSummaryMessageComponent",
    "CompactionSummaryMessageComponent",
    "ConfigSelectorComponent",
    "CustomEditorComponent",
    "CustomMessageComponent",
    "DiffComponent",
    "DynamicBorder",
    "ExtensionEditorComponent",
    "ExtensionInputComponent",
    "ExtensionSelectorComponent",
    "KeybindingHintsComponent",
    "OAuthSelectorComponent",
    "ScopedModelsSelectorComponent",
    "ShowImagesSelectorComponent",
    "ThemeSelectorComponent",
    "TreeSelectorComponent",
    "UserMessageSelectorComponent",
    "VisualTruncateComponent",
    "VisualTruncateResult",
    "truncate_to_visual_lines",
    "render_diff",
]
