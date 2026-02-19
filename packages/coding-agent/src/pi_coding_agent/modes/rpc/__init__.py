"""RPC mode subpackage — mirrors modes/rpc/ in the TypeScript source."""
from .client import RpcClient, RpcClientOptions, RpcEventListener
from .mode import run_rpc_mode
from .types import (
    RpcCommand,
    RpcExtensionUIRequest,
    RpcExtensionUIResponse,
    RpcResponse,
    RpcResponseError,
    RpcResponseSuccess,
    RpcSessionState,
    RpcSlashCommand,
)

__all__ = [
    "RpcClient",
    "RpcClientOptions",
    "RpcCommand",
    "RpcEventListener",
    "RpcExtensionUIRequest",
    "RpcExtensionUIResponse",
    "RpcResponse",
    "RpcResponseError",
    "RpcResponseSuccess",
    "RpcSessionState",
    "RpcSlashCommand",
    "run_rpc_mode",
]
