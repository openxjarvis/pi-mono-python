"""
Open a URL in the platform browser — mirrors packages/coding-agent/src/utils/open-browser.ts
"""
from __future__ import annotations

import os
import subprocess


def open_browser(target: str) -> None:
    if os.name == "nt":
        cmd, args = "rundll32", ["url.dll,FileProtocolHandler", target]
    elif os.uname().sysname == "Darwin":
        cmd, args = "open", [target]
    else:
        cmd, args = "xdg-open", [target]
    try:
        subprocess.Popen([cmd, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        pass
