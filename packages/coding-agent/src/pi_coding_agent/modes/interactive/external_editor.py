"""Open $EDITOR for long input. Mirrors external-editor.ts"""
from __future__ import annotations

import os
import subprocess
import tempfile


def edit_in_external_editor(initial: str = "") -> str:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as fh:
        fh.write(initial)
        path = fh.name
    try:
        subprocess.run([editor, path], check=False)
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
