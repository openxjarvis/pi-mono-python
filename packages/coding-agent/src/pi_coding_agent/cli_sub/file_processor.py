"""
Process @file CLI arguments into text content and image attachments.

Mirrors packages/coding-agent/src/cli/file-processor.ts
"""
from __future__ import annotations

import base64
import mimetypes
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
}


@dataclass
class ProcessedFiles:
    text: str = ""
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessFileOptions:
    auto_resize_images: bool = True


def _detect_image_mime_type(path: str) -> str | None:
    """Detect image MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type and mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return mime_type
    return None


async def process_file_arguments(
    file_args: list[str],
    options: ProcessFileOptions | None = None,
) -> ProcessedFiles:
    """Process @file arguments into text content and image attachments."""
    opts = options or ProcessFileOptions()
    result = ProcessedFiles()

    for file_arg in file_args:
        # Expand ~ and resolve relative paths
        abs_path = str(Path(os.path.expanduser(file_arg)).resolve())

        if not os.path.exists(abs_path):
            print(f"Error: File not found: {abs_path}", file=sys.stderr)
            sys.exit(1)

        file_size = os.path.getsize(abs_path)
        if file_size == 0:
            continue

        mime_type = _detect_image_mime_type(abs_path)

        if mime_type:
            with open(abs_path, "rb") as f:
                content = f.read()
            b64 = base64.b64encode(content).decode()
            image: dict[str, Any] = {"type": "image", "mimeType": mime_type, "data": b64}
            result.images.append(image)
            result.text += f'<file name="{abs_path}"></file>\n'
        else:
            try:
                with open(abs_path, encoding="utf-8") as f:
                    content_str = f.read()
                result.text += f'<file name="{abs_path}">\n{content_str}\n</file>\n'
            except Exception as e:
                print(f"Error: Could not read file {abs_path}: {e}", file=sys.stderr)
                sys.exit(1)

    return result
