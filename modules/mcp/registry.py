"""
MCP tool naming, schema compression, and argument validation constants.
"""

import hashlib
import re
from typing import Any

MAX_DESC_CHARS = 300
MAX_TOOL_ARGUMENT_CHARS = 16000


def sanitize_tool_name(server_name: str, tool_name: str) -> str:
    """
    Combines server name and tool name into a valid identifier matching ^[a-zA-Z0-9_-]{1,64}$.
    Appends a deterministic 8-char SHA-256 hash when punctuation is replaced or length exceeds 64,
    preventing collisions while strictly adhering to OpenAI/Anthropic naming specifications.
    """
    combined = f"{server_name}__{tool_name}"
    is_clean = bool(re.match(r"^[a-zA-Z0-9_-]+$", combined))
    if is_clean and len(combined) <= 64 and "__" not in server_name:
        return combined

    clean_base = re.sub(r"[^a-zA-Z0-9_-]", "_", combined)
    h = hashlib.sha256(f"{server_name}:{tool_name}".encode("utf-8")).hexdigest()[:8]
    prefix = clean_base[:55]
    return f"{prefix}_{h}"


def compress_schema(val: Any) -> Any:
    """Recursively removes excess whitespace and trims overly verbose property descriptions (>300 chars)."""
    if isinstance(val, dict):
        cleaned = {}
        for k, v in val.items():
            if k == "description" and isinstance(v, str):
                s = " ".join(v.split()).strip()
                cleaned[k] = s[:MAX_DESC_CHARS - 3] + "..." if len(s) > MAX_DESC_CHARS else s
            else:
                cleaned[k] = compress_schema(v)
        return cleaned
    elif isinstance(val, list):
        return [compress_schema(item) for item in val]
    return val
