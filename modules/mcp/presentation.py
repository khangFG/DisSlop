"""
MCP tool output formatting, sanitization, and truncation presentation.
"""

from dataclasses import dataclass
import json
import re
from typing import Any, Optional

MAX_TOOL_RESULT_CHARS = 128000
HEAD_CHARS = 60000
MID_CHARS = 35000
TAIL_CHARS = 30000

MARKER_MID = "\n...[Middle section sampled]...\n"
MARKER_TAIL = "\n...[Concluding section sampled]...\n"
OVERHEAD_CHARS = len(MARKER_MID) + len(MARKER_TAIL)


@dataclass
class ToolOutcome:
    """Structured representation of tool execution result."""
    text: str
    is_error: bool = False
    structured_content: Optional[Any] = None


def format_tool_result(tool_name: str, text: str) -> str:
    """
    Format-aware tool presentation:
    - Replaces large inline base64 blobs with placeholder.
    - Preserves short source code (<script>, HTML, Python indentation, JSON) untouched.
    - Never infers disposable content from a tool name.
    - Strictly bounds text within MAX_TOOL_RESULT_CHARS (128,000) including all markers.
    """
    if not text:
        return ""

    out = text

    # Strip large base64 payloads regardless of tool type
    out = re.sub(r"data:[a-zA-Z0-9/+-]+;base64,[A-Za-z0-9+/=]{30,}", "[base64_data_omitted]", out)

    if len(out) <= MAX_TOOL_RESULT_CHARS:
        return out

    # Head + Middle + Tail sampling strictly bounded by MAX_TOOL_RESULT_CHARS
    avail = MAX_TOOL_RESULT_CHARS - OVERHEAD_CHARS
    head_len = min(HEAD_CHARS, avail // 3)
    mid_len = min(MID_CHARS, avail // 3)
    tail_len = avail - head_len - mid_len

    head_part = out[:head_len]
    mid_start = max(head_len, (len(out) // 2) - (mid_len // 2))
    mid_part = out[mid_start:mid_start + mid_len]
    tail_part = out[-tail_len:]

    final_result = f"{head_part}{MARKER_MID}{mid_part}{MARKER_TAIL}{tail_part}"
    return final_result[:MAX_TOOL_RESULT_CHARS]


def format_tool_outcome(tool_name: str, outcome: ToolOutcome) -> str:
    """Renders a ToolOutcome into bounded text preserving error status and structured content."""
    text_content = outcome.text
    if outcome.structured_content is not None:
        try:
            structured = json.dumps(outcome.structured_content, ensure_ascii=False)
        except Exception:
            structured = "(Structured result could not be serialized)"
        text_content = f"{text_content}\n{structured}" if text_content else structured

    if outcome.is_error:
        text_content = f"Tool returned an error:\n{text_content}"
    return format_tool_result(tool_name, text_content)


# Compatibility alias
compact_tool_output = format_tool_result
