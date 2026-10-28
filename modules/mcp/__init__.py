"""
MCP Subsystem: Process isolation, registry naming, schema compression, and presentation.
"""

from .environment import build_isolated_env
from .registry import sanitize_tool_name, compress_schema, MAX_DESC_CHARS, MAX_TOOL_ARGUMENT_CHARS
from .presentation import (
    ToolOutcome,
    format_tool_result,
    format_tool_outcome,
    compact_tool_output,
    MAX_TOOL_RESULT_CHARS,
    HEAD_CHARS,
    MID_CHARS,
    TAIL_CHARS,
    MARKER_MID,
    MARKER_TAIL,
    OVERHEAD_CHARS,
)

__all__ = [
    "build_isolated_env",
    "sanitize_tool_name",
    "compress_schema",
    "MAX_DESC_CHARS",
    "MAX_TOOL_ARGUMENT_CHARS",
    "ToolOutcome",
    "format_tool_result",
    "format_tool_outcome",
    "compact_tool_output",
    "MAX_TOOL_RESULT_CHARS",
    "HEAD_CHARS",
    "MID_CHARS",
    "TAIL_CHARS",
    "MARKER_MID",
    "MARKER_TAIL",
    "OVERHEAD_CHARS",
]
