"""
LLM Subsystem: DSML parsing, reasoning extraction, routing, and prompt caching.
"""

from .dsml import parse_dsml_tool_calls, strip_dsml_tags, resolve_tool_name
from .reasoning import _extract_reasoning_text
from .routing import (
    SEARCH_EFFICIENCY_RULES,
    _build_litellm_params,
    _prepare_anthropic_messages,
    should_enable_tools,
)

__all__ = [
    "parse_dsml_tool_calls",
    "strip_dsml_tags",
    "resolve_tool_name",
    "_extract_reasoning_text",
    "SEARCH_EFFICIENCY_RULES",
    "_build_litellm_params",
    "_prepare_anthropic_messages",
    "should_enable_tools",
]
