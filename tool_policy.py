import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass(frozen=True)
class RequestContext:
    """Immutable identity context carried from Discord interaction/message."""
    user_id: str
    guild_id: Optional[str] = None
    channel_id: Optional[str] = None
    is_admin: bool = False


@dataclass
class ToolPolicy:
    """Explicit authorization policy for tool exposure and dispatch."""
    admin_only: bool = True
    allowed_users: Set[str] = field(default_factory=set)
    allowed_channels: Set[str] = field(default_factory=set)
    allowed_tools: Set[str] = field(default_factory=set)

    @classmethod
    def from_config(cls, config: Any) -> "ToolPolicy":
        if hasattr(config, "get_tool_policy"):
            raw = config.get_tool_policy()
        elif hasattr(config, "tool_policy"):
            raw = getattr(config, "tool_policy")
        else:
            raw = {}

        if not isinstance(raw, dict):
            raw = {}

        return cls(
            admin_only=raw.get("admin_only", True),
            allowed_users=set(str(u).strip() for u in raw.get("allowed_users", []) if str(u).strip()),
            allowed_channels=set(str(c).strip() for c in raw.get("allowed_channels", []) if str(c).strip()),
            allowed_tools=set(str(t).strip() for t in raw.get("allowed_tools", []) if str(t).strip()),
        )


def is_user_authorized_for_tools(context: Optional[RequestContext], config: Any = None) -> bool:
    """Checks whether the user context has permission to use any MCP tools."""
    if context is None or not isinstance(context, RequestContext):
        return False
    if not context.user_id:
        return False

    if context.is_admin:
        return True

    policy = ToolPolicy.from_config(config)
    if policy.admin_only:
        return False

    # Non-admin access requires explicit user allowlist
    if context.user_id not in policy.allowed_users:
        return False

    if policy.allowed_channels and context.channel_id not in policy.allowed_channels:
        return False

    return True


def _matches_allowed_tool(tool_name: str, allowed_pattern: str) -> bool:
    """Matches a tool name against an allowlist pattern (exact name, wildcard, or server prefix)."""
    pat = allowed_pattern.strip()
    if not pat:
        return False
    if pat == "*" or tool_name == pat:
        return True
    if "*" in pat or "?" in pat:
        return fnmatch.fnmatch(tool_name, pat)
    # Server-level allow: e.g. "ddg" allows "ddg__duckduckgo_search"
    if tool_name.startswith(f"{pat}__"):
        return True
    return False


def is_tool_allowed(context: Optional[RequestContext], tool_name: str, config: Any = None) -> bool:
    """
    Checks whether a specific tool dispatch is authorized for the given request context.
    Enforces default-deny on missing identity and unauthorized non-admin callers.
    """
    if not is_user_authorized_for_tools(context, config):
        return False

    policy = ToolPolicy.from_config(config)

    # A blank allowlist grants nothing, including for administrators. Tools can
    # mutate data regardless of their names or server-supplied annotations.
    if not policy.allowed_tools:
        return False

    return any(_matches_allowed_tool(tool_name, pat) for pat in policy.allowed_tools)


def filter_tools_for_request(
    context: Optional[RequestContext],
    tools: List[Dict[str, Any]],
    config: Any = None,
) -> List[Dict[str, Any]]:
    """Filters a tool schema list (OpenAI or Anthropic format) according to caller authorization."""
    if not is_user_authorized_for_tools(context, config):
        return []

    policy = ToolPolicy.from_config(config)
    if not policy.allowed_tools:
        return []

    allowed = []
    for t in tools:
        # Extract tool name from OpenAI format or Anthropic format
        name = ""
        if "function" in t and isinstance(t["function"], dict):
            name = t["function"].get("name", "")
        else:
            name = t.get("name", "")

        if is_tool_allowed(context, name, config):
            allowed.append(t)

    return allowed
