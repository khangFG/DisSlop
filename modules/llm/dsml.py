import json
import re
import uuid
from typing import Any, Dict, List, Set


def parse_dsml_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Parses DeepSeek Markup Language (DSML) tool call tokens emitted in text by models
    like SenseNova, DeepSeek, or other third-party OpenAI-compatible providers:
    < | DSML | tool_calls>
    < | DSML | invoke name="tool_name">
    < | DSML | parameter name="param_name" string="true">value</ | DSML | parameter>
    </ | DSML | invoke>
    </ | DSML | tool_calls>
    """
    if not content or ("DSML" not in content and "tool_calls" not in content):
        return []

    m = re.search(
        r"<\s*\|\s*DSML\s*\|\s*tool_calls\s*>(.*?)</\s*\|\s*DSML\s*\|\s*tool_calls\s*>",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []

    body = m.group(1)
    invokes = re.findall(
        r"<\s*\|\s*DSML\s*\|\s*invoke\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</\s*\|\s*DSML\s*\|\s*invoke\s*>",
        body,
        re.DOTALL | re.IGNORECASE,
    )

    tool_calls = []
    for idx, (tool_name, invoke_body) in enumerate(invokes):
        params = {}
        param_matches = re.findall(
            r"<\s*\|\s*DSML\s*\|\s*parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</\s*\|\s*DSML\s*\|\s*parameter\s*>",
            invoke_body,
            re.DOTALL | re.IGNORECASE,
        )
        for p_name, p_val in param_matches:
            val_clean = p_val.strip()
            try:
                params[p_name] = json.loads(val_clean)
            except Exception:
                params[p_name] = val_clean

        tool_calls.append(
            {
                "id": f"call_dsml_{idx}_{uuid.uuid4().hex[:6]}",
                "type": "function",
                "function": {
                    "name": tool_name.strip(),
                    "arguments": json.dumps(params),
                },
            }
        )
    return tool_calls


def strip_dsml_tags(content: str) -> str:
    """Removes DSML / XML tool execution tags from text before presenting to user."""
    if not content or "DSML" not in content:
        return content
    cleaned = re.sub(
        r"<\s*\|\s*DSML\s*\|\s*tool_calls\s*>.*?</\s*\|\s*DSML\s*\|\s*tool_calls\s*>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"<\s*\|\s*DSML\s*\|\s*[a-zA-Z_]+[^>]*>", "", cleaned)
    cleaned = re.sub(r"</\s*\|\s*DSML\s*\|\s*[a-zA-Z_]+[^>]*>", "", cleaned)
    return cleaned.strip()


def resolve_tool_name(raw_name: str, available_tools: Set[str]) -> str:
    """
    Resolves unprefixed tool names or common aliases (e.g. fetch_page -> fetch__fetch)
    to the exact offered MCP tool name.
    """
    if raw_name in available_tools:
        return raw_name
    # 1. Server-prefix omitted: e.g. "search_wikipedia" -> "wiki__search_wikipedia"
    for t in available_tools:
        if t.split("__", 1)[-1].lower() == raw_name.lower():
            return t
    # 2. Known common aliases for web fetch / search
    clean = raw_name.lower().replace("-", "_")
    if clean in ("fetch_page", "fetch_url", "web_fetch", "fetch", "read_url", "browse_url"):
        for t in available_tools:
            if t in ("fetch__fetch", "parallel__web_fetch") or t.endswith("__web_fetch") or t.endswith("__fetch"):
                return t
    if clean in ("search", "web_search", "google_search", "search_web", "wikipedia_search"):
        for t in available_tools:
            if t in ("wiki__search_wikipedia", "parallel__web_search") or t.endswith("__search") or t.endswith("__web_search"):
                return t
    return raw_name
