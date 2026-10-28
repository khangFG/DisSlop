import re
from typing import Any, Dict, List

from config_manager import resolve_litellm_model

DEFAULT_MAX_TOKENS = 32768

SEARCH_EFFICIENCY_RULES = (
    "\n\nTool Usage Guidelines:\n"
    "- When using search tools, prefer searching first and reading the returned snippets. "
    "Only call full webpage fetch tools on the 1 best URL if the search snippets do not already contain the answer.\n"
    "- If a search tool supports a `max_results` or `count` parameter, set it to 5."
)


def should_enable_tools(user_message: str) -> bool:
    """
    Lightweight filter to avoid attaching heavy tool schemas for casual chat.
    If the message is short casual chat (<4 words, no URLs, no question/action words),
    returns False so replies cost <200 tokens and respond instantly.
    """
    if not user_message:
        return False

    text = user_message.strip().lower()

    # 1. Any link indicates browsing or fetching
    if "http://" in text or "https://" in text:
        return True

    # 2. Check for action / query trigger words that typically need tools
    action_triggers = (
        "search", "fetch", "find", "lookup", "look up", "browse", "read", "scrape",
        "check", "run", "execute", "list", "summarize", "weather", "github", "repo",
        "database", "sqlite", "query", "sql", "file", "download", "calc", "calculate",
        "price", "stock", "news", "website", "youtube", "transcript", "video",
        "tool", "tools", "mcp", "wiki", "wikipedia", "ddg", "duckduckgo",
        "continue", "tell me", "show me", "explain", "details", "detail"
    )
    for trigger in action_triggers:
        if trigger in text:
            return True

    words = re.findall(r"\b\w+\b", text)

    # 3. Question words or question mark
    question_triggers = {"what", "who", "where", "when", "why", "how", "can", "could", "please", "which"}
    if "?" in text or any(w in question_triggers for w in words):
        return True

    # 4. Short greetings / acknowledgments without action words
    casual_words = {
        "hi", "hello", "hey", "sup", "yo", "thanks", "thx", "thank", "you",
        "bye", "ok", "okay", "cool", "nice", "gm", "gn", "good", "morning", "night"
    }
    if len(words) < 4 and set(words).issubset(casual_words):
        return False

    return len(words) >= 4


def _prepare_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Coalesces consecutive same-role text messages and ensures the conversation
    starts with a 'user' role to satisfy Anthropic's message validation rules.
    Preserves list content (tool results / content blocks).
    """
    cleaned: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role not in ("user", "assistant"):
            role = "user"

        if isinstance(content, str):
            if not content.strip():
                continue
            if cleaned and cleaned[-1]["role"] == role and isinstance(cleaned[-1].get("content"), str):
                cleaned[-1]["content"] = f"{cleaned[-1]['content']}\n\n{content}"
            else:
                cleaned.append({"role": role, "content": content})
        else:
            cleaned.append({"role": role, "content": content})

    if not cleaned:
        cleaned.append({"role": "user", "content": "Hello"})
    elif cleaned[0]["role"] != "user":
        cleaned.insert(0, {"role": "user", "content": "Context start."})

    return cleaned


def _build_litellm_params(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic helper converting a profile dict into kwargs for litellm.acompletion:
    - model: resolved via resolve_litellm_model(api_format, model, base_url)
    - api_base: mapped from profile["base_url"]
    - api_key: mapped from profile["api_key"]
    - max_tokens: mapped from profile["max_tokens"]
    """
    api_format = (profile.get("api_format") or "litellm").lower().strip()
    raw_model = (profile.get("model") or "").strip()
    base_url = profile.get("base_url") or None
    api_key = (profile.get("api_key") or "").strip()
    max_tokens = int(profile.get("max_tokens") or DEFAULT_MAX_TOKENS)

    resolved_model = resolve_litellm_model(api_format, raw_model, base_url)

    params: Dict[str, Any] = {
        "model": resolved_model,
        "max_tokens": max_tokens,
    }
    if api_key:
        params["api_key"] = api_key
    if base_url:
        params["api_base"] = base_url

    return params
