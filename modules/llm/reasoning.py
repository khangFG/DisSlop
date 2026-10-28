from typing import Any, Optional


def _extract_reasoning_text(message: Any) -> Optional[str]:
    """
    Extract reasoning or thinking text from models that emit chain-of-thought
    (DeepSeek-R1, GLM, Claude thinking, Gemini thoughts, etc.).
    Standardized across LiteLLM reasoning_content and provider-specific structures.
    """
    if message is None:
        return None

    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    provider_fields = getattr(message, "provider_specific_fields", None)
    if isinstance(provider_fields, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            val = provider_fields.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    content = getattr(message, "content", None)
    if isinstance(content, list):
        thinking_parts = []
        for b in content:
            if getattr(b, "type", None) == "thinking" or hasattr(b, "thinking"):
                t = getattr(b, "thinking", "")
                if isinstance(t, str) and t.strip():
                    thinking_parts.append(t.strip())
        if thinking_parts:
            return "\n\n".join(thinking_parts).strip()

    return None
