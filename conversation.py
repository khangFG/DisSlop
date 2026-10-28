import re
from typing import Any, Dict, List


MAX_HISTORY_MESSAGE_CHARS = 4000
DEFAULT_MAX_INPUT_CHARS = 4000


def clean_mention(content: str, bot_user_id: int) -> str:
    """Removes the bot mention prefix/tag from message content."""
    pattern = re.compile(rf"<@!?{bot_user_id}>")
    return pattern.sub("", content).strip()


def build_conversation_context(
    prior_history: List[Any],
    trigger_message: Any,
    bot_user_id: int,
    max_history_messages: int = 6,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> List[Dict[str, str]]:
    """
    Builds conversational context strictly bounded before the trigger message,
    then appends the current trigger message once with explicit length truncation.
    Filters history to the requester and bot to avoid sharing bystander messages.
    """
    context: List[Dict[str, str]] = []
    requester_id = getattr(trigger_message.author, "id", None)
    history_by_id = {getattr(msg, "id", None): msg for msg in prior_history}

    # Filter and format prior messages
    for msg in prior_history[-max_history_messages:]:
        if getattr(msg, "id", 0) >= getattr(trigger_message, "id", 0):
            continue
        author_id = getattr(msg.author, "id", None)
        is_bot = author_id == bot_user_id
        is_requester = author_id == requester_id

        # Only include messages from the requester or the bot
        if not (is_bot or is_requester):
            continue
        if is_bot:
            # Bot messages can contain another user's tool results. Only use
            # replies whose original requester can be established locally.
            reference = getattr(msg, "reference", None)
            original = getattr(reference, "resolved", None)
            if getattr(original, "author", None) is None:
                original = history_by_id.get(getattr(reference, "message_id", None))
            if getattr(getattr(original, "author", None), "id", None) != requester_id:
                continue

        raw_content = getattr(msg, "content", "")
        cleaned = clean_mention(raw_content, bot_user_id)
        if not cleaned:
            continue

        # Cap prior history messages to prevent token bloat
        if len(cleaned) > MAX_HISTORY_MESSAGE_CHARS:
            cleaned = cleaned[:MAX_HISTORY_MESSAGE_CHARS] + "..."

        role = "assistant" if is_bot else "user"
        context.append({"role": role, "content": cleaned})

    # Format trigger message
    raw_trigger = getattr(trigger_message, "content", "")
    cleaned_trigger = clean_mention(raw_trigger, bot_user_id) or raw_trigger.strip()

    # Enforce input length truncation policy
    if len(cleaned_trigger) > max_input_chars:
        cleaned_trigger = (
            cleaned_trigger[:max_input_chars] + f"\n...[Input truncated to {max_input_chars} characters]"
        )

    context.append({"role": "user", "content": cleaned_trigger})
    return context
