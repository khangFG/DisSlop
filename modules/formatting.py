"""
Discord message chunking, mention suppression, and administrative input parsing helpers.
"""

import json
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Set

import discord


def _parse_admin_ids(raw: Any) -> Set[str]:
    """Parses single ID, comma-separated IDs, or collections of admin IDs."""
    if not raw:
        return set()
    if isinstance(raw, (set, list, tuple)):
        return {str(x).strip() for x in raw if str(x).strip()}
    return {uid.strip() for uid in re.split(r"[,;\s]+", str(raw)) if uid.strip()}


def split_into_chunks(text: str, max_chars: int = 1950) -> List[str]:
    """Splits text on newline/space boundaries so it fits Discord's message length limits."""
    if not text:
        return ["(Empty response)"]
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        split_idx = remaining.rfind("\n", 0, max_chars)
        if split_idx == -1 or split_idx < max_chars // 2:
            split_idx = remaining.rfind(" ", 0, max_chars)
        if split_idx == -1 or split_idx < max_chars // 2:
            split_idx = max_chars

        chunk = remaining[:split_idx].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_idx:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


async def send_chunked_reply(message: discord.Message, text: str) -> None:
    """Sends long responses cleanly across multiple messages without triggering length limits or pinging mentions."""
    safe_mentions = discord.AllowedMentions.none()
    chunks = split_into_chunks(text, max_chars=1950)
    for i, chunk in enumerate(chunks):
        if i == 0:
            await message.reply(chunk, mention_author=False, allowed_mentions=safe_mentions)
        else:
            await message.channel.send(chunk, reference=message, allowed_mentions=safe_mentions)


def parse_admin_arguments(raw_str: Optional[str]) -> List[str]:
    """
    Parses command arguments supporting both structured JSON arrays and command-line strings.
    Preserves Windows backslashes and path syntax without corruption.
    """
    if not raw_str or not raw_str.strip():
        return []
    s = raw_str.strip()
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                if not all(isinstance(x, str) for x in data):
                    raise ValueError("JSON arguments array must contain only strings.")
                return data
        except json.JSONDecodeError as jde:
            raise ValueError(f"Malformed JSON argument array: {jde}")

    try:
        # Use posix=False on Windows to preserve backslashes in Windows file paths
        is_posix = os.name != "nt"
        parts = shlex.split(s, posix=is_posix)
        return [p[1:-1] if not is_posix and len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]
    except Exception as e:
        raise ValueError(f"Failed to parse arguments: {e}")


def parse_admin_mapping(raw_str: Optional[str]) -> Dict[str, str]:
    """
    Parses key-value mappings supporting JSON maps or KEY=VALUE entries.
    Preserves values with spaces such as 'Authorization=Bearer my_token'.
    """
    if not raw_str or not raw_str.strip():
        return {}
    s = raw_str.strip()
    if s.startswith("{"):
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                if not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
                    raise ValueError("JSON map must contain only string keys and string values.")
                return data
        except json.JSONDecodeError as jde:
            raise ValueError(f"Malformed JSON map: {jde}")

    parsed = {}
    # Split on newlines, semicolons, or commas between pairs (when comma precedes a key=)
    entries = re.split(r"[\r\n;]+|,\s*(?=[A-Za-z0-9_\-]+=)", s)
    for entry in entries:
        item = entry.strip()
        if "=" in item:
            k, v = item.split("=", 1)
            k_clean = k.strip()
            if k_clean:
                parsed[k_clean] = v.strip()
            else:
                raise ValueError("Mapping keys cannot be empty.")
        else:
            raise ValueError("Expected KEY=VALUE entries or a JSON object.")
    return parsed


def parse_key_value_pairs(raw_str: Optional[str]) -> Dict[str, str]:
    """Legacy alias preserved for backward compatibility."""
    return parse_admin_mapping(raw_str)
