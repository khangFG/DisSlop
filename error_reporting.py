import os
import re
import urllib.parse
import uuid
from typing import Any, List, Optional, Tuple


SAFE_SYSTEM_ENV_VARS = {
    # Windows standard
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP",
    "COMSPEC", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA",
    "LOCALAPPDATA", "WINDIR", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)",
    # POSIX standard
    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TMPDIR", "TERM",
}


def validate_remote_url(url: str, allow_http_localhost: bool = True) -> bool:
    """
    Validates that remote API/MCP endpoints use HTTPS.
    Allows HTTP only for local loopback development endpoints.
    """
    if not url or not isinstance(url, str) or any(c.isspace() for c in url):
        return False

    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            return False
        parsed.port  # Validate malformed/out-of-range ports before use.
    except Exception:
        return False

    if parsed.scheme == "https":
        return True

    if parsed.scheme == "http":
        hostname = (parsed.hostname or "").lower()
        if allow_http_localhost and hostname in ("localhost", "127.0.0.1", "::1"):
            return True
        return False

    return False


def redact_sensitive_text(text: str, secrets: Optional[List[str]] = None) -> str:
    """
    Scrubs known secrets, embedded URL credentials, and common API key patterns from text.
    """
    if not text:
        return ""

    out = str(text)

    # 1. Redact explicit secrets
    if secrets:
        for s in secrets:
            if s and isinstance(s, str) and len(s) >= 4:
                out = out.replace(s, "[REDACTED_SECRET]")

    # 2. Redact environment secrets if present
    for env_var in ("DISCORD_BOT_TOKEN", "DISCORD_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        val = os.getenv(env_var)
        if val and len(val) >= 4:
            out = out.replace(val, "[REDACTED_SECRET]")

    # 3. Redact credentials in URLs: http://user:pass@host
    out = re.sub(r"://([^/:@\s]+):([^/@\s]+)@", "://[REDACTED_USER]:[REDACTED_PASS]@", out)
    # Also http://user@host
    out = re.sub(r"://([^/:@\s]+)@", "://[REDACTED_USER]@", out)

    # 4. Redact query-parameter API keys in URLs (e.g. ?key=AIza... or &key=... or api_key=...)
    out = re.sub(r"([?&](?:key|api_key|token|access_token)=)[^&\s\"\'\)>]+", r"\1[REDACTED_KEY]", out, flags=re.IGNORECASE)

    # 5. Redact standard API key patterns
    out = re.sub(r"sk-[a-zA-Z0-9_\-]{15,}", "[REDACTED_API_KEY]", out)
    out = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}", "Bearer [REDACTED_TOKEN]", out, flags=re.IGNORECASE)

    return out


# Alias for explicit secret scrubbing
redact_secrets = redact_sensitive_text


def format_public_error(internal_message: str, error_prefix: str = "An unexpected error occurred.") -> Tuple[str, str]:
    """
    Returns (public_user_message, correlation_id).
    The public message contains only the generic prefix and correlation ID, protecting
    internal details, paths, and provider errors from public exposure.
    """
    cid = uuid.uuid4().hex[:10]
    public_msg = f"❌ **Error**: {error_prefix} (Reference ID: `{cid}`)"
    return public_msg, cid


def log_sanitized_error(cid: str, context: str, exc: Any = None, extra: str = "") -> None:
    """
    Logs an internal error with correlation ID and redacted secrets to standard diagnostic output.
    """
    # Exception bodies and arbitrary extra payloads can contain credentials not
    # present in the environment. Log the category, never the remote payload.
    exc_text = f": {type(exc).__name__}" if exc else ""
    raw_log = f"[ErrorRef:{cid}] {context}{exc_text}".strip()
    sanitized_log = redact_sensitive_text(raw_log)
    print(sanitized_log)
