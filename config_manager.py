import copy
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Set
from error_reporting import validate_remote_url

CONFIG_FILE = "runtime_config.json"
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a helpful, concise AI assistant inside a Discord server."
)

VALID_API_FORMATS: Set[str] = {
    "litellm",
    "openai",
    "anthropic",
    "gemini",
    "vertex_ai",
    "bedrock",
    "ollama",
    "openrouter",
    "deepseek",
    "groq",
    "xai",
}


def resolve_litellm_model(api_format: str, model: str, base_url: Optional[str] = None) -> str:
    """
    Normalizes profile configurations into LiteLLM-compatible model identifier strings.
    - If api_format == "openai" and model has no provider prefix (or uses custom proxy base_url),
      normalizes to f"openai/{model}".
    - If api_format == "anthropic", normalizes to f"anthropic/{model}" (if not already prefixed).
    - If api_format == "litellm", passes model string directly.
    - Other known providers (gemini, ollama, etc.) prefix with provider if not already prefixed.
    """
    fmt = (api_format or "litellm").strip().lower()
    m = (model or "").strip()
    if not m:
        return ""

    if fmt == "litellm":
        return m

    if fmt == "openai":
        if m.startswith("openai/"):
            return m
        return f"openai/{m}"

    if fmt == "anthropic":
        if m.startswith("anthropic/"):
            return m
        return f"anthropic/{m}"

    if fmt in VALID_API_FORMATS:
        prefix = f"{fmt}/"
        if m.startswith(prefix):
            return m
        return f"{prefix}{m}"

    return m


class ConfigManager:
    """Thread-safe configuration manager for Discord LLM profiles, MCP servers, and runtime settings."""

    def __init__(self, config_path: Optional[str] = CONFIG_FILE):
        self.config_path = config_path
        self._lock = threading.Lock()

        # Defaults
        self.active_profile: str = "openai"
        self.system_prompt: str = os.getenv("DEFAULT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        self.profiles: Dict[str, Dict[str, Any]] = {
            "openai": {
                "api_format": "openai",
                "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
                "base_url": None,
                "model": "gpt-4o-mini",
            },
            "anthropic": {
                "api_format": "anthropic",
                "api_key": os.getenv("ANTHROPIC_API_KEY", "").strip(),
                "base_url": None,
                "model": "claude-3-5-haiku-latest",
            },
        }
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}
        self.tool_policy = {"admin_only": True, "allowed_users": [], "allowed_channels": [], "allowed_tools": []}

        self._load()

    @staticmethod
    def _validate_state(data: Dict[str, Any]) -> None:
        """Validate the complete candidate before disk writes or publication."""
        if not isinstance(data, dict):
            raise ValueError("Configuration must be an object.")
        profiles, servers = data.get("profiles"), data.get("mcp_servers")
        if not isinstance(profiles, dict) or not isinstance(servers, dict):
            raise ValueError("Profiles and MCP servers must be objects.")
        for name, profile in profiles.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(profile, dict):
                raise ValueError("Invalid provider entry.")
            if profile.get("api_format") not in VALID_API_FORMATS:
                raise ValueError("Invalid provider format.")
            if not all(isinstance(profile.get(k), str) for k in ("model", "api_key")):
                raise ValueError("Provider model and key must be strings.")
            if "max_tokens" in profile and profile["max_tokens"] is not None:
                if not isinstance(profile["max_tokens"], int) or profile["max_tokens"] <= 0:
                    raise ValueError("Provider max_tokens must be a positive integer.")
            url = profile.get("base_url")
            if url is not None and not validate_remote_url(url):
                raise ValueError("Provider URL must use HTTPS or loopback HTTP without embedded credentials.")
        if data.get("active_profile") not in profiles:
            raise ValueError("Active profile does not exist.")
        if not isinstance(data.get("system_prompt"), str):
            raise ValueError("System prompt must be a string.")
        for name, server in servers.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(server, dict):
                raise ValueError("Invalid MCP entry.")
            if type(server.get("enabled", True)) is not bool:
                raise ValueError("MCP enabled must be a boolean.")
            if server.get("transport") == "stdio":
                if not isinstance(server.get("command"), str) or not server["command"].strip():
                    raise ValueError("MCP stdio command is required.")
            elif server.get("transport") in ("sse", "http", "streamable_http"):
                if not validate_remote_url(server.get("url")):
                    raise ValueError("MCP URL must use HTTPS or loopback HTTP without embedded credentials.")
            else:
                raise ValueError("Unsupported MCP transport.")
            args = server.get("args", [])
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise ValueError("MCP args must be a list of strings.")
            for key in ("env", "headers"):
                mapping = server.get(key, {})
                if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
                    raise ValueError("MCP env/headers must contain string keys and values.")
        policy = data.get("tool_policy", {})
        if not isinstance(policy, dict) or type(policy.get("admin_only", True)) is not bool:
            raise ValueError("Invalid tool policy.")
        for key in ("allowed_users", "allowed_channels", "allowed_tools"):
            values = policy.get(key, [])
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError("Policy allowlists must be lists of nonempty strings.")

    def _load(self) -> None:
        if self.config_path is None or not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Configuration must be an object.")
            profiles = copy.deepcopy(self.profiles)
            raw_profiles = data.get("profiles", {})
            raw_servers = data.get("mcp_servers", {})
            if not isinstance(raw_profiles, dict) or not isinstance(raw_servers, dict):
                raise ValueError("Invalid configuration mappings.")
            for name, value in raw_profiles.items():
                if not isinstance(value, dict):
                    raise ValueError("Invalid provider entry.")
                name = name.strip().lower()
                profile = profiles.get(name, {"api_format": "openai", "api_key": "", "base_url": None, "model": "default-model"})
                profiles[name] = {**profile, **value}
            servers = {name.strip().lower(): value for name, value in raw_servers.items()}
            candidate = {
                "profiles": profiles, "mcp_servers": servers,
                "active_profile": data.get("active_profile", self.active_profile),
                "system_prompt": data.get("system_prompt", self.system_prompt),
                "tool_policy": data.get("tool_policy", self.tool_policy),
            }
            self._validate_state(candidate)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            # Do not publish partially loaded state or allow later writes over a
            # damaged file. The operator must repair the existing file first.
            raise ValueError("Configuration could not be loaded; original file preserved.") from exc
        self.profiles = candidate["profiles"]
        self.mcp_servers = candidate["mcp_servers"]
        self.active_profile = candidate["active_profile"]
        self.system_prompt = candidate["system_prompt"]
        self.tool_policy = candidate["tool_policy"]

    def get_tool_policy(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.tool_policy)

    def _atomic_persist_and_publish(
        self,
        candidate_profiles: Optional[Dict[str, Any]] = None,
        candidate_mcp: Optional[Dict[str, Any]] = None,
        candidate_active: Optional[str] = None,
        candidate_prompt: Optional[str] = None,
    ) -> None:
        """
        Transactional persistence: writes candidate state to a temporary file,
        replaces the target file atomically, and only then publishes updates to in-memory state.
        If disk write or replace fails, in-memory state remains completely unchanged.
        """
        new_profiles = candidate_profiles if candidate_profiles is not None else self.profiles
        new_mcp = candidate_mcp if candidate_mcp is not None else self.mcp_servers
        new_active = candidate_active if candidate_active is not None else self.active_profile
        new_prompt = candidate_prompt if candidate_prompt is not None else self.system_prompt

        data = {
            "active_profile": new_active,
            "system_prompt": new_prompt,
            "profiles": new_profiles,
            "mcp_servers": new_mcp,
            "tool_policy": self.tool_policy,
        }

        self._validate_state(data)
        if self.config_path is not None:
            temp_path = f"{self.config_path}.tmp"
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, self.config_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Published to memory only after successful disk persistence
        if candidate_profiles is not None:
            self.profiles = new_profiles
        if candidate_mcp is not None:
            self.mcp_servers = new_mcp
        if candidate_active is not None:
            self.active_profile = new_active
        if candidate_prompt is not None:
            self.system_prompt = new_prompt

    def _save_unlocked(self) -> None:
        """Writes current config to file (must be called with _lock held or during init)."""
        self._atomic_persist_and_publish()

    # ----------------------------------------------------
    # LLM Profile Methods
    # ----------------------------------------------------

    def upsert_profile(
        self,
        name: str,
        api_format: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Creates or updates a profile and immediately persists it transactionally."""
        norm_name = name.strip().lower()
        if not norm_name:
            raise ValueError("Profile name cannot be empty.")

        with self._lock:
            existing = self.profiles.get(norm_name)

            # Normalize base_url
            processed_base_url: Optional[str]
            if base_url is not None:
                stripped_url = base_url.strip()
                if stripped_url.lower() in ("default", "none", ""):
                    processed_base_url = None
                else:
                    processed_base_url = stripped_url.rstrip("/")
            else:
                processed_base_url = existing.get("base_url") if existing else None

            if processed_base_url is not None and not validate_remote_url(processed_base_url):
                raise ValueError("Provider URL must use HTTPS or loopback HTTP without embedded credentials.")

            # Format validation
            target_format: str
            if api_format is not None:
                fmt_clean = api_format.strip().lower()
                if fmt_clean not in VALID_API_FORMATS:
                    raise ValueError(f"api_format must be one of: {', '.join(sorted(VALID_API_FORMATS))}.")
                target_format = fmt_clean
            elif existing:
                target_format = existing.get("api_format", "openai")
            else:
                target_format = "openai"

            target_key = (
                api_key.strip()
                if api_key is not None
                else (existing.get("api_key", "") if existing else "")
            )

            target_model = (
                model.strip()
                if model is not None
                else (
                    existing.get("model", "")
                    if existing
                    else ("gpt-4o-mini" if target_format == "openai" else "claude-3-5-haiku-latest")
                )
            )

            target_max_tokens: Optional[int]
            if max_tokens is not None:
                if max_tokens <= 0:
                    raise ValueError("max_tokens must be a positive integer.")
                target_max_tokens = int(max_tokens)
            elif existing and "max_tokens" in existing:
                target_max_tokens = existing.get("max_tokens")
            else:
                target_max_tokens = None

            profile_data = {
                "api_format": target_format,
                "api_key": target_key,
                "base_url": processed_base_url,
                "model": target_model,
            }
            if target_max_tokens is not None:
                profile_data["max_tokens"] = target_max_tokens

            candidate_profiles = copy.deepcopy(self.profiles)
            candidate_profiles[norm_name] = profile_data

            self._atomic_persist_and_publish(candidate_profiles=candidate_profiles)
            return copy.deepcopy(profile_data)

    def set_active_profile(self, name: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Sets the active profile and optionally overrides its model transactionally."""
        norm_name = name.strip().lower()
        with self._lock:
            if norm_name not in self.profiles:
                raise ValueError(f"Profile '{norm_name}' does not exist.")

            candidate_profiles = copy.deepcopy(self.profiles)
            if model_override and model_override.strip():
                candidate_profiles[norm_name]["model"] = model_override.strip()

            self._atomic_persist_and_publish(
                candidate_profiles=candidate_profiles,
                candidate_active=norm_name,
            )
            return copy.deepcopy(self.profiles[norm_name])

    def delete_profile(self, name: str) -> bool:
        """Deletes a profile transactionally. Active profile cannot be deleted."""
        norm_name = name.strip().lower()
        with self._lock:
            if norm_name not in self.profiles:
                raise ValueError(f"Profile '{norm_name}' does not exist.")
            if norm_name == self.active_profile:
                raise ValueError(f"Cannot delete active profile '{norm_name}'. Switch to another profile first.")

            candidate_profiles = copy.deepcopy(self.profiles)
            del candidate_profiles[norm_name]

            self._atomic_persist_and_publish(candidate_profiles=candidate_profiles)
            return True

    def set_system_prompt(self, prompt: str) -> str:
        """Updates the system prompt and immediately persists it transactionally."""
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("System prompt cannot be empty.")

        with self._lock:
            self._atomic_persist_and_publish(candidate_prompt=clean_prompt)
            return self.system_prompt

    def get_active_profile(self) -> Tuple[str, Dict[str, Any]]:
        """Returns deep snapshot of (active_profile_name, profile_dict)."""
        with self._lock:
            profile = copy.deepcopy(self.profiles.get(self.active_profile, {}))
            return self.active_profile, profile

    # ----------------------------------------------------
    # MCP Server Methods
    # ----------------------------------------------------

    def upsert_mcp_server(
        self,
        name: str,
        transport: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """Adds or updates an MCP server configuration transactionally."""
        norm_name = name.strip().lower()
        if not norm_name:
            raise ValueError("MCP server name cannot be empty.")

        trans_clean = transport.strip().lower()
        if trans_clean not in ("stdio", "sse", "http", "streamable_http"):
            raise ValueError("Transport must be either 'stdio', 'sse', or 'http'.")

        if trans_clean == "stdio" and not (command and command.strip()):
            raise ValueError("A command is required for 'stdio' transport (e.g. 'npx', 'uvx', 'python').")

        if trans_clean in ("sse", "http", "streamable_http") and not (url and url.strip()):
            raise ValueError("A URL is required for remote transport (e.g. 'https://...').")

        # Nested type validations
        if args is not None:
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ValueError("args must be a list of strings.")

        if env is not None:
            if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise ValueError("env must be a dictionary with string keys and values.")

        if headers is not None:
            if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                raise ValueError("headers must be a dictionary with string keys and values.")

        with self._lock:
            existing = self.mcp_servers.get(norm_name)
            server_data = {
                "transport": trans_clean,
                "command": command.strip() if command else (existing.get("command") if existing else None),
                "args": copy.deepcopy(args) if args is not None else (existing.get("args", []) if existing else []),
                "url": url.strip() if url else (existing.get("url") if existing else None),
                "env": copy.deepcopy(env) if env is not None else (existing.get("env", {}) if existing else {}),
                "headers": copy.deepcopy(headers) if headers is not None else (existing.get("headers", {}) if existing else {}),
                "enabled": enabled,
            }

            candidate_mcp = copy.deepcopy(self.mcp_servers)
            candidate_mcp[norm_name] = server_data

            self._atomic_persist_and_publish(candidate_mcp=candidate_mcp)
            return copy.deepcopy(server_data)

    def remove_mcp_server(self, name: str) -> bool:
        """Removes an MCP server by name transactionally."""
        norm_name = name.strip().lower()
        with self._lock:
            if norm_name not in self.mcp_servers:
                raise ValueError(f"MCP server '{norm_name}' does not exist.")

            candidate_mcp = copy.deepcopy(self.mcp_servers)
            del candidate_mcp[norm_name]

            self._atomic_persist_and_publish(candidate_mcp=candidate_mcp)
            return True

    def get_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Returns a deep snapshot of all configured MCP servers."""
        with self._lock:
            return copy.deepcopy(self.mcp_servers)

    # ----------------------------------------------------
    # Status & Utility Methods
    # ----------------------------------------------------

    @staticmethod
    def mask_key(key: str) -> str:
        """Masks sensitive API key (e.g. sk-proj...9a8b) or returns 'Not Set'."""
        if not key or not key.strip():
            return "Not Set"
        k = key.strip()
        if len(k) <= 8:
            return f"{k[:2]}...{k[-2:]}"
        return f"{k[:7]}...{k[-4:]}"

    def get_status_report(self) -> str:
        """Returns a clean, Markdown-formatted overview of system status, profiles, and MCP servers."""
        with self._lock:
            active_name = self.active_profile
            active_p = self.profiles.get(active_name, {})
            active_format = active_p.get("api_format", "unknown")
            active_model = active_p.get("model", "unknown")
            active_url = active_p.get("base_url") or "SDK Default"
            active_target = resolve_litellm_model(active_format, active_model, active_p.get("base_url"))
            active_key_masked = self.mask_key(active_p.get("api_key", ""))

            report = [
                "### 🤖 LLM Router Status Report",
                f"- **Active Profile:** `{active_name}`",
                f"- **Active Format:** `{active_format}`",
                f"- **Active Model:** `{active_model}`",
                f"- **LiteLLM Target:** `{active_target}`",
                f"- **Base URL:** `{active_url}`",
                f"- **API Key Status:** `{active_key_masked}`",
                f"- **System Prompt:**\n> {self.system_prompt}",
                "",
                "#### 📋 Configured Profiles",
                "| Profile | Format | Model | Target | Base URL | Key Status | Active |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
            ]

            for name, data in sorted(self.profiles.items()):
                is_act = "✅" if name == active_name else ""
                url = data.get("base_url") or "SDK Default"
                m_key = self.mask_key(data.get("api_key", ""))
                fmt = data.get("api_format", "openai")
                mdl = data.get("model", "unknown")
                tgt = resolve_litellm_model(fmt, mdl, data.get("base_url"))
                report.append(f"| `{name}` | `{fmt}` | `{mdl}` | `{tgt}` | `{url}` | `{m_key}` | {is_act} |")

            report.append("")
            report.append("#### 🔌 Configured MCP Servers")
            if not self.mcp_servers:
                report.append("*No MCP servers currently configured. Use `/add_mcp_stdio` or `/add_mcp_sse`.*")
            else:
                report.append("| Server | Transport | Target / Command | Enabled |")
                report.append("| :--- | :--- | :--- | :---: |")
                for s_name, s_data in sorted(self.mcp_servers.items()):
                    trans = s_data.get("transport", "stdio")
                    en_mark = "✅" if s_data.get("enabled", True) else "❌"
                    if trans == "stdio":
                        cmd_line = f"{s_data.get('command', '')} {' '.join(s_data.get('args') or [])}".strip()
                        target = f"`{cmd_line}`" if cmd_line else "*(empty)*"
                    else:
                        target = f"`{s_data.get('url', '')}`"
                    report.append(f"| `{s_name}` | `{trans}` | {target} | {en_mark} |")

            return "\n".join(report)
