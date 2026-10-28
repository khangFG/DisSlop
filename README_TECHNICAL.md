# Discord Multi-Provider LLM Bot with Model Context Protocol (MCP)

An asynchronous Discord bot in Python with offline-tested security controls and full MCP tool integration:
1. **Unified Multi-Provider LLM Engine (LiteLLM)**: Single unified agentic loop supporting OpenAI, Anthropic, Google Gemini, Vertex AI, AWS Bedrock, Ollama, OpenRouter, DeepSeek, Groq, xAI, and custom OpenAI/Anthropic-compatible proxies (Logflare, NVIDIA NIM, SenseNova, ShareLLM).
2. **Model Context Protocol (MCP) Integration**: Connect to local (`stdio`), remote (`sse`), and modern `streamable_http` MCP servers on the fly, expose their tools to the active LLM, and execute multi-step agentic tool loops inside Discord.
3. **Reasoning Standardization & DSML Interception**: Extracts chain-of-thought thinking blocks normalized across providers (`reasoning_content`, Anthropic `thinking` blocks, Gemini thoughts). Intercepts and executes DeepSeek Markup Language (`< | DSML | tool_calls>`) emitted by reasoning models and sanitizes raw XML from user replies.
4. **Default-Deny Tool Security**: MCP tools require an explicit tool allowlist (`allowed_tools` supporting server names, wildcards, and exact tool names). Non-admin users receive conversational chat unless explicitly allowed by policy.
5. **Subprocess & Credential Isolation**: Child MCP processes receive a clean OS execution environment without inheriting sensitive `.env` secrets (`DISCORD_BOT_TOKEN`, provider keys). Public Discord error messages redact secrets (including URL query parameter keys) and output generic correlation IDs (`[ErrorRef:xxxxxxxxxx]`).
6. **Multi-Admin Support**: Supports multiple Discord administrators configured via `.env` (`ADMIN_USER_ID=id1, id2`).
7. **Admission Control & Generous Research Limits**: Monotonic per-user cooldowns, concurrency limits, 10 agentic iterations per request, up to 20 tool calls, 128,000 characters per tool result, 250,000 characters total turn budget, and a 180-second generation deadline.
8. **Mention Suppression**: All Discord bot messages and replies enforce `AllowedMentions.none()` to prevent ping spam or notification abuse.

---

## 📁 System Architecture

```text
discord-llm-bot/
├── .env                  # Bootstrap secrets (Discord Token, Admin Discord IDs, optional seed keys)
├── .env.example          # Setup template
├── .gitignore            # Protects .env, runtime_config.json, and .venv/
├── requirements.txt      # Direct runtime dependencies
├── requirements.lock     # Authoritative pinned dependency lockfile
├── runtime_config.json   # Persistent transactional store for profiles and MCP servers
├── config_manager.py     # Thread-safe transactional state manager with atomic writes
├── tool_policy.py        # Default-deny authorization policy & request context
├── mcp_manager.py        # Facade re-exporting modules.mcp and managing worker lifecycle
├── llm_service.py        # Facade re-exporting modules.llm and running unified LiteLLM loop
├── error_reporting.py    # Secret redaction (keys, query params), correlation IDs, URL checks
├── request_limits.py     # Monotonic cooldowns, per-user limits, and admission control
├── conversation.py       # Deterministic conversation history builder anchored before trigger
├── modules/              # Subsystem logic packages
│   ├── llm/              # DSML parser, reasoning token extractor, LiteLLM routing
│   ├── mcp/              # Process isolation, tool naming, schema compression, presentation
│   └── formatting.py     # Discord chunked reply splitting, mention suppression, argument parsing
├── cogs/                 # Discord.py Cogs subsystem
│   ├── base.py           # Shared base cog, admin authorization checks, autocompletes
│   ├── providers.py      # /configure_provider, /switch_provider, /delete_provider, /set_prompt, /llm_status
│   ├── mcp.py            # /add_mcp_stdio, /add_mcp_sse, /add_mcp_http, /remove_mcp, /mcp_tools
│   └── chat.py           # @mention conversational listener, history context, and !sync prefix command
├── main.py               # Lean application entry point and coordinator (~175 lines)
├── bot.py                # Backward-compatible wrapper re-exporting main and bot
├── run.bat / run.ps1     # Directory-anchored start scripts using .venv
└── tests/                # Comprehensive offline regression test suite (100 tests across 17 files)
```

---

## 🚀 Quick Start Guide

### 1. Requirements
- Python 3.11+ (Python 3.14.10 tested and verified with `uv`)
- Node.js (if running `npx`-based MCP servers) or `uv` / `python`

### 2. Discord Developer Portal Setup
1. Create a bot in the [Discord Developer Portal](https://discord.com/developers/applications).
2. Under **Bot**:
   - Copy your `DISCORD_BOT_TOKEN`.
   - Enable **Message Content Intent** under *Privileged Gateway Intents*.
3. Under **OAuth2 > URL Generator**:
   - Scopes: Select `bot` **AND** `applications.commands`.
   - Bot Permissions: Select `Send Messages`, `Read Messages/View Channels`, `Read Message History`.
   - Copy the invite URL and authorize the bot for your Discord server.

### 3. Get Your Discord User ID (`ADMIN_USER_ID`)
1. Enable **Developer Mode** in Discord (**User Settings > Advanced**).
2. Right-click your name in Discord and choose **Copy User ID**.
3. For multiple admins, separate IDs with commas or spaces: `ADMIN_USER_ID=123456789012345678, 234567890123456789`.

### 4. Create Virtual Environment & Install Dependencies
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Or Windows Command Prompt
.\.venv\Scripts\activate.bat

# Install locked dependencies
pip install -r requirements.lock
```

### 5. Configure `.env`
Ensure your `.env` contains:
```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here
ADMIN_USER_ID=your_discord_user_id_here
DEFAULT_SYSTEM_PROMPT=You are a helpful, concise AI assistant inside a Discord server.
```

### 6. Run the Bot
The provided launcher scripts automatically anchor the working directory to the project root:
```powershell
# Windows Batch
.\run.bat

# Windows PowerShell
.\run.ps1

# Or directly with Python:
.\.venv\Scripts\python.exe main.py
```

---

## Tool Permissions & Policy

By default, tool execution requires an explicit allowlist. To permit tools for admins or non-admins, configure the `tool_policy` section in `runtime_config.json`:

```json
"tool_policy": {
  "admin_only": false,
  "allowed_users": [],
  "allowed_channels": [],
  "allowed_tools": ["wiki", "parallel", "context7", "fetch"]
}
```

- `allowed_tools`: Supports server names (e.g. `"wiki"`, `"parallel"`), wildcards (`"*"` for all tools), or exact tool names (`"wiki__get_article"`).
- `admin_only`: When `false`, users listed in `allowed_users` (or all users if empty) can use the tools listed in `allowed_tools` within `allowed_channels`.
- Admin callers bypass user and channel restrictions but still obey `allowed_tools`.

---

## 🔌 Model Context Protocol (MCP) Slash Commands

All MCP configuration commands are restricted to bot administrators (`ADMIN_USER_ID`) and respond with `ephemeral=True`.

### 1. Connect a Local MCP Server (`stdio`)
```text
/add_mcp_stdio name:[str] command:[str] args:[str] env:[optional str]
```
- Automatically removes inadvertent `-y` flags if command is `uvx`.
- **Wikipedia MCP:**
  ```text
  /add_mcp_stdio name:wiki command:uvx args:wikipedia-mcp
  ```
- **Fetch MCP:**
  ```text
  /add_mcp_stdio name:fetch command:uvx args:mcp-server-fetch
  ```
- **Context7 Docs MCP:**
  ```text
  /add_mcp_stdio name:context7 command:npx args:-y @upstash/context7-mcp --api-key ctx7sk-your-key
  ```

### 2. Connect a Remote Streamable HTTP MCP Server
```text
/add_mcp_http name:[str] url:[str] api_key:[optional str] headers:[optional str]
```
- Modern JSON-RPC over HTTP POST (e.g. Parallel.ai):
  ```text
  /add_mcp_http name:parallel url:https://mcp.parallel.ai/mcp headers:Authorization=Bearer your_key
  ```

### 3. Connect a Remote SSE MCP Server
```text
/add_mcp_sse name:[str] url:[str] api_key:[optional str] headers:[optional str]
```
- Automatically falls back to Streamable HTTP if upstream returns `405 Method Not Allowed`.

### 4. View All Active Servers and Discovered Tools
```text
/mcp_tools
```
Displays registered MCP servers, transport type, status, and all exposed tools.

### 5. Remove an MCP Server
```text
/remove_mcp name:[str]
```

---

## ⚡ LLM Provider Slash Commands

### 1. Add or Update a Provider
`/configure_provider` (or alias `/provider_set`):

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `profile` | `str` | Profile name (e.g. `openai`, `anthropic`, `gemini`, `freevip`) |
| `api_format` | `choice` | `litellm` (Auto/Prefix), `openai`, `anthropic`, `gemini`, `openrouter`, `ollama` (also supports `vertex_ai`, `bedrock`, `deepseek`, `groq`, `xai`) |
| `api_key` | `str` | Provider API key |
| `base_url` | `str` | Custom endpoint (e.g. `https://api.groq.com/openai/v1`, `https://sharellm.net/v1`, `https://logfare.ai/v1`) |
| `model` | `str` | Target model name (e.g. `glm-5.3-flash`, `sensenova`, `gemini-2.5-pro`, `claude-3-5-haiku-latest`) |
| `max_tokens` | `int` | Maximum generation tokens (default: `32768`) |

### 2. Switch Active Provider / Model
```text
/switch_provider profile:freevip model:sensenova
```

### 3. Delete a Provider
```text
/delete_provider profile:old_profile
```

### 4. Change Persona / System Prompt
```text
/set_prompt prompt:You are an expert research assistant.
```

### 5. Check Live Router & MCP Status
```text
/llm_status
```
Displays the active profile, configured API format, base URL, target model, resolved LiteLLM routing target, and registered MCP server statuses.

---

## 🧪 Testing & Verification

The codebase includes an extensive offline regression test suite that uses synthetic configurations, fakes, and mock transports without contacting external APIs or spawning untrusted subprocesses.

### Running Offline Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
All **100 regression and safety tests** pass in ~6.3s across 17 test modules (including `test_litellm_routing.py`).

### Compilation Check
```powershell
.\.venv\Scripts\python.exe -m py_compile config_manager.py mcp_manager.py llm_service.py main.py bot.py llm_engine.py tool_policy.py error_reporting.py request_limits.py conversation.py
```
Passes with exit code 0.
