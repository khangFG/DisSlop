# 🗑️ disslop

> **The world's most over-engineered pipeline for high-velocity, artisanal Discord slop.**

[![Slop Quality: 100% Free-Range](https://img.shields.io/badge/Slop%20Quality-100%25%20Organic-orange.svg)](#)
[![Status: 500 Internal Server Error](https://img.shields.io/badge/Status-500%20Internal%20Error%20(Feature)-red.svg)](#)
[![Python: 3.14 (Why Run Stable?)](https://img.shields.io/badge/Python-3.14.10%20(Bleeding%20Edge)-blue.svg)](#)
[![Tests: 100/100 Over-Engineered](https://img.shields.io/badge/Tests-100%2F100%20Green-brightgreen.svg)](#)
[![Architecture: Extreme Modular Overkill](https://img.shields.io/badge/Architecture-Cogs%20%2B%20Subsystems-purple.svg)](#)

---

## 🧐 What is `disslop`?

Have you ever looked at a quiet, peaceful Discord server and thought:

> *"What this chat really needs is a 26-billion-parameter open-weights model reading a 128,000-character Wikipedia article about Soviet tank transmissions for 92 seconds before crashing with a `500 Internal Server Error`?"*

Look no further. **`DisSlop`** turns what could have been a 15-line `discord.py` bot into an enterprise-grade, asynchronous Model Context Protocol (MCP) orchestrator designed to pump unadulterated AI slop directly into your `#general` channel.

Why write a simple bot when you can build a **multi-process subprocess sandbox with monotonic cooldown algorithms, LiteLLM unified dispatch, and 100 unit tests** just to deliver three paragraphs of hallucinated text?

---

## ✨ Groundbreaking Features

* **Artisanal 91.66-Second Suspense Engine**:
  Will it answer? Will it time out? Will the upstream provider melt down? Every `@mention` is an exhilarating, high-stakes game of Russian roulette!
* **Format-Aware Tool Result Compaction**:
  Fetches two Wikipedia articles totaling 250,000 characters, chops them into Head/Middle/Tail samples with surgical precision, and feeds them into an LLM context window that was never meant to hold them.
* **100% Contract-Tested Slop Delivery**:
  Over **100 offline unit tests** ensuring that when your bot inevitably drops a generic reference card (`❌ Error: AI Provider returned an API error. (Reference ID: 1099a13ed2)`), not a single byte of your API key is leaked to the peasants.
* **Extreme Modular Overkill**:
  Decomposed into Discord.py Cogs (`cogs/providers.py`, `cogs/mcp.py`, `cogs/chat.py`), decoupled state stores (`modules/state.py`), and subprocess isolation suites. Because God forbid any file have more than 200 lines.
* **Monotonic Admission Limiter**:
  Strictly regulates how often users can request slop, ensuring everyone waits their fair 10-second turn before receiving another 1,950-character block of AI prose.

---

## 🏗️ The Cathedral of Slop (Architecture)

```text
disslop/
├── cogs/
│   ├── base.py           # Checks if you're worthy (admin) to summon slop
│   ├── providers.py      # Switch between cheap slop, expensive slop, and free VIP slop
│   ├── mcp.py            # Spawns background Node & Python children to scrape the web
│   └── chat.py           # Listens for @mentions and types anxiously while the model stalls
├── modules/
│   ├── llm/              # Translates human queries into tokens and thoughts into excuses
│   ├── mcp/              # Sanitizes 100k-character dumps so the bot doesn't explode
│   ├── formatting.py     # Chops slop into Discord-safe 1,950-character chunks
│   └── state.py          # Solves Python __main__ duality because imports are hard
├── main.py               # The conductor orchestrating this entire circus (~175 lines)
└── tests/                # 100 automated tests proving this slop is strictly certified
```

---

## 🚀 Quick Slop Setup

### 1. Requirements
* **Python 3.14+** 
* A Discord bot token from the [Developer Portal](https://discord.com/developers/applications) with **Message Content Intent**, or just go YOLO with **Administrators** enabled.
* An API key from whatever third-party proxy endpoint is currently not returning `502 Bad Gateway`.

### 2. Spawn Your Virtual Isolation Chamber (`.venv`)
Isolate this slop machine from your global Python environment before it decides to upgrade your entire operating system:

```powershell
# Step A: Conjure the virtual environment out of thin air
python -m venv .venv

# Step B: Enter the matrix
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# (If PowerShell complains about execution policies: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# On Windows Command Prompt (cmd.exe):
.\.venv\Scripts\activate.bat

# On Linux / macOS (if you somehow escaped Windows):
source .venv/bin/activate

# Step C: Ingest the locked dependencies
pip install -r requirements.lock
```

*(Or if you are a speed demon with `uv`: `uv venv && uv pip install -r requirements.lock`)*

### 3. Configure Your Secrets
Copy `.env.example` to `.env`:
```env
DISCORD_BOT_TOKEN=your_token_here
ADMIN_USER_ID=your_id_here
```

### 4. Launch the Slop Machine
```powershell
# Windows
.\run.bat
# Sorry but this shit only have executable for winslop because the owner is a dumb fuck
# Or via PowerShell
.\run.ps1
```

---

## 🕹️ Slash Commands for Connoisseurs

| Command | What It Actually Does |
|---|---|
| `/configure_provider` | Choose your preferred flavor of slop (OpenAI, Anthropic, Gemini, Ollama, or sketchy proxies). |
| `/switch_provider` | Instantly jump to a profile that isn't throwing 500 errors right now. |
| `/add_mcp_stdio` | Launch an unmonitored `npx` or `uvx` child process on your computer to crawl the web. |
| `/mcp_tools` | View all the external weapons currently loaded into the bot's arsenal. |
| `/set_prompt` | Teach the bot a new personality so its slop feels fresh and bespoke. |
| `/llm_status` | Stare at your active token limit and calculate how much money you burned today. |
| `!sync` | Screams at Discord's gateway until your slash commands finally show up. |

---

## 💡 How to Trigger Peak Slop

Mention the bot in your server:
```text
@yourbot fetch https://en.wikipedia.org/wiki/T-62
```
Sit back, grab a coffee, watch Discord's typing indicator pulse for 90 seconds, and marvel at the technological achievement of our generation.

---

## 📜 Philosophy

> *"If a Discord bot can be written in a single 30-line script, it simply doesn't have enough layers of abstraction."*

---

*For actual technical documentation and architecture audits, see [README_TECHNICAL.md](README_TECHNICAL.md) and [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).*
