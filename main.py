"""
Discord LLM Bot Entrypoint & Main Coordinator.
Modularized coordinator wiring Discord.py Cogs, ConfigManager, MCPManager, and LLM services.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Set

import discord
from discord.ext import commands
from dotenv import load_dotenv

from config_manager import ConfigManager, resolve_litellm_model
from mcp_manager import MCPManager
from tool_policy import RequestContext
from error_reporting import format_public_error, log_sanitized_error
from request_limits import RequestLimiter, RequestLimitError, REQUEST_TIMEOUT_SECONDS
from conversation import build_conversation_context
import llm_service
from modules.formatting import (
    _parse_admin_ids,
    split_into_chunks,
    send_chunked_reply,
    parse_admin_arguments,
    parse_admin_mapping,
    parse_key_value_pairs,
)
from modules.state import state
from cogs.base import (
    is_admin_user,
    is_admin,
    verify_admin_interaction,
    profile_autocomplete,
    mcp_server_autocomplete,
    get_current_config,
    get_current_mcp_manager,
)
from cogs import ProviderCog, MCPCog, ChatCog

# Discord Bot Token and Admin ID
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "").strip()
_admin_user_ids: Set[str] = _parse_admin_ids(ADMIN_USER_ID)

# Importing commands must not read credentials or the live configuration.
# Runtime state is loaded explicitly by main(); tests can inject their own.
config = state.config
mcp_manager = state.mcp_manager


def set_app_state(
    cfg: Optional[ConfigManager] = None,
    mgr: Optional[MCPManager] = None,
    admin_id: Optional[Any] = None,
) -> None:
    """Injectable construction seam for testing and startup without writing files or spawning processes."""
    global config, mcp_manager, ADMIN_USER_ID, _admin_user_ids
    state.set_state(cfg=cfg, mgr=mgr, admin_id=admin_id)
    if cfg is not None:
        config = cfg
    if mgr is not None:
        mcp_manager = mgr
    if admin_id is not None:
        ADMIN_USER_ID = str(admin_id).strip()
        _admin_user_ids = _parse_admin_ids(admin_id)

    # Synchronize across all module scopes (handles __main__ vs main duality)
    for mod_name in ("__main__", "main"):
        mod = sys.modules.get(mod_name)
        if mod and mod is not sys.modules.get(__name__):
            if cfg is not None:
                mod.config = cfg
            if mgr is not None:
                mod.mcp_manager = mgr
            if admin_id is not None:
                mod.ADMIN_USER_ID = ADMIN_USER_ID
                mod._admin_user_ids = _admin_user_ids


# Admission & rate limiter for requests
request_limiter = RequestLimiter(cooldown_seconds=10.0)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents, allowed_mentions=discord.AllowedMentions.none())
_background_tasks: set[asyncio.Task] = set()


def schedule_tool_refresh() -> None:
    """Schedules background refresh of active MCP servers."""
    active_mgr = get_current_mcp_manager()
    task = asyncio.create_task(active_mgr.refresh_tools())
    _background_tasks.add(task)

    def finished(done):
        _background_tasks.discard(done)
        if not done.cancelled() and done.exception() is not None:
            log_sanitized_error("refresh", "MCP refresh failed", done.exception())

    task.add_done_callback(finished)


# Ensure MCP background sessions are gracefully closed when the bot shuts down
_orig_bot_close = bot.close


async def _bot_close_with_mcp_cleanup() -> None:
    try:
        pending = list(_background_tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        active_mgr = get_current_mcp_manager()
        if active_mgr:
            await active_mgr.close_all()
    except Exception as e:
        log_sanitized_error("shutdown", "MCP shutdown failed", e)
    finally:
        await _orig_bot_close()


bot.close = _bot_close_with_mcp_cleanup


# Register modular Discord Cogs onto the bot tree and command registry
async def _register_cogs() -> None:
    await bot.add_cog(ProviderCog(bot))
    await bot.add_cog(MCPCog(bot))
    await bot.add_cog(ChatCog(bot))


try:
    loop = asyncio.get_running_loop()
    loop.create_task(_register_cogs())
except RuntimeError:
    asyncio.run(_register_cogs())


@bot.event
async def setup_hook() -> None:
    """Runs during startup to register slash commands globally."""
    try:
        synced = await bot.tree.sync()
        print(f"[Bot] Successfully synced {len(synced)} global slash commands.")
    except Exception as e:
        print(f"[Bot] Failed to sync global slash commands: {e}")


@bot.event
async def on_ready() -> None:
    """Triggered when bot connects to Discord."""
    active_cfg = get_current_config()
    active_admin_ids = _admin_user_ids or state.admin_user_ids
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"Active LLM Profile: {active_cfg.active_profile}")
    admin_list = ", ".join(sorted(active_admin_ids)) if active_admin_ids else "Not Configured"
    print(f"Configured Admin ID(s): {admin_list}")

    # Instant Guild Sync so commands appear immediately
    for guild in bot.guilds:
        try:
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
            print(f"[Bot] Synced {len(synced)} slash commands instantly to server '{guild.name}' ({guild.id})")
        except Exception as e:
            print(f"[Bot] Note: Could not sync to server '{guild.name}': {e}")

    # Warm up persistent MCP tool connections in background
    schedule_tool_refresh()


def main() -> None:
    """Main application entry point."""
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")
    admin_id_env = os.getenv("ADMIN_USER_ID", "").strip()
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token or token == "your_discord_bot_token_here":
        print("❌ Error: DISCORD_BOT_TOKEN is not configured in .env.")
        print("Please edit .env with your bot token from the Discord Developer Portal.")
        return

    # Load persistent runtime configuration and synchronize application state
    config_file = project_dir / "runtime_config.json"
    loaded_cfg = ConfigManager(config_path=str(config_file))
    loaded_mgr = MCPManager(loaded_cfg)
    set_app_state(cfg=loaded_cfg, mgr=loaded_mgr, admin_id=admin_id_env)

    print("🤖 Starting Discord AI Bot...")
    try:
        bot.run(token)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")


if __name__ == "__main__":
    main()
