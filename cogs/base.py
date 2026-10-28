"""
Base Cog with shared admin verification and autocomplete routines.
Uses canonical AppState with fallback module inspection to eliminate __main__ vs main duality.
"""

import sys
from typing import Any, List, Set
import discord
from discord import app_commands
from discord.ext import commands

from modules.state import state


def get_current_config():
    """Returns the active ConfigManager, checking __main__, main, and state."""
    for mod_name in ("__main__", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "config") and mod.config is not None:
            if getattr(mod.config, "config_path", None) is not None or mod.config.profiles:
                return mod.config
    return state.config


def get_current_mcp_manager():
    """Returns the active MCPManager, checking __main__, main, and state."""
    for mod_name in ("__main__", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "mcp_manager") and mod.mcp_manager is not None:
            return mod.mcp_manager
    return state.mcp_manager


def get_current_admin_ids() -> Set[str]:
    """Returns the active admin user ID set, checking __main__, main, and state."""
    for mod_name in ("__main__", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "_admin_user_ids") and mod._admin_user_ids:
            return mod._admin_user_ids
    return state.admin_user_ids


def get_current_admin_user_id() -> str:
    """Returns the active admin user ID string, checking __main__, main, and state."""
    for mod_name in ("__main__", "main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "ADMIN_USER_ID") and mod.ADMIN_USER_ID:
            return mod.ADMIN_USER_ID
    return state.admin_user_id


def is_admin_user(user_id: Any) -> bool:
    """Checks if a user ID matches ADMIN_USER_ID or is in the set of configured admin IDs."""
    if not user_id:
        return False
    u_str = str(user_id).strip()
    if not u_str:
        return False
    from modules.formatting import _parse_admin_ids

    primary_id = get_current_admin_user_id()
    admin_ids = get_current_admin_ids()

    if primary_id and (u_str == primary_id or u_str in _parse_admin_ids(primary_id)):
        return True
    return u_str in admin_ids


def is_admin(interaction: discord.Interaction) -> bool:
    """Verifies that the interacting user matches one of the configured admin IDs."""
    if not interaction.user:
        return False
    return is_admin_user(interaction.user.id)


async def verify_admin_interaction(interaction: discord.Interaction) -> bool:
    """Checks admin authorization and sends ephemeral warning if unauthorized."""
    if not is_admin(interaction):
        await interaction.response.send_message(
            "⛔ Unauthorized: Only the configured bot admin can run this command.",
            ephemeral=True,
        )
        return False
    return True


async def profile_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """Dynamic profile name autocomplete helper."""
    if not is_admin(interaction):
        return []
    cfg = get_current_config()
    choices = []
    for name in cfg.profiles.keys():
        if current.lower() in name.lower():
            choices.append(app_commands.Choice(name=name, value=name))
    return choices[:25]


async def mcp_server_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """Dynamic MCP server name autocomplete helper."""
    if not is_admin(interaction):
        return []
    cfg = get_current_config()
    choices = []
    for name in cfg.get_mcp_servers().keys():
        if current.lower() in name.lower():
            choices.append(app_commands.Choice(name=name, value=name))
    return choices[:25]


class BaseCog(commands.Cog):
    """Base class for all Discord bot cogs."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @property
    def config(self):
        return get_current_config()

    @property
    def mcp_manager(self):
        return get_current_mcp_manager()
