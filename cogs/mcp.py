"""
MCP server management cog: slash commands for adding stdio, SSE, and HTTP MCP servers,
removing servers, and inspecting exposed tools.
"""

import os
from typing import Optional
import discord
from discord import app_commands

from modules.formatting import (
    parse_admin_arguments,
    parse_admin_mapping,
    split_into_chunks,
)
from .base import (
    BaseCog,
    verify_admin_interaction,
    mcp_server_autocomplete,
)


class MCPCog(BaseCog):
    """Cog for managing Model Context Protocol (MCP) server connections."""

    @app_commands.command(
        name="add_mcp_stdio",
        description="Connect a local stdio MCP server on the fly (e.g. npx, uvx, python).",
    )
    @app_commands.describe(
        name="Identifier name for this MCP server (e.g. brave, github, fetch, sqlite)",
        command="Executable command (e.g. npx, uvx, python)",
        args="Command-line arguments (e.g. -y @modelcontextprotocol/server-brave-search or JSON array)",
        env="Environment variables/API keys for the tool (e.g. BRAVE_API_KEY=xxx or JSON map)",
    )
    async def add_mcp_stdio(
        self,
        interaction: discord.Interaction,
        name: str,
        command: str,
        args: str = "",
        env: Optional[str] = None,
    ) -> None:
        if not await verify_admin_interaction(interaction):
            return

        clean_name = name.strip().lower()
        try:
            split_args = parse_admin_arguments(args)
            cmd_clean = command.strip()
            if os.path.basename(cmd_clean).lower() in ("uvx", "uvx.exe", "uv", "uv.exe"):
                split_args = [a for a in split_args if a != "-y"]
            env_dict = parse_admin_mapping(env)

            self.config.upsert_mcp_server(
                name=clean_name,
                transport="stdio",
                command=cmd_clean,
                args=split_args,
                env=env_dict,
                enabled=True,
            )
            import main
            main.schedule_tool_refresh()
            masked_env_keys = ", ".join(env_dict.keys()) if env_dict else "None"
            await interaction.response.send_message(
                f"✅ **MCP Server `{clean_name}` (stdio) added successfully!**\n"
                f"- **Command:** `{command.strip()}`\n"
                f"- **Arguments:** `{split_args}`\n"
                f"- **Configured Keys:** `{masked_env_keys}`\n\n"
                f"Run `/mcp_tools` to discover and verify exposed tools.",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ **Failed to add MCP server:** {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="add_mcp_sse",
        description="Connect a remote SSE or Streamable HTTP MCP server on the fly.",
    )
    @app_commands.describe(
        name="Identifier name for this MCP server",
        url="Remote URL endpoint (e.g. https://mcp.parallel.ai/mcp or http://localhost:8000/sse)",
        api_key="Bearer token / API key for the remote server (optional)",
        headers="Custom HTTP headers (e.g. X-API-Key=xxx,Auth=yyy or JSON map) (optional)",
    )
    async def add_mcp_sse(
        self,
        interaction: discord.Interaction,
        name: str,
        url: str,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
    ) -> None:
        if not await verify_admin_interaction(interaction):
            return

        clean_name = name.strip().lower()
        try:
            headers_dict = parse_admin_mapping(headers)
            if api_key and api_key.strip():
                headers_dict["Authorization"] = f"Bearer {api_key.strip()}"

            self.config.upsert_mcp_server(
                name=clean_name,
                transport="sse",
                url=url.strip(),
                headers=headers_dict,
                enabled=True,
            )
            import main
            main.schedule_tool_refresh()
            has_auth = "Configured (Bearer / Custom Headers)" if headers_dict else "None"
            await interaction.response.send_message(
                f"✅ **MCP Server `{clean_name}` (Remote SSE / HTTP) added successfully!**\n"
                f"- **URL:** `{url.strip()}`\n"
                f"- **Auth Headers:** `{has_auth}`\n\n"
                f"Run `/mcp_tools` to discover and verify exposed tools.",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ **Failed to add MCP server:** {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="add_mcp_http",
        description="Alias for connecting remote Streamable HTTP or SSE MCP servers.",
    )
    @app_commands.describe(
        name="Identifier name for this MCP server",
        url="Remote Streamable HTTP URL endpoint",
        api_key="Bearer token / API key (optional)",
        headers="Custom HTTP headers (optional)",
    )
    async def add_mcp_http(
        self,
        interaction: discord.Interaction,
        name: str,
        url: str,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
    ) -> None:
        await self.add_mcp_sse(interaction, name, url, api_key, headers)

    @app_commands.command(
        name="remove_mcp",
        description="Remove a configured MCP server.",
    )
    @app_commands.describe(name="Name of the MCP server to remove")
    @app_commands.autocomplete(name=mcp_server_autocomplete)
    async def remove_mcp(
        self,
        interaction: discord.Interaction,
        name: str,
    ) -> None:
        if not await verify_admin_interaction(interaction):
            return

        clean_name = name.strip().lower()
        try:
            self.config.remove_mcp_server(clean_name)
            import main
            main.schedule_tool_refresh()
            await interaction.response.send_message(
                f"🗑️ MCP server `{clean_name}` was successfully removed.",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ **Failed to remove MCP server:** {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="mcp_tools",
        description="Connects to enabled MCP servers and lists all active servers and their exposed tools.",
    )
    async def mcp_tools(self, interaction: discord.Interaction) -> None:
        if not await verify_admin_interaction(interaction):
            return

        # Defer response as MCP tool discovery involves network/process I/O
        await interaction.response.defer(ephemeral=True)

        try:
            overview = await self.mcp_manager.get_overview_report()
            chunks = split_into_chunks(overview, max_chars=1950)
            for chunk in chunks:
                await interaction.followup.send(chunk, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error inspecting MCP tools: {e}", ephemeral=True)
