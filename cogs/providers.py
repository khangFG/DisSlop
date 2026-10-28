"""
Provider management cog: slash commands for configuring, switching, deleting LLM profiles,
updating system prompts, and viewing provider status.
"""

from typing import Optional
import discord
from discord import app_commands
from discord.ext import commands

from config_manager import ConfigManager, resolve_litellm_model
from modules.formatting import split_into_chunks
from .base import (
    BaseCog,
    verify_admin_interaction,
    profile_autocomplete,
)


class ProviderCog(BaseCog):
    """Cog for managing LLM profiles and provider configurations."""

    async def _handle_provider_set(
        self,
        interaction: discord.Interaction,
        profile: str,
        api_format: Optional[app_commands.Choice[str]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        make_active: bool = True,
        max_tokens: Optional[int] = None,
    ) -> None:
        if not await verify_admin_interaction(interaction):
            return

        clean_profile = profile.strip().lower()
        fmt_val = api_format.value if api_format else None

        if clean_profile not in self.config.profiles and fmt_val is None:
            fmt_val = "openai"

        try:
            saved = self.config.upsert_profile(
                name=clean_profile,
                api_format=fmt_val,
                api_key=api_key,
                model=model,
                base_url=base_url,
                max_tokens=max_tokens,
            )

            active_note = ""
            if make_active:
                self.config.set_active_profile(clean_profile)
                active_note = " and set as **Active**"

            masked_key = ConfigManager.mask_key(saved.get("api_key", ""))
            url_display = saved.get("base_url") or "SDK Default"
            tokens_display = str(saved.get("max_tokens") or "Default (4096)")
            resolved_tgt = resolve_litellm_model(
                saved.get("api_format", "openai"),
                saved.get("model", ""),
                saved.get("base_url"),
            )

            await interaction.response.send_message(
                f"✅ **Profile `{clean_profile}` saved successfully{active_note}!**\n"
                f"- **Format:** `{saved.get('api_format')}`\n"
                f"- **Model:** `{saved.get('model')}`\n"
                f"- **LiteLLM Target:** `{resolved_tgt}`\n"
                f"- **Base URL:** `{url_display}`\n"
                f"- **Max Tokens:** `{tokens_display}`\n"
                f"- **API Key:** `{masked_key}`",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ **Failed to update profile:** {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="configure_provider",
        description="Configure or add an LLM provider profile (OpenAI, Anthropic, or 3rd-party).",
    )
    @app_commands.describe(
        profile="Profile identifier name (e.g. openai, anthropic, openrouter, groq, deepseek)",
        api_format="SDK protocol format (litellm, openai, anthropic, gemini, openrouter, ollama)",
        api_key="API secret key for this provider",
        model="Model name (e.g. gpt-4o-mini, claude-3-5-sonnet-latest, deepseek-chat)",
        base_url="Custom endpoint URL or 'default' to reset to official SDK URL",
        make_active="Automatically set this profile as active (default True)",
        max_tokens="Maximum output tokens budget (e.g. 4096, 8192)",
    )
    @app_commands.choices(
        api_format=[
            app_commands.Choice(name="LiteLLM (Auto / Prefix-based)", value="litellm"),
            app_commands.Choice(name="OpenAI Compatible", value="openai"),
            app_commands.Choice(name="Anthropic Compatible", value="anthropic"),
            app_commands.Choice(name="Google Gemini", value="gemini"),
            app_commands.Choice(name="OpenRouter", value="openrouter"),
            app_commands.Choice(name="Ollama (Local)", value="ollama"),
        ]
    )
    @app_commands.autocomplete(profile=profile_autocomplete)
    async def configure_provider(
        self,
        interaction: discord.Interaction,
        profile: str,
        api_format: Optional[app_commands.Choice[str]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        make_active: bool = True,
        max_tokens: Optional[int] = None,
    ) -> None:
        await self._handle_provider_set(
            interaction, profile, api_format, api_key, model, base_url, make_active, max_tokens
        )

    @app_commands.command(
        name="provider_set",
        description="Alias for /configure_provider.",
    )
    @app_commands.describe(
        profile="Profile identifier name",
        api_format="SDK protocol format (litellm, openai, anthropic, gemini, openrouter, ollama)",
        api_key="API secret key",
        model="Model name",
        base_url="Custom endpoint URL or 'default'",
        make_active="Set as active profile",
        max_tokens="Maximum output tokens budget (e.g. 4096, 8192)",
    )
    @app_commands.choices(
        api_format=[
            app_commands.Choice(name="LiteLLM (Auto / Prefix-based)", value="litellm"),
            app_commands.Choice(name="OpenAI Compatible", value="openai"),
            app_commands.Choice(name="Anthropic Compatible", value="anthropic"),
            app_commands.Choice(name="Google Gemini", value="gemini"),
            app_commands.Choice(name="OpenRouter", value="openrouter"),
            app_commands.Choice(name="Ollama (Local)", value="ollama"),
        ]
    )
    @app_commands.autocomplete(profile=profile_autocomplete)
    async def provider_set(
        self,
        interaction: discord.Interaction,
        profile: str,
        api_format: Optional[app_commands.Choice[str]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        make_active: bool = True,
        max_tokens: Optional[int] = None,
    ) -> None:
        await self._handle_provider_set(
            interaction, profile, api_format, api_key, model, base_url, make_active, max_tokens
        )

    async def _handle_provider_switch(
        self,
        interaction: discord.Interaction,
        profile: str,
        model: Optional[str] = None,
    ) -> None:
        if not await verify_admin_interaction(interaction):
            return

        clean_profile = profile.strip().lower()
        try:
            active_p = self.config.set_active_profile(clean_profile, model_override=model)
            url_display = active_p.get("base_url") or "SDK Default"
            masked_key = ConfigManager.mask_key(active_p.get("api_key", ""))
            resolved_tgt = resolve_litellm_model(
                active_p.get("api_format", "openai"),
                active_p.get("model", ""),
                active_p.get("base_url"),
            )

            await interaction.response.send_message(
                f"🔄 **Active profile switched to `{clean_profile}`!**\n"
                f"- **Format:** `{active_p.get('api_format')}`\n"
                f"- **Model:** `{active_p.get('model')}`\n"
                f"- **LiteLLM Target:** `{resolved_tgt}`\n"
                f"- **Base URL:** `{url_display}`\n"
                f"- **API Key:** `{masked_key}`",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ **Switch failed:** {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="switch_provider",
        description="Instantly switch the active provider profile and/or model.",
    )
    @app_commands.describe(
        profile="Name of the saved profile to activate",
        model="Optional model override for the profile",
    )
    @app_commands.autocomplete(profile=profile_autocomplete)
    async def switch_provider(
        self,
        interaction: discord.Interaction,
        profile: str,
        model: Optional[str] = None,
    ) -> None:
        await self._handle_provider_switch(interaction, profile, model)

    @app_commands.command(
        name="provider_switch",
        description="Alias for /switch_provider.",
    )
    @app_commands.describe(
        profile="Name of the saved profile to activate",
        model="Optional model override for the profile",
    )
    @app_commands.autocomplete(profile=profile_autocomplete)
    async def provider_switch(
        self,
        interaction: discord.Interaction,
        profile: str,
        model: Optional[str] = None,
    ) -> None:
        await self._handle_provider_switch(interaction, profile, model)

    async def _handle_provider_delete(
        self,
        interaction: discord.Interaction,
        profile: str,
    ) -> None:
        if not await verify_admin_interaction(interaction):
            return

        clean_profile = profile.strip().lower()
        try:
            self.config.delete_profile(clean_profile)
            await interaction.response.send_message(
                f"🗑️ Profile `{clean_profile}` was successfully deleted.",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ **Delete failed:** {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="delete_provider",
        description="Remove a saved custom provider profile.",
    )
    @app_commands.describe(profile="Name of the profile to delete")
    @app_commands.autocomplete(profile=profile_autocomplete)
    async def delete_provider(
        self,
        interaction: discord.Interaction,
        profile: str,
    ) -> None:
        await self._handle_provider_delete(interaction, profile)

    @app_commands.command(
        name="provider_delete",
        description="Alias for /delete_provider.",
    )
    @app_commands.describe(profile="Name of the profile to delete")
    @app_commands.autocomplete(profile=profile_autocomplete)
    async def provider_delete(
        self,
        interaction: discord.Interaction,
        profile: str,
    ) -> None:
        await self._handle_provider_delete(interaction, profile)

    @app_commands.command(
        name="set_prompt",
        description="Change the bot's system prompt / persona on the fly.",
    )
    @app_commands.describe(prompt="The new system prompt for the AI")
    async def set_prompt(
        self,
        interaction: discord.Interaction,
        prompt: str,
    ) -> None:
        if not await verify_admin_interaction(interaction):
            return

        try:
            updated = self.config.set_system_prompt(prompt)
            await interaction.response.send_message(
                f"🧠 **System prompt updated successfully!**\n> {updated}",
                ephemeral=True,
            )
        except Exception as e:
            await interaction.response.send_message(
                f"❌ **Failed to update system prompt:** {e}",
                ephemeral=True,
            )

    @app_commands.command(
        name="llm_status",
        description="View active provider, model, base_url, masked keys, and MCP servers status.",
    )
    async def llm_status(self, interaction: discord.Interaction) -> None:
        if not await verify_admin_interaction(interaction):
            return

        report = self.config.get_status_report()
        chunks = split_into_chunks(report, max_chars=1950)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await interaction.response.send_message(chunk, ephemeral=True)
            else:
                await interaction.followup.send(chunk, ephemeral=True)
