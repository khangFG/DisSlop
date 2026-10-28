"""
Chat interaction cog: @mention conversational listener, history context anchoring,
rate/concurrency admission control, typing indicator, and administrative !sync prefix command.
"""

import asyncio
from typing import List
import discord
from discord.ext import commands

from tool_policy import RequestContext
from error_reporting import format_public_error, log_sanitized_error
from request_limits import RequestLimitError, REQUEST_TIMEOUT_SECONDS
from conversation import build_conversation_context
from modules.formatting import send_chunked_reply
import llm_service
from .base import BaseCog, is_admin_user


class ChatCog(BaseCog):
    """Cog handling user chat messages, mentions, and administrative prefix commands."""

    @commands.command(
        name="sync",
        help="Instantly synchronizes all slash commands to the current Discord server.",
    )
    async def sync(self, ctx: commands.Context) -> None:
        """Admin fallback command to force instant slash command sync to the current server."""
        if not is_admin_user(ctx.author.id):
            await ctx.send("⛔ Unauthorized: Only configured bot admins can run `!sync`.")
            return

        if ctx.guild is None:
            await ctx.send("⚠️ Please run `!sync` inside a server channel, not in DMs.")
            return

        async with ctx.typing():
            try:
                self.bot.tree.copy_global_to(guild=ctx.guild)
                synced = await self.bot.tree.sync(guild=ctx.guild)
                await ctx.send(
                    f"✅ Successfully synced {len(synced)} slash commands to **{ctx.guild.name}**! "
                    "They should now be visible when typing `/`."
                )
            except Exception as e:
                await ctx.send(f"❌ Failed to sync slash commands: {e}")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages sent by any bot (including itself)
        if message.author.bot:
            return

        # Check if bot was mentioned
        if self.bot.user not in message.mentions:
            await self.bot.process_commands(message)
            return

        import main

        # Enforce monotonic cooldown (Admins bypass cooldown)
        user_id_str = str(message.author.id)
        user_is_admin = is_admin_user(user_id_str)

        if not user_is_admin:
            allowed, reason = main.request_limiter.check_and_record_cooldown(user_id_str)
            if not allowed:
                await message.reply(
                    reason,
                    mention_author=False,
                    delete_after=4.0,
                )
                return

        try:
            async with main.request_limiter.acquire(user_id_str), asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                # Show Discord's typing indicator while generating response
                async with message.channel.typing():
                    try:
                        # Build conversational context anchored strictly BEFORE the trigger message
                        raw_history: List[discord.Message] = []
                        async for h_msg in message.channel.history(limit=10, before=message):
                            raw_history.append(h_msg)

                        # Chronological order (oldest to newest)
                        raw_history.reverse()

                        context_messages = build_conversation_context(
                            prior_history=raw_history,
                            trigger_message=message,
                            bot_user_id=self.bot.user.id if self.bot.user else 0,
                        )

                        # Build immutable request identity
                        request_ctx = RequestContext(
                            user_id=user_id_str,
                            guild_id=str(message.guild.id) if message.guild else None,
                            channel_id=str(message.channel.id),
                            is_admin=user_is_admin,
                        )

                        # Dispatch to LLM service with caller identity and full MCP multi-step tool loop
                        reply_text = await llm_service.generate_response(
                            context_messages,
                            self.config,
                            self.mcp_manager,
                            context=request_ctx,
                        )

                        # Send response using chunked reply helper
                        await send_chunked_reply(message, reply_text)

                    except Exception as e:
                        public_err, cid = format_public_error("An unexpected error occurred while processing your request.")
                        log_sanitized_error(cid, "Error processing mention in on_message", e)
                        await message.reply(
                            public_err,
                            mention_author=False,
                            allowed_mentions=discord.AllowedMentions.none(),
                        )

        except RequestLimitError as rerr:
            await message.reply(str(rerr), mention_author=False)
        except TimeoutError:
            await message.reply("The request timed out. Check any tool operation before retrying.", mention_author=False)

        await self.bot.process_commands(message)
