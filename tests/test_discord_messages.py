import unittest
from types import SimpleNamespace
import discord
from unittest.mock import AsyncMock

from tests.fakes import FakeChannel, FakeMessage, FakeUser


class TestDiscordMessages(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 9: History anchoring, input policy, and mention suppression (F9)."""

    def test_history_anchoring_and_single_trigger_append(self) -> None:
        """Prior history must be anchored strictly before the trigger message; trigger must be appended once at end."""
        from conversation import build_conversation_context

        bot_user = FakeUser(user_id=999, name="Bot", bot=True)
        requester = FakeUser(user_id=123, name="Alice")
        bystander = FakeUser(user_id=456, name="Bob")
        channel = FakeChannel(channel_id=111)

        # Messages occurring BEFORE the trigger
        m1 = FakeMessage(content="Hello bot", author=requester, channel=channel, message_id=1)
        m2 = FakeMessage(content="Hi Alice, how can I help?", author=bot_user, channel=channel, message_id=2)
        m2.reference = SimpleNamespace(message_id=m1.id, resolved=m1)

        # The actual trigger message
        trigger = FakeMessage(content="<@999> Tell me a joke", author=requester, channel=channel, message_id=3)

        # Build context from prior history list [m1, m2] and trigger
        context = build_conversation_context(
            prior_history=[m1, m2],
            trigger_message=trigger,
            bot_user_id=999,
        )

        # Must have exactly 3 messages: m1, m2, and trigger
        self.assertEqual(len(context), 3)
        self.assertEqual(context[0], {"role": "user", "content": "Hello bot"})
        self.assertEqual(context[1], {"role": "assistant", "content": "Hi Alice, how can I help?"})
        # Trigger message must be the final message, cleaned of bot mention
        self.assertEqual(context[2], {"role": "user", "content": "Tell me a joke"})

    def test_long_input_truncation_policy(self) -> None:
        """Trigger message exceeding max_input_chars must be safely capped with a clear notice."""
        from conversation import build_conversation_context

        requester = FakeUser(user_id=123, name="Alice")
        channel = FakeChannel(channel_id=111)

        giant_input = "A" * 6000
        trigger = FakeMessage(content=f"<@999> {giant_input}", author=requester, channel=channel, message_id=10)

        context = build_conversation_context(
            prior_history=[],
            trigger_message=trigger,
            bot_user_id=999,
            max_input_chars=4000,
        )

        self.assertEqual(len(context), 1)
        content = context[0]["content"]
        self.assertLessEqual(len(content), 4100)
        self.assertIn("truncated", content.lower())

    async def test_send_chunked_reply_suppresses_all_mentions(self) -> None:
        """All replies and subsequent chunks must suppress user, role, and everyone mentions."""
        from main import send_chunked_reply

        channel = FakeChannel(channel_id=222)
        requester = FakeUser(user_id=123, name="Alice")
        msg = FakeMessage(content="Ping test", author=requester, channel=channel)

        malicious_response = "@everyone @here <@&987654321> <@123456789> dangerous ping!"
        await send_chunked_reply(msg, malicious_response)

        self.assertEqual(len(msg.replies), 1)
        sent = msg.replies[0]
        # allowed_mentions must have been set and have all pings disabled
        allowed_mentions = sent.get("allowed_mentions")
        self.assertIsNotNone(allowed_mentions)
        self.assertFalse(allowed_mentions.everyone)
        self.assertFalse(allowed_mentions.roles)
        self.assertFalse(allowed_mentions.users)
        self.assertFalse(allowed_mentions.replied_user)


if __name__ == "__main__":
    unittest.main()
