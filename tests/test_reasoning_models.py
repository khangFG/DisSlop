import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import llm_service
from tool_policy import RequestContext


class TestReasoningModels(unittest.IsolatedAsyncioTestCase):
    """Verifies that reasoning/thinking model outputs (DeepSeek-R1, GLM, Claude thinking)

    are preserved and formatted when 'content' is empty or token limit is reached.
    """

    async def test_openai_extracts_reasoning_when_content_is_none(self) -> None:
        cfg = MagicMock()
        cfg.get_active_profile.return_value = (
            "test_profile",
            {
                "api_format": "openai",
                "api_key": "fake_key",
                "base_url": None,
                "model": "glm-5.3-flash",
                "max_tokens": 4096,
            },
        )
        cfg.system_prompt = "You are a bot"

        fake_msg = MagicMock()
        fake_msg.content = None
        fake_msg.tool_calls = None
        fake_msg.reasoning_content = "This is the model's detailed reasoning breakdown."
        fake_msg.model_extra = {}

        fake_choice = MagicMock()
        fake_choice.finish_reason = "stop"
        fake_choice.message = fake_msg

        fake_resp = MagicMock()
        fake_resp.choices = [fake_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = fake_resp

        with patch("llm_service.AsyncOpenAI", return_value=mock_client):
            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Tell me about tanks"}],
                cfg,
                None,
                context=RequestContext(user_id=123, is_admin=False),
            )

        self.assertIn("This is the model's detailed reasoning breakdown.", reply)
        self.assertNotIn("No response content", reply)

    async def test_openai_token_limit_reached_during_reasoning_includes_warning(self) -> None:
        cfg = MagicMock()
        cfg.get_active_profile.return_value = (
            "test_profile",
            {
                "api_format": "openai",
                "api_key": "fake_key",
                "base_url": None,
                "model": "glm-5.3-flash",
            },
        )
        cfg.system_prompt = "You are a bot"

        fake_msg = MagicMock()
        fake_msg.content = ""
        fake_msg.tool_calls = None
        fake_msg.reasoning_content = None
        fake_msg.model_extra = {"reasoning": "Analyzing all T-80 variants..."}

        fake_choice = MagicMock()
        fake_choice.finish_reason = "length"
        fake_choice.message = fake_msg

        fake_resp = MagicMock()
        fake_resp.choices = [fake_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = fake_resp

        with patch("llm_service.AsyncOpenAI", return_value=mock_client):
            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Tell me about T-80"}],
                cfg,
                None,
                context=RequestContext(user_id=123, is_admin=False),
            )

        self.assertIn("Token limit reached during model reasoning", reply)
        self.assertIn("Analyzing all T-80 variants...", reply)

    async def test_anthropic_extracts_thinking_block_when_text_empty(self) -> None:
        cfg = MagicMock()
        cfg.get_active_profile.return_value = (
            "test_anthropic",
            {
                "api_format": "anthropic",
                "api_key": "fake_key",
                "base_url": None,
                "model": "claude-3-7-sonnet",
            },
        )
        cfg.system_prompt = "You are a bot"

        fake_thinking_block = MagicMock()
        fake_thinking_block.type = "thinking"
        fake_thinking_block.thinking = "Deep thought about the user question."

        fake_resp = MagicMock()
        fake_resp.stop_reason = "end_turn"
        fake_resp.content = [fake_thinking_block]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = fake_resp

        with patch("llm_service.AsyncAnthropic", return_value=mock_client):
            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Tell me a story"}],
                cfg,
                None,
                context=RequestContext(user_id=123, is_admin=False),
            )

        self.assertEqual(reply, "Deep thought about the user question.")


if __name__ == "__main__":
    unittest.main()
