import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import config_manager
import error_reporting
import llm_service
from config_manager import ConfigManager, resolve_litellm_model
from tool_policy import RequestContext


class TestLiteLLMRouting(unittest.IsolatedAsyncioTestCase):
    """Verifies LiteLLM unified model routing, parameter normalization, and client lifecycle."""

    def test_model_prefix_normalization_openai_custom_base_url(self) -> None:
        """SensNova or OpenAI proxy models must normalize to 'openai/<model>' with api_base preserved."""
        # 1. Bare model name with custom proxy
        model = resolve_litellm_model("openai", "sensenova", "https://sharellm.net/v1")
        self.assertEqual(model, "openai/sensenova")

        # 2. Already prefixed model
        model_prefixed = resolve_litellm_model("openai", "openai/gpt-4o", None)
        self.assertEqual(model_prefixed, "openai/gpt-4o")

        # 3. Model with sub-vendor path (e.g. NVIDIA integrate NIM)
        model_subvendor = resolve_litellm_model("openai", "minimaxai/minimax-m3.1-flash", "https://integrate.api.nvidia.com/v1")
        self.assertEqual(model_subvendor, "openai/minimaxai/minimax-m3.1-flash")

        # 4. _build_litellm_params output verification
        profile = {
            "api_format": "openai",
            "model": "sensenova",
            "base_url": "https://sharellm.net/v1",
            "api_key": "sk-fake-sharellm",
            "max_tokens": 32000,
        }
        params = llm_service._build_litellm_params(profile)
        self.assertEqual(params["model"], "openai/sensenova")
        self.assertEqual(params["api_base"], "https://sharellm.net/v1")
        self.assertEqual(params["api_key"], "sk-fake-sharellm")
        self.assertEqual(params["max_tokens"], 32000)

    def test_native_anthropic_and_gemini_prefix_resolution(self) -> None:
        """Anthropic, Gemini, Ollama, OpenRouter, and LiteLLM auto formats resolve correctly."""
        # Anthropic
        self.assertEqual(resolve_litellm_model("anthropic", "claude-3-5-haiku-latest"), "anthropic/claude-3-5-haiku-latest")
        self.assertEqual(resolve_litellm_model("anthropic", "anthropic/claude-3-7-sonnet"), "anthropic/claude-3-7-sonnet")

        # Gemini
        self.assertEqual(resolve_litellm_model("gemini", "gemini-2.5-pro"), "gemini/gemini-2.5-pro")
        self.assertEqual(resolve_litellm_model("gemini", "gemini/gemini-2.5-flash"), "gemini/gemini-2.5-flash")

        # Ollama
        self.assertEqual(resolve_litellm_model("ollama", "llama3"), "ollama/llama3")

        # OpenRouter
        self.assertEqual(resolve_litellm_model("openrouter", "google/gemini-flash-1.5"), "openrouter/google/gemini-flash-1.5")

        # LiteLLM format (passes model string directly)
        self.assertEqual(resolve_litellm_model("litellm", "xai/grok-3"), "xai/grok-3")
        self.assertEqual(resolve_litellm_model("litellm", "bedrock/anthropic.claude-v2"), "bedrock/anthropic.claude-v2")

    def test_litellm_local_model_cost_map_enforced(self) -> None:
        """Verifies zero telemetry and local cost map are enforced on LiteLLM setup."""
        self.assertEqual(os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"), "True")
        llm = llm_service._get_litellm()
        self.assertFalse(llm.telemetry)
        self.assertTrue(llm.suppress_debug_info)
        self.assertTrue(llm.drop_params)

    async def test_async_http_client_closed_on_success_and_error(self) -> None:
        """The underlying httpx.AsyncClient must be closed deterministically in all exit branches."""
        cfg = ConfigManager(config_path=None)
        cfg.profiles["openai"]["api_key"] = "sk-fake-key"
        cfg.profiles["openai"]["model"] = "gpt-4o-mini"
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        # 1. Success path
        fake_msg = SimpleNamespace(content="Unified success", tool_calls=None)
        fake_choice = SimpleNamespace(finish_reason="stop", message=fake_msg)
        fake_resp = SimpleNamespace(choices=[fake_choice])

        with patch.object(llm_service, "AsyncHTTPClient", return_value=mock_http_client), \
             patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = fake_resp

            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Hello"}],
                cfg,
                None,
                context=admin_ctx,
            )
            self.assertEqual(reply, "Unified success")
            mock_http_client.aclose.assert_awaited_once()

        mock_http_client.aclose.reset_mock()

        # 2. Error path
        with patch.object(llm_service, "AsyncHTTPClient", return_value=mock_http_client), \
             patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.side_effect = RuntimeError("Fatal network drop")

            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Hello"}],
                cfg,
                None,
                context=admin_ctx,
            )
            self.assertIn("Error", reply)
            mock_http_client.aclose.assert_awaited_once()

    async def test_unified_tool_call_serialization_across_multistep_turns(self) -> None:
        """Unified loop executes MCP tool and appends OpenAI-formatted tool result message."""
        cfg = ConfigManager(config_path=None)
        cfg.profiles["openai"]["api_key"] = "sk-fake-key"
        cfg.profiles["openai"]["model"] = "gpt-4o-mini"
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        # Setup mock MCP Manager
        mock_mgr = MagicMock()
        mock_mgr.get_openai_tools = AsyncMock(return_value=[
            {
                "type": "function",
                "function": {
                    "name": "wiki__search_wikipedia",
                    "description": "Search wikipedia",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ])
        mock_mgr.call_tool = AsyncMock(return_value="T-54 was a main battle tank.")

        # Step 1: Model calls tool
        tc = SimpleNamespace(
            id="call_12345",
            type="function",
            function=SimpleNamespace(name="wiki__search_wikipedia", arguments='{"query": "T-54"}'),
        )
        msg1 = SimpleNamespace(content=None, tool_calls=[tc])
        choice1 = SimpleNamespace(finish_reason="tool_calls", message=msg1)
        resp1 = SimpleNamespace(choices=[choice1])

        # Step 2: Model gives final reply
        msg2 = SimpleNamespace(content="The T-54 is a famous tank.", tool_calls=None)
        choice2 = SimpleNamespace(finish_reason="stop", message=msg2)
        resp2 = SimpleNamespace(choices=[choice2])

        responses = [resp1, resp2]
        call_records = []

        async def fake_acompletion(**kwargs):
            call_records.append(kwargs)
            return responses.pop(0)

        with patch("litellm.acompletion", side_effect=fake_acompletion):
            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Tell me about T-54"}],
                cfg,
                mock_mgr,
                context=admin_ctx,
            )

        self.assertEqual(reply, "The T-54 is a famous tank.")
        mock_mgr.call_tool.assert_awaited_once_with("wiki__search_wikipedia", {"query": "T-54"}, context=admin_ctx)

        # Verify second turn received the serialized tool message
        second_turn_messages = call_records[1]["messages"]
        tool_messages = [m for m in second_turn_messages if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_12345")
        self.assertEqual(tool_messages[0]["name"], "wiki__search_wikipedia")
        self.assertEqual(tool_messages[0]["content"], "T-54 was a main battle tank.")

    async def test_dsml_tool_calls_interception_and_tag_stripping(self) -> None:
        """DSML tokens emitted in content are intercepted as tool calls and tags stripped from output."""
        cfg = ConfigManager(config_path=None)
        cfg.profiles["openai"]["api_key"] = "sk-fake-key"
        cfg.profiles["openai"]["model"] = "sensenova"
        cfg.profiles["openai"]["base_url"] = "https://sharellm.net/v1"
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        mock_mgr = MagicMock()
        mock_mgr.get_openai_tools = AsyncMock(return_value=[
            {
                "type": "function",
                "function": {
                    "name": "wiki__search_wikipedia",
                    "description": "Search wikipedia",
                    "parameters": {"type": "object"},
                },
            }
        ])
        mock_mgr.call_tool = AsyncMock(return_value="SenseNova result.")

        # Turn 1: Model emits DSML inside content
        dsml_text = (
            "Thinking...\n"
            "< | DSML | tool_calls>\n"
            "< | DSML | invoke name=\"search_wikipedia\">\n"
            "< | DSML | parameter name=\"query\" string=\"true\">T-55</ | DSML | parameter>\n"
            "</ | DSML | invoke>\n"
            "</ | DSML | tool_calls>"
        )
        msg1 = SimpleNamespace(content=dsml_text, tool_calls=None)
        choice1 = SimpleNamespace(finish_reason="stop", message=msg1)
        resp1 = SimpleNamespace(choices=[choice1])

        # Turn 2: Final response
        msg2 = SimpleNamespace(content="Found details about T-55.", tool_calls=None)
        choice2 = SimpleNamespace(finish_reason="stop", message=msg2)
        resp2 = SimpleNamespace(choices=[choice2])

        responses = [resp1, resp2]

        async def fake_acompletion(**kwargs):
            return responses.pop(0)

        with patch("litellm.acompletion", side_effect=fake_acompletion):
            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Search T-55"}],
                cfg,
                mock_mgr,
                context=admin_ctx,
            )

        self.assertEqual(reply, "Found details about T-55.")
        mock_mgr.call_tool.assert_awaited_once_with("wiki__search_wikipedia", {"query": "T-55"}, context=admin_ctx)

    async def test_litellm_exception_redaction_and_correlation_id(self) -> None:
        """LiteLLM exceptions are caught, scrubbed of query param API keys, and reference ID generated."""
        cfg = ConfigManager(config_path=None)
        cfg.profiles["openai"]["api_key"] = "sk-sensitive-123456789"
        cfg.profiles["openai"]["model"] = "gemini-2.5-pro"
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        llm = llm_service._get_litellm()

        # Simulate Gemini LiteLLM exception containing query parameter key
        secret_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro?key=AIzaSySecretApiKey12345"
        exc = llm.exceptions.BadRequestError(
            message=f"Invalid request to {secret_url}",
            model="gemini-2.5-pro",
            llm_provider="gemini",
        )

        with patch("litellm.acompletion", side_effect=exc):
            reply = await llm_service.generate_response(
                [{"role": "user", "content": "Hi"}],
                cfg,
                None,
                context=admin_ctx,
            )

        self.assertNotIn("AIzaSySecretApiKey12345", reply)
        self.assertIn("Reference ID:", reply)
        self.assertIn("Error", reply)


if __name__ == "__main__":
    unittest.main()
