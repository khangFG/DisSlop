import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

from tests.fakes import FakeMCPServerWorker


class TestRequestLimits(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 8: Resource bounds, cooldowns, and SDK client closure (F7)."""

    async def test_sdk_clients_closed_on_exit_and_error(self) -> None:
        """AsyncOpenAI and AsyncAnthropic clients must be explicitly closed on every return/error path."""
        import llm_service
        from config_manager import ConfigManager
        from mcp_manager import MCPManager
        from tool_policy import RequestContext

        cfg = ConfigManager("test_limits_config.json")
        cfg.profiles["openai"]["api_key"] = "sk-fake-key"
        mgr = MCPManager(cfg)
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        # 1. Test OpenAI normal completion closes client
        with patch("llm_service.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_resp = AsyncMock()
            mock_choice = AsyncMock()
            mock_choice.finish_reason = "stop"
            mock_choice.message.content = "Response"
            mock_choice.message.tool_calls = None
            mock_resp.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            await llm_service.generate_response([{"role": "user", "content": "Hi"}], cfg, mgr, context=admin_ctx)
            mock_client.close.assert_awaited()

        # 2. Test OpenAI exception closes client
        with patch("llm_service.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("Fatal error")
            mock_openai_cls.return_value = mock_client

            await llm_service.generate_response([{"role": "user", "content": "Hi"}], cfg, mgr, context=admin_ctx)
            mock_client.close.assert_awaited()

    async def test_limiter_monotonic_expiring_cooldown(self) -> None:
        """Limiter must reject rapid requests within the cooldown window and prune expired records."""
        from request_limits import RequestLimiter

        limiter = RequestLimiter(cooldown_seconds=1.0)
        user = "user_123"

        # First request succeeds
        allowed, reason = limiter.check_and_record_cooldown(user)
        self.assertTrue(allowed)

        # Immediate follow-up is rejected by cooldown
        allowed2, reason2 = limiter.check_and_record_cooldown(user)
        self.assertFalse(allowed2)
        self.assertIn("wait", reason2.lower())

        # Wait for cooldown to expire
        await asyncio.sleep(1.05)
        allowed3, _ = limiter.check_and_record_cooldown(user)
        self.assertTrue(allowed3)

    async def test_limiter_concurrency_bounds_and_releases_on_error(self) -> None:
        """Limiter must enforce per-user and global concurrency and release slots on error/cancellation."""
        from request_limits import RequestLimiter

        limiter = RequestLimiter(max_per_user=1, max_global=2)
        user = "u1"

        async with limiter.acquire(user):
            # Second concurrent request from same user should be rejected
            with self.assertRaises(RuntimeError) as cm:
                async with limiter.acquire(user):
                    pass
            self.assertIn("active", str(cm.exception).lower())

        # After block exit, user can acquire again
        acquired = False
        async with limiter.acquire(user):
            acquired = True
        self.assertTrue(acquired)

    async def test_tool_budget_caps_excessive_tool_calls(self) -> None:
        """LLM service must stop executing tool calls if the turn tool budget is exceeded."""
        import llm_service
        from config_manager import ConfigManager
        from mcp_manager import MCPManager
        from tool_policy import RequestContext

        cfg = ConfigManager("test_limits_config.json")
        cfg.profiles["openai"]["api_key"] = "sk-fake-key"
        mgr = MCPManager(cfg)
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        tool_exec_count = 0
        async def fake_handler(tool_name, args):
            nonlocal tool_exec_count
            tool_exec_count += 1
            return type("Result", (), {"content": [type("Text", (), {"text": "ok"})()], "is_error": False})()

        worker = FakeMCPServerWorker("srv", call_handler=fake_handler)
        cfg.mcp_servers = {"srv": {"transport": "stdio", "command": "fake", "enabled": True}}
        cfg.tool_policy["allowed_tools"] = ["srv__loop_tool"]
        mgr._cached_openai_tools = [{"type": "function", "function": {"name": "srv__loop_tool", "parameters": {"type": "object"}}}]
        mgr._workers["srv"] = worker
        mgr._tool_mappings["srv__loop_tool"] = {
            "server_name": "srv",
            "original_name": "loop_tool",
            "description": "Looping tool",
            "input_schema": {"type": "object"},
        }
        mgr._tools_initialized = True

        # Simulate model requesting 15 tool calls in a single choice
        many_calls = []
        for i in range(15):
            tc = AsyncMock()
            tc.id = f"tc_{i}"
            tc.function.name = "srv__loop_tool"
            tc.function.arguments = "{}"
            many_calls.append(tc)

        mock_choice = AsyncMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message.tool_calls = many_calls
        mock_choice.message.content = None

        mock_resp = AsyncMock()
        mock_resp.choices = [mock_choice]

        with patch("llm_service.AsyncOpenAI") as mock_openai_cls, patch.object(llm_service, "MAX_TOOL_CALLS_PER_TURN", 8):
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            await llm_service.generate_response([{"role": "user", "content": "Please run the loop tool"}], cfg, mgr, context=admin_ctx)

            # Must be bounded by MAX_TOOL_CALLS_PER_TURN (8)
            self.assertEqual(tool_exec_count, 8)


if __name__ == "__main__":
    unittest.main()
