import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from tests.fakes import FakeMCPServerWorker, FakeTextContent, FakeToolResult


class TestToolPolicy(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 2: Caller and tool authorization enforcement."""

    async def asyncSetUp(self) -> None:
        from config_manager import ConfigManager
        from mcp_manager import MCPManager
        from tool_policy import RequestContext

        self.cfg = ConfigManager(config_path="test_policy_config.json")
        self.cfg.profiles["openai"]["api_key"] = "synthetic-key"
        self.cfg.tool_policy["allowed_tools"] = ["fs__read_file"]
        self.cfg.mcp_servers = {"fs": {"transport": "stdio", "command": "dummy", "enabled": True}}
        self.mgr = MCPManager(self.cfg)

        # Setup fake worker with a privileged tool
        worker = FakeMCPServerWorker(
            name="fs",
            tools=[type("Tool", (), {"name": "read_file", "description": "Reads file", "input_schema": {}})()],
        )
        self.mgr._workers["fs"] = worker
        self.mgr._tool_mappings["fs__read_file"] = {
            "server_name": "fs",
            "original_name": "read_file",
            "description": "Reads file",
            "input_schema": {},
        }
        self.mgr._tools_initialized = True

        self.admin_ctx = RequestContext(user_id="111", is_admin=True)
        self.non_admin_ctx = RequestContext(user_id="222", is_admin=False)

    async def test_missing_identity_denies_dispatch(self) -> None:
        """Missing request identity must deny tool dispatch."""
        res = await self.mgr.call_tool("fs__read_file", {"path": "/etc/passwd"}, context=None)
        self.assertIn("Permission denied", res)
        # Verify worker received zero calls
        self.assertEqual(len(self.mgr._workers["fs"].calls), 0)

    async def test_non_admin_denied_by_default(self) -> None:
        """Non-admin caller must be denied by default."""
        res = await self.mgr.call_tool("fs__read_file", {"path": "/etc/passwd"}, context=self.non_admin_ctx)
        self.assertIn("Permission denied", res)
        self.assertEqual(len(self.mgr._workers["fs"].calls), 0)

    async def test_authorized_admin_succeeds(self) -> None:
        """Authorized admin caller succeeds."""
        res = await self.mgr.call_tool("fs__read_file", {"path": "/allowed"}, context=self.admin_ctx)
        self.assertNotIn("Permission denied", res)
        self.assertEqual(len(self.mgr._workers["fs"].calls), 1)

    async def test_tool_exposure_filters_for_non_admin(self) -> None:
        """Tool exposure list must be empty for non-admin callers by default."""
        from tool_policy import filter_tools_for_request

        tools = [{"name": "fs__read_file", "type": "function"}]
        filtered_non_admin = filter_tools_for_request(self.non_admin_ctx, tools, self.cfg)
        self.assertEqual(filtered_non_admin, [])

        filtered_none = filter_tools_for_request(None, tools, self.cfg)
        self.assertEqual(filtered_none, [])

        filtered_admin = filter_tools_for_request(self.admin_ctx, tools, self.cfg)
        self.assertEqual(filtered_admin, tools)

    async def test_llm_service_does_not_call_tool_for_non_admin(self) -> None:
        """In llm_service, non-admin caller receives no tools and cannot execute fabricated tool calls."""
        import llm_service

        mock_choice = AsyncMock()
        mock_choice.finish_reason = "tool_calls"
        mock_tc = AsyncMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "fs__read_file"
        mock_tc.function.arguments = '{"path": "secret.txt"}'
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.message.content = "Let me read that."

        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]

        with patch("llm_service.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            messages = [{"role": "user", "content": "Please read secret.txt"}]
            # Non-admin context
            res = await llm_service.generate_response(messages, self.cfg, self.mgr, context=self.non_admin_ctx)

            mock_client.chat.completions.create.assert_awaited()
            # Worker must NOT have been called
            self.assertEqual(len(self.mgr._workers["fs"].calls), 0)

    async def test_server_name_pattern_allows_prefixed_tools(self) -> None:
        """Specifying server name e.g. 'fs' in allowed_tools allows 'fs__read_file'."""
        from tool_policy import is_tool_allowed, filter_tools_for_request

        self.cfg.tool_policy["allowed_tools"] = ["fs"]
        tools = [{"name": "fs__read_file"}, {"name": "other__tool"}]
        filtered = filter_tools_for_request(self.admin_ctx, tools, self.cfg)
        self.assertEqual(filtered, [{"name": "fs__read_file"}])
        self.assertTrue(is_tool_allowed(self.admin_ctx, "fs__read_file", self.cfg))
        self.assertFalse(is_tool_allowed(self.admin_ctx, "other__tool", self.cfg))

    async def test_wildcard_pattern_allows_all_tools(self) -> None:
        """Specifying '*' in allowed_tools allows all tools."""
        from tool_policy import is_tool_allowed, filter_tools_for_request

        self.cfg.tool_policy["allowed_tools"] = ["*"]
        tools = [{"name": "fs__read_file"}, {"name": "other__tool"}]
        filtered = filter_tools_for_request(self.admin_ctx, tools, self.cfg)
        self.assertEqual(filtered, tools)
        self.assertTrue(is_tool_allowed(self.admin_ctx, "fs__read_file", self.cfg))
        self.assertTrue(is_tool_allowed(self.admin_ctx, "other__tool", self.cfg))


if __name__ == "__main__":
    unittest.main()
