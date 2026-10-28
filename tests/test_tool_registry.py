import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from tests.fakes import FakeMCPServerWorker, FakeTextContent, FakeToolResult


class TestToolRegistry(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 3: Exact registry and argument validation (F4)."""

    async def asyncSetUp(self) -> None:
        from config_manager import ConfigManager
        from mcp_manager import MCPManager
        from tool_policy import RequestContext

        self.cfg = ConfigManager(config_path="test_registry_config.json")
        self.cfg.profiles["openai"]["api_key"] = "synthetic-key"
        self.cfg.tool_policy["allowed_tools"] = ["srv__unregistered_command", "srv__do_task", "srv__write_data", "srv__query_db"]
        self.cfg.mcp_servers = {"srv": {"transport": "stdio", "command": "dummy", "enabled": True}}
        self.mgr = MCPManager(self.cfg)
        self.admin_ctx = RequestContext(user_id="admin_1", is_admin=True)

    async def test_rejects_unregistered_guessed_tool_names(self) -> None:
        """Calling an unregistered/unexposed tool name must be rejected without dispatching to any worker."""
        worker = FakeMCPServerWorker(name="srv", tools=[])
        self.mgr._workers["srv"] = worker
        self.mgr._tools_initialized = True
        self.mgr._tool_mappings.clear()

        # Attacker tries to guess an unexposed tool name on the server
        res = await self.mgr.call_tool("srv__unregistered_command", {"arg": "val"}, context=self.admin_ctx)
        self.assertIn("not registered", res.lower())
        self.assertEqual(len(worker.calls), 0)

    async def test_collision_resistant_naming_for_punctuation(self) -> None:
        """Punctuation variations like 'tool.a' and 'tool_a' must not collide or silently overwrite."""
        t1 = type("Tool", (), {
            "name": "read.file",
            "description": "Reads file with dot",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        })()
        t2 = type("Tool", (), {
            "name": "read_file",
            "description": "Reads file with underscore",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        })()

        worker = FakeMCPServerWorker(name="srv", tools=[t1, t2])
        self.mgr._workers["srv"] = worker

        # Refresh tool cache
        await self.mgr.refresh_tools()

        # Both tools must have unique registered IDs in _tool_mappings
        orig_names = [v["original_name"] for v in self.mgr._tool_mappings.values()]
        self.assertIn("read.file", orig_names)
        self.assertIn("read_file", orig_names)
        # Ensure they did not overwrite each other into a single tool
        self.assertEqual(len(self.mgr._tool_mappings), 2)

    async def test_collision_resistant_naming_for_long_names(self) -> None:
        """Tools with identical prefixes >64 characters must not collide or silently overwrite."""
        long_base = "a" * 60
        t1 = type("Tool", (), {
            "name": f"{long_base}_one",
            "description": "Long tool 1",
            "input_schema": {"type": "object"},
        })()
        t2 = type("Tool", (), {
            "name": f"{long_base}_two",
            "description": "Long tool 2",
            "input_schema": {"type": "object"},
        })()

        worker = FakeMCPServerWorker(name="srv", tools=[t1, t2])
        self.mgr._workers["srv"] = worker

        await self.mgr.refresh_tools()

        self.assertEqual(len(self.mgr._tool_mappings), 2)
        # All generated identifiers must be <= 64 characters and match identifier rules
        for tool_id in self.mgr._tool_mappings.keys():
            self.assertLessEqual(len(tool_id), 64)
            self.assertTrue(tool_id.replace("_", "").replace("-", "").isalnum())

    async def test_rejects_non_dict_arguments(self) -> None:
        """Tool calls with null, string, array, or scalar arguments must be rejected before dispatch."""
        t = type("Tool", (), {
            "name": "do_task",
            "description": "Task tool",
            "input_schema": {"type": "object", "properties": {"msg": {"type": "string"}}},
        })()
        worker = FakeMCPServerWorker(name="srv", tools=[t])
        self.mgr._workers["srv"] = worker
        await self.mgr.refresh_tools()

        tool_id = list(self.mgr._tool_mappings.keys())[0]

        # Call with None
        res_none = await self.mgr.call_tool(tool_id, None, context=self.admin_ctx)  # type: ignore
        self.assertIn("invalid", res_none.lower())

        # Call with list
        res_list = await self.mgr.call_tool(tool_id, ["not", "a", "dict"], context=self.admin_ctx)  # type: ignore
        self.assertIn("invalid", res_list.lower())

        # Call with string
        res_str = await self.mgr.call_tool(tool_id, "hello", context=self.admin_ctx)  # type: ignore
        self.assertIn("invalid", res_str.lower())

        self.assertEqual(len(worker.calls), 0)

    async def test_rejects_schema_violating_arguments(self) -> None:
        """Tool arguments violating required properties or types must fail before worker dispatch."""
        t = type("Tool", (), {
            "name": "write_data",
            "description": "Writes data",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        })()
        worker = FakeMCPServerWorker(name="srv", tools=[t])
        self.mgr._workers["srv"] = worker
        await self.mgr.refresh_tools()

        tool_id = list(self.mgr._tool_mappings.keys())[0]

        # Missing required parameter 'file_path'
        res_missing = await self.mgr.call_tool(tool_id, {"count": 5}, context=self.admin_ctx)
        self.assertIn("schema", res_missing.lower())
        self.assertEqual(len(worker.calls), 0)

        # Invalid type for 'count'
        res_wrong_type = await self.mgr.call_tool(tool_id, {"file_path": "a.txt", "count": "not_an_int"}, context=self.admin_ctx)
        self.assertIn("schema", res_wrong_type.lower())
        self.assertEqual(len(worker.calls), 0)

    async def test_llm_service_rejects_malformed_json_arguments(self) -> None:
        """In llm_service, malformed JSON arguments from the model must not be converted to empty dict."""
        import llm_service

        t = type("Tool", (), {
            "name": "query_db",
            "description": "Queries database",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        })()
        worker = FakeMCPServerWorker(name="srv", tools=[t])
        self.mgr._workers["srv"] = worker
        await self.mgr.refresh_tools()

        mock_choice = AsyncMock()
        mock_choice.finish_reason = "tool_calls"
        mock_tc = AsyncMock()
        mock_tc.id = "call_broken_json"
        tool_id = list(self.mgr._tool_mappings.keys())[0]
        mock_tc.function.name = tool_id
        mock_tc.function.arguments = "MALFORMED { JSON"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.message.content = None

        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]

        with patch("llm_service.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            messages = [{"role": "user", "content": "Run query"}]
            # Admin context
            res = await llm_service.generate_response(messages, self.cfg, self.mgr, context=self.admin_ctx)

            mock_client.chat.completions.create.assert_awaited()
            # Worker must NOT have been called with empty dict
            self.assertEqual(len(worker.calls), 0)


if __name__ == "__main__":
    unittest.main()
