import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch


class TestWorkerLifecycle(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 5: Worker lifecycle, cancellation, and replay prevention (F3)."""

    async def test_expired_queued_call_executes_zero_times(self) -> None:
        """A queued call that is canceled or times out before worker pickup must NOT be executed."""
        from mcp_manager import MCPServerWorker

        s_data = {"transport": "stdio", "command": "dummy", "enabled": True}
        worker = MCPServerWorker("test_srv", s_data)

        executed_tools = []

        fake_session = AsyncMock()
        async def fake_call_tool(tool_name, tool_args):
            executed_tools.append(tool_name)
            return type("Result", (), {"content": [], "is_error": False})()

        fake_session.call_tool = fake_call_tool
        fake_session.initialize = AsyncMock()
        fake_session.list_tools = AsyncMock(return_value=type("ToolsResult", (), {"tools": []})())

        # Put an already-canceled future into the queue
        loop = asyncio.get_running_loop()
        canceled_fut = loop.create_future()
        canceled_fut.cancel()

        deadline_past = time.monotonic() - 10.0
        await worker.queue.put(("call_tool", ("write_to_db", {"id": 1}), canceled_fut, deadline_past))

        # Stop marker
        await worker.queue.put(None)

        # Run consumer
        await worker._consume_session(fake_session)

        # Tool must NOT have been executed
        self.assertEqual(len(executed_tools), 0)

    async def test_worker_stop_settles_all_pending_waiters(self) -> None:
        """Stopping or failing a worker must reject all pending futures without hanging."""
        from mcp_manager import MCPServerWorker

        s_data = {"transport": "stdio", "command": "dummy", "enabled": True}
        worker = MCPServerWorker("test_srv", s_data)

        loop = asyncio.get_running_loop()
        pending_fut1 = loop.create_future()
        pending_fut2 = loop.create_future()

        await worker.queue.put(("call_tool", ("tool1", {}), pending_fut1, None))
        await worker.queue.put(("call_tool", ("tool2", {}), pending_fut2, None))

        # Stop worker
        await worker.stop()

        # Both futures must now be settled (done) with an exception
        self.assertTrue(pending_fut1.done())
        self.assertTrue(pending_fut2.done())
        with self.assertRaises(Exception):
            pending_fut1.result()
        with self.assertRaises(Exception):
            pending_fut2.result()

    async def test_no_blanket_replay_on_failure(self) -> None:
        """When a tool call raises an exception or times out, mcp_manager must not replay the call."""
        from config_manager import ConfigManager
        from mcp_manager import MCPManager
        from tool_policy import RequestContext
        from tests.fakes import FakeMCPServerWorker

        cfg = ConfigManager("test_replay_config.json")
        cfg.tool_policy["allowed_tools"] = ["srv__commit_order"]
        cfg.mcp_servers = {"srv": {"transport": "stdio", "command": "dummy", "enabled": True}}
        mgr = MCPManager(cfg)
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        call_count = 0
        async def failing_call_handler(tool_name, args):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Database connection dropped mid-transaction")

        worker = FakeMCPServerWorker("srv", call_handler=failing_call_handler)
        mgr._workers["srv"] = worker
        mgr._tool_mappings["srv__commit_order"] = {
            "server_name": "srv",
            "original_name": "commit_order",
            "description": "Commits order",
            "input_schema": {"type": "object"},
        }
        mgr._tools_initialized = True

        res = await mgr.call_tool("srv__commit_order", {"order_id": 999}, context=admin_ctx)

        # Call must execute exactly once (NO replay!)
        self.assertEqual(call_count, 1)
        self.assertIn("error", res.lower())


if __name__ == "__main__":
    unittest.main()
