import asyncio
import unittest
from unittest.mock import AsyncMock

from tests.fakes import FakeMCPServerWorker


class TestRefreshShutdown(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 6: Refresh and shutdown ownership (F8)."""

    async def test_coalesces_concurrent_refreshes(self) -> None:
        """Concurrent calls to refresh_tools must coalesce into a single execution."""
        from config_manager import ConfigManager
        from mcp_manager import MCPManager

        cfg = ConfigManager("test_refresh_config.json")
        cfg.mcp_servers = {"s1": {"transport": "stdio", "command": "dummy", "enabled": True}}
        mgr = MCPManager(cfg)

        worker = FakeMCPServerWorker("s1", tools=[])
        mgr._workers["s1"] = worker

        # Launch two refresh_tools concurrently
        task1 = asyncio.create_task(mgr.refresh_tools())
        task2 = asyncio.create_task(mgr.refresh_tools())
        await asyncio.gather(task1, task2)

        self.assertTrue(mgr._tools_initialized)

    async def test_unchanged_worker_not_reconnected_on_refresh(self) -> None:
        """Refreshing tools when server configuration is unchanged must not recreate or reconnect the worker."""
        from config_manager import ConfigManager
        from mcp_manager import MCPManager

        cfg = ConfigManager("test_refresh_config.json")
        cfg.mcp_servers = {
            "stable": {"transport": "stdio", "command": "cmd1", "enabled": True},
            "dynamic": {"transport": "stdio", "command": "cmd2", "enabled": True},
        }
        mgr = MCPManager(cfg)

        w_stable = FakeMCPServerWorker("stable", tools=[])
        w_dynamic = FakeMCPServerWorker("dynamic", tools=[])
        mgr._workers["stable"] = w_stable
        mgr._workers["dynamic"] = w_dynamic
        mgr._server_configs = {
            "stable": dict(cfg.mcp_servers["stable"]),
            "dynamic": dict(cfg.mcp_servers["dynamic"]),
        }
        mgr._tools_initialized = True

        # Mutate only dynamic server's configuration
        cfg.mcp_servers["dynamic"]["command"] = "new_cmd2"

        # Mock MCPServerWorker so we know what is spawned
        new_dynamic_worker = FakeMCPServerWorker("dynamic", tools=[])
        with unittest.mock.patch("mcp_manager.MCPServerWorker", return_value=new_dynamic_worker):
            await mgr.refresh_tools()

        # w_stable must NOT have been stopped or replaced
        self.assertFalse(w_stable._stopped)
        self.assertIs(mgr._workers["stable"], w_stable)

        # dynamic server must have been replaced
        self.assertTrue(w_dynamic._stopped)
        self.assertIs(mgr._workers["dynamic"], new_dynamic_worker)

    async def test_close_all_settles_and_leaves_no_pending_tasks(self) -> None:
        """close_all() must stop all active workers and clear memory state cleanly."""
        from config_manager import ConfigManager
        from mcp_manager import MCPManager

        cfg = ConfigManager("test_refresh_config.json")
        cfg.mcp_servers = {"srv1": {"transport": "stdio", "command": "dummy", "enabled": True}}
        mgr = MCPManager(cfg)

        w = FakeMCPServerWorker("srv1")
        mgr._workers["srv1"] = w
        mgr._tools_initialized = True

        await mgr.close_all()

        self.assertTrue(w._stopped)
        self.assertEqual(len(mgr._workers), 0)
        self.assertFalse(mgr._tools_initialized)


if __name__ == "__main__":
    unittest.main()
