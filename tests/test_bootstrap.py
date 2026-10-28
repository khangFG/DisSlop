import os
import sys
import unittest
import builtins
import runpy
from pathlib import Path
from unittest.mock import patch

from tests.fakes import FakeInteraction, FakeMessage, FakeUser


class TestBootstrap(unittest.TestCase):
    """Verifies safe offline bootstrap: no import-time writes, no spawned children, command aliases intact."""

    def test_imports_do_not_spawn_subprocesses_or_network(self) -> None:
        """Verifies importing core modules does not initiate network or spawn processes."""
        with patch("subprocess.Popen") as mock_popen, \
             patch("asyncio.create_subprocess_exec") as mock_subproc, \
             patch("socket.socket") as mock_sock:

            import config_manager
            import mcp_manager
            import llm_service
            import llm_engine
            import main
            import bot

            mock_popen.assert_not_called()
            mock_subproc.assert_not_called()
            mock_sock.assert_not_called()

    def test_config_manager_init_does_not_write_missing_file(self) -> None:
        """Verifies initializing ConfigManager on a nonexistent path does not create the file on disk."""
        nonexistent = "test_nonexistent_runtime_config_probe.json"
        if os.path.exists(nonexistent):
            os.remove(nonexistent)

        try:
            from config_manager import ConfigManager
            cfg = ConfigManager(config_path=nonexistent)
            self.assertFalse(os.path.exists(nonexistent), "ConfigManager created file on read initialization!")
            self.assertEqual(cfg.active_profile, "openai")
        finally:
            if os.path.exists(nonexistent):
                os.remove(nonexistent)

    def test_command_aliases_registered_on_bot(self) -> None:
        """Verifies all primary and alias slash commands and prefix commands are registered on the bot."""
        import main

        tree_commands = {cmd.name: cmd for cmd in main.bot.tree.get_commands()}

        expected_slash_commands = [
            "configure_provider",
            "provider_set",
            "switch_provider",
            "provider_switch",
            "delete_provider",
            "provider_delete",
            "set_prompt",
            "llm_status",
            "add_mcp_stdio",
            "add_mcp_sse",
            "add_mcp_http",
            "remove_mcp",
            "mcp_tools",
        ]

        for cmd_name in expected_slash_commands:
            self.assertIn(cmd_name, tree_commands, f"Missing registered slash command: /{cmd_name}")

        prefix_commands = [c.name for c in main.bot.commands]
        self.assertIn("sync", prefix_commands, "Missing prefix command '!sync'")

    def test_bot_entrypoint_re_exports(self) -> None:
        """Verifies bot.py properly exposes bot and main from main.py."""
        import bot
        import main

        self.assertIs(bot.bot, main.bot)
        self.assertIs(bot.main, main.main)

    def test_injectable_seam(self) -> None:
        """Verifies main.set_app_state allows injecting synthetic state without disk writes."""
        import main
        from config_manager import ConfigManager
        from mcp_manager import MCPManager

        fake_cfg = ConfigManager(config_path="synthetic_test_config.json")
        fake_mgr = MCPManager(fake_cfg)

        orig_cfg = main.config
        orig_mgr = main.mcp_manager
        orig_admin = main.ADMIN_USER_ID

        try:
            main.set_app_state(cfg=fake_cfg, mgr=fake_mgr, admin_id="synthetic_admin_123")
            self.assertIs(main.config, fake_cfg)
            self.assertIs(main.mcp_manager, fake_mgr)
            self.assertEqual(main.ADMIN_USER_ID, "synthetic_admin_123")
        finally:
            main.set_app_state(cfg=orig_cfg, mgr=orig_mgr, admin_id=orig_admin)


    def test_fresh_entrypoint_execution_does_not_read_live_config(self) -> None:
        real_open = builtins.open
        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and Path(path).name in ("runtime_config.json", ".env"):
                raise AssertionError("Import attempted to read live credentials/config")
            return real_open(path, *args, **kwargs)
        with patch("builtins.open", side_effect=guarded_open):
            state = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="offline_import")
        self.assertIsNone(state["config"].config_path)



if __name__ == "__main__":
    unittest.main()
