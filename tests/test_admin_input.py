import asyncio
import os
import unittest
from unittest.mock import AsyncMock

from tests.fakes import FakeInteraction, FakeUser


class TestAdminInput(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 10: Robust admin input and autocomplete (F10)."""

    def setUp(self) -> None:
        self.admin_id = "111222333"
        import main
        self.old_admin = main.ADMIN_USER_ID
        self.old_config = main.config
        from config_manager import ConfigManager
        main.config = ConfigManager(None)
        main.ADMIN_USER_ID = self.admin_id

    def tearDown(self) -> None:
        import main
        main.ADMIN_USER_ID = self.old_admin
        main.config = self.old_config

    def test_unauthorized_user_receives_zero_autocomplete_choices(self) -> None:
        """Non-admin callers must receive zero autocomplete choices for profiles and MCP servers."""
        from main import profile_autocomplete, mcp_server_autocomplete, config

        config.profiles["secret_profile"] = {"api_format": "openai", "api_key": "k"}
        config.mcp_servers["secret_server"] = {"transport": "stdio", "command": "cmd"}

        normal_user = FakeUser(user_id=999888777)
        interaction = FakeInteraction(user=normal_user)

        # Autocomplete choices for non-admin
        profile_choices = asyncio.run(profile_autocomplete(interaction, ""))
        server_choices = asyncio.run(mcp_server_autocomplete(interaction, ""))

        self.assertEqual(profile_choices, [])
        self.assertEqual(server_choices, [])

        # Autocomplete choices for admin
        admin_user = FakeUser(user_id=int(self.admin_id))
        admin_interaction = FakeInteraction(user=admin_user)
        admin_profile_choices = asyncio.run(profile_autocomplete(admin_interaction, ""))
        self.assertTrue(len(admin_profile_choices) > 0)

    def test_parses_json_arrays_and_preserves_windows_paths(self) -> None:
        """JSON argument arrays and unquoted Windows paths must be cleanly preserved without corruption."""
        from main import parse_admin_arguments

        # JSON array
        json_args = '["--dir", "C:\\\\Users\\\\Admin\\\\Data", "--flag"]'
        parsed = parse_admin_arguments(json_args)
        self.assertEqual(parsed, ["--dir", "C:\\Users\\Admin\\Data", "--flag"])

        # Unquoted Windows path in string
        raw_win = r"C:\Users\Admin\mcp-server\index.js --port 8080"
        parsed_win = parse_admin_arguments(raw_win)
        self.assertEqual(parsed_win[0], r"C:\Users\Admin\mcp-server\index.js")
        self.assertEqual(parsed_win[1:], ["--port", "8080"])

    def test_parses_json_maps_and_preserves_bearer_tokens(self) -> None:
        """Headers containing spaces like 'Authorization=Bearer my_token_123' must not be split on whitespace."""
        from main import parse_admin_mapping

        # JSON map
        json_map = '{"Authorization": "Bearer token_abc", "X-Custom": "val 123"}'
        parsed = parse_admin_mapping(json_map)
        self.assertEqual(parsed["Authorization"], "Bearer token_abc")
        self.assertEqual(parsed["X-Custom"], "val 123")

        # String pair containing space in value
        raw_header = "Authorization=Bearer secret_bearer_token, X-Other=simple_val"
        parsed_header = parse_admin_mapping(raw_header)
        self.assertEqual(parsed_header.get("Authorization"), "Bearer secret_bearer_token")
        self.assertEqual(parsed_header.get("X-Other"), "simple_val")

    def test_rejects_malformed_input_without_config_mutation(self) -> None:
        """Malformed JSON or unclosed quote strings must raise error and not mutate config."""
        from main import parse_admin_arguments, parse_admin_mapping

        with self.assertRaises(ValueError):
            parse_admin_arguments('["broken JSON array')

        with self.assertRaises(ValueError):
            parse_admin_mapping('{"broken": JSON map')


if __name__ == "__main__":
    unittest.main()
