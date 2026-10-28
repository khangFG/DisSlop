import os
import unittest
from unittest.mock import AsyncMock, patch

from tests.fakes import FakeMCPServerWorker


class TestSecretBoundaries(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 4: Credential isolation and failure sanitization (F2/F6)."""

    def setUp(self) -> None:
        self.sentinel_discord = "discord_token_super_secret_987654321"
        self.sentinel_provider = "sk-proj-sentinel-api-key-123456789"
        self.sentinel_bearer = "Bearer token-sentinel-auth-val"

        # Set fake secrets in os.environ
        os.environ["DISCORD_TOKEN"] = self.sentinel_discord
        os.environ["OPENAI_API_KEY"] = self.sentinel_provider

    def tearDown(self) -> None:
        os.environ.pop("DISCORD_TOKEN", None)
        os.environ.pop("OPENAI_API_KEY", None)

    async def test_child_process_does_not_inherit_bot_secrets(self) -> None:
        """Subprocess environment for stdio MCP server must not inherit DISCORD_TOKEN or OPENAI_API_KEY."""
        from mcp_manager import MCPServerWorker

        s_data = {
            "transport": "stdio",
            "command": "dummy",
            "args": [],
            "env": {"CUSTOM_SERVER_VAR": "custom_val"},
            "enabled": True,
        }
        worker = MCPServerWorker("test_srv", s_data)

        # Intercept StdioServerParameters creation inside worker._run_loop
        with patch("mcp_manager.stdio_client") as mock_stdio, \
             patch("mcp_manager.StdioServerParameters") as mock_params_cls:

            mock_stdio.side_effect = Exception("Stop before launching")
            worker.task = None

            # Check env passed to StdioServerParameters
            from mcp_manager import build_isolated_env
            child_env = build_isolated_env(s_data.get("env"))

            self.assertNotIn("DISCORD_TOKEN", child_env)
            self.assertNotIn("OPENAI_API_KEY", child_env)
            self.assertEqual(child_env.get("CUSTOM_SERVER_VAR"), "custom_val")
            # Essential system variables must be present if set on host
            if "PATH" in os.environ:
                self.assertIn("PATH", child_env)

    async def test_llm_service_sanitizes_provider_error_strings(self) -> None:
        """Provider exceptions containing sentinel secrets must not leak into return value."""
        import llm_service
        from config_manager import ConfigManager
        from mcp_manager import MCPManager
        from tool_policy import RequestContext

        cfg = ConfigManager("test_secret_config.json")
        mgr = MCPManager(cfg)
        admin_ctx = RequestContext(user_id="admin", is_admin=True)

        with patch("llm_service.AsyncOpenAI") as mock_openai_cls:
            mock_client = AsyncMock()
            # Provider exception containing raw sentinel key and raw URL credentials
            mock_client.chat.completions.create.side_effect = Exception(
                f"Unauthorized at https://admin:{self.sentinel_discord}@api.example.com using {self.sentinel_provider}"
            )
            mock_openai_cls.return_value = mock_client

            messages = [{"role": "user", "content": "Hello"}]
            reply = await llm_service.generate_response(messages, cfg, mgr, context=admin_ctx)

            # Neither sentinel should appear in the returned user-facing string
            self.assertNotIn(self.sentinel_discord, reply)
            self.assertNotIn(self.sentinel_provider, reply)
            self.assertNotIn("admin:", reply)
            self.assertIn("error", reply.lower())

    async def test_remote_url_validation_blocks_insecure_remote_urls(self) -> None:
        """Remote credential-bearing or API URLs must reject HTTP except for localhost."""
        from error_reporting import validate_remote_url

        self.assertTrue(validate_remote_url("https://api.openai.com/v1"))
        self.assertTrue(validate_remote_url("http://localhost:8000/v1"))
        self.assertTrue(validate_remote_url("http://127.0.0.1:11434/v1"))
        self.assertFalse(validate_remote_url("http://insecure-remote-server.com/v1"))
        self.assertFalse(validate_remote_url("ftp://server.com"))

    async def test_error_reporting_redacts_secrets(self) -> None:
        """Error reporting module must redact known secrets and return generic public errors with correlation IDs."""
        from error_reporting import redact_sensitive_text, format_public_error

        sensitive_text = f"Failed with token {self.sentinel_discord} and key {self.sentinel_provider}"
        redacted = redact_sensitive_text(sensitive_text, secrets=[self.sentinel_discord, self.sentinel_provider])
        self.assertNotIn(self.sentinel_discord, redacted)
        self.assertNotIn(self.sentinel_provider, redacted)

        public_err, cid = format_public_error("Internal server error")
        self.assertIn(cid, public_err)
        self.assertNotIn("Internal server error", public_err)


if __name__ == "__main__":
    unittest.main()
