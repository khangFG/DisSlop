import copy
import json
import os
import unittest
from unittest.mock import patch


class TestConfigTransactions(unittest.IsolatedAsyncioTestCase):
    """Regression tests for Task 7: Transactional configuration and deep snapshots (F5)."""

    def setUp(self) -> None:
        self.test_cfg_path = "test_transaction_config.json"
        if os.path.exists(self.test_cfg_path):
            os.remove(self.test_cfg_path)

    def tearDown(self) -> None:
        if os.path.exists(self.test_cfg_path):
            os.remove(self.test_cfg_path)
        tmp = f"{self.test_cfg_path}.tmp"
        if os.path.exists(tmp):
            os.remove(tmp)

    def test_corrupt_config_remains_byte_for_byte_intact(self) -> None:
        """When a config file contains corrupt or malformed JSON, loading it must NOT overwrite the file."""
        from config_manager import ConfigManager

        corrupt_bytes = b"{\n  \"active_profile\": \"openai\",\n  \"profiles\": { TRUNCATED CORRUPT"
        with open(self.test_cfg_path, "wb") as f:
            f.write(corrupt_bytes)

        # Initialize ConfigManager on the corrupt file
        with self.assertRaises(ValueError):
            ConfigManager(self.test_cfg_path)

        # File on disk must remain byte-for-byte identical to the original
        with open(self.test_cfg_path, "rb") as f:
            disk_bytes = f.read()

        self.assertEqual(disk_bytes, corrupt_bytes)

    def test_mutation_persists_before_updating_memory(self) -> None:
        """If atomic file write/replace fails during upsert, in-memory state must remain unchanged."""
        from config_manager import ConfigManager

        cfg = ConfigManager(self.test_cfg_path)
        initial_profiles = copy.deepcopy(cfg.profiles)

        with patch("os.replace", side_effect=OSError("Disk write failed / simulated I/O failure")):
            with self.assertRaises(OSError):
                cfg.upsert_profile("new_profile", api_format="openai", api_key="sk-test", model="gpt-4o")

        # In-memory profiles must NOT contain 'new_profile'
        self.assertNotIn("new_profile", cfg.profiles)
        self.assertEqual(cfg.profiles, initial_profiles)

    def test_deep_snapshot_prevents_nested_mutation(self) -> None:
        """Mutating returned objects from get_mcp_servers or get_active_profile must not alter manager state."""
        from config_manager import ConfigManager

        cfg = ConfigManager(self.test_cfg_path)
        cfg.upsert_mcp_server(
            name="fs",
            transport="stdio",
            command="npx",
            args=["-y", "mcp-server"],
            env={"AUTH": "token123"},
            headers={"X-Test": "val"},
        )

        servers = cfg.get_mcp_servers()
        # Caller attempts to mutate nested dicts
        servers["fs"]["args"].append("--injected-arg")
        servers["fs"]["env"]["AUTH"] = "MUTATED"
        servers["fs"]["headers"]["X-Test"] = "MUTATED"

        # Manager's internal state must remain pristine
        fresh_servers = cfg.get_mcp_servers()
        self.assertEqual(fresh_servers["fs"]["args"], ["-y", "mcp-server"])
        self.assertEqual(fresh_servers["fs"]["env"]["AUTH"], "token123")
        self.assertEqual(fresh_servers["fs"]["headers"]["X-Test"], "val")

    def test_rejects_wrong_nested_types(self) -> None:
        """Passing non-dict or invalid nested types must be rejected before any disk or memory change."""
        from config_manager import ConfigManager

        cfg = ConfigManager(self.test_cfg_path)

        with self.assertRaises(ValueError):
            cfg.upsert_mcp_server(
                name="invalid",
                transport="stdio",
                command="npx",
                args="not a list",  # type: ignore
            )

        with self.assertRaises(ValueError):
            cfg.upsert_mcp_server(
                name="invalid",
                transport="stdio",
                command="npx",
                env="not a dict",  # type: ignore
            )


if __name__ == "__main__":
    unittest.main()
