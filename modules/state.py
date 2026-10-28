"""
Global application state store shared across main, cogs, and tests.
Provides a single canonical source of truth, eliminating __main__ vs main module duality.
"""

from typing import Any, Optional, Set
from config_manager import ConfigManager
from modules.formatting import _parse_admin_ids


class AppState:
    """Canonical singleton holding live configuration, MCP manager, and admin identity."""

    def __init__(self) -> None:
        self.config: ConfigManager = ConfigManager(config_path=None)
        self._mcp_manager: Optional[Any] = None
        self.admin_user_id: str = ""
        self.admin_user_ids: Set[str] = set()

    @property
    def mcp_manager(self) -> Any:
        if self._mcp_manager is None:
            from mcp_manager import MCPManager
            self._mcp_manager = MCPManager(self.config)
        return self._mcp_manager

    @mcp_manager.setter
    def mcp_manager(self, mgr: Any) -> None:
        self._mcp_manager = mgr

    def set_state(
        self,
        cfg: Optional[ConfigManager] = None,
        mgr: Optional[Any] = None,
        admin_id: Optional[Any] = None,
    ) -> None:
        if cfg is not None:
            self.config = cfg
            if self._mcp_manager is not None and getattr(self._mcp_manager, "config", None) is not cfg:
                self._mcp_manager.config = cfg
        if mgr is not None:
            self._mcp_manager = mgr
        if admin_id is not None:
            self.admin_user_id = str(admin_id).strip()
            self.admin_user_ids = _parse_admin_ids(admin_id)


state = AppState()
