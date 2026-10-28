"""
Discord.py Cogs subsystem for modular command and event handling.
"""

from .base import BaseCog
from .providers import ProviderCog
from .mcp import MCPCog
from .chat import ChatCog

__all__ = ["BaseCog", "ProviderCog", "MCPCog", "ChatCog"]
