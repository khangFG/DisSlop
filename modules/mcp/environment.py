"""
Environment isolation utilities for MCP subprocesses.
"""

import os
from typing import Dict, Optional

from error_reporting import SAFE_SYSTEM_ENV_VARS


def build_isolated_env(explicit_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Builds a minimal subprocess execution environment containing only essential OS variables
    and explicitly configured server environment variables.
    Prevents leaking Discord tokens and provider API keys to child processes.
    """
    base_env: Dict[str, str] = {}
    for var in SAFE_SYSTEM_ENV_VARS:
        val = os.environ.get(var)
        if val is not None:
            base_env[var] = val
        if os.name == "nt":
            for k, v in os.environ.items():
                if k.upper() == var.upper():
                    base_env[k] = v

    if explicit_env and isinstance(explicit_env, dict):
        for k, v in explicit_env.items():
            if isinstance(k, str) and isinstance(v, str):
                base_env[k] = v

    return base_env
