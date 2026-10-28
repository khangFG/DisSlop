import asyncio
import copy
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import time
from typing import Any, Callable, Dict, List, Optional

import jsonschema
from referencing import Registry
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
import httpx2
from config_manager import ConfigManager
from tool_policy import RequestContext, is_tool_allowed, is_user_authorized_for_tools, filter_tools_for_request
from error_reporting import (
    SAFE_SYSTEM_ENV_VARS,
    validate_remote_url,
    redact_sensitive_text,
    format_public_error,
    log_sanitized_error,
)
from request_limits import MAX_WORKER_QUEUE_SIZE

DEFAULT_TOOL_TIMEOUT = 20.0


from modules.mcp import (
    build_isolated_env,
    sanitize_tool_name,
    compress_schema,
    ToolOutcome,
    format_tool_result,
    format_tool_outcome,
    compact_tool_output,
    MAX_TOOL_RESULT_CHARS,
    HEAD_CHARS,
    MID_CHARS,
    TAIL_CHARS,
    MARKER_MID,
    MARKER_TAIL,
    OVERHEAD_CHARS,
    MAX_DESC_CHARS,
    MAX_TOOL_ARGUMENT_CHARS,
)


class MCPServerWorker:
    """
    Dedicated worker task managing an MCP server connection.
    Enters and exits anyio cancel scopes entirely within its own task to prevent
    'Attempted to exit cancel scope in a different task' errors.
    """

    def __init__(self, name: str, s_data: Dict[str, Any]):
        self.name = name
        self.s_data = s_data
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_WORKER_QUEUE_SIZE)
        self.ready_event: asyncio.Event = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.tools: List[Any] = []
        self.error: Optional[str] = None
        self._stopped: bool = False

    async def start(self, timeout: float = 25.0) -> bool:
        self.task = asyncio.create_task(self._run_loop(), name=f"mcp_worker_{self.name}")
        try:
            await asyncio.wait_for(self.ready_event.wait(), timeout=timeout)
            if self.error is not None:
                await self.stop()
            return self.error is None
        except asyncio.TimeoutError:
            self.error = f"Connection to '{self.name}' timed out after {timeout}s"
            await self.stop()
            return False
        except asyncio.CancelledError:
            await self.stop()
            raise

    def _drain_queue(self, exc: Optional[Exception] = None) -> None:
        """Settles any remaining items in the queue with an exception."""
        err = exc or RuntimeError(f"MCPServerWorker '{self.name}' stopped before request could execute")
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if item is not None and len(item) >= 3:
                    fut = item[2]
                    if isinstance(fut, asyncio.Future) and not fut.done() and not fut.cancelled():
                        fut.set_exception(err)
            except asyncio.QueueEmpty:
                break

    async def _consume_session(self, session: ClientSession) -> None:
        """Shared session consumer loop with expired request skip and deadline bounds."""
        await session.initialize()
        tools_result = await session.list_tools()
        self.tools = tools_result.tools
        self.ready_event.set()

        while not self._stopped:
            item = await self.queue.get()
            if item is None:
                break

            action = item[0]
            req_args = item[1]
            fut = item[2]
            deadline = item[3] if len(item) > 3 else None
            authorize = item[4] if len(item) > 4 else None

            # Skip cancelled, completed, or expired work without executing it
            if fut.done() or fut.cancelled() or (deadline is not None and time.monotonic() > deadline):
                if not fut.done() and not fut.cancelled():
                    fut.set_exception(asyncio.TimeoutError("Request deadline expired before worker execution began"))
                continue

            remaining = max(0, deadline - time.monotonic()) if deadline is not None else DEFAULT_TOOL_TIMEOUT
            scope = asyncio.timeout(remaining)
            scope_state = {"active": True}
            def cancel_abandoned(done, active_scope=scope, state=scope_state):
                if state["active"] and done.cancelled() and not active_scope.expired():
                    active_scope.reschedule(asyncio.get_running_loop().time())
            fut.add_done_callback(cancel_abandoned)
            try:
                # Timeout and session lifecycle stay in this worker task.
                async with scope:
                    if authorize is not None and not authorize():
                        raise PermissionError("Tool authorization changed before execution.")
                    if action == "call_tool":
                        tool_name, tool_args = req_args
                        res = await session.call_tool(tool_name, tool_args or {})
                    elif action == "list_tools":
                        res = await session.list_tools()
                        self.tools = res.tools
                    else:
                        raise ValueError("Unsupported worker action")
                if not fut.done():
                    fut.set_result(res)
            except asyncio.CancelledError:
                if not fut.done():
                    fut.set_exception(RuntimeError("Worker stopped; tool outcome is unknown."))
                raise
            except (Exception, BaseExceptionGroup) as e:
                if not fut.done():
                    fut.set_exception(e)
            finally:
                scope_state["active"] = False
                fut.remove_done_callback(cancel_abandoned)

    async def _run_loop(self) -> None:
        transport = self.s_data.get("transport", "stdio")
        try:
            if transport == "stdio":
                raw_cmd = self.s_data.get("command") or ""
                cmd = shutil.which(raw_cmd) or raw_cmd
                args = list(self.s_data.get("args") or [])
                if os.path.basename(raw_cmd).lower() in ("uvx", "uvx.exe", "uv", "uv.exe"):
                    args = [a for a in args if a != "-y"]
                env_vars = build_isolated_env(self.s_data.get("env"))
                params = StdioServerParameters(command=cmd, args=args, env=env_vars)

                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await self._consume_session(session)

            elif transport in ("sse", "http", "streamable_http"):
                url = self.s_data.get("url") or ""
                if not validate_remote_url(url):
                    self.error = f"Insecure or invalid MCP server URL '{url}'. Remote servers must use HTTPS."
                    self.ready_event.set()
                    return
                headers = self.s_data.get("headers") or {}

                if transport in ("http", "streamable_http"):
                    async with httpx2.AsyncClient(headers=headers or None) as http_client:
                        async with streamable_http_client(url, http_client=http_client) as (read, write):
                            async with ClientSession(read, write) as session:
                                await self._consume_session(session)
                else:
                    # transport is "sse": try sse_client first; fallback to streamable_http_client on 405 Method Not Allowed
                    try:
                        async with sse_client(url, headers=headers or None) as (read, write):
                            async with ClientSession(read, write) as session:
                                await self._consume_session(session)
                    except Exception as sse_err:
                        status_code = getattr(sse_err, "status_code", None)
                        if status_code is None and hasattr(sse_err, "response"):
                            status_code = getattr(sse_err.response, "status_code", None)
                        if status_code == 405 or "405" in str(sse_err):
                            async with httpx2.AsyncClient(headers=headers or None) as http_client:
                                async with streamable_http_client(url, http_client=http_client) as (read, write):
                                    async with ClientSession(read, write) as session:
                                        await self._consume_session(session)
                        else:
                            raise sse_err
            else:
                self.error = f"Unsupported transport '{transport}'"
                self.ready_event.set()

        except Exception as e:
            self.error = str(e)
            self.ready_event.set()
            _, cid = format_public_error("")
            log_sanitized_error(cid, "MCP worker failed", e)
            self._drain_queue(e)
        finally:
            self._drain_queue()
            if not self.ready_event.is_set():
                self.ready_event.set()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = DEFAULT_TOOL_TIMEOUT,
                        authorize: Optional[Callable[[], bool]] = None) -> Any:
        if self._stopped:
            raise RuntimeError(f"Server '{self.name}' worker is stopped")
        if self.error:
            raise RuntimeError(f"Server '{self.name}' is in error state: {self.error}")
        if self.task is None or self.task.done():
            raise RuntimeError(f"Server '{self.name}' worker is not running")

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        deadline = time.monotonic() + timeout
        try:
            self.queue.put_nowait(("call_tool", (tool_name, arguments), fut, deadline, authorize))
        except asyncio.QueueFull:
            raise RuntimeError("MCP worker is busy at capacity.") from None
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            if not fut.done() and not fut.cancelled():
                fut.cancel()
            raise asyncio.TimeoutError(f"Tool '{tool_name}' timed out after {timeout}s on worker '{self.name}'")

    async def stop(self) -> None:
        self._stopped = True
        self._drain_queue()
        if self.task and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        self._drain_queue()


class MCPManager:
    """
    High-performance MCP Manager with task-isolated workers,
    memory-cached tool schemas, schema compression, and smart result compaction.
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self._lock = asyncio.Lock()

        # Server workers: server_name -> MCPServerWorker
        self._workers: Dict[str, MCPServerWorker] = {}
        self._server_configs: Dict[str, Dict[str, Any]] = {}

        # Cached tool mappings & formatted schemas
        self._tool_mappings: Dict[str, Dict[str, Any]] = {}
        self._cached_openai_tools: List[Dict[str, Any]] = []
        self._cached_anthropic_tools: List[Dict[str, Any]] = []
        self._tools_initialized: bool = False
        self._refresh_task: Optional[asyncio.Task] = None
        self._closed = False

    async def _stop_worker_unlocked(self, s_name: str) -> None:
        """Stops and removes an existing worker."""
        worker = self._workers.pop(s_name, None)
        self._server_configs.pop(s_name, None)
        if worker:
            try:
                await worker.stop()
            except Exception as e:
                log_sanitized_error("shutdown", "MCP worker cleanup failed", e)

    async def refresh_tools(self, force_reconnect: bool = False) -> None:
        """
        Refreshes server connections and caches optimized tool schemas in memory.
        Coalesces concurrent requests and reconnects only changed/removed workers.
        """
        if self._closed:
            return
        if self._refresh_task and not self._refresh_task.done():
            await asyncio.shield(self._refresh_task)
            if not force_reconnect:
                return

        task = asyncio.create_task(self._refresh_until_current(force_reconnect))
        self._refresh_task = task
        def finished(done):
            if self._refresh_task is task:
                self._refresh_task = None
            if not done.cancelled() and done.exception() is not None:
                log_sanitized_error("refresh", "MCP refresh failed", done.exception())
        task.add_done_callback(finished)
        await asyncio.shield(task)

    async def _refresh_until_current(self, force_reconnect: bool) -> None:
        while not self._closed:
            before = self.config.get_mcp_servers()
            await self._do_refresh_tools(force_reconnect)
            if before == self.config.get_mcp_servers():
                return
            force_reconnect = False

    async def _do_refresh_tools(self, force_reconnect: bool = False) -> None:
        async with self._lock:
            if self._closed:
                return
            servers = self.config.get_mcp_servers()

            # Disconnect removed, disabled, or changed servers
            for s_name in list(self._workers.keys()):
                if s_name not in servers or not servers[s_name].get("enabled", True):
                    await self._stop_worker_unlocked(s_name)
                elif force_reconnect:
                    await self._stop_worker_unlocked(s_name)
                elif s_name in self._server_configs and servers[s_name] != self._server_configs.get(s_name):
                    await self._stop_worker_unlocked(s_name)

            new_tool_mappings: Dict[str, Dict[str, Any]] = {}
            new_openai_tools: List[Dict[str, Any]] = []
            new_anthropic_tools: List[Dict[str, Any]] = []

            for s_name, s_data in servers.items():
                if not s_data.get("enabled", True):
                    continue

                worker = self._workers.get(s_name)
                if worker is None or (worker.task and worker.task.done()) or worker.error:
                    await self._stop_worker_unlocked(s_name)
                    worker = MCPServerWorker(s_name, s_data)
                    started = await worker.start()
                    if started:
                        self._workers[s_name] = worker
                        self._server_configs[s_name] = copy.deepcopy(s_data)
                    else:
                        log_sanitized_error("startup", "MCP worker failed to start")
                        continue
                else:
                    self._server_configs[s_name] = copy.deepcopy(s_data)

                for t in worker.tools:
                    pref_name = sanitize_tool_name(s_name, t.name)

                    if pref_name in new_tool_mappings:
                        existing = new_tool_mappings[pref_name]
                        if (existing["server_name"], existing["original_name"]) != (s_name, t.name):
                            print(f"[MCPManager] Collision detected: '{s_name}:{t.name}' collides with '{existing['server_name']}:{existing['original_name']}' as '{pref_name}'. Registration rejected.")
                            continue

                    # Clean and truncate tool descriptions (>300 chars)
                    raw_desc = " ".join((t.description or "").split()).strip()
                    cleaned_desc = raw_desc[:MAX_DESC_CHARS - 3] + "..." if len(raw_desc) > MAX_DESC_CHARS else raw_desc

                    # Preserve original validation schema and create compressed schema for LLM
                    raw_schema = t.input_schema if hasattr(t, "input_schema") else getattr(t, "inputSchema", {})
                    original_schema = raw_schema if isinstance(raw_schema, dict) else {"type": "object", "properties": {}}
                    cleaned_schema = compress_schema(raw_schema) or {"type": "object", "properties": {}}

                    new_tool_mappings[pref_name] = {
                        "server_name": s_name,
                        "original_name": t.name,
                        "description": cleaned_desc,
                        "input_schema": original_schema,
                    }

                    new_openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": pref_name,
                            "description": cleaned_desc,
                            "parameters": cleaned_schema,
                        },
                    })

                    new_anthropic_tools.append({
                        "name": pref_name,
                        "description": cleaned_desc,
                        "input_schema": cleaned_schema,
                    })

            self._tool_mappings = new_tool_mappings
            self._cached_openai_tools = new_openai_tools
            self._cached_anthropic_tools = new_anthropic_tools
            self._tools_initialized = True

    async def get_openai_tools(self, context: Optional[RequestContext] = None) -> List[Dict[str, Any]]:
        """Returns cached OpenAI function definitions authorized for caller without spawning subprocesses."""
        if self._closed or not is_user_authorized_for_tools(context, self.config):
            return []
        if not self._tools_initialized:
            await self.refresh_tools()
        return filter_tools_for_request(context, copy.deepcopy(self._cached_openai_tools), self.config)

    async def get_anthropic_tools(self, context: Optional[RequestContext] = None) -> List[Dict[str, Any]]:
        """Returns cached Anthropic tool definitions authorized for caller without spawning subprocesses."""
        if self._closed or not is_user_authorized_for_tools(context, self.config):
            return []
        if not self._tools_initialized:
            await self.refresh_tools()
        copied = copy.deepcopy(self._cached_anthropic_tools)
        return filter_tools_for_request(context, copied, self.config)

    async def call_tool(
        self,
        prefixed_tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[RequestContext] = None,
    ) -> str:
        """
        Executes a tool on the persistent server worker.
        Enforces caller authorization, smart output compaction and head+middle+tail sampling.
        """
        # Strict caller authorization boundary before any discovery or worker interaction
        if self._closed or not is_tool_allowed(context, prefixed_tool_name, self.config):
            return f"Error: Permission denied. Caller is not authorized to execute tool '{prefixed_tool_name}'."
        if not self._tools_initialized:
            await self.refresh_tools()

        info = self._tool_mappings.get(prefixed_tool_name)
        if not info:
            return f"Error: Tool '{prefixed_tool_name}' is not registered."

        server_name = info["server_name"]
        original_tool_name = info["original_name"]

        # Validate arguments type
        if not isinstance(arguments, dict):
            return f"Error: Invalid arguments for tool '{prefixed_tool_name}'. Arguments must be a JSON object (dict)."
        try:
            if len(json.dumps(arguments, allow_nan=False)) > MAX_TOOL_ARGUMENT_CHARS:
                return "Error: Tool arguments exceed the size limit."
        except (ValueError, TypeError, RecursionError):
            return "Error: Tool arguments must be finite JSON data."

        # Validate arguments against original schema
        schema = info.get("input_schema")
        if schema is not None:
            try:
                validator = jsonschema.validators.validator_for(schema)
                validator.check_schema(schema)
                validator(schema, registry=Registry()).validate(arguments)
            except jsonschema.ValidationError:
                return "Error: Arguments do not match the tool schema."
            except Exception:
                return "Error: Tool schema could not be validated locally."

        servers = self.config.get_mcp_servers()
        s_data = servers.get(server_name)
        if not s_data:
            return f"Error: MCP server '{server_name}' is not configured."
        if not s_data.get("enabled", True):
            return f"Error: MCP server '{server_name}' is currently disabled."

        async with self._lock:
            # Recheck after acquiring the lock: another caller may have already
            # recovered the worker, or shutdown/reconfiguration may have won.
            if self._closed or self.config.get_mcp_servers().get(server_name) != s_data:
                return "Error: MCP configuration changed; request was not executed."
            worker = self._workers.get(server_name)
            stale = server_name in self._server_configs and self._server_configs[server_name] != s_data
            if stale:
                return "Error: MCP refresh is pending; request was not executed."
            if worker is None or (worker.task and worker.task.done()) or worker.error:
                await self._stop_worker_unlocked(server_name)
                worker = MCPServerWorker(server_name, s_data)
                started = await worker.start()
                if started:
                    self._workers[server_name] = worker
                    self._server_configs[server_name] = copy.deepcopy(s_data)
                else:
                    return "Error: Could not connect to the MCP server."

        if (not is_tool_allowed(context, prefixed_tool_name, self.config)
                or self.config.get_mcp_servers().get(server_name) != s_data):
            return "Error: Tool permission or server configuration changed. Please retry."

        # Execute single tool call; no blanket retry to prevent duplicate side-effects
        try:
            def still_authorized():
                return (not self._closed
                        and is_tool_allowed(context, prefixed_tool_name, self.config)
                        and self.config.get_mcp_servers().get(server_name) == s_data
                        and self._tool_mappings.get(prefixed_tool_name) == info)
            result = await worker.call_tool(original_tool_name, arguments, timeout=DEFAULT_TOOL_TIMEOUT, authorize=still_authorized)

            text_parts = []
            if hasattr(result, "content") and result.content:
                for c in result.content:
                    if hasattr(c, "text") and c.text:
                        text_parts.append(c.text)
                    elif hasattr(c, "data"):
                        text_parts.append("(Binary tool content omitted)")
                    else:
                        text_parts.append("(Non-text tool content omitted)")

            out = "\n".join(text_parts)
            is_err = bool(getattr(result, "is_error", False))
            struct_content = getattr(result, "structured_content", None)

            if is_err:
                return "Error: MCP tool reported a failure."
            if not out and struct_content is None:
                out = "(Tool executed successfully with no output)"
            outcome = ToolOutcome(text=out, is_error=is_err, structured_content=struct_content)
            return format_tool_outcome(original_tool_name, outcome)

        except asyncio.TimeoutError:
            return f"Error: MCP tool execution timed out after {int(DEFAULT_TOOL_TIMEOUT)}s. Upstream server took too long to respond."
        except Exception as e:
            public, cid = format_public_error("", "Tool execution failed; outcome may be unknown. Do not automatically retry.")
            log_sanitized_error(cid, "MCP tool execution failed", e)
            return public

    async def get_overview_report(self) -> str:
        """Returns a Markdown overview of all servers and their exposed tools using cached mappings."""
        if not self._tools_initialized:
            await self.refresh_tools()

        servers = self.config.get_mcp_servers()
        if not servers:
            return "ℹ️ No MCP servers are configured yet. Use `/add_mcp_stdio` or `/add_mcp_sse` to connect one."

        lines = ["### 🔌 Active MCP Servers & Tools Overview\n"]
        for s_name, s_data in sorted(servers.items()):
            trans = s_data.get("transport", "stdio")
            enabled = s_data.get("enabled", True)
            status_badge = "🟢 Enabled" if enabled else "🔴 Disabled"

            if trans == "stdio":
                target = f"`{s_data.get('command')} {' '.join(s_data.get('args') or [])}`"
            else:
                target = f"`{s_data.get('url')}`"

            lines.append(f"#### **{s_name}** ({trans.upper()}) - {status_badge}")
            lines.append(f"- **Target:** {target}")

            server_tools = [
                (k, v) for k, v in self._tool_mappings.items() if v["server_name"] == s_name
            ]
            if not server_tools:
                lines.append("- *No tools exposed (or server currently unreachable)*\n")
            else:
                lines.append(f"- **Exposed Tools ({len(server_tools)}):**")
                for pref_name, info in server_tools:
                    raw_desc = (info["description"] or "No description").strip()
                    first_line = raw_desc.split("\n")[0].strip()
                    short_desc = first_line[:100] + "..." if len(first_line) > 100 else first_line
                    lines.append(f"  - `{pref_name}`: {short_desc}")
                lines.append("")

        return "\n".join(lines)

    async def close_all(self) -> None:
        """Closes all background server workers cleanly and resets cached state."""
        self._closed = True
        refresh_task = self._refresh_task
        if refresh_task and not refresh_task.done():
            refresh_task.cancel()
            try:
                await refresh_task
            except (asyncio.CancelledError, Exception):
                pass
            self._refresh_task = None

        async with self._lock:
            for s_name in list(self._workers.keys()):
                await self._stop_worker_unlocked(s_name)
            self._workers.clear()
            self._server_configs.clear()
            self._tool_mappings.clear()
            self._cached_openai_tools.clear()
            self._cached_anthropic_tools.clear()
            self._tools_initialized = False
