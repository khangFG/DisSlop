"""Regressions from the second review; no live services or real configuration."""
import asyncio
import contextlib
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from config_manager import ConfigManager
from mcp_manager import MCPManager, MCPServerWorker
from tool_policy import RequestContext, is_tool_allowed
from tests.fakes import FakeMCPServerWorker


def config():
    cfg = ConfigManager("unused_followup_config.json")
    cfg.profiles["openai"]["api_key"] = "synthetic-key"
    cfg.mcp_servers = {"srv": {"transport": "stdio", "command": "fake", "enabled": True}}
    cfg.tool_policy = {"admin_only": True, "allowed_tools": ["srv__read"]}
    return cfg


def manager(schema=None, result=None):
    cfg = config()
    mgr = MCPManager(cfg)
    worker = FakeMCPServerWorker("srv", call_handler=AsyncMock(return_value=result))
    mgr._workers["srv"] = worker
    mgr._tool_mappings["srv__read"] = {
        "server_name": "srv", "original_name": "read",
        "input_schema": schema if schema is not None else {"type": "object"},
    }
    mgr._tools_initialized = True
    return mgr, worker


ADMIN = RequestContext("admin", is_admin=True)


class ConfigAndPolicyTests(unittest.TestCase):
    def test_history_excludes_bot_replies_to_other_users(self):
        from conversation import build_conversation_context
        from tests.fakes import FakeMessage, FakeUser
        alice, bob, bot = FakeUser(1), FakeUser(2), FakeUser(99, bot=True)
        question = FakeMessage("Bob's question", bob, message_id=10)
        answer = FakeMessage("Bob's private tool result", bot, message_id=11)
        answer.reference = SimpleNamespace(message_id=10, resolved=question)
        trigger = FakeMessage("Hi", alice, message_id=12)
        context = build_conversation_context([question, answer], trigger, 99)
        self.assertEqual(context, [{"role": "user", "content": "Hi"}])

    def test_delimiter_in_server_name_does_not_alias_another_tool(self):
        from mcp_manager import sanitize_tool_name
        self.assertNotEqual(sanitize_tool_name("a__b", "c"), sanitize_tool_name("a", "b__c"))

    def test_error_result_cap_includes_error_prefix(self):
        from mcp_manager import format_tool_outcome, ToolOutcome, MAX_TOOL_RESULT_CHARS
        self.assertLessEqual(len(format_tool_outcome("read", ToolOutcome("x" * (MAX_TOOL_RESULT_CHARS + 5000), True))), MAX_TOOL_RESULT_CHARS)

    def test_search_code_preserves_source(self):
        from mcp_manager import format_tool_result
        text = '<script>important_code()</script>\n    indented'
        self.assertEqual(format_tool_result("search_code", text), text)

    def test_empty_allowlist_denies_even_admin(self):
        cfg = config()
        cfg.tool_policy["allowed_tools"] = []
        self.assertFalse(is_tool_allowed(ADMIN, "srv__delete", cfg))

    def test_policy_survives_load_and_unrelated_save(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "config.json"
            policy = {"admin_only": False, "allowed_users": ["42"],
                      "allowed_channels": ["7"], "allowed_tools": ["srv__read"]}
            path.write_text(json.dumps({"tool_policy": policy}), encoding="utf-8")
            cfg = ConfigManager(str(path))
            self.assertTrue(is_tool_allowed(RequestContext("42", channel_id="7"), "srv__read", cfg))
            cfg.set_system_prompt("Updated")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["tool_policy"], policy)

    def test_invalid_config_fails_closed_without_overwrite(self):
        for content in ('{broken', '{"profiles":{"bad":[]}}', '{"tool_policy":{"admin_only":"false"}}'):
            with self.subTest(content=content), tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                path = Path(directory) / "config.json"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    ConfigManager(str(path))
                self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_explicit_empty_key_overrides_environment(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"profiles":{"openai":{"api_key":""}}}', encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-environment-key"}):
                cfg = ConfigManager(str(path))
            self.assertEqual(cfg.get_active_profile()[1]["api_key"], "")

    def test_rejects_insecure_provider_before_persist(self):
        cfg = config()
        with patch.object(cfg, "_atomic_persist_and_publish") as persist:
            with self.assertRaises(ValueError):
                cfg.upsert_profile("bad", base_url="http://remote.example/v1")
            persist.assert_not_called()

    def test_url_requires_host_valid_port_and_no_credentials(self):
        from error_reporting import validate_remote_url
        for url in ("https://", "https:///path", "https://host:bad", "https://user:pass@host", "http://0.0.0.0"):
            with self.subTest(url=url):
                self.assertFalse(validate_remote_url(url))

    def test_diagnostic_log_does_not_echo_unknown_secret(self):
        from error_reporting import log_sanitized_error
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            log_sanitized_error("ref", "Provider failure", RuntimeError("custom-token-SENTINEL"))
        self.assertNotIn("custom-token-SENTINEL", output.getvalue())


class AsyncBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_transports_close_in_the_owning_task(self):
        for transport, transport_func in (("stdio", "stdio_client"), ("sse", "sse_client")):
            with self.subTest(transport=transport):
                owners = []
                session = SimpleNamespace(initialize=AsyncMock(), list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])), call_tool=AsyncMock(return_value="ok"))
                @contextlib.asynccontextmanager
                async def stream(*args, **kwargs):
                    owners.append(asyncio.current_task())
                    try:
                        yield (None, None)
                    finally:
                        owners.append(asyncio.current_task())
                @contextlib.asynccontextmanager
                async def client(*args, **kwargs):
                    owners.append(asyncio.current_task())
                    try:
                        yield session
                    finally:
                        owners.append(asyncio.current_task())
                worker = MCPServerWorker("fake", {"transport": transport, "command": "fake", "url": "https://example.invalid"})
                with patch("mcp_manager." + transport_func, stream), patch("mcp_manager.ClientSession", client):
                    self.assertTrue(await worker.start())
                    self.assertEqual(await worker.call_tool("read", {}), "ok")
                    await worker.stop()
                self.assertEqual(len(owners), 4)
                self.assertTrue(all(task is worker.task for task in owners))
                self.assertTrue(worker.task.done())

    async def test_queued_authorization_is_checked_at_execution(self):
        worker = MCPServerWorker("fake", {})
        fut = asyncio.get_running_loop().create_future()
        worker.queue.put_nowait(("call_tool", ("read", {}), fut, None, lambda: False))
        worker.queue.put_nowait(None)
        session = SimpleNamespace(initialize=AsyncMock(), list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])), call_tool=AsyncMock())
        await worker._consume_session(session)
        with self.assertRaises(PermissionError):
            await fut
        session.call_tool.assert_not_called()

    async def test_both_providers_enforce_exact_total_result_budget(self):
        import llm_service
        for fmt, cls in (("openai", "AsyncOpenAI"), ("anthropic", "AsyncAnthropic")):
            with self.subTest(fmt=fmt):
                result = SimpleNamespace(content=[SimpleNamespace(text="x" * 12000)], is_error=False)
                mgr, worker = manager(result=result)
                mgr.config.profiles["openai"]["api_format"] = fmt
                definition = {"name": "srv__read", "input_schema": {"type": "object"}}
                mgr._cached_anthropic_tools = [definition]
                mgr._cached_openai_tools = [{"type": "function", "function": {"name": "srv__read", "parameters": {"type": "object"}}}]
                calls = [SimpleNamespace(id=str(i), function=SimpleNamespace(name="srv__read", arguments="{}")) for i in range(6)]
                blocks = [SimpleNamespace(type="tool_use", id=str(i), name="srv__read", input={}) for i in range(6)]
                initial = (SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(content=None, tool_calls=calls))]) if fmt == "openai"
                           else SimpleNamespace(stop_reason="tool_use", content=blocks))
                final = (SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="done", tool_calls=None))]) if fmt == "openai"
                         else SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")]))
                requests = []
                async def create(**kwargs):
                    requests.append(copy.deepcopy(kwargs))
                    return initial if len(requests) == 1 else final
                client = AsyncMock()
                client.chat.completions.create.side_effect = create
                client.messages.create.side_effect = create
                with patch.object(llm_service, cls, return_value=client), patch.object(llm_service, "MAX_TOTAL_TOOL_OUTPUT_CHARS", 40000):
                    reply = await llm_service.generate_response([{"role":"user", "content":"Please read the tool"}], mgr.config, mgr, ADMIN)
                self.assertEqual(reply, "done")
                self.assertEqual(len(worker.calls), 4)
                messages = requests[1]["messages"]
                outputs = ([m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "tool"] if fmt == "openai"
                           else [b["content"] for m in messages if isinstance(m.get("content"), list) for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"])
                self.assertEqual(sum(map(len, outputs)), 40000)
                client.close.assert_awaited_once()

    async def test_concurrent_recovery_reuses_new_worker(self):
        mgr, _ = manager()
        mgr._workers.clear()
        entered, release = asyncio.Event(), asyncio.Event()
        worker = FakeMCPServerWorker("srv")
        async def start():
            entered.set()
            await release.wait()
            return True
        worker.start = AsyncMock(side_effect=start)
        with patch("mcp_manager.MCPServerWorker", return_value=worker):
            first = asyncio.create_task(mgr.call_tool("srv__read", {}, ADMIN))
            await entered.wait()
            second = asyncio.create_task(mgr.call_tool("srv__read", {}, ADMIN))
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, second)
        worker.start.assert_awaited_once()
        self.assertFalse(worker._stopped)

    async def test_cancelled_request_cancels_active_execution(self):
        worker = MCPServerWorker("fake", {})
        entered, canceled = asyncio.Event(), asyncio.Event()
        async def slow(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()
        session = SimpleNamespace(initialize=AsyncMock(), list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])), call_tool=slow)
        worker.task = asyncio.create_task(worker._consume_session(session))
        request = asyncio.create_task(worker.call_tool("read", {}))
        try:
            await entered.wait()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            await asyncio.wait_for(canceled.wait(), 0.3)
        finally:
            await worker.stop()

    async def test_full_queue_rejects_without_unbounded_wait(self):
        worker = MCPServerWorker("fake", {})
        worker.queue = asyncio.Queue(maxsize=1)
        worker.queue.put_nowait(None)
        worker.task = asyncio.create_task(asyncio.Event().wait())
        try:
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                await asyncio.wait_for(worker.call_tool("read", {}, timeout=0.05), 0.2)
        finally:
            await worker.stop()

    async def test_remote_schema_reference_does_not_fetch_network(self):
        mgr, worker = manager({"$ref": "https://example.invalid/schema"})
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network is forbidden")) as network:
            reply = await mgr.call_tool("srv__read", {}, ADMIN)
        network.assert_not_called()
        self.assertIn("Error", reply)
        self.assertEqual(worker.calls, [])

    async def test_policy_revoked_during_reconnect_denies_dispatch(self):
        mgr, _ = manager()
        mgr._workers.clear()
        worker = FakeMCPServerWorker("srv")
        async def start():
            mgr.config.tool_policy["allowed_tools"] = []
            return True
        worker.start = start
        with patch("mcp_manager.MCPServerWorker", return_value=worker):
            reply = await mgr.call_tool("srv__read", {}, ADMIN)
        self.assertIn("Error", reply)
        self.assertEqual(worker.calls, [])

    async def test_canceling_refresh_waiter_does_not_cancel_shared_refresh(self):
        mgr, _ = manager()
        entered, release = asyncio.Event(), asyncio.Event()
        async def refresh(force_reconnect=False):
            entered.set()
            await release.wait()
        with patch.object(mgr, "_do_refresh_tools", side_effect=refresh):
            waiter = asyncio.create_task(mgr.refresh_tools())
            await entered.wait()
            shared = mgr._refresh_task
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            self.assertFalse(shared.cancelled())
            release.set()
            await shared
        await mgr.close_all()

    async def test_overload_rejects_instead_of_waiting(self):
        from request_limits import RequestLimiter
        limiter = RequestLimiter(max_global=1)
        async with limiter.acquire("first"):
            async def second():
                with self.assertRaisesRegex(RuntimeError, "busy|capacity"):
                    async with limiter.acquire("second"):
                        self.fail("Overloaded request admitted")
            await asyncio.wait_for(second(), 0.2)
        self.assertEqual(limiter._user_active_jobs, {})

    async def test_active_tool_execution_honors_deadline(self):
        worker = MCPServerWorker("fake", {})
        entered, canceled = asyncio.Event(), asyncio.Event()
        async def slow(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()
        session = SimpleNamespace(initialize=AsyncMock(), list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])), call_tool=slow)
        consumer = asyncio.create_task(worker._consume_session(session))
        fut = asyncio.get_running_loop().create_future()
        worker.queue.put_nowait(("call_tool", ("write", {}), fut, time.monotonic() + 0.05))
        try:
            await asyncio.wait_for(entered.wait(), 0.2)
            await asyncio.wait_for(canceled.wait(), 0.3)
            with self.assertRaises(TimeoutError):
                await fut
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            if fut.done() and not fut.cancelled():
                fut.exception()

    async def test_canceling_consumer_settles_active_future(self):
        worker = MCPServerWorker("fake", {})
        entered = asyncio.Event()
        async def slow(*args):
            entered.set()
            await asyncio.Event().wait()
        session = SimpleNamespace(initialize=AsyncMock(), list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])), call_tool=slow)
        consumer = asyncio.create_task(worker._consume_session(session))
        fut = asyncio.get_running_loop().create_future()
        worker.queue.put_nowait(("call_tool", ("write", {}), fut, None))
        await entered.wait()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        self.assertTrue(fut.done())
        if fut.done() and not fut.cancelled():
            self.assertIsNotNone(fut.exception())

    async def test_shutdown_during_refresh_stops_existing_workers(self):
        mgr, worker = manager()
        mgr._refresh_task = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)
        await mgr.close_all()
        self.assertTrue(worker._stopped)
        self.assertEqual(mgr._workers, {})

    async def test_schema_without_top_level_type_is_validated(self):
        mgr, worker = manager({"allOf": [{"required": ["path"]}]})
        reply = await mgr.call_tool("srv__read", {}, ADMIN)
        self.assertIn("Error", reply)
        self.assertEqual(worker.calls, [])

    async def test_oversize_arguments_never_reach_worker(self):
        mgr, worker = manager()
        reply = await mgr.call_tool("srv__read", {"data": "x" * 100000}, ADMIN)
        self.assertIn("Error", reply)
        self.assertEqual(worker.calls, [])

    async def test_structured_only_result_not_replaced_by_success_placeholder(self):
        result = SimpleNamespace(content=[], structured_content={"answer": 42}, is_error=False)
        mgr, _ = manager(result=result)
        reply = await mgr.call_tool("srv__read", {}, ADMIN)
        self.assertIn('"answer": 42', reply)

    async def test_binary_block_is_not_dumped_into_model_context(self):
        result = SimpleNamespace(content=[SimpleNamespace(type="image", data="SENTINEL_BINARY" * 200)], is_error=False)
        mgr, _ = manager(result=result)
        reply = await mgr.call_tool("srv__read", {}, ADMIN)
        self.assertNotIn("SENTINEL_BINARY", reply)

    async def test_tool_exception_is_not_passed_to_model(self):
        mgr, worker = manager()
        worker.call_handler.side_effect = RuntimeError("SENTINEL_TOOL_SECRET")
        reply = await mgr.call_tool("srv__read", {}, ADMIN)
        self.assertNotIn("SENTINEL_TOOL_SECRET", reply)
        self.assertIn("Error", reply)

    async def test_generation_deadline_closes_client_for_both_formats(self):
        import llm_service
        for fmt, cls in (("openai", "AsyncOpenAI"), ("anthropic", "AsyncAnthropic")):
            with self.subTest(fmt=fmt):
                cfg = config()
                cfg.profiles["openai"]["api_format"] = fmt
                client = AsyncMock()
                async def slow(**kwargs):
                    await asyncio.Event().wait()
                client.chat.completions.create.side_effect = slow
                client.messages.create.side_effect = slow
                with patch.object(llm_service, cls, return_value=client), patch.object(llm_service, "REQUEST_TIMEOUT_SECONDS", 0.02):
                    reply = await asyncio.wait_for(llm_service.generate_response([{"role":"user", "content":"Hi"}], cfg), 0.3)
                self.assertIn("tim", reply.lower())
                client.close.assert_awaited_once()
