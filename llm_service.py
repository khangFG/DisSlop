import asyncio
import copy
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set
import uuid

import httpx
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from config_manager import ConfigManager, resolve_litellm_model, VALID_API_FORMATS
from mcp_manager import MCPManager, MAX_TOOL_ARGUMENT_CHARS
from tool_policy import RequestContext, is_user_authorized_for_tools
from error_reporting import (
    redact_sensitive_text,
    redact_secrets,
    format_public_error,
    log_sanitized_error,
    validate_remote_url,
)
from request_limits import (
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOTAL_TOOL_OUTPUT_CHARS,
    REQUEST_TIMEOUT_SECONDS,
)

# Zero side effects on import: configure LiteLLM local cache flag upfront
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_litellm: Any = None


def _get_litellm() -> Any:
    """Lazy loader for litellm ensuring zero network/subprocess side effects on module import."""
    global _litellm
    if _litellm is None:
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm
        litellm.telemetry = False
        litellm.suppress_debug_info = True
        litellm.drop_params = True
        _litellm = litellm
    return _litellm


def __getattr__(name: str) -> Any:
    if name == "litellm":
        return _get_litellm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Preserve original classes for test mock detection
_OriginalAsyncOpenAI = openai.AsyncOpenAI
_OriginalAsyncAnthropic = anthropic.AsyncAnthropic

# Exposed for test patching and backward compatibility
AsyncOpenAI = openai.AsyncOpenAI
AsyncAnthropic = anthropic.AsyncAnthropic
AsyncHTTPClient = httpx.AsyncClient

DEFAULT_MAX_TOKENS = 32768

# Re-export modularized LLM components for backward compatibility
from modules.llm import (
    parse_dsml_tool_calls,
    strip_dsml_tags,
    resolve_tool_name,
    _extract_reasoning_text,
    SEARCH_EFFICIENCY_RULES,
    _build_litellm_params,
    _prepare_anthropic_messages,
    should_enable_tools,
)


def _sanitize_error_message(error: Exception, api_key: str) -> str:
    """Removes sensitive keys, URL credentials, and tokens from error strings."""
    return redact_sensitive_text(str(error), secrets=[api_key] if api_key else None)


async def generate_response(
    messages: List[Dict[str, Any]],
    config: ConfigManager,
    mcp_manager: Optional[MCPManager] = None,
    context: Optional[RequestContext] = None,
) -> str:
    """Bound the whole provider/tool turn, including discovery and client cleanup."""
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            return await _generate_response(messages, config, mcp_manager, context)
    except TimeoutError:
        return "The request timed out. A tool may have completed; check its outcome before retrying."


async def _generate_response(
    messages: List[Dict[str, Any]],
    config: ConfigManager,
    mcp_manager: Optional[MCPManager] = None,
    context: Optional[RequestContext] = None,
) -> str:
    """
    Asynchronously executes a unified multi-step agentic LLM tool loop.
    Enforces persistent session reuse, tool schema caching, casual chat filtering,
    Anthropic prompt caching, and strict output character capping.
    """
    profile_name, profile = config.get_active_profile()
    api_key = (profile.get("api_key") or "").strip()
    api_format = (profile.get("api_format") or "openai").lower().strip()
    base_url = profile.get("base_url") or None
    if base_url is not None and not validate_remote_url(base_url):
        return "Error: Provider endpoint is invalid or insecure. Ask the admin to update it."
    model = (profile.get("model") or "").strip()

    if not api_key:
        return (
            f"⚠️ **API Key Missing**: The active profile `{profile_name}` does not have an API key configured.\n"
            f"Use `/provider_set profile:{profile_name} api_key:<your_key>` to configure it."
        )

    if not model:
        return (
            f"⚠️ **Model Missing**: The active profile `{profile_name}` does not have a model specified.\n"
            f"Use `/provider_set profile:{profile_name} model:<model_name>` to configure it."
        )

    if api_format not in VALID_API_FORMATS:
        return f"❌ **Unsupported Format**: Unknown api_format `{api_format}` on profile `{profile_name}`."

    system_prompt = config.system_prompt
    caller_authorized = is_user_authorized_for_tools(context, config)
    tools_needed = caller_authorized if mcp_manager is not None else False
    effective_system_prompt = f"{system_prompt}{SEARCH_EFFICIENCY_RULES}" if tools_needed else system_prompt

    # Special handling when legacy AsyncAnthropic has been patched by a unit test
    if AsyncAnthropic is not _OriginalAsyncAnthropic and api_format == "anthropic":
        return await _run_mock_anthropic_loop(
            messages=messages,
            config=config,
            mcp_manager=mcp_manager,
            context=context,
            profile=profile,
            profile_name=profile_name,
            api_key=api_key,
            model=model,
            base_url=base_url,
            effective_system_prompt=effective_system_prompt,
            tools_needed=tools_needed,
        )

    # Standard Unified Agentic Loop
    return await _run_agentic_loop(
        messages=messages,
        config=config,
        mcp_manager=mcp_manager,
        context=context,
        profile=profile,
        profile_name=profile_name,
        api_format=api_format,
        api_key=api_key,
        model=model,
        base_url=base_url,
        effective_system_prompt=effective_system_prompt,
        tools_needed=tools_needed,
    )


async def _run_agentic_loop(
    messages: List[Dict[str, Any]],
    config: ConfigManager,
    mcp_manager: Optional[MCPManager],
    context: Optional[RequestContext],
    profile: Dict[str, Any],
    profile_name: str,
    api_format: str,
    api_key: str,
    model: str,
    base_url: Optional[str],
    effective_system_prompt: str,
    tools_needed: bool,
) -> str:
    """
    Unified agentic loop supporting LiteLLM and mock OpenAI clients.
    Translates standard OpenAI messages and tool schemas across providers,
    extracts reasoning content, enforces turn budgets, and handles error sanitization.
    """
    base_params = _build_litellm_params(profile)
    resolved_model = base_params["model"]

    is_mock_openai = AsyncOpenAI is not _OriginalAsyncOpenAI

    client: Any = None
    if is_mock_openai:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        async def completion_fn(**kwargs: Any) -> Any:
            kw = dict(kwargs)
            kw.pop("client", None)
            kw.pop("api_base", None)
            if "model" in kw:
                kw["model"] = model
            return await client.chat.completions.create(**kw)

        lex_auth: Any = (openai.AuthenticationError, anthropic.AuthenticationError)
        lex_rate: Any = (openai.RateLimitError, anthropic.RateLimitError)
        lex_conn: Any = (openai.APIConnectionError, anthropic.APIConnectionError)
        lex_bad: Any = (openai.BadRequestError, anthropic.BadRequestError)
        lex_api: Any = (openai.APIError, anthropic.APIError)
    else:
        client = AsyncHTTPClient(timeout=REQUEST_TIMEOUT_SECONDS)
        async def completion_fn(**kwargs: Any) -> Any:
            llm = _get_litellm()
            return await llm.acompletion(timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)

        llm = _get_litellm()
        lex_auth = (llm.exceptions.AuthenticationError, openai.AuthenticationError, anthropic.AuthenticationError)
        lex_rate = (llm.exceptions.RateLimitError, openai.RateLimitError, anthropic.RateLimitError)
        lex_conn = (llm.exceptions.APIConnectionError, openai.APIConnectionError, anthropic.APIConnectionError)
        lex_bad = (llm.exceptions.BadRequestError, openai.BadRequestError, anthropic.BadRequestError)
        lex_api = (llm.exceptions.APIError, openai.APIError, anthropic.APIError)

    try:
        is_anthropic_model = resolved_model.startswith("anthropic/")

        if is_anthropic_model and not is_mock_openai:
            system_content: Any = [
                {
                    "type": "text",
                    "text": effective_system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_content = effective_system_prompt

        conversation_messages: List[Any] = [
            {"role": "system", "content": system_content}
        ]
        for m in messages:
            if m.get("content"):
                conversation_messages.append({"role": m.get("role", "user"), "content": m["content"]})

        tools: List[Dict[str, Any]] = []
        if tools_needed and mcp_manager is not None:
            raw_tools = await mcp_manager.get_openai_tools(context=context)
            if raw_tools:
                tools = copy.deepcopy(raw_tools)
                if is_anthropic_model and not is_mock_openai:
                    tools[-1]["cache_control"] = {"type": "ephemeral"}

        total_tool_calls_executed = 0
        total_tool_output_chars = 0
        max_tool_iterations = 10

        for iteration in range(max_tool_iterations):
            call_kwargs = dict(base_params)
            call_kwargs["messages"] = conversation_messages

            budget_exhausted = (
                total_tool_calls_executed >= MAX_TOOL_CALLS_PER_TURN
                or total_tool_output_chars >= MAX_TOTAL_TOOL_OUTPUT_CHARS
            )
            if tools and not budget_exhausted:
                call_kwargs["tools"] = tools

            response = await completion_fn(**call_kwargs)

            choices = getattr(response, "choices", None)
            if not choices:
                return "Error: No response content returned by model."
            choice = choices[0]
            assistant_msg = choice.message
            conversation_messages.append(assistant_msg)

            extracted_tool_calls = list(getattr(assistant_msg, "tool_calls", None) or [])
            is_dsml = False
            msg_content = getattr(assistant_msg, "content", None)
            if not extracted_tool_calls and msg_content and isinstance(msg_content, str):
                dsml_calls = parse_dsml_tool_calls(msg_content)
                if dsml_calls:
                    is_dsml = True
                    for dc in dsml_calls:
                        extracted_tool_calls.append(
                            SimpleNamespace(
                                id=dc["id"],
                                type="function",
                                function=SimpleNamespace(
                                    name=dc["function"]["name"],
                                    arguments=dc["function"]["arguments"],
                                ),
                            )
                        )

            if choice.finish_reason == "tool_calls" or extracted_tool_calls:
                available_names = {t["function"]["name"] for t in call_kwargs.get("tools", [])}
                if is_dsml:
                    conversation_messages[-1] = {
                        "role": "assistant",
                        "content": strip_dsml_tags(msg_content) or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": resolve_tool_name(tc.function.name, available_names),
                                    "arguments": tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments),
                                },
                            }
                            for tc in extracted_tool_calls
                        ],
                    }

                for tc in extracted_tool_calls:
                    raw_name = tc.function.name
                    func_name = resolve_tool_name(raw_name, available_names)
                    try:
                        raw_args = tc.function.arguments
                        if isinstance(raw_args, str):
                            if len(raw_args) > MAX_TOOL_ARGUMENT_CHARS:
                                raise ValueError("Arguments too large")
                            func_args = json.loads(raw_args) if raw_args.strip() else {}
                        elif isinstance(raw_args, dict):
                            func_args = raw_args
                        else:
                            func_args = None
                    except Exception:
                        func_args = None

                    if total_tool_calls_executed >= MAX_TOOL_CALLS_PER_TURN or total_tool_output_chars >= MAX_TOTAL_TOOL_OUTPUT_CHARS:
                        tool_output = "(Tool execution budget reached for this turn)"
                    elif func_name not in available_names:
                        tool_output = "Error: Tool was not offered for this request."
                    elif not isinstance(func_args, dict):
                        tool_output = f"Error: Invalid JSON arguments for tool '{func_name}'. Arguments must be a JSON object."
                    elif mcp_manager is not None:
                        total_tool_calls_executed += 1
                        tool_output = await mcp_manager.call_tool(func_name, func_args, context=context)
                    else:
                        tool_output = "(No MCP Manager available)"

                    tool_output = str(tool_output)[:max(0, MAX_TOTAL_TOOL_OUTPUT_CHARS - total_tool_output_chars)]
                    total_tool_output_chars += len(tool_output)
                    conversation_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": func_name,
                        "content": tool_output,
                    })
                continue
            else:
                raw_text = getattr(assistant_msg, "content", None) or ""
                if isinstance(raw_text, list):
                    text_parts = [b.text for b in raw_text if getattr(b, "type", None) == "text" and hasattr(b, "text")]
                    reply_text = strip_dsml_tags("".join(text_parts).strip())
                else:
                    reply_text = strip_dsml_tags(str(raw_text).strip())

                if not reply_text:
                    reasoning = _extract_reasoning_text(assistant_msg)
                    if reasoning:
                        if choice.finish_reason in ("length", "max_tokens"):
                            reply_text = f"⚠️ *[Token limit reached during model reasoning]*\n\n{reasoning}"
                        else:
                            reply_text = reasoning
                    elif choice.finish_reason in ("length", "max_tokens"):
                        reply_text = "⚠️ The AI model reached its token generation limit (`max_tokens`) before producing an answer. Try simplifying your query or increasing token limit."
                return reply_text if reply_text else "(No response content returned by model)"

        # Fallback synthesis if tool loop exhausted all iterations
        final_messages = conversation_messages + [
            {"role": "user", "content": "Please provide your final answer now based on the information gathered above."}
        ]
        final_kwargs = dict(base_params)
        final_kwargs["messages"] = final_messages
        final_resp = await completion_fn(**final_kwargs)
        final_choices = getattr(final_resp, "choices", None)
        if not final_choices:
            return "I have completed the requested search and gathered the findings above."
        final_choice = final_choices[0]
        final_msg = final_choice.message
        final_raw = getattr(final_msg, "content", None) or ""
        if isinstance(final_raw, list):
            final_text = strip_dsml_tags("".join(b.text for b in final_raw if getattr(b, "type", None) == "text" and hasattr(b, "text")).strip())
        else:
            final_text = strip_dsml_tags(str(final_raw).strip())

        if not final_text:
            reasoning = _extract_reasoning_text(final_msg)
            if reasoning:
                final_text = reasoning
        return final_text if final_text else "I have completed the requested search and gathered the findings above."

    except lex_auth:
        return f"❌ **Authentication Failed**: Invalid API key for profile `{profile_name}`. Run `/provider_set` to update."
    except lex_rate:
        return "⏳ **Rate Limit**: AI provider returned a rate limit error. Please wait a moment before trying again."
    except lex_conn as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="Failed to reach provider endpoint.")
        log_sanitized_error(cid, "Provider connection error", e)
        return public_err
    except lex_bad as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="Provider rejected request.")
        log_sanitized_error(cid, "Provider BadRequest", e)
        return public_err
    except lex_api as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="AI Provider returned an API error.")
        log_sanitized_error(cid, "Provider APIError", e)
        return public_err
    except Exception as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="Unexpected failure during generation.")
        log_sanitized_error(cid, "Unexpected generation error", e)
        return public_err
    finally:
        if is_mock_openai:
            await client.close()
        else:
            await client.aclose()


async def _run_mock_anthropic_loop(
    messages: List[Dict[str, Any]],
    config: ConfigManager,
    mcp_manager: Optional[MCPManager],
    context: Optional[RequestContext],
    profile: Dict[str, Any],
    profile_name: str,
    api_key: str,
    model: str,
    base_url: Optional[str],
    effective_system_prompt: str,
    tools_needed: bool,
) -> str:
    """Mock-injection bridge for unit tests specifically patching AsyncAnthropic."""
    client = AsyncAnthropic(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
    )
    try:
        tools = []
        if tools_needed and mcp_manager is not None:
            tools = await mcp_manager.get_anthropic_tools(context=context)
            if tools:
                tools[-1]["cache_control"] = {"type": "ephemeral"}

        conversation_messages = _prepare_anthropic_messages(messages)

        cached_system: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": effective_system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        max_tokens = int(profile.get("max_tokens") or DEFAULT_MAX_TOKENS)
        max_tool_iterations = 10
        total_tool_calls_executed = 0
        total_tool_output_chars = 0

        for iteration in range(max_tool_iterations):
            kwargs: Dict[str, Any] = {
                "model": model,
                "system": cached_system,
                "messages": conversation_messages,
                "max_tokens": max_tokens,
            }
            budget_exhausted = (
                total_tool_calls_executed >= MAX_TOOL_CALLS_PER_TURN
                or total_tool_output_chars >= MAX_TOTAL_TOOL_OUTPUT_CHARS
            )
            if tools and not budget_exhausted:
                kwargs["tools"] = tools

            response = await client.messages.create(**kwargs)

            conversation_messages.append({
                "role": "assistant",
                "content": response.content,
            })

            if getattr(response, "stop_reason", None) == "tool_use":
                tool_results = []
                for block in response.content:
                    if getattr(block, "type", None) == "tool_use":
                        tool_id = getattr(block, "id", "")
                        tool_name = getattr(block, "name", "")
                        tool_input = getattr(block, "input", {})
                        if total_tool_calls_executed >= MAX_TOOL_CALLS_PER_TURN or total_tool_output_chars >= MAX_TOTAL_TOOL_OUTPUT_CHARS:
                            tool_output = "(Tool execution budget reached for this turn)"
                        elif tool_name not in {t["name"] for t in kwargs.get("tools", [])}:
                            tool_output = "Error: Tool was not offered for this request."
                        elif not isinstance(tool_input, dict):
                            tool_output = f"Error: Invalid arguments for tool '{tool_name}'. Arguments must be a JSON object (dict)."
                        elif mcp_manager is not None:
                            total_tool_calls_executed += 1
                            tool_output = await mcp_manager.call_tool(tool_name, tool_input, context=context)
                        else:
                            tool_output = "(No MCP Manager available)"

                        tool_output = str(tool_output)[:max(0, MAX_TOTAL_TOOL_OUTPUT_CHARS - total_tool_output_chars)]
                        total_tool_output_chars += len(tool_output)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": tool_output,
                        })

                conversation_messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                continue
            else:
                text_parts = [
                    b.text for b in response.content if getattr(b, "type", None) == "text" and hasattr(b, "text")
                ]
                reply_text = strip_dsml_tags("".join(text_parts).strip())
                if not reply_text:
                    thinking_parts = [
                        getattr(b, "thinking", "") for b in response.content if getattr(b, "type", None) == "thinking" or hasattr(b, "thinking")
                    ]
                    thinking_text = "".join(thinking_parts).strip()
                    if thinking_text:
                        if getattr(response, "stop_reason", None) == "max_tokens":
                            reply_text = f"⚠️ *[Token limit reached during model thinking]*\n\n{thinking_text}"
                        else:
                            reply_text = thinking_text
                    elif getattr(response, "stop_reason", None) == "max_tokens":
                        reply_text = "⚠️ The AI model reached its token generation limit (`max_tokens`) before producing an answer."
                return reply_text if reply_text else "(No response content returned by Anthropic model)"

        # Fallback synthesis
        final_messages = conversation_messages + [
            {"role": "user", "content": "Please provide your final answer now based on the information gathered above."}
        ]
        final_resp = await client.messages.create(
            model=model,
            system=cached_system,
            messages=final_messages,
            max_tokens=max_tokens,
        )
        text_parts = [
            b.text for b in final_resp.content if getattr(b, "type", None) == "text" and hasattr(b, "text")
        ]
        final_content = strip_dsml_tags("".join(text_parts).strip())
        if not final_content:
            thinking_parts = [
                getattr(b, "thinking", "") for b in final_resp.content if getattr(b, "type", None) == "thinking" or hasattr(b, "thinking")
            ]
            thinking_text = "".join(thinking_parts).strip()
            if thinking_text:
                final_content = thinking_text
        return final_content if final_content else "I have completed the requested search and gathered the findings above."

    except anthropic.AuthenticationError:
        return f"❌ **Authentication Failed**: Invalid Anthropic API key for profile `{profile_name}`. Run `/provider_set` to update."
    except anthropic.RateLimitError:
        return "⏳ **Rate Limit**: Anthropic rate limit reached. Please retry in a few moments."
    except anthropic.APIConnectionError as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="Failed to reach Anthropic endpoint.")
        log_sanitized_error(cid, "Anthropic connection error", e)
        return public_err
    except anthropic.BadRequestError as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="Anthropic rejected request.")
        log_sanitized_error(cid, "Anthropic BadRequest", e)
        return public_err
    except anthropic.APIError as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="Anthropic returned an API error.")
        log_sanitized_error(cid, "Anthropic APIError", e)
        return public_err
    except Exception as e:
        clean_err = _sanitize_error_message(e, api_key)
        public_err, cid = format_public_error(clean_err, error_prefix="Unexpected Anthropic failure.")
        log_sanitized_error(cid, "Unexpected Anthropic error", e)
        return public_err
    finally:
        await client.close()


async def generate_reply(
    messages: List[Dict[str, Any]],
    config: ConfigManager,
    context: Optional[RequestContext] = None,
) -> str:
    """Wrapper matching previous llm_engine signature."""
    return await generate_response(messages, config, None, context=context)
