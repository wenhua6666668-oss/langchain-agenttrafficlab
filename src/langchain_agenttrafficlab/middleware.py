"""A bounded ATL decision layer for LangChain agents."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import json
import threading
from typing import Any, Callable, Mapping, Sequence

import requests
from langchain.agents.middleware import AgentMiddleware


DecisionClient = Callable[[str, list[str]], Any]


@dataclass(frozen=True)
class _Decision:
    selected_tools: tuple[str, ...]


class ATLMiddleware(AgentMiddleware):
    """Use ATL to narrow existing LangChain tools before a model call.

    v0.1 is deliberately non-discovery: ATL may select only names present in
    the current LangChain request. A failed or invalid ATL decision leaves the
    original tool list untouched.
    """

    def __init__(
        self,
        *,
        endpoint: str = "https://mcp.agenttrafficlab.com/mcp",
        timeout: float = 2.0,
        fail_open: bool = True,
        decision_client: DecisionClient | None = None,
        cache_size: int = 128,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.endpoint = endpoint
        self.timeout = timeout
        self.fail_open = fail_open
        self._decision_client = decision_client or self._http_decide
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[int, str, tuple[str, ...]], _Decision] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="atl")

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Ask ATL once per invocation/task, then continue the normal loop."""
        original_tools = list(getattr(request, "tools", ()) or ())
        tool_names = tuple(self._tool_name(tool) for tool in original_tools)
        tool_names = tuple(name for name in tool_names if name)
        if not tool_names:
            return handler(request)

        task = self._task_from_request(request)
        invocation = getattr(request, "runtime", None)
        invocation_key = id(invocation) if invocation is not None else id(request)
        cache_key = (invocation_key, task, tool_names)
        decision = self._cache_get(cache_key)
        if decision is None:
            decision = self._bounded_decide(task, list(tool_names))
            if decision is not None:
                self._cache_put(cache_key, decision)

        selected = tuple(name for name in (decision.selected_tools if decision else ()) if name in tool_names)
        if not selected:
            return handler(request)

        selected_set = set(selected)
        narrowed_tools = [tool for tool in original_tools if self._tool_name(tool) in selected_set]
        try:
            request = request.override(tools=narrowed_tools)
        except AttributeError:
            # A malformed/non-LangChain request must not block the agent loop.
            return handler(request)
        return handler(request)

    def _bounded_decide(self, task: str, tools: list[str]) -> _Decision | None:
        future = self._executor.submit(self._decision_client, task, tools)
        try:
            raw = future.result(timeout=self.timeout)
        except (FutureTimeoutError, Exception):
            future.cancel()
            return None if self.fail_open else None
        selected = self._parse_selected_tools(raw, tools)
        return _Decision(tuple(selected)) if selected else None

    def _http_decide(self, task: str, tools: list[str]) -> Any:
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        initialize = {
            "jsonrpc": "2.0", "id": "atl-init", "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                        "clientInfo": {"name": "langchain-agenttrafficlab", "version": "0.1.0"}},
        }
        init_response = requests.post(self.endpoint, json=initialize, headers=headers, timeout=self.timeout)
        init_response.raise_for_status()
        session_id = init_response.headers.get("Mcp-Session-Id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        call = {"jsonrpc": "2.0", "id": "atl-decide", "method": "tools/call",
                "params": {"name": "atl_decide", "arguments": {"task": task}}}
        response = requests.post(self.endpoint, json=call, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return self._decode_response(response)

    @staticmethod
    def _decode_response(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
            raise

    @staticmethod
    def _tool_name(tool: Any) -> str:
        if isinstance(tool, Mapping):
            return str(tool.get("name") or "")
        return str(getattr(tool, "name", "") or "")

    @staticmethod
    def _task_from_request(request: Any) -> str:
        messages = getattr(request, "messages", ()) or ()
        if not messages:
            return ""
        message = next(
            (
                candidate
                for candidate in messages
                if getattr(candidate, "type", "") in {"human", "user"}
            ),
            messages[0],
        )
        content = getattr(message, "content", message if isinstance(message, str) else "")
        if isinstance(content, str):
            return content
        return json.dumps(content, sort_keys=True, default=str)

    @staticmethod
    def _parse_selected_tools(raw: Any, registered: Sequence[str]) -> list[str]:
        value = raw
        if isinstance(value, Mapping):
            value = value.get("result", value)
            if isinstance(value, Mapping):
                value = value.get("structuredContent", value.get("content", value))
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    value = item.get("text", item.get("selected_tool", item.get("tool", item)))
                    break
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return [value] if value in registered else []
        if isinstance(value, Mapping):
            value = value.get("selected_tool", value.get("tool", value.get("name", "")))
        if isinstance(value, str):
            return [value] if value in registered else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item) in registered]
        return []

    def _cache_get(self, key: tuple[int, str, tuple[str, ...]]) -> _Decision | None:
        with self._cache_lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)
            return value

    def _cache_put(self, key: tuple[int, str, tuple[str, ...]], value: _Decision) -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
