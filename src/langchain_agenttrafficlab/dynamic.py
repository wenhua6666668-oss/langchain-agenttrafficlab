"""Two-stage dynamic execution for ATL executable MCP handoffs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import ipaddress
import json
import socket
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlsplit

import requests
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class HandoffError(ValueError):
    """The ATL response is not a usable executable handoff."""


class EndpointRejected(HandoffError):
    """The handoff endpoint fails the adapter's network policy."""


class AuthRequired(HandoffError):
    """The provider requires caller-supplied credentials."""


class ATLUnavailable(RuntimeError):
    """ATL could not produce a decision; a caller may use its safe fallback."""


class DynamicLoadError(RuntimeError):
    """The selected provider/tool could not be loaded safely."""


CredentialProvider = Callable[["ATLHandoff"], Mapping[str, str] | Awaitable[Mapping[str, str]]]
OutcomeReporter = Callable[["ATLHandoff", str, Mapping[str, Any] | None], Any]


@dataclass(frozen=True)
class ATLHandoff:
    provider_id: str
    provider_name: str
    capability: str
    endpoint: str
    transport: str
    tool_name: str
    input_schema: Mapping[str, Any]
    auth_required: bool
    auth_status: str
    decision_reference: str | None = None
    outcome_correlation_token: str | None = None
    expires_at_ms: int | None = None


@dataclass(frozen=True)
class ExecutionResult:
    """A truthful, ATL-compatible summary of one provider execution."""

    result_class: str
    outcome_status: str
    error_code: str | None = None
    failure_type: str | None = None
    http_status: int | None = None
    value: Any = None

    def attempt_outcome(self, provider_id: str) -> dict[str, Any]:
        attempt = {
            "provider_id": provider_id,
            "outcome_status": self.outcome_status,
        }
        if self.error_code:
            attempt["error_code"] = self.error_code
        if self.failure_type:
            attempt["failure_type"] = self.failure_type
        if self.http_status is not None:
            attempt["http_status"] = self.http_status
        return attempt


@dataclass(frozen=True)
class RetryContext:
    """Adapter-owned context carried into an existing ATL task request."""

    failed_provider_id: str
    decision_reference: str | None
    outcome_correlation_token: str | None
    failure: ExecutionResult
    attempt_history: tuple[Mapping[str, Any], ...] = ()

    def to_decision_task(self, task: str) -> str:
        context = {
            "failed_provider_id": self.failed_provider_id,
            "decision_reference": self.decision_reference,
            "outcome_correlation_token": self.outcome_correlation_token,
            "failure_class": self.failure.result_class,
            "attempt_history": [dict(item) for item in self.attempt_history],
        }
        return f"{task}\nATL retry context (adapter-generated, non-authoritative): {json.dumps(context, sort_keys=True)}"


def classify_execution_error(error: BaseException) -> ExecutionResult:
    """Map an execution exception to existing ATL outcome vocabulary."""
    status = _exception_http_status(error)
    if status == 402:
        return ExecutionResult("PAYMENT_REQUIRED", "FAILURE", "PAYMENT_REQUIRED", "payment_required", 402)
    if status in {401, 403}:
        return ExecutionResult("AUTH_FAILURE", "FAILURE", "AUTH_INVALID", "auth_failure", status)
    if _is_timeout(error):
        return ExecutionResult("TIMEOUT", "FAILURE", "TIMEOUT", "provider_timeout")
    if isinstance(status, int) and status >= 500:
        return ExecutionResult("PROVIDER_FAILURE", "FAILURE", "PROVIDER_5XX", "provider_5xx", status)
    return ExecutionResult("UNKNOWN_FAILURE", "FAILURE", "UNKNOWN_FAILURE", "unknown_failure")


def _exception_http_status(error: BaseException) -> int | None:
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            status = _exception_http_status(child)
            if status is not None:
                return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(error, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _is_timeout(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return any(_is_timeout(child) for child in error.exceptions)
    return isinstance(error, (TimeoutError, asyncio.TimeoutError)) or "timeout" in type(error).__name__.lower()


def parse_handoff(payload: Mapping[str, Any]) -> ATLHandoff:
    """Parse an ATL MCP response without accepting model/task endpoint input."""
    if not isinstance(payload, Mapping):
        raise HandoffError("ATL response must be an object")
    value: Any = payload
    if isinstance(value.get("result"), Mapping):
        value = value["result"]
    if isinstance(value, Mapping) and isinstance(value.get("structuredContent"), Mapping):
        value = value["structuredContent"]
    if isinstance(value, Mapping) and isinstance(value.get("content"), list):
        for item in value["content"]:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                try:
                    value = json.loads(item["text"])
                except json.JSONDecodeError as exc:
                    raise HandoffError("ATL content is not valid JSON") from exc
                break
    if isinstance(value, Mapping) and isinstance(value.get("executable_handoff"), Mapping):
        handoff = value["executable_handoff"]
        outer = value
    else:
        handoff = value
        outer = payload

    if not isinstance(handoff, Mapping):
        raise HandoffError("ATL executable handoff is missing")
    provider_id = _required_string(handoff, "provider_id")
    endpoint = _required_string(handoff, "endpoint")
    validate_endpoint(endpoint)
    transport = _normalise_transport(_required_string(handoff, "transport"))
    action = handoff.get("action")
    if not isinstance(action, Mapping):
        raise HandoffError("ATL handoff action is missing")
    tool_name = _required_string(action, "tool_name")
    input_schema = action.get("input_schema")
    if not isinstance(input_schema, Mapping) or input_schema.get("type") not in (None, "object"):
        raise HandoffError("ATL handoff input_schema must be an object schema")
    auth = handoff.get("auth") or {}
    if not isinstance(auth, Mapping):
        raise HandoffError("ATL handoff auth must be an object")
    expires_at_ms = handoff.get("expires_at_ms")
    if expires_at_ms is not None:
        try:
            expires_at_ms = int(expires_at_ms)
        except (TypeError, ValueError) as exc:
            raise HandoffError("ATL handoff expiry is invalid") from exc
        if expires_at_ms <= _now_ms():
            raise HandoffError("ATL handoff has expired")

    outcome = outer.get("outcome") if isinstance(outer, Mapping) else None
    if not isinstance(outcome, Mapping):
        outcome = {}
    return ATLHandoff(
        provider_id=provider_id,
        provider_name=str(handoff.get("provider_name") or ""),
        capability=str(handoff.get("capability") or ""),
        endpoint=endpoint,
        transport=transport,
        tool_name=tool_name,
        input_schema=dict(input_schema),
        auth_required=bool((auth.get("required") is True) or auth.get("status") == "required"),
        auth_status=str(auth.get("status") or "not_required"),
        decision_reference=_optional_string(outer, "decision_id") or _optional_string(outer, "decision_reference"),
        outcome_correlation_token=_optional_string(outcome, "outcome_correlation_token") or _optional_string(outer, "outcome_correlation_token"),
        expires_at_ms=expires_at_ms,
    )


def validate_endpoint(endpoint: str) -> None:
    """Require a public HTTPS endpoint and reject obvious SSRF targets."""
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise EndpointRejected("MCP endpoint must be an HTTPS URL without embedded credentials")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise EndpointRejected("local MCP endpoints are not allowed")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        _reject_non_public(literal)
        return
    try:
        addresses = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        raise EndpointRejected("MCP endpoint DNS lookup failed") from exc
    if not addresses:
        raise EndpointRejected("MCP endpoint has no resolved address")
    for address in addresses:
        _reject_non_public(address)


def _reject_non_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_multicast or address.is_unspecified:
        raise EndpointRejected("MCP endpoint resolves to a non-public address")


def _normalise_transport(value: str) -> str:
    value = value.lower().replace("_", "-")
    if value in {"http", "streamable-http"}:
        return "http"
    raise HandoffError("unsupported MCP transport")


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = str(value.get(key) or "").strip()
    if not result:
        raise HandoffError(f"ATL handoff field {key} is required")
    return result


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    result = str(value.get(key) or "").strip()
    return result or None


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class ATLClient:
    """Minimal MCP client for ATL decision and outcome calls."""

    def __init__(
        self,
        endpoint: str = "https://mcp.agenttrafficlab.com/mcp",
        timeout: float = 3.0,
        tenant_id: str = "public-mcp",
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        validate_endpoint(endpoint)
        self.endpoint = endpoint
        self.timeout = timeout
        self.tenant_id = tenant_id

    async def decide(self, task: str, retry_context: RetryContext | None = None) -> Mapping[str, Any]:
        try:
            decision_task = retry_context.to_decision_task(task) if retry_context else task
            return await asyncio.to_thread(self._call, "atl_decide", {"task": decision_task}, "decide")
        except Exception as exc:
            raise ATLUnavailable("ATL decision was unavailable") from exc

    async def report_outcome(self, handoff: ATLHandoff, status: str, details: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        arguments = {
            "decision_reference": handoff.decision_reference,
            "outcome_correlation_token": handoff.outcome_correlation_token,
            "provider_id": handoff.provider_id,
            "final_provider_id": handoff.provider_id,
            "outcome_status": status,
            "tenant_id": self.tenant_id,
        }
        if details:
            attempt = dict(details)
            for key in ("error_code", "failure_type", "http_status"):
                if attempt.get(key) is not None:
                    arguments[key] = attempt[key]
            arguments["attempt_outcomes"] = [attempt]
        return await asyncio.to_thread(self._call, "atl_outcome", arguments, "outcome")

    def _call(self, tool_name: str, arguments: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        init = self._post({"jsonrpc": "2.0", "id": f"atl-{request_id}-init", "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "langchain-agenttrafficlab", "version": "0.2.0"}}}, headers)
        session_id = ((init.get("result") or {}).get("session") or {}).get("id") if isinstance(init, Mapping) else None
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        return self._post({"jsonrpc": "2.0", "id": f"atl-{request_id}", "method": "tools/call", "params": {"name": tool_name, "arguments": dict(arguments)}}, headers)

    def _post(self, body: Mapping[str, Any], headers: Mapping[str, str]) -> Mapping[str, Any]:
        response = requests.post(self.endpoint, json=body, headers=dict(headers), timeout=self.timeout, allow_redirects=False)
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError:
            result = next((json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith("data:")), None)
        if not isinstance(result, Mapping) or "error" in result:
            raise RuntimeError("ATL returned an invalid MCP response")
        return result


class TwoStageATLExecutor:
    """Call ATL first, then build a new execution-stage agent around one MCP tool."""

    def __init__(
        self,
        *,
        decision_client: Callable[[str], Any],
        credential_provider: CredentialProvider | None = None,
        mcp_client_factory: Callable[..., Any] = MultiServerMCPClient,
        outcome_reporter: OutcomeReporter | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.decision_client = decision_client
        self.credential_provider = credential_provider
        self.mcp_client_factory = mcp_client_factory
        self.outcome_reporter = outcome_reporter
        self.timeout = timeout

    async def decide(self, task: str, retry_context: RetryContext | None = None) -> ATLHandoff:
        try:
            decision_task = retry_context.to_decision_task(task) if retry_context else task
            raw = await _maybe_await(self.decision_client(decision_task))
        except Exception as exc:
            raise ATLUnavailable("ATL decision was unavailable") from exc
        return parse_handoff(raw)

    async def load_selected_tool(self, handoff: ATLHandoff) -> BaseTool:
        validate_endpoint(handoff.endpoint)
        headers: Mapping[str, str] = {}
        if handoff.auth_required:
            if self.credential_provider is None:
                raise AuthRequired("provider credentials are required")
            supplied = await _maybe_await(self.credential_provider(handoff))
            if not supplied:
                raise AuthRequired("provider credentials are required")
            headers = dict(supplied)
        connection: dict[str, Any] = {"transport": handoff.transport, "url": handoff.endpoint}
        if headers:
            connection["headers"] = headers
        client = self.mcp_client_factory({"atl-selected": connection}, handle_tool_errors=False)
        try:
            tools = await asyncio.wait_for(client.get_tools(server_name="atl-selected"), timeout=self.timeout)
        except Exception as exc:
            raise DynamicLoadError("selected MCP provider could not be loaded") from exc
        selected = [tool for tool in tools if getattr(tool, "name", "") == handoff.tool_name]
        if len(selected) != 1:
            raise DynamicLoadError("ATL-selected MCP tool was not loaded exactly once")
        return selected[0]

    async def run(self, task: str, *, model: Any, original_agent: Any = None) -> Any:
        try:
            handoff = await self.decide(task)
        except ATLUnavailable:
            if original_agent is None:
                raise
            return await _invoke_agent(original_agent, task)
        try:
            return await self._execute_stage(handoff, task, model)
        except Exception as exc:
            if self.outcome_reporter is not None:
                try:
                    await self.report_execution_result(handoff, classify_execution_error(exc))
                except Exception:
                    pass
            raise

    async def _execute_stage(self, handoff: ATLHandoff, task: str, model: Any) -> Any:
        tool = await self.load_selected_tool(handoff)
        agent = create_agent(model=model, tools=[tool])
        return await asyncio.wait_for(agent.ainvoke({"messages": [{"role": "user", "content": task}]}), timeout=self.timeout)

    async def report_outcome(self, handoff: ATLHandoff, status: str, details: Mapping[str, Any] | None = None) -> Any:
        if self.outcome_reporter is None:
            return None
        return await _maybe_await(self.outcome_reporter(handoff, status, details))

    async def report_execution_result(self, handoff: ATLHandoff, result: ExecutionResult) -> Any:
        """Report a result without dropping provider failure classification."""
        return await self.report_outcome(handoff, result.outcome_status, result.attempt_outcome(handoff.provider_id))

    @staticmethod
    def build_retry_context(
        handoff: ATLHandoff,
        result: ExecutionResult,
        attempt_history: tuple[Mapping[str, Any], ...] = (),
    ) -> RetryContext:
        return RetryContext(
            failed_provider_id=handoff.provider_id,
            decision_reference=handoff.decision_reference,
            outcome_correlation_token=handoff.outcome_correlation_token,
            failure=result,
            attempt_history=attempt_history + (result.attempt_outcome(handoff.provider_id),),
        )


async def _invoke_agent(agent: Any, task: str) -> Any:
    if hasattr(agent, "ainvoke"):
        return await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    if callable(agent) and not hasattr(agent, "invoke"):
        result = agent({"messages": [{"role": "user", "content": task}]})
        return await result if inspect.isawaitable(result) else result
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    return await result if inspect.isawaitable(result) else result


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value
