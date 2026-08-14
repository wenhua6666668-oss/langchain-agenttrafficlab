import asyncio
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import ClassVar

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import langchain_agenttrafficlab.dynamic as dynamic
from langchain_agenttrafficlab import (
    ATLHandoff,
    AuthRequired,
    DynamicLoadError,
    ExecutionResult,
    EndpointRejected,
    RetryContext,
    TwoStageATLExecutor,
    classify_execution_error,
    parse_handoff,
)


def response(tool_name="selected_tool", *, auth=None, expires=None):
    handoff = {
        "provider_id": "real:verified-provider",
        "provider_name": "Verified Provider",
        "capability": "summarize",
        "endpoint": "https://93.184.216.34/mcp",
        "transport": "streamable-http",
        "action": {"method": "tools/call", "tool_name": tool_name, "input_schema": {"type": "object"}},
        "auth": auth or {"required": False, "status": "not_required"},
    }
    if expires is not None:
        handoff["expires_at_ms"] = expires
    return {
        "result": {
            "structuredContent": {
                "ok": True,
                "decision_id": "decision-1",
                "outcome": {"outcome_correlation_token": "outcome-1"},
                "executable_handoff": handoff,
            }
        }
    }


def test_valid_handoff_parses():
    parsed = parse_handoff(response(expires=4102444800000))
    assert parsed.provider_id == "real:verified-provider"
    assert parsed.endpoint == "https://93.184.216.34/mcp"
    assert parsed.transport == "http"
    assert parsed.tool_name == "selected_tool"
    assert parsed.decision_reference == "decision-1"
    assert parsed.outcome_correlation_token == "outcome-1"


def test_invalid_endpoint_rejected():
    with pytest.raises(EndpointRejected):
        parse_handoff({"executable_handoff": {"provider_id": "p", "endpoint": "http://provider.example", "transport": "http", "action": {"tool_name": "x", "input_schema": {"type": "object"}}}})
    with pytest.raises(EndpointRejected):
        dynamic.validate_endpoint("http://provider.example/mcp")


def test_private_and_localhost_endpoints_rejected():
    for endpoint in ("https://localhost/mcp", "https://127.0.0.1/mcp", "https://10.0.0.2/mcp", "https://[::1]/mcp"):
        with pytest.raises(EndpointRejected):
            dynamic.validate_endpoint(endpoint)


class FakeMCPClient:
    tools = []
    seen = []
    should_fail = False

    def __init__(self, connections, **kwargs):
        type(self).seen.append((connections, kwargs))

    async def get_tools(self, *, server_name=None):
        if type(self).should_fail:
            raise RuntimeError("provider unavailable")
        return list(type(self).tools)


@tool
def selected_tool(text: str) -> str:
    """Selected provider tool."""
    return f"selected:{text}"


@tool
def unrelated_tool(text: str) -> str:
    """Unrelated provider tool."""
    return f"unrelated:{text}"


def executor(**kwargs):
    return TwoStageATLExecutor(
        decision_client=lambda task: response(),
        mcp_client_factory=FakeMCPClient,
        **kwargs,
    )


def test_exact_selected_mcp_tool_is_loaded_and_unrelated_filtered():
    FakeMCPClient.tools = [selected_tool, unrelated_tool]
    FakeMCPClient.seen = []
    loaded = asyncio.run(executor().load_selected_tool(parse_handoff(response())))
    assert loaded is selected_tool
    assert FakeMCPClient.seen[0][0]["atl-selected"]["url"] == "https://93.184.216.34/mcp"


def test_missing_auth_returns_auth_required():
    handoff = parse_handoff(response(auth={"required": True, "status": "required"}))
    with pytest.raises(AuthRequired):
        asyncio.run(executor().load_selected_tool(handoff))


def test_provider_load_failure_is_safe():
    FakeMCPClient.tools = [selected_tool]
    FakeMCPClient.should_fail = True
    try:
        with pytest.raises(DynamicLoadError):
            asyncio.run(executor().load_selected_tool(parse_handoff(response())))
    finally:
        FakeMCPClient.should_fail = False


def test_selected_tool_executes_in_execution_stage_agent():
    class LocalModel(BaseChatModel):
        calls: ClassVar[int] = 0

        @property
        def _llm_type(self):
            return "dynamic-test-model"

        @property
        def _identifying_params(self):
            return {}

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            type(self).calls += 1
            message = (
                AIMessage(content="", tool_calls=[{"name": "selected_tool", "args": {"text": "hello"}, "id": "call-1", "type": "tool_call"}])
                if type(self).calls == 1 else AIMessage(content="complete")
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

    FakeMCPClient.tools = [selected_tool, unrelated_tool]
    result = asyncio.run(executor().run("summarize this", model=LocalModel()))
    assert result["messages"][-1].content == "complete"
    assert any(message.name == "selected_tool" for message in result["messages"] if hasattr(message, "name"))


def test_decision_unavailable_can_use_original_agent():
    async def fallback(input_value):
        return {"fallback": input_value["messages"][0]["content"]}

    runner = TwoStageATLExecutor(decision_client=lambda task: (_ for _ in ()).throw(ConnectionError("down")))
    result = asyncio.run(runner.run("hello", model=object(), original_agent=fallback))
    assert result == {"fallback": "hello"}


def test_outcome_hook_can_be_called():
    calls = []
    runner = TwoStageATLExecutor(
        decision_client=lambda task: response(),
        outcome_reporter=lambda handoff, status, details: calls.append((handoff.decision_reference, status, details)) or {"accepted": True},
    )
    handoff = parse_handoff(response())
    result = asyncio.run(runner.report_outcome(handoff, "SUCCESS", {"latency_ms": 12}))
    assert result == {"accepted": True}
    assert calls == [("decision-1", "SUCCESS", {"latency_ms": 12})]


def test_expired_handoff_is_rejected():
    with pytest.raises(dynamic.HandoffError):
        parse_handoff(response(expires=1))


def test_atl_core_app_is_untouched():
    core = Path(__file__).resolve().parents[2] / "agent-traffic-lab"
    if not (core / ".git").exists():
        pytest.skip("ATL core checkout is not adjacent to this package")
    changed = subprocess.run(
        ["git", "-C", str(core), "status", "--porcelain", "--", "app.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert changed == ""


class HTTPErrorForTest(Exception):
    def __init__(self, status_code):
        self.response = SimpleNamespace(status_code=status_code)
        super().__init__(f"HTTP {status_code}")


def test_http_402_maps_to_payment_required():
    result = classify_execution_error(ExceptionGroup("tool failure", [HTTPErrorForTest(402)]))
    assert result == ExecutionResult("PAYMENT_REQUIRED", "FAILURE", "PAYMENT_REQUIRED", "payment_required", 402)


def test_outcome_receives_structured_402_evidence_and_correlation():
    calls = []
    handoff = parse_handoff(response())
    runner = TwoStageATLExecutor(
        decision_client=lambda task: response(),
        outcome_reporter=lambda current, status, details: calls.append((current, status, details)),
    )
    failure = classify_execution_error(HTTPErrorForTest(402))
    asyncio.run(runner.report_execution_result(handoff, failure))
    current, status, details = calls[0]
    assert current.decision_reference == "decision-1"
    assert current.outcome_correlation_token == "outcome-1"
    assert status == "FAILURE"
    assert details == {
        "provider_id": "real:verified-provider",
        "outcome_status": "FAILURE",
        "error_code": "PAYMENT_REQUIRED",
        "failure_type": "payment_required",
        "http_status": 402,
    }


def test_atl_client_report_outcome_preserves_structured_fields():
    captured = []
    client = object.__new__(dynamic.ATLClient)
    client._call = lambda tool, arguments, request_id: captured.append((tool, arguments)) or {"ok": True}
    handoff = parse_handoff(response())
    asyncio.run(client.report_outcome(handoff, "FAILURE", {
        "provider_id": handoff.provider_id,
        "outcome_status": "FAILURE",
        "error_code": "PAYMENT_REQUIRED",
        "failure_type": "payment_required",
        "http_status": 402,
    }))
    tool_name, arguments = captured[0]
    assert tool_name == "atl_outcome"
    assert arguments["decision_reference"] == "decision-1"
    assert arguments["outcome_correlation_token"] == "outcome-1"
    assert arguments["provider_id"] == "real:verified-provider"
    assert arguments["error_code"] == "PAYMENT_REQUIRED"
    assert arguments["failure_type"] == "payment_required"
    assert arguments["http_status"] == 402
    assert arguments["attempt_outcomes"][0]["provider_id"] == "real:verified-provider"


def test_retry_context_preserves_failed_provider_and_attempt_history():
    handoff = parse_handoff(response())
    failure = classify_execution_error(HTTPErrorForTest(402))
    context = TwoStageATLExecutor.build_retry_context(handoff, failure)
    assert isinstance(context, RetryContext)
    assert context.failed_provider_id == "real:verified-provider"
    assert context.decision_reference == "decision-1"
    assert context.outcome_correlation_token == "outcome-1"
    retry_task = context.to_decision_task("retry the same task")
    assert "real:verified-provider" in retry_task
    assert "PAYMENT_REQUIRED" in retry_task
    assert "real:other-provider" not in retry_task
    assert context.attempt_history[0]["http_status"] == 402


def test_retry_decision_client_receives_context_without_fallback_selection():
    seen = []
    runner = TwoStageATLExecutor(decision_client=lambda task: seen.append(task) or response())
    handoff = parse_handoff(response())
    failure = classify_execution_error(HTTPErrorForTest(402))
    context = runner.build_retry_context(handoff, failure)
    asyncio.run(runner.decide("same task", retry_context=context))
    assert len(seen) == 1
    assert "failed_provider_id" in seen[0]
    assert "real:verified-provider" in seen[0]
    assert "real:other-provider" not in seen[0]


def test_unknown_execution_error_stays_unknown():
    result = classify_execution_error(RuntimeError("provider returned an unusable response"))
    assert result.result_class == "UNKNOWN_FAILURE"
    assert result.error_code == "UNKNOWN_FAILURE"
    assert result.failure_type == "unknown_failure"
    assert result.http_status is None


def test_run_automatically_reports_structured_execution_failure():
    calls = []

    class FailingExecutor(TwoStageATLExecutor):
        async def _execute_stage(self, handoff, task, model):
            raise ExceptionGroup("provider call", [HTTPErrorForTest(402)])

    runner = FailingExecutor(
        decision_client=lambda task: response(),
        outcome_reporter=lambda handoff, status, details: calls.append((handoff, status, details)),
    )
    with pytest.raises(ExceptionGroup):
        asyncio.run(runner.run("same task", model=object()))
    handoff, status, details = calls[0]
    assert handoff.provider_id == "real:verified-provider"
    assert status == "FAILURE"
    assert details["error_code"] == "PAYMENT_REQUIRED"
    assert details["failure_type"] == "payment_required"
    assert details["http_status"] == 402
