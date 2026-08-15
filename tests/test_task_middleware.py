import asyncio

import pytest
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from langchain_agenttrafficlab import ATLHandoff, ExecutionResult, TwoStageATLExecutor
from langchain_agenttrafficlab.task_middleware import ATLTaskMiddleware


def _handoff(provider_id="real:provider-a", tool_name="tool_a", decision_ref="decision-1", token="outcome-1"):
    return ATLHandoff(
        provider_id=provider_id,
        provider_name=provider_id,
        capability="summarize",
        endpoint="https://93.184.216.34/mcp",
        transport="http",
        tool_name=tool_name,
        input_schema={"type": "object"},
        auth_required=False,
        auth_status="not_required",
        decision_reference=decision_ref,
        outcome_correlation_token=token,
    )


@tool
def tool_a(text: str) -> str:
    """Provider A's tool."""
    return f"a:{text}"


@tool
def tool_b(text: str) -> str:
    """Provider B's tool."""
    return f"b:{text}"


@tool
def tool_b_failing(text: str) -> str:
    """Provider B's tool, configured to also fail (for retry-cap testing)."""
    raise RuntimeError("provider B also failed")


class FakeExecutor(TwoStageATLExecutor):
    """A TwoStageATLExecutor stand-in that avoids any real network/MCP calls."""

    def __init__(self, *, decisions=None, tools=None, decide_calls=None, outcome_calls=None):
        # Bypass the real __init__: this fake never touches HTTP/MCP.
        self.decisions = list(decisions or [])
        self.tools = list(tools or [])
        self.decide_calls = decide_calls if decide_calls is not None else []
        self.outcome_calls = outcome_calls if outcome_calls is not None else []
        self.timeout = 5.0

    async def decide(self, task, retry_context=None):
        self.decide_calls.append((task, retry_context))
        if not self.decisions:
            raise AssertionError("no more decisions configured")
        result = self.decisions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def load_selected_tool(self, handoff):
        if not self.tools:
            raise AssertionError("no more tools configured")
        result = self.tools.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def report_execution_result(self, handoff, result):
        self.outcome_calls.append((handoff, result))
        return None


class FakeRequest:
    """Minimal stand-in for LangChain's ModelRequest/ToolCallRequest."""

    def __init__(self, *, messages=(), tools=(), tool_call=None, runtime=None):
        self.messages = list(messages)
        self.tools = list(tools)
        self.tool_call = tool_call
        self.runtime = runtime

    def override(self, **overrides):
        new = FakeRequest(
            messages=overrides.get("messages", self.messages),
            tools=overrides.get("tools", self.tools),
            tool_call=overrides.get("tool_call", self.tool_call),
            runtime=self.runtime,
        )
        return new


async def _handler_returns(value):
    async def handler(request):
        return value
    return handler


def test_decision_hook_calls_atl_decide_once_and_injects_tool():
    fake = FakeExecutor(decisions=[_handoff()], tools=[tool_a])
    mw = ATLTaskMiddleware(executor=fake)
    request = FakeRequest(messages=[HumanMessage(content="do the task")], runtime=object())
    captured = {}

    async def handler(req):
        captured["tools"] = req.tools
        return "ok"

    result = asyncio.run(mw.awrap_model_call(request, handler))
    assert result == "ok"
    assert captured["tools"] == [tool_a]
    assert len(fake.decide_calls) == 1
    assert fake.decide_calls[0][0] == "do the task"


def test_need_discovery_selection_collapse_into_one_decision():
    fake = FakeExecutor(decisions=[_handoff()], tools=[tool_a])
    mw = ATLTaskMiddleware(executor=fake)
    runtime = object()
    request = FakeRequest(messages=[HumanMessage(content="task")], runtime=runtime)

    async def handler(req):
        return "ok"

    asyncio.run(mw.awrap_model_call(request, handler))
    assert len(fake.decide_calls) == 1


def test_duplicate_trigger_dedupe_same_attempt():
    fake = FakeExecutor(decisions=[_handoff()], tools=[tool_a, tool_a])
    mw = ATLTaskMiddleware(executor=fake)
    runtime = object()
    request = FakeRequest(messages=[HumanMessage(content="task")], runtime=runtime)

    async def handler(req):
        return "ok"

    asyncio.run(mw.awrap_model_call(request, handler))
    # A second model-call phase within the same attempt (same runtime) must not
    # trigger a second atl_decide call.
    asyncio.run(mw.awrap_model_call(request, handler))
    assert len(fake.decide_calls) == 1


def test_no_unnecessary_atl_call_when_tools_already_present():
    fake = FakeExecutor()
    mw = ATLTaskMiddleware(executor=fake)
    request = FakeRequest(messages=[HumanMessage(content="task")], tools=[tool_a])

    async def handler(req):
        return req.tools

    result = asyncio.run(mw.awrap_model_call(request, handler))
    assert result == [tool_a]
    assert fake.decide_calls == []


def test_no_provider_returned_graceful_fallback():
    fake = FakeExecutor(decisions=[ExceptionForATLUnavailable()])
    mw = ATLTaskMiddleware(executor=fake)
    request = FakeRequest(messages=[HumanMessage(content="task")], runtime=object())

    async def handler(req):
        return req.tools

    result = asyncio.run(mw.awrap_model_call(request, handler))
    assert result == []
    assert len(fake.decide_calls) == 1


def ExceptionForATLUnavailable():
    from langchain_agenttrafficlab import ATLUnavailable
    return ATLUnavailable("no provider")


def test_successful_attempt_outcome_reported():
    fake = FakeExecutor(decisions=[_handoff()], tools=[tool_a])
    mw = ATLTaskMiddleware(executor=fake)
    runtime = object()
    model_request = FakeRequest(messages=[HumanMessage(content="task")], runtime=runtime)

    async def model_handler(req):
        return None

    asyncio.run(mw.awrap_model_call(model_request, model_handler))

    tool_call = {"name": "tool_a", "args": {"text": "x"}, "id": "call-1"}
    tool_request = FakeRequest(tool_call=tool_call, runtime=runtime)

    async def tool_handler(req):
        return "tool-result"

    result = asyncio.run(mw.awrap_tool_call(tool_request, tool_handler))
    assert result == "tool-result"
    assert len(fake.outcome_calls) == 1
    handoff, outcome_result = fake.outcome_calls[0]
    assert outcome_result.outcome_status == "SUCCESS"
    assert handoff.provider_id == "real:provider-a"


def test_failed_attempt_outcome_includes_failed_provider_and_no_alternate():
    fake = FakeExecutor(decisions=[_handoff(), ExceptionForATLUnavailable()], tools=[tool_a])
    mw = ATLTaskMiddleware(executor=fake)
    runtime = object()
    model_request = FakeRequest(messages=[HumanMessage(content="task")], runtime=runtime)

    async def model_handler(req):
        return None

    asyncio.run(mw.awrap_model_call(model_request, model_handler))

    tool_call = {"name": "tool_a", "args": {"text": "x"}, "id": "call-1"}
    tool_request = FakeRequest(tool_call=tool_call, runtime=runtime)

    async def tool_handler(req):
        raise TimeoutError("provider timed out")

    with pytest.raises(TimeoutError):
        asyncio.run(mw.awrap_tool_call(tool_request, tool_handler))

    assert len(fake.outcome_calls) == 1
    handoff, result = fake.outcome_calls[0]
    assert handoff.provider_id == "real:provider-a"
    assert result.outcome_status == "FAILURE"
    # No alternate provider was available (decide() raised ATLUnavailable), so
    # exactly one redecision attempt occurred and no further looping happened.
    assert len(fake.decide_calls) == 2


def test_one_redecision_and_alternate_provider_retry_produces_two_outcomes():
    fake = FakeExecutor(
        decisions=[_handoff(provider_id="real:provider-a", tool_name="tool_a"), _handoff(provider_id="real:provider-b", tool_name="tool_b", decision_ref="decision-2", token="outcome-2")],
        tools=[tool_a, tool_b],
    )
    mw = ATLTaskMiddleware(executor=fake)
    runtime = object()
    model_request = FakeRequest(messages=[HumanMessage(content="task")], runtime=runtime)

    async def model_handler(req):
        return None

    asyncio.run(mw.awrap_model_call(model_request, model_handler))

    tool_call = {"name": "tool_a", "args": {"text": "x"}, "id": "call-1"}
    tool_request = FakeRequest(tool_call=tool_call, runtime=runtime)

    async def tool_handler(req):
        raise RuntimeError("provider A failed")

    result = asyncio.run(mw.awrap_tool_call(tool_request, tool_handler))
    assert result.content == "b:x"
    assert len(fake.decide_calls) == 2
    assert len(fake.outcome_calls) == 2
    first_handoff, first_result = fake.outcome_calls[0]
    second_handoff, second_result = fake.outcome_calls[1]
    assert first_handoff.provider_id == "real:provider-a"
    assert first_result.outcome_status == "FAILURE"
    assert second_handoff.provider_id == "real:provider-b"
    assert second_result.outcome_status == "SUCCESS"
    # Decision/attempt correlation preserved across the redecision.
    assert first_handoff.decision_reference == "decision-1"
    assert second_handoff.decision_reference == "decision-2"


def test_retry_cap_is_exactly_one():
    fake = FakeExecutor(
        decisions=[_handoff(provider_id="real:provider-a", tool_name="tool_a"), _handoff(provider_id="real:provider-b", tool_name="tool_b_failing", decision_ref="decision-2", token="outcome-2")],
        tools=[tool_a, tool_b_failing],
    )
    mw = ATLTaskMiddleware(executor=fake)
    runtime = object()
    model_request = FakeRequest(messages=[HumanMessage(content="task")], runtime=runtime)

    async def model_handler(req):
        return None

    asyncio.run(mw.awrap_model_call(model_request, model_handler))

    tool_call = {"name": "tool_a", "args": {"text": "x"}, "id": "call-1"}
    tool_request = FakeRequest(tool_call=tool_call, runtime=runtime)

    async def failing_handler(req):
        raise RuntimeError("provider A failed")

    with pytest.raises(RuntimeError):
        asyncio.run(mw.awrap_tool_call(tool_request, failing_handler))

    # Exactly one redecision (2 total decide calls), never a third.
    assert len(fake.decide_calls) == 2
    assert len(fake.outcome_calls) == 2
    assert fake.outcome_calls[0][1].outcome_status == "FAILURE"
    assert fake.outcome_calls[1][1].outcome_status == "FAILURE"


def test_dynamic_selected_tool_injection_uses_streamable_http_transport():
    handoff = _handoff()
    assert handoff.transport == "http"
    fake = FakeExecutor(decisions=[handoff], tools=[tool_a])
    mw = ATLTaskMiddleware(executor=fake)
    request = FakeRequest(messages=[HumanMessage(content="task")], runtime=object())

    async def handler(req):
        return req.tools

    injected = asyncio.run(mw.awrap_model_call(request, handler))
    assert injected == [tool_a]


def test_tenant_and_session_context_untouched_by_middleware():
    # The middleware never reads/writes tenant_id or caller_id; those remain
    # entirely owned by the ATLClient/executor it wraps.
    fake = FakeExecutor(decisions=[_handoff()], tools=[tool_a])
    fake.tenant_marker = "tenant-a"
    mw = ATLTaskMiddleware(executor=fake)
    request = FakeRequest(messages=[HumanMessage(content="task")], runtime=object())

    async def handler(req):
        return None

    asyncio.run(mw.awrap_model_call(request, handler))
    assert fake.tenant_marker == "tenant-a"
