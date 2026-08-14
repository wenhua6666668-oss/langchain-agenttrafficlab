from types import SimpleNamespace
import time

from langchain_agenttrafficlab import ATLMiddleware


class Tool:
    def __init__(self, name):
        self.name = name


class Request:
    def __init__(self, runtime, tools, task="lookup"):
        self.runtime = runtime
        self.tools = tools
        self.messages = [SimpleNamespace(content=task)]

    def override(self, **changes):
        result = Request(self.runtime, self.tools)
        result.messages = self.messages
        for key, value in changes.items():
            setattr(result, key, value)
        return result


def run(middleware, request):
    seen = []
    middleware.wrap_model_call(request, lambda current: seen.append(current.tools) or "ok")
    return seen[0]


def test_atl_selects_existing_registered_tool():
    calls = []
    middleware = ATLMiddleware(decision_client=lambda task, tools: calls.append(tools) or {"selected_tool": "search"})
    tools = [Tool("search"), Tool("calculator")]
    assert [tool.name for tool in run(middleware, Request(object(), tools))] == ["search"]
    assert calls == [["search", "calculator"]]


def test_atl_failure_fails_open():
    middleware = ATLMiddleware(decision_client=lambda task, tools: (_ for _ in ()).throw(RuntimeError("down")))
    tools = [Tool("search"), Tool("calculator")]
    assert run(middleware, Request(object(), tools)) == tools


def test_timeout_fails_open():
    def slow_decide(task, tools):
        time.sleep(0.2)
        return {"selected_tool": "search"}

    middleware = ATLMiddleware(timeout=0.01, decision_client=slow_decide)
    tools = [Tool("search"), Tool("calculator")]
    assert run(middleware, Request(object(), tools)) == tools


def test_atl_cannot_inject_unregistered_tool():
    middleware = ATLMiddleware(decision_client=lambda task, tools: {"selected_tool": "not_registered"})
    tools = [Tool("search")]
    assert run(middleware, Request(object(), tools)) == tools


def test_cache_prevents_repeated_calls_within_invocation():
    calls = []
    runtime = object()
    middleware = ATLMiddleware(decision_client=lambda task, tools: calls.append(1) or {"selected_tool": "search"})
    tools = [Tool("search"), Tool("calculator")]
    assert [tool.name for tool in run(middleware, Request(runtime, tools))] == ["search"]
    assert [tool.name for tool in run(middleware, Request(runtime, tools))] == ["search"]
    assert len(calls) == 1
