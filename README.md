# langchain-agenttrafficlab

A local LangChain adapter for the Agent Traffic Lab decision layer. Version 0.2 adds a bounded two-stage path for ATL-selected, executable MCP providers.

## Scope

`ATLMiddleware` uses LangChain's `wrap_model_call` hook to ask ATL which tool to use, then narrows the tools already registered on the current model request. The normal LangChain agent loop continues unchanged.

v0.1 is deliberately **not** dynamic discovery: ATL cannot inject an unknown provider or an unregistered tool. ATL failure and timeout fail open by default and preserve the original registered tools. There are no hidden retries or infinite loops. Decisions are cached per agent invocation/task.

```python
from langchain.agents import create_agent
from langchain_agenttrafficlab import ATLMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool, calculator_tool],
    middleware=[ATLMiddleware(timeout=1.5)],
)
```

The v0.1 path is:

`task -> ATL decision -> narrow/rank existing registered tools -> normal LangChain execution`

The v0.2 dynamic path is a two-stage orchestration. It does not inject a tool into an already-created agent:

`task -> atl_decide -> validate executable_handoff -> load exact MCP tool -> create execution agent -> execute -> atl_outcome`

```python
from langchain_agenttrafficlab import ATLClient, TwoStageATLExecutor

atl = ATLClient(timeout=3.0)
executor = TwoStageATLExecutor(decision_client=atl.decide)
result = await executor.run(task, model=model, original_agent=existing_agent)
```

Only ATL-returned HTTPS MCP endpoints and the exact ATL-returned tool name are accepted. Private/local endpoints, expired or malformed handoffs, unrelated provider tools, missing required credentials, and provider load failures are rejected. Credentials must come from an explicit caller callback and are never persisted by this package.

When execution is complete, the caller can report the selected provider using `await executor.report_outcome(handoff, "SUCCESS", details)` with an outcome reporter such as `atl.report_outcome`.

When an execution-stage MCP error is raised and an outcome reporter is configured, `run` preserves the original exception while automatically propagating existing ATL fields such as `error_code`, `failure_type`, `http_status`, provider identity, decision reference, outcome token, and `attempt_outcomes`. HTTP 402 is reported as `PAYMENT_REQUIRED`; unknown errors remain `UNKNOWN_FAILURE`.

The v0.2 dynamic path does not claim caller-local tools are ATL candidates, does not infer provider identity from names, and does not provide automatic fallback or reputation management.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
```

This package is not published by this repository. License: MIT.
