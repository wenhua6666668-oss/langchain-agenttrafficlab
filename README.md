# langchain-agenttrafficlab

`langchain-agenttrafficlab` connects a LangChain/LangGraph application to the Agent Traffic Lab (ATL) decision layer. ATL chooses an eligible, executable provider before the adapter loads and runs the provider's MCP tool.

This is a decision integration, not a normal wrapper that exposes `atl_decide` as an agent tool. The adapter keeps ATL at the provider-selection boundary, validates the returned execution contract, and exposes only the exact tool ATL selected.

## Install

```bash
python -m pip install langchain-agenttrafficlab
```

The package supports Python 3.10+ and uses LangChain 1.x, `langchain-mcp-adapters` 0.3.x, and `requests`.

## Minimal Usage

For v0.1, use the middleware with tools that are already registered on the agent:

```python
from langchain.agents import create_agent
from langchain_agenttrafficlab import ATLMiddleware

agent = create_agent(
    model=model,
    tools=[search_tool, calculator_tool],
    middleware=[ATLMiddleware(timeout=1.5)],
)
```

ATL can narrow the current registered tool set, but v0.1 does not inject unknown providers or tools.

## Dynamic Provider Flow

The v0.2 path is deliberately two-stage. It does not inject a tool into an agent that has already been created:

```text
task
  -> ATL atl_decide
  -> validate executable handoff
  -> connect to the ATL-returned MCP endpoint
  -> load the exact selected tool
  -> create the execution-stage agent
  -> execute
  -> atl_outcome
```

```python
from langchain_agenttrafficlab import ATLClient, TwoStageATLExecutor

atl = ATLClient(timeout=3.0)
executor = TwoStageATLExecutor(
    decision_client=atl.decide,
    outcome_reporter=atl.report_outcome,
)

result = await executor.run(
    "Extract the title from a public webpage into JSON.",
    model=model,
    original_agent=existing_agent,
)
```

ATL remains the decision authority. The adapter never hard-codes a fallback provider, maps providers by name or semantic similarity, or treats caller-local tools as ATL candidates. Dynamic loading is limited to ATL-known providers with a validated executable handoff.

## Failures And Failover

ATL decision failure may fail open to `original_agent` when the caller supplies a safe fallback. A malformed, expired, unverifiable, or security-rejected handoff fails closed for dynamic loading. Provider connection or tool-loading failure never substitutes an unverified tool.

Execution errors are preserved and classified using existing ATL outcome fields. For example, HTTP 402 becomes `PAYMENT_REQUIRED` with `failure_type=payment_required` and `http_status=402`. The caller can create retry context from the failed handoff and ask ATL for a fresh decision:

```python
from langchain_agenttrafficlab import TwoStageATLExecutor, classify_execution_error

failure = classify_execution_error(provider_exception)
retry_context = executor.build_retry_context(handoff, failure)
next_handoff = await executor.decide(task, retry_context=retry_context)
```

ATL chooses any next provider. The adapter validates the new handoff and does not force a particular alternative.

## Outcome Reporting

`ATLClient.report_outcome` preserves the decision reference, outcome correlation token, provider identity, failure code, failure type, HTTP status, and attempt history. For custom reporters, the same payload is available through `executor.report_execution_result(handoff, result)`.

Credentials, if required by the selected provider, must come from an explicit caller-supplied credential provider:

```python
async def credentials_for(handoff):
    return {"Authorization": caller_managed_authorization}

executor = TwoStageATLExecutor(
    decision_client=atl.decide,
    credential_provider=credentials_for,
    outcome_reporter=atl.report_outcome,
)
```

This package does not invent, persist, or log credentials. Payment and authentication requirements are never bypassed.

## Security Model

- Only validated HTTPS MCP endpoints returned by ATL are used.
- User or model text cannot override the provider identity, endpoint, transport, or selected tool.
- Localhost, loopback, private, link-local, reserved, and other non-public endpoint addresses are rejected.
- The execution-stage agent receives only the exact ATL-selected tool; unrelated tools from the provider are filtered out.
- Credentials are caller-supplied only and are not persisted or logged by this package.
- Connection, loading, and execution work is bounded by short timeouts.
- Provider failures, including payment and auth failures, are reported truthfully and are not silently converted into success.

## Limitations

ATL selects from its verified, executable provider universe. This package does not register caller-local providers with ATL, invent provider identities, or claim that a natural-language match is an authorization to connect. Automatic reputation processing and broader fallback policy remain ATL responsibilities.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
```

License: MIT.
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
