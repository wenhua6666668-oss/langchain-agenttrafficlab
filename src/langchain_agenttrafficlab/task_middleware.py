"""ATLTaskMiddleware: Decision -> Execute -> Recover -> Learn at LangChain task time.

This module implements the three-execution-layer architecture on top of the
existing (unmodified) ``ATLClient`` / ``TwoStageATLExecutor`` primitives:

- Decision Hook (``awrap_model_call``): consolidates Need + Discovery +
  Selection into at most one ``atl_decide`` call per attempt.
- Recovery Hook (``awrap_tool_call``): on a genuine execution failure of the
  ATL-selected tool, reports the failed attempt and requests exactly one
  fresh ATL decision (auto-failover budget = 1) before retrying once.
- Learning Hook: every actual provider attempt (success or failure) is
  reported to ``atl_outcome`` separately.

ATL core is never modified by this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .dynamic import (
    ATLClient,
    ATLHandoff,
    ATLUnavailable,
    ExecutionResult,
    TwoStageATLExecutor,
    classify_execution_error,
)


@dataclass
class _AttemptState:
    """Bookkeeping for one task attempt's ATL decision + failover budget.

    Kept intentionally small: just enough to dedupe repeated Decision Hook
    triggers for the same attempt and to enforce the v0.1 failover budget of
    exactly one automatic redecision.
    """

    task: str
    handoff: ATLHandoff
    tool_name: str
    failover_used: bool = False
    attempt_index: int = 1
    last_result: ExecutionResult | None = None
    attempt_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class ATLTaskMiddleware(AgentMiddleware):
    """Automatic ATL routing for LangChain agents: Decision, Recovery, Learning.

    Add this middleware once; ATL then participates automatically whenever a
    task arrives with no tools already registered on the request. If tools
    are already present, ATL is not consulted (avoids unnecessary calls when
    intervention is clearly not needed).

    This middleware is async-only (``ainvoke``/``astream``), matching the
    async-native ``TwoStageATLExecutor`` it wraps.
    """

    def __init__(
        self,
        *,
        executor: TwoStageATLExecutor | None = None,
        endpoint: str = "https://mcp.agenttrafficlab.com/mcp",
        timeout: float = 5.0,
    ) -> None:
        if executor is not None:
            self._executor = executor
        else:
            atl = ATLClient(endpoint=endpoint, timeout=timeout)
            self._executor = TwoStageATLExecutor(
                decision_client=atl.decide,
                outcome_reporter=atl.report_outcome,
                timeout=timeout,
            )
        # Keyed by a per-attempt fingerprint; cleared once an attempt reaches
        # a terminal (reported) outcome.
        self._attempts: dict[Any, _AttemptState] = {}

    # ------------------------------------------------------------------
    # Decision Hook: Need + Discovery + Selection, deduped to one call.
    # ------------------------------------------------------------------

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        existing_tools = list(getattr(request, "tools", ()) or ())
        if existing_tools:
            # Need: tools are already registered/selected by the caller, so
            # ATL intervention is not needed for this request.
            return await handler(request)

        task = self._extract_task(request)
        if not task:
            return await handler(request)

        fingerprint = self._fingerprint(request)
        state = self._attempts.get(fingerprint)
        if state is not None:
            # Same attempt already has a primary decision; reuse it instead
            # of triggering a duplicate atl_decide call.
            try:
                tool = await self._executor.load_selected_tool(state.handoff)
            except Exception:
                return await handler(request)
            return await handler(request.override(tools=[tool]))

        try:
            handoff = await self._executor.decide(task)
        except ATLUnavailable:
            # Discovery: ATL returned no executable provider; preserve
            # normal LangChain behavior by leaving the request untouched.
            return await handler(request)
        try:
            tool = await self._executor.load_selected_tool(handoff)
        except Exception:
            return await handler(request)

        self._attempts[fingerprint] = _AttemptState(task=task, handoff=handoff, tool_name=tool.name)
        return await handler(request.override(tools=[tool]))

    # ------------------------------------------------------------------
    # Recovery Hook + Learning Hook.
    # ------------------------------------------------------------------

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        fingerprint = self._fingerprint(request)
        state = self._attempts.get(fingerprint)
        tool_call = getattr(request, "tool_call", None) or {}
        if state is None or tool_call.get("name") != state.tool_name:
            # Not an ATL-selected tool call; leave it to normal handling.
            return await handler(request)

        try:
            result = await handler(request)
        except Exception as exc:
            return await self._recover(state, fingerprint, tool_call, exc)

        await self._report_attempt(state, ExecutionResult("SUCCESS", "SUCCESS"))
        self._attempts.pop(fingerprint, None)
        return result

    async def _recover(self, state: _AttemptState, fingerprint: Any, tool_call: dict, exc: BaseException) -> Any:
        classified = classify_execution_error(exc)
        await self._report_attempt(state, classified)

        if state.failover_used:
            # Auto-failover budget (exactly 1) already spent; do not loop.
            self._attempts.pop(fingerprint, None)
            raise exc

        retry_context = self._executor.build_retry_context(
            state.handoff, classified, attempt_history=state.attempt_history
        )
        try:
            new_handoff = await self._executor.decide(state.task, retry_context=retry_context)
            new_tool = await self._executor.load_selected_tool(new_handoff)
        except Exception:
            # No alternate provider available; stop, do not loop further.
            self._attempts.pop(fingerprint, None)
            raise exc

        state.attempt_history = state.attempt_history + (classified.attempt_outcome(state.handoff.provider_id),)
        state.handoff = new_handoff
        state.tool_name = new_tool.name
        state.failover_used = True
        state.attempt_index = 2

        try:
            output = await new_tool.ainvoke(dict(tool_call.get("args") or {}))
        except Exception as retry_exc:
            retry_classified = classify_execution_error(retry_exc)
            await self._report_attempt(state, retry_classified)
            self._attempts.pop(fingerprint, None)
            raise retry_exc

        await self._report_attempt(state, ExecutionResult("SUCCESS", "SUCCESS"))
        self._attempts.pop(fingerprint, None)
        content = output if isinstance(output, str) else json.dumps(output, default=str)
        return ToolMessage(content=content, tool_call_id=tool_call.get("id") or "", name=new_tool.name)

    async def _report_attempt(self, state: _AttemptState, result: ExecutionResult) -> None:
        # Learning Hook: every actual provider attempt is reported separately,
        # never collapsed into a single final outcome.
        await self._executor.report_execution_result(state.handoff, result)

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_task(request: Any) -> str:
        messages = getattr(request, "messages", ()) or ()
        if not messages:
            return ""
        message = next(
            (m for m in messages if getattr(m, "type", "") in {"human", "user"}),
            messages[0],
        )
        content = getattr(message, "content", message if isinstance(message, str) else "")
        if isinstance(content, str):
            return content
        return json.dumps(content, sort_keys=True, default=str)

    @staticmethod
    def _fingerprint(request: Any) -> Any:
        # Stable across the model-call and tool-call phases of the same
        # attempt (same LangGraph invocation), without introducing any new
        # public state or changing the ATL public API.
        runtime = getattr(request, "runtime", None)
        if runtime is not None:
            return id(runtime)
        state = getattr(request, "state", None)
        if state is not None:
            return id(state)
        return id(request)
