from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


HookCallback = Callable[[dict[str, Any]], object | Awaitable[object]]


@dataclass(slots=True)
class HookInvocation:
    event: str
    callback_name: str
    result: object | None


class GraphHookRegistry:
    """Lifecycle hook registry for graph execution."""

    def __init__(self) -> None:
        self._callbacks: dict[str, list[HookCallback]] = defaultdict(list)

    def register(self, event: str, callback: HookCallback) -> None:
        self._callbacks[event].append(callback)

    def clear(self, event: str | None = None) -> None:
        if event is None:
            self._callbacks.clear()
            return
        self._callbacks.pop(event, None)

    def callbacks_for(self, event: str) -> list[HookCallback]:
        return list(self._callbacks.get(event, []))

    async def emit(self, event: str, context: dict[str, Any]) -> list[HookInvocation]:
        invocations: list[HookInvocation] = []
        for callback in self._callbacks.get(event, []):
            callback_name = getattr(callback, "__name__", callback.__class__.__name__)
            result = callback(context)
            if inspect.isawaitable(result):
                result = await result
            invocations.append(
                HookInvocation(
                    event=event,
                    callback_name=str(callback_name),
                    result=result,
                )
            )
        return invocations


DEFAULT_HOOK_EVENTS = {
    "before_run",
    "after_run",
    "before_node",
    "after_node",
    "on_error",
}
