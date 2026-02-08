from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class CredentialRecord:
    user_id: str
    provider: str
    access_token: str
    refresh_token: str | None
    expires_at: str | None


class CredentialProvider(Protocol):
    async def get_user_credential(self, user_id: str, provider: str) -> CredentialRecord | None: ...


class ExecutionBackend(Protocol):
    async def run_graph(
        self,
        *,
        graph: dict[str, Any],
        input_payload: object,
        guarantee_mode: str,
        trigger_source: str,
    ) -> object: ...

    async def replay_run(self, *, run_id: str) -> object: ...

    async def resume_run(self, *, run_id: str) -> object: ...
