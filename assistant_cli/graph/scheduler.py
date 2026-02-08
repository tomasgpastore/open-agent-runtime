from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from assistant_cli.graph.state_store import GraphScheduleRecord, GraphStateStore


@dataclass(slots=True)
class ScheduleTriggerResult:
    schedule_id: str
    graph_id: str
    status: str
    detail: str


class GraphScheduler:
    """In-app cron scheduler for graph execution."""

    def __init__(
        self,
        *,
        state_store: GraphStateStore,
        execute_callback: Callable[[str, str, object, str], Awaitable[object]],
    ) -> None:
        self._state_store = state_store
        self._execute_callback = execute_callback

    def add_schedule(
        self,
        *,
        graph_id: str,
        cron_expr: str,
        guarantee_mode: str,
        input_payload: object,
        enabled: bool = True,
        now: datetime | None = None,
    ) -> str:
        anchor = now or datetime.now().astimezone()
        next_run = self._next_run_after(anchor, cron_expr)
        return self._state_store.create_schedule(
            graph_id=graph_id,
            cron_expr=cron_expr,
            guarantee_mode=guarantee_mode,
            input_payload=input_payload,
            next_run_at=next_run,
            enabled=enabled,
        )

    def list_schedules(self, enabled_only: bool = False, limit: int = 200) -> list[GraphScheduleRecord]:
        return self._state_store.list_schedules(enabled_only=enabled_only, limit=limit)

    def set_enabled(self, schedule_id: str, enabled: bool) -> None:
        self._state_store.set_schedule_enabled(schedule_id, enabled)

    def delete(self, schedule_id: str) -> None:
        self._state_store.delete_schedule(schedule_id)

    async def trigger_due(self, now: datetime | None = None, limit: int = 20) -> list[ScheduleTriggerResult]:
        ts = now or datetime.now().astimezone()
        due = self._state_store.due_schedules(ts, limit=limit)
        results: list[ScheduleTriggerResult] = []

        for schedule in due:
            results.append(await self._trigger(schedule=schedule, now=ts))

        return results

    async def trigger(self, schedule_id: str, now: datetime | None = None) -> ScheduleTriggerResult:
        schedule = self._state_store.get_schedule(schedule_id)
        if schedule is None:
            raise ValueError(f"Schedule '{schedule_id}' was not found.")
        ts = now or datetime.now().astimezone()
        return await self._trigger(schedule=schedule, now=ts)

    async def _trigger(self, *, schedule: GraphScheduleRecord, now: datetime) -> ScheduleTriggerResult:
        try:
            await self._execute_callback(
                schedule.graph_id,
                schedule.guarantee_mode,
                schedule.input_payload,
                schedule.schedule_id,
            )
            last_error = None
            status = "ok"
            detail = "Run triggered successfully."
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            status = "error"
            detail = last_error

        next_run = self._next_run_after(now, schedule.cron_expr)
        self._state_store.update_schedule_after_run(
            schedule_id=schedule.schedule_id,
            next_run_at=next_run,
            last_error=last_error,
            last_run_at=now,
        )

        return ScheduleTriggerResult(
            schedule_id=schedule.schedule_id,
            graph_id=schedule.graph_id,
            status=status,
            detail=detail,
        )

    def _next_run_after(self, now: datetime, cron_expr: str) -> datetime:
        minute_bucket = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for offset in range(0, 60 * 24 * 370):
            candidate = minute_bucket + timedelta(minutes=offset)
            if self._matches_cron(candidate, cron_expr):
                return candidate
        raise ValueError(f"Could not compute next run for cron expression '{cron_expr}'.")

    def _matches_cron(self, dt: datetime, cron_expr: str) -> bool:
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression '{cron_expr}'. Expected 5 fields: minute hour day month weekday"
            )

        minute, hour, day, month, weekday = parts

        cron_weekday = (dt.weekday() + 1) % 7
        if not self._matches_field(dt.minute, minute, 0, 59):
            return False
        if not self._matches_field(dt.hour, hour, 0, 23):
            return False
        if not self._matches_field(dt.day, day, 1, 31):
            return False
        if not self._matches_field(dt.month, month, 1, 12):
            return False
        if not self._matches_field(cron_weekday, weekday, 0, 7, normalize_weekday=True):
            return False
        return True

    def _matches_field(
        self,
        value: int,
        pattern: str,
        minimum: int,
        maximum: int,
        normalize_weekday: bool = False,
    ) -> bool:
        allowed = self._parse_field(pattern, minimum, maximum, normalize_weekday=normalize_weekday)
        return value in allowed

    def _parse_field(
        self,
        pattern: str,
        minimum: int,
        maximum: int,
        normalize_weekday: bool = False,
    ) -> set[int]:
        normalized_pattern = pattern.strip()
        if normalized_pattern == "*":
            return set(range(minimum, maximum + 1))

        values: set[int] = set()
        for token in normalized_pattern.split(","):
            token = token.strip()
            if not token:
                continue

            if token.startswith("*/"):
                step = int(token[2:])
                if step <= 0:
                    raise ValueError(f"Invalid cron step in token '{token}'.")
                for value in range(minimum, maximum + 1):
                    if (value - minimum) % step == 0:
                        values.add(value)
                continue

            if "-" in token:
                start_str, end_str = token.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    raise ValueError(f"Invalid cron range '{token}'.")
                for value in range(start, end + 1):
                    values.add(self._normalize_field_value(value, normalize_weekday))
                continue

            raw_value = int(token)
            values.add(self._normalize_field_value(raw_value, normalize_weekday))

        for item in values:
            if item < minimum or item > maximum:
                raise ValueError(
                    f"Cron value '{item}' out of bounds [{minimum}, {maximum}] for pattern '{pattern}'."
                )

        if not values:
            raise ValueError(f"Cron field '{pattern}' resolved to empty value set.")

        return values

    def _normalize_field_value(self, value: int, normalize_weekday: bool) -> int:
        if normalize_weekday and value == 7:
            return 0
        return value
