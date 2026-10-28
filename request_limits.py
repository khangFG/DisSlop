import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional, Tuple


MAX_CONCURRENT_REQUESTS = 10
MAX_USER_CONCURRENT_REQUESTS = 2
MAX_WORKER_QUEUE_SIZE = 50
MAX_TOOL_CALLS_PER_TURN = 20
MAX_TOTAL_TOOL_OUTPUT_CHARS = 250000
REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_COOLDOWN_SECONDS = 10.0


class RequestLimitError(RuntimeError):
    """Expected admission refusal, safe to show to the requesting user."""


class RequestLimiter:
    """
    Event-loop task-safe admission limiter managing global concurrency,
    per-user active jobs, and monotonic expiring cooldowns.
    """

    def __init__(
        self,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_per_user: int = MAX_USER_CONCURRENT_REQUESTS,
        max_global: int = MAX_CONCURRENT_REQUESTS,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_per_user = max_per_user
        self.max_global = max_global

        self._lock = asyncio.Lock()
        self._user_last_timestamp: Dict[str, float] = {}
        self._user_active_jobs: Dict[str, int] = {}
        self._active_jobs = 0

    def check_and_record_cooldown(self, user_id: str) -> Tuple[bool, str]:
        """
        Validates whether user is off cooldown and records current monotonic timestamp.
        Prunes expired timestamps to keep memory bounded.
        """
        now = time.monotonic()
        # Prune expired entries
        expired = [uid for uid, ts in self._user_last_timestamp.items() if now - ts >= self.cooldown_seconds * 2]
        for uid in expired:
            self._user_last_timestamp.pop(uid, None)

        last_ts = self._user_last_timestamp.get(user_id)
        if last_ts is not None:
            elapsed = now - last_ts
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                return False, f"⏳ Please wait {remaining:.1f}s before sending another request."

        self._user_last_timestamp[user_id] = now
        return True, ""

    @asynccontextmanager
    async def acquire(self, user_id: str) -> AsyncIterator[None]:
        """
        Acquires an admission slot for global and per-user concurrency.
        Releases all slots on any exception, return, or cancellation.
        """
        async with self._lock:
            if self._active_jobs >= self.max_global:
                raise RequestLimitError("The bot is busy at capacity. Please try again shortly.")
            active = self._user_active_jobs.get(user_id, 0)
            if active >= self.max_per_user:
                raise RequestLimitError(
                    f"⚠️ You already have {active} active request(s) processing. Please wait for them to finish."
                )
            self._user_active_jobs[user_id] = active + 1
            self._active_jobs += 1

        try:
            yield
        finally:
            async with self._lock:
                self._active_jobs -= 1
                current = self._user_active_jobs.get(user_id, 1)
                if current <= 1:
                    self._user_active_jobs.pop(user_id, None)
                else:
                    self._user_active_jobs[user_id] = current - 1
