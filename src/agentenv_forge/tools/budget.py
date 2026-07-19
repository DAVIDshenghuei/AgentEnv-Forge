from threading import Lock


class ActionBudgetExhaustedError(ValueError):
    """The episode action budget has been exhausted."""


class ActionBudget:
    """A thread-safe, monotonically consumed episode action budget."""

    __slots__ = ("_limit", "_used", "_lock")

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("invalid action budget")
        self._limit = limit
        self._used = 0
        self._lock = Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._limit - self._used

    def charge(self) -> None:
        with self._lock:
            if self._used >= self._limit:
                raise ActionBudgetExhaustedError(
                    "episode action budget exhausted"
                )
            self._used += 1
