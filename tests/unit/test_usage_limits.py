"""Limites d'usage quotidiennes par utilisateur (questions, envois)."""

from __future__ import annotations

import threading
from datetime import date

from doc_assistant.services.usage_limits import DailyUsageLimiter


class Clock:
    def __init__(self) -> None:
        self.day = date(2026, 10, 21)

    def __call__(self) -> date:
        return self.day


def test_each_user_and_kind_has_its_own_daily_budget() -> None:
    limiter = DailyUsageLimiter({"question": 2, "upload": 1}, today=Clock())

    assert [limiter.try_consume("alice", "question") for _ in range(3)] == [True, True, False]
    assert limiter.try_consume("bob", "question")  # le budget d'Alice ne touche pas Bob
    assert limiter.try_consume("alice", "upload")  # ni ses envois
    assert not limiter.try_consume("alice", "upload")
    assert limiter.remaining("alice", "question") == 0
    assert limiter.remaining("bob", "question") == 1


def test_a_refused_use_is_not_counted() -> None:
    limiter = DailyUsageLimiter({"question": 1, "upload": 1}, today=Clock())
    limiter.try_consume("alice", "question")

    for _ in range(5):
        assert not limiter.try_consume("alice", "question")
    assert limiter.remaining("alice", "question") == 0


def test_budgets_reset_the_next_day() -> None:
    clock = Clock()
    limiter = DailyUsageLimiter({"question": 1, "upload": 1}, today=clock)
    limiter.try_consume("alice", "question")
    assert not limiter.try_consume("alice", "question")

    clock.day = date(2026, 10, 22)

    assert limiter.try_consume("alice", "question")


def test_concurrent_uses_never_exceed_the_limit() -> None:
    limiter = DailyUsageLimiter({"question": 50, "upload": 1}, today=Clock())
    granted: list[bool] = []
    lock = threading.Lock()

    def ask() -> None:
        for _ in range(20):
            ok = limiter.try_consume("alice", "question")
            with lock:
                granted.append(ok)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert granted.count(True) == 50 and len(granted) == 160
