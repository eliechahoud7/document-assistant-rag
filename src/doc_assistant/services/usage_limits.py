"""Limites d'usage quotidiennes par utilisateur connecté (questions, envois).

Protège le coût (appels Gemini payants, écritures Chroma Cloud, calcul des embeddings) contre
un usage excessif d'un compte. Les compteurs sont gardés en mémoire dans le processus et
remis à zéro chaque jour (UTC) : un redémarrage de l'application les remet aussi à zéro. C'est
un garde-fou de coût, pas une garantie contractuelle.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Literal

UsageKind = Literal["question", "upload"]


def _today() -> date:
    return datetime.now(UTC).date()


class DailyUsageLimiter:
    def __init__(self, limits: dict[UsageKind, int], *, today: Callable[[], date] = _today) -> None:
        self._limits = dict(limits)
        self._today = today
        self._day = today()
        self._used: dict[tuple[str, UsageKind], int] = {}
        self._lock = threading.Lock()

    def try_consume(self, user_key: str, kind: UsageKind) -> bool:
        """Compte une utilisation et renvoie ``True``, ou ``False`` (rien n'est compté) si la
        limite du jour est atteinte pour cet utilisateur."""
        with self._lock:
            self._roll_over()
            used = self._used.get((user_key, kind), 0)
            if used >= self._limits[kind]:
                return False
            self._used[(user_key, kind)] = used + 1
            return True

    def remaining(self, user_key: str, kind: UsageKind) -> int:
        with self._lock:
            self._roll_over()
            return max(0, self._limits[kind] - self._used.get((user_key, kind), 0))

    def limit(self, kind: UsageKind) -> int:
        return self._limits[kind]

    def _roll_over(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._used = today, {}
