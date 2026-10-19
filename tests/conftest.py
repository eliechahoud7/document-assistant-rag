"""Fixtures partagées par toute la suite de tests."""

from __future__ import annotations

import socket
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn

import pytest

from pdf_factory import write_pdf

MakePdf = Callable[..., Path]


class NetworkAccessError(RuntimeError):
    """Levée quand un test tente d'ouvrir une connexion réseau."""


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interdit le réseau, sauf pour les tests marqués ``@pytest.mark.network``.

    Garantit que la suite ne dépend ni d'Internet, ni d'une API payante, ni d'un
    téléchargement de modèle caché.
    """
    if request.node.get_closest_marker("network"):
        return

    def guard(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise NetworkAccessError(
            f"Accès réseau interdit dans {request.node.nodeid} "
            "(ajoutez @pytest.mark.network si c'est voulu)."
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Fabrique de PDF de test dans un dossier temporaire propre à chaque test."""

    def factory(pages: Sequence[Sequence[str]], name: str = "test.pdf") -> Path:
        return write_pdf(tmp_path / name, pages)

    return factory
