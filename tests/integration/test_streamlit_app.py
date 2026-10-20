"""Interface Streamlit (app.py) exécutée hors ligne : faux embedder, faux index, faux LLM.

Le dossier courant est un dossier temporaire : les vrais index (data/chroma,
data/courses/chroma) ne sont jamais ouverts.
"""

from __future__ import annotations

import _socket
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import NetworkAccessError
from doc_assistant import bootstrap
from doc_assistant.config import Settings
from doc_assistant.domain.errors import LLMConfigurationError, LLMQuotaError
from doc_assistant.domain.models import Chunk
from doc_assistant.domain.ports import AnswerGenerator, Embedder
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.qa_service import QAService
from fakes import FakeEmbedder, InMemoryVectorStore, ScriptedAnswerGenerator

APP = Path(__file__).resolve().parents[2] / "app.py"
TIMEOUT_S = 60


class Pipeline:
    """Remplace les constructeurs de ``bootstrap`` ; chaque collection a son propre texte."""

    def __init__(self, script: dict[str, str | Exception]) -> None:
        self.script = script
        self.built_for: list[Path] = []
        self.generator_error: Exception | None = None

    def build_embedder(self, _settings: Settings) -> Embedder:
        return FakeEmbedder()

    def build_generator(self, _settings: Settings) -> AnswerGenerator:
        if self.generator_error is not None:
            raise self.generator_error
        return ScriptedAnswerGenerator(self.script, model_name="fake-gemini")

    def build_qa_service(
        self, settings: Settings, embedder: Embedder, generator: AnswerGenerator
    ) -> QAService:
        self.built_for.append(settings.data_dir)
        name = "cv.pdf" if settings.data_dir == Path("data") else "cours.pdf"
        text = f"Contenu de {name} : les chats dorment beaucoup."
        store = InMemoryVectorStore()
        store.add([Chunk(f"{name}:0", name, name, 2, 0, 0, text)], embedder.embed_documents([text]))
        return QAService(Retriever(embedder, store, default_top_k=1), generator)


@pytest.fixture
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sous Windows, la boucle asyncio de Streamlit crée un canal interne sur 127.0.0.1 :
    on l'autorise, tout autre accès réseau reste interdit.
    """

    def guarded(original: Any) -> Any:
        def connect(sock: socket.socket, address: Any) -> Any:
            if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
                return original(sock, address)
            raise NetworkAccessError(f"accès réseau interdit : {address!r}")

        return connect

    monkeypatch.setattr(socket.socket, "connect", guarded(_socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(_socket.socket.connect_ex))


@pytest.fixture
def pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loopback_only: None
) -> Iterator[Pipeline]:
    for data_dir in ("data", "data/courses"):
        (tmp_path / data_dir / "chroma").mkdir(parents=True)
        (tmp_path / data_dir / "chroma" / "chroma.sqlite3").touch()
    monkeypatch.chdir(tmp_path)
    for name in ("DOCASSIST_DATA_DIR", "DOCASSIST_LLM_PROVIDER", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    fake = Pipeline({"Que font les chats ?": "Ils dorment [1].", "Inconnu ?": "abstain"})
    monkeypatch.setattr(bootstrap, "build_embedder", fake.build_embedder)
    monkeypatch.setattr(bootstrap, "build_generator", fake.build_generator)
    monkeypatch.setattr(bootstrap, "build_qa_service", fake.build_qa_service)
    st.cache_resource.clear()
    yield fake
    st.cache_resource.clear()


def start() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    app.run()
    assert not app.exception
    return app


def ask(app: AppTest, question: str) -> None:
    app.chat_input[0].set_value(question).run()
    assert not app.exception


def test_answer_shows_model_and_cited_source(pipeline: Pipeline) -> None:
    app = start()

    ask(app, "Que font les chats ?")

    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert "Ils dorment [1]." in app.chat_message[1].markdown[0].value
    assert any("fake-gemini" in md.value for md in app.chat_message[1].markdown)
    assert app.expander[0].label == "[1] cv.pdf — page 2"
    assert "les chats dorment" in app.expander[0].markdown[0].value
    assert pipeline.built_for == [Path("data")]


def test_abstention_is_explicit(pipeline: Pipeline) -> None:
    app = start()

    ask(app, "Inconnu ?")

    assert "pas trouvé la réponse" in app.chat_message[1].info[0].value
    assert len(app.expander) == 0


def test_switching_collection_resets_history_and_uses_its_own_index(pipeline: Pipeline) -> None:
    app = start()
    ask(app, "Que font les chats ?")

    app.sidebar.radio(key="collection").set_value("courses").run()

    assert len(app.chat_message) == 0
    ask(app, "Que font les chats ?")
    assert app.expander[0].label == "[1] cours.pdf — page 2"
    assert pipeline.built_for == [Path("data"), Path("data/courses")]

    app.sidebar.radio(key="collection").set_value("cv").run()
    ask(app, "Que font les chats ?")
    assert app.expander[0].label == "[1] cv.pdf — page 2"
    assert len(pipeline.built_for) == 2  # services en cache, chacun pour sa collection


def test_clear_button_empties_the_conversation(pipeline: Pipeline) -> None:
    app = start()
    ask(app, "Que font les chats ?")

    app.sidebar.button[0].click().run()

    assert len(app.chat_message) == 0


def test_api_error_is_shown_without_crashing(pipeline: Pipeline) -> None:
    pipeline.script["Que font les chats ?"] = LLMQuotaError("quota gratuit Gemini atteint")
    app = start()

    ask(app, "Que font les chats ?")

    assert "Quota Gemini atteint" in app.chat_message[1].error[0].value


def test_missing_configuration_disables_the_chat(pipeline: Pipeline) -> None:
    pipeline.generator_error = LLMConfigurationError(
        "variable d'environnement GEMINI_API_KEY absente"
    )

    app = start()

    assert "GEMINI_API_KEY absente" in app.error[0].value
    assert app.chat_input[0].disabled


def test_missing_index_is_never_created(pipeline: Pipeline, tmp_path: Path) -> None:
    (tmp_path / "data" / "courses" / "chroma" / "chroma.sqlite3").unlink()
    app = start()

    app.sidebar.radio(key="collection").set_value("courses").run()

    assert "index introuvable" in app.error[0].value
    assert not (tmp_path / "data" / "courses" / "chroma" / "chroma.sqlite3").exists()
    assert pipeline.built_for == [Path("data")]
