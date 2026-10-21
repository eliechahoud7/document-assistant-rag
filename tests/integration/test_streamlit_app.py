"""Interface Streamlit (app.py) exécutée hors ligne : faux embedder, faux index, faux LLM.

Le dossier courant est un dossier temporaire : les vrais index (data/chroma,
data/courses/chroma) ne sont jamais ouverts. La bibliothèque, son registre, l'indexation, le
chargeur PDF et le découpeur sont les vrais ; seul l'index est en mémoire (il survit aux
« redémarrages » simulés, comme un index sur disque).
"""

from __future__ import annotations

import _socket
import hashlib
import html
import io
import json
import logging
import os
import re
import shutil
import socket
import time
import tomllib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

from conftest import NetworkAccessError
from doc_assistant import bootstrap, web_identity
from doc_assistant.config import Settings
from doc_assistant.domain.errors import (
    LLMConfigurationError,
    LLMQuotaError,
    TranslationModelNotFoundError,
)
from doc_assistant.domain.models import Answer, Chunk, SearchResult
from doc_assistant.domain.ports import AnswerGenerator, Embedder
from doc_assistant.indexing import model_download
from doc_assistant.indexing.document_registry import DocumentRegistry
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.library_service import DocumentLibrary
from doc_assistant.services.qa_service import QAService
from doc_assistant.services.upload_service import MAX_UPLOAD_BYTES, MEGABYTE
from doc_assistant.services.user_libraries import UserIdentity
from fakes import FakeEmbedder, InMemoryVectorStore, ScriptedAnswerGenerator
from pdf_factory import write_pdf

APP = Path(__file__).resolve().parents[2] / "app.py"
# Constructeurs réels, capturés avant que la fixture ne les remplace.
REAL_BUILD_LIBRARY = bootstrap.build_document_library
REAL_BUILD_QA_SERVICE = bootstrap.build_qa_service
REAL_BUILD_STORE = bootstrap.build_store
TIMEOUT_S = 60
CLI_DOC = "aaaaaaaaaaaaaaaa"  # document « indexé avec la CLI » présent au démarrage


class RecordingGenerator(ScriptedAnswerGenerator):
    """Générateur scripté qui garde les passages reçus : ce qui serait envoyé à Gemini."""

    def __init__(
        self,
        script: dict[str, str | Exception],
        model_name: str,
        log: list[list[SearchResult]] | None = None,
    ) -> None:
        super().__init__(script, model_name=model_name)
        self.passages: list[list[SearchResult]] = []
        self._log = log  # journal commun à tous les générateurs (plusieurs utilisateurs)

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer:
        self.passages.append(list(passages))
        if self._log is not None:
            self._log.append(list(passages))
        return super().generate(question, passages)


class Pipeline:
    """Remplace les constructeurs de ``bootstrap`` autour d'un index en mémoire partagé."""

    def __init__(self, script: dict[str, str | Exception]) -> None:
        self.script = script
        self.generator_error: Exception | None = None
        self.embedder_error: Exception | None = None
        self.store = InMemoryVectorStore()
        self.libraries_built = 0
        self.qa_built_for: list[Path] = []
        self.generator: RecordingGenerator | None = None
        self.sent: list[list[SearchResult]] = []  # passages de chaque question, tous générateurs
        text = "Contenu de cours.pdf : les chats dorment beaucoup."
        chunk = Chunk(f"{CLI_DOC}:0", CLI_DOC, "cours.pdf", 2, 0, 0, text)
        self.store.add([chunk], FakeEmbedder().embed_documents([text]))

    def build_embedder(self, _settings: Settings) -> Embedder:
        if self.embedder_error is not None:
            raise self.embedder_error
        return FakeEmbedder()

    def build_generator(self, _settings: Settings) -> AnswerGenerator:
        if self.generator_error is not None:
            raise self.generator_error
        self.generator = RecordingGenerator(self.script, "fake-gemini", self.sent)
        return self.generator

    def build_store(self, _settings: Settings, _embedder: Embedder) -> InMemoryVectorStore:
        return self.store

    def build_qa_service(
        self,
        settings: Settings,
        embedder: Embedder,
        generator: AnswerGenerator,
        *,
        store: object = None,
    ) -> QAService:
        self.qa_built_for.append(settings.data_dir)
        return QAService(Retriever(embedder, self.store, default_top_k=3), generator)

    def build_document_library(
        self, settings: Settings, embedder: Embedder, **options: Any
    ) -> DocumentLibrary:
        """``options`` : documents protégés et limites d'une bibliothèque de visiteur (l'index
        fourni, ``store``, est toujours le faux index partagé)."""
        options.pop("store", None)
        self.libraries_built += 1
        # Comme ChromaVectorStore : ouvrir la bibliothèque crée l'index s'il n'existe pas.
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        (settings.chroma_dir / "chroma.sqlite3").touch()
        indexing = IndexingService(PypdfLoader(), RecursiveChunker(800, 100), embedder, self.store)
        registry = DocumentRegistry.for_data_dir(settings.data_dir)
        return DocumentLibrary(indexing, self.store, registry, settings.uploads_dir, **options)


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
    (tmp_path / "data" / "chroma").mkdir(parents=True)
    (tmp_path / "data" / "chroma" / "chroma.sqlite3").touch()
    monkeypatch.chdir(tmp_path)
    for name in (
        "DOCASSIST_DATA_DIR",
        "DOCASSIST_LLM_PROVIDER",
        "GEMINI_API_KEY",
        "DOCASSIST_APP_UPLOADS",
        "DOCASSIST_DEMO_DOCS_DIR",
        "DOCASSIST_MODEL_AUTO_DOWNLOAD",
        "DOCASSIST_SESSION_UPLOADS",
        "DOCASSIST_UPLOAD_MAX_MB",
        "DOCASSIST_UPLOAD_MAX_PAGES",
        "DOCASSIST_UPLOAD_MAX_FILES",
        "DOCASSIST_MAX_SESSIONS",
        "DOCASSIST_USER_LIBRARIES",
        "DOCASSIST_USER_STORE",
        "DOCASSIST_ACCOUNT_MAX_DOCUMENTS",
        "CHROMA_API_KEY",
        "CHROMA_TENANT",
        "CHROMA_DATABASE",
        "DOCASSIST_PRIVACY_URL",
        "DOCASSIST_ACCOUNT_DAILY_QUESTIONS",
        "DOCASSIST_ACCOUNT_DAILY_UPLOADS",
        "DOCASSIST_MAX_ACCOUNTS",
    ):
        monkeypatch.delenv(name, raising=False)
    fake = Pipeline({"Que font les chats ?": "Ils dorment [1].", "Inconnu ?": "abstain"})
    monkeypatch.setattr(bootstrap, "build_embedder", fake.build_embedder)
    monkeypatch.setattr(bootstrap, "build_generator", fake.build_generator)
    monkeypatch.setattr(bootstrap, "build_qa_service", fake.build_qa_service)
    monkeypatch.setattr(bootstrap, "build_document_library", fake.build_document_library)
    monkeypatch.setattr(bootstrap, "build_store", fake.build_store)
    st.cache_resource.clear()
    yield fake
    st.cache_resource.clear()


def start() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    app.run()
    assert not app.exception
    return app


def restart() -> AppTest:
    """Nouveau processus simulé : caches vidés, nouvelle session (le disque reste)."""
    st.cache_resource.clear()
    return start()


def ask(app: AppTest, question: str) -> None:
    app.chat_input[0].set_value(question).run()
    assert not app.exception


def sidebar_text(app: AppTest) -> str:
    return "\n".join(m.value for m in app.sidebar.markdown)


def pdf(tmp_path: Path, name: str, line: str) -> bytes:
    return write_pdf(tmp_path / f"src-{name}", [[line]]).read_bytes()


def upload(app: AppTest, files: list[tuple[str, bytes]]) -> None:
    """Comme dans le navigateur : la sélection relance la page (bouton activé), puis clic."""
    app.sidebar.file_uploader[0].set_value([(n, d, "application/pdf") for n, d in files]).run()
    selection = next(c.value for c in app.sidebar.caption if c.value.startswith("Sélection"))
    assert all(f"« {name} »" in selection for name, _ in files)  # noms affichés avant indexation
    app.sidebar.button(key="index-files").click().run()
    assert not app.exception


OTTERS = "Les loutres de mer dorment en se tenant la patte."
VOLCANOES = "Un volcan rejette de la lave lors d'une éruption."


# ---------------------------------------------------------------- questions et réponses


def test_answer_shows_cited_source_without_the_model_name(pipeline: Pipeline) -> None:
    app = start()

    ask(app, "Que font les chats ?")

    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert "Ils dorment [1]." in app.chat_message[1].markdown[0].value
    assert not any("fake-gemini" in md.value for md in app.chat_message[1].markdown)
    assert app.main.expander[0].label == "[1] cours.pdf — page 2"
    assert "les chats dorment" in app.main.expander[0].markdown[0].value
    assert pipeline.qa_built_for == [Path("data")]  # l'index de DOCASSIST_DATA_DIR


def test_no_collection_selector_is_shown(pipeline: Pipeline) -> None:
    app = start()

    assert len(app.radio) == 0
    assert "CV" not in sidebar_text(app)


def test_abstention_is_explicit(pipeline: Pipeline) -> None:
    app = start()

    ask(app, "Inconnu ?")

    assert "pas trouvé la réponse dans vos documents" in app.chat_message[1].info[0].value
    assert len(app.main.expander) == 0


def test_clear_button_empties_the_conversation(pipeline: Pipeline) -> None:
    app = start()
    ask(app, "Que font les chats ?")

    next(b for b in app.sidebar.button if b.label == "Nouvelle conversation").click().run()

    assert len(app.chat_message) == 0


def test_api_error_is_shown_without_crashing(pipeline: Pipeline) -> None:
    pipeline.script["Que font les chats ?"] = LLMQuotaError("quota gratuit Gemini atteint")
    app = start()

    ask(app, "Que font les chats ?")

    error = app.chat_message[1].error[0].value
    assert "Limite de requêtes atteinte" in error
    assert "Gemini" not in error  # le message technique reste dans le terminal


def test_missing_configuration_disables_the_chat_without_technical_details(
    pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline.generator_error = LLMConfigurationError(
        "variable d'environnement GEMINI_API_KEY absente"
    )

    app = start()

    assert "Réponses indisponibles" in app.error[0].value
    assert "GEMINI_API_KEY" not in app.error[0].value
    assert len(app.code) == 0
    assert app.chat_input[0].disabled
    assert "GEMINI_API_KEY absente" in caplog.text  # diagnostic conservé dans le terminal


def test_missing_translation_model_is_reported_without_internal_paths(
    pipeline: Pipeline,
) -> None:
    pipeline.embedder_error = TranslationModelNotFoundError(
        "modèle de traduction introuvable dans models/opus-mt-fr-en"
    )

    app = start()

    # L'embedder (avec traduction) est partagé : c'est le chargement des documents qui échoue.
    assert "Documents indisponibles" in app.error[0].value
    assert "models/opus-mt-fr-en" not in app.error[0].value
    assert "erreur inattendue" not in app.error[0].value
    assert app.chat_input[0].disabled


def test_sidebar_hides_storage_and_provider_details(pipeline: Pipeline) -> None:
    app = start()

    text = sidebar_text(app)
    assert "cours.pdf" in text  # la bibliothèque reste visible
    for hidden in ("État", "data/chroma", "Fournisseur", "Modèle", "GEMINI_API_KEY", "Gemini"):
        assert hidden not in text


# ---------------------------------------------------------------- « Mes documents »


def test_existing_documents_are_listed(pipeline: Pipeline) -> None:
    app = start()

    text = sidebar_text(app)
    assert "Mes documents" in text
    assert "cours.pdf" in text
    assert "1 passage · indexé avec la CLI" in text
    assert "Indexé" in text


def test_missing_index_is_never_created_and_chat_waits_for_a_document(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    (tmp_path / "data" / "chroma" / "chroma.sqlite3").unlink()

    app = start()

    assert "bibliothèque est vide" in app.info[0].value
    assert app.chat_input[0].disabled
    assert not (tmp_path / "data" / "chroma" / "chroma.sqlite3").exists()
    assert pipeline.libraries_built == 0


def test_first_upload_creates_the_library_and_enables_questions(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    (tmp_path / "data" / "chroma" / "chroma.sqlite3").unlink()
    pipeline.store.delete_document(CLI_DOC)
    app = start()

    upload(app, [("Loutres.pdf", pdf(tmp_path, "a.pdf", OTTERS))])

    assert [t.value for t in app.toast] == ["« Loutres.pdf » ajouté à vos documents."]
    assert len(app.success) == 0  # plus de bloc de succès encombrant
    assert "1 page · 1 passage" in sidebar_text(app)
    assert not app.chat_input[0].disabled
    ask(app, "Comment dorment les loutres de mer ?")
    assert app.main.expander[0].label == "[1] Loutres.pdf — page 1"


def test_several_pdfs_are_indexed_and_listed_persistently(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()

    upload(
        app,
        [
            ("Loutres.pdf", pdf(tmp_path, "a.pdf", OTTERS)),
            ("Volcans.pdf", pdf(tmp_path, "b.pdf", VOLCANOES)),
        ],
    )

    assert [t.value for t in app.toast] == [
        "« Loutres.pdf » ajouté à vos documents.",
        "« Volcans.pdf » ajouté à vos documents.",
    ]
    assert len(app.success) == 0
    restarted = restart()
    text = sidebar_text(restarted)
    for name in ("cours.pdf", "Loutres.pdf", "Volcans.pdf"):
        assert name in text
    assert "1 page · 1 passage · ajouté le" in text
    registry = json.loads((tmp_path / "data" / "documents.json").read_text("utf-8"))
    assert sorted(d["filename"] for d in registry["documents"]) == ["Loutres.pdf", "Volcans.pdf"]


def test_duplicate_upload_is_not_reindexed(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    content = pdf(tmp_path, "a.pdf", OTTERS)
    upload(app, [("Loutres.pdf", content)])

    upload(app, [("Copie.pdf", content)])

    assert "« Copie.pdf » est déjà dans vos documents (même contenu que « Loutres.pdf »)" in (
        app.sidebar.info[0].value
    )
    assert pipeline.store.count() == 2  # cours.pdf + Loutres.pdf, pas de troisième


def test_invalid_file_is_reported_and_valid_ones_still_indexed(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()

    upload(
        app, [("faux.pdf", b"Ceci n'est pas un PDF"), ("Loutres.pdf", pdf(tmp_path, "a", OTTERS))]
    )

    assert "faux.pdf : ce fichier n'est pas un PDF valide" in app.sidebar.error[0].value
    assert [t.value for t in app.toast] == ["« Loutres.pdf » ajouté à vos documents."]
    assert pipeline.store.count() == 2


def test_upload_works_without_gemini_configuration(pipeline: Pipeline, tmp_path: Path) -> None:
    pipeline.generator_error = LLMConfigurationError("GEMINI_API_KEY absente")
    app = start()

    upload(app, [("Loutres.pdf", pdf(tmp_path, "a.pdf", OTTERS))])

    assert [t.value for t in app.toast] == ["« Loutres.pdf » ajouté à vos documents."]
    assert app.chat_input[0].disabled  # les réponses exigent toujours Gemini


def test_deleting_a_document_keeps_the_others(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    upload(
        app,
        [
            ("Loutres.pdf", pdf(tmp_path, "a.pdf", OTTERS)),
            ("Volcans.pdf", pdf(tmp_path, "b.pdf", VOLCANOES)),
        ],
    )
    otters = next(d for d in pipeline.store.list_documents() if d.filename == "Loutres.pdf")

    app.sidebar.button(key=f"delete-{otters.doc_id}").click().run()

    assert [t.value for t in app.toast] == ["« Loutres.pdf » supprimé de vos documents."]
    assert sorted(d.filename for d in pipeline.store.list_documents()) == [
        "Volcans.pdf",
        "cours.pdf",
    ]
    assert not (tmp_path / "data" / "uploads" / otters.doc_id).exists()
    assert "Loutres.pdf" not in sidebar_text(restart())


def test_document_management_can_be_disabled(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_APP_UPLOADS", "false")

    app = start()

    assert len(app.sidebar.file_uploader) == 0
    assert not any(b.key and b.key.startswith("delete-") for b in app.sidebar.button)
    assert "cours.pdf" in sidebar_text(app)  # la liste reste visible


def test_invalid_interface_configuration_is_reported(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_APP_UPLOADS", "peut-être")

    app = start()

    assert "DOCASSIST_APP_UPLOADS" in app.error[0].value
    assert pipeline.libraries_built == 0


class UnloadedEmbedder(FakeEmbedder):
    """Tout accès qui chargerait le modèle d'embeddings échoue."""

    @property
    def dimension(self) -> int:
        raise AssertionError("le modèle d'embeddings a été chargé à l'ouverture de la page")


def test_opening_the_page_lists_documents_without_loading_the_embedding_model(
    pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Régression : la page restait sur « Chargement de vos documents… » le temps de charger
    le modèle E5 (≈ 9 s mesurées), alors que lister les documents n'en a pas besoin."""
    shutil.rmtree(tmp_path / "data" / "chroma")  # vrai index ChromaDB à la place du faux
    monkeypatch.setattr(bootstrap, "build_store", REAL_BUILD_STORE)
    real = Settings(data_dir=Path("data"))
    library = REAL_BUILD_LIBRARY(real, FakeEmbedder())
    library.add("Loutres.pdf", pdf(tmp_path, "a.pdf", OTTERS))
    monkeypatch.setattr(bootstrap, "build_document_library", REAL_BUILD_LIBRARY)
    monkeypatch.setattr(bootstrap, "build_qa_service", REAL_BUILD_QA_SERVICE)
    monkeypatch.setattr(bootstrap, "build_embedder", lambda _settings: UnloadedEmbedder())
    st.cache_resource.clear()

    app = start()

    assert "Loutres.pdf" in sidebar_text(app)
    assert "1 page · 1 passage · ajouté le" in sidebar_text(app)
    assert not app.chat_input[0].disabled  # service de réponses prêt, modèle toujours absent
    assert not app.error


# ---------------------------------------------------------------- choix du document interrogé


def doc_id_of(pipeline: Pipeline, filename: str) -> str:
    return next(d.doc_id for d in pipeline.store.list_documents() if d.filename == filename)


def sent_to_llm(pipeline: Pipeline) -> list[str]:
    """``doc_id`` de chaque passage transmis au générateur lors de la dernière question."""
    assert pipeline.generator is not None
    return [p.chunk.doc_id for p in pipeline.generator.passages[-1]]


def labels(app: AppTest) -> list[str]:
    """Libellés affichés (``options`` d'AppTest = texte montré ; ``value`` = ``doc_id``)."""
    return list(app.selectbox(key="scope").options)


def two_documents(app: AppTest, tmp_path: Path) -> None:
    upload(
        app,
        [
            ("Loutres.pdf", pdf(tmp_path, "a.pdf", OTTERS)),
            ("Volcans.pdf", pdf(tmp_path, "b.pdf", VOLCANOES)),
        ],
    )


def test_technical_loading_message_is_gone() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "Recherche des passages et rédaction de la réponse" not in source
    assert "charge aussi le modèle d'embeddings" not in source
    assert 'st.spinner("Réflexion en cours…")' in source


def test_selector_lists_all_documents_by_filename(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    two_documents(app, tmp_path)

    assert app.selectbox(key="scope").value == "__all__"
    # Même ordre que « Mes documents » : nom de fichier, sans tenir compte de la casse.
    assert labels(app) == ["Tous les documents", "cours.pdf", "Loutres.pdf", "Volcans.pdf"]


def test_selected_document_is_the_only_context_sent_to_the_llm(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()
    two_documents(app, tmp_path)
    volcanoes = doc_id_of(pipeline, "Volcans.pdf")
    app.selectbox(key="scope").set_value(volcanoes).run()

    ask(app, "Comment dorment les loutres de mer ?")  # mots du document Loutres, pas Volcans

    assert sent_to_llm(pipeline)
    assert set(sent_to_llm(pipeline)) == {volcanoes}
    assert {e.label.split(" — ")[0] for e in app.main.expander} == {"[1] Volcans.pdf"}
    assert any("Document : Volcans.pdf" in md.value for md in app.chat_message[1].markdown)
    assert app.chat_input[0].placeholder == "Posez une question sur « Volcans.pdf »…"


def test_all_documents_keeps_cross_document_retrieval(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    two_documents(app, tmp_path)

    ask(app, "loutres de mer et volcan de lave")

    sent = set(sent_to_llm(pipeline))
    assert {doc_id_of(pipeline, "Loutres.pdf"), doc_id_of(pipeline, "Volcans.pdf")} <= sent
    assert any("Tous les documents" in md.value for md in app.chat_message[1].markdown)


def test_switching_the_selection_changes_the_next_retrieval(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()
    two_documents(app, tmp_path)
    otters, volcanoes = doc_id_of(pipeline, "Loutres.pdf"), doc_id_of(pipeline, "Volcans.pdf")

    app.selectbox(key="scope").set_value(otters).run()
    ask(app, "loutres de mer et volcan de lave")
    first = set(sent_to_llm(pipeline))
    app.selectbox(key="scope").set_value(volcanoes).run()
    ask(app, "loutres de mer et volcan de lave")

    assert (first, set(sent_to_llm(pipeline))) == ({otters}, {volcanoes})
    assert app.selectbox(key="scope").value == volcanoes  # sélection conservée


def test_duplicate_filenames_never_mix_documents(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    upload(
        app,
        [
            ("notes.pdf", pdf(tmp_path, "a.pdf", OTTERS)),
            ("notes.pdf", pdf(tmp_path, "b.pdf", VOLCANOES)),  # même nom, autre contenu
        ],
    )
    ids = [d.doc_id for d in pipeline.store.list_documents() if d.filename == "notes.pdf"]

    assert len(ids) == 2
    assert sorted(x for x in labels(app) if x.startswith("notes.pdf")) == sorted(
        f"notes.pdf · {doc_id[:8]}" for doc_id in ids
    )
    for doc_id in ids:
        app.selectbox(key="scope").set_value(doc_id).run()
        ask(app, "loutres de mer et volcan de lave")
        assert set(sent_to_llm(pipeline)) == {doc_id}


def test_newly_indexed_document_is_selectable_immediately(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()
    assert labels(app) == ["Tous les documents", "cours.pdf"]

    upload(app, [("Loutres.pdf", pdf(tmp_path, "a.pdf", OTTERS))])

    otters = doc_id_of(pipeline, "Loutres.pdf")
    assert "Loutres.pdf" in labels(app)  # sans redémarrer Streamlit
    app.selectbox(key="scope").set_value(otters).run()
    ask(app, "Comment dorment les loutres de mer ?")
    assert set(sent_to_llm(pipeline)) == {otters}


def test_deleting_the_selected_document_falls_back_to_all_documents(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()
    two_documents(app, tmp_path)
    otters, volcanoes = doc_id_of(pipeline, "Loutres.pdf"), doc_id_of(pipeline, "Volcans.pdf")
    app.selectbox(key="scope").set_value(otters).run()

    app.sidebar.button(key=f"delete-{otters}").click().run()

    assert app.selectbox(key="scope").value == "__all__"
    assert "Loutres.pdf" not in labels(app)
    assert {d.doc_id for d in pipeline.store.list_documents()} == {CLI_DOC, volcanoes}


def test_abstention_names_the_selected_document(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    two_documents(app, tmp_path)
    app.selectbox(key="scope").set_value(doc_id_of(pipeline, "Volcans.pdf")).run()

    ask(app, "Inconnu ?")

    assert "pas trouvé la réponse dans « Volcans.pdf »" in app.chat_message[1].info[0].value


# ---------------------------------------------------------------- un historique par document


def questions(app: AppTest) -> list[str]:
    return [m.markdown[0].value for m in app.chat_message if m.name == "user"]


def assistant_count(app: AppTest) -> int:
    return sum(m.name == "assistant" for m in app.chat_message)


def select(app: AppTest, scope: str) -> None:
    app.selectbox(key="scope").set_value(scope).run()
    assert not app.exception


def guide_and_lecture(app: AppTest, tmp_path: Path, pipeline: Pipeline) -> tuple[str, str]:
    upload(
        app,
        [
            ("Guide.pdf", pdf(tmp_path, "guide.pdf", OTTERS)),
            ("lecture1.pdf", pdf(tmp_path, "lecture1.pdf", VOLCANOES)),
        ],
    )
    return doc_id_of(pipeline, "Guide.pdf"), doc_id_of(pipeline, "lecture1.pdf")


def test_each_document_keeps_its_own_conversation(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    guide, lecture = guide_and_lecture(app, tmp_path, pipeline)

    select(app, guide)
    ask(app, "Comment dorment les loutres ?")
    ask(app, "Que tiennent-elles ?")
    assert questions(app) == ["Comment dorment les loutres ?", "Que tiennent-elles ?"]

    select(app, lecture)  # la conversation de Guide.pdf disparaît : écran d'accueil
    assert len(app.chat_message) == 0
    assert any("Que voulez-vous savoir" in md.value for md in app.main.markdown)

    ask(app, "Que rejette un volcan ?")
    select(app, guide)  # ses deux questions et réponses reviennent, avec leurs sources
    assert questions(app) == ["Comment dorment les loutres ?", "Que tiennent-elles ?"]
    assert assistant_count(app) == 2
    assert {e.label for e in app.main.expander} == {"[1] Guide.pdf — page 1"}
    assert all(
        any("Document : Guide.pdf" in md.value for md in m.markdown)
        for m in app.chat_message
        if m.name == "assistant"
    )

    select(app, lecture)
    assert questions(app) == ["Que rejette un volcan ?"]
    assert {e.label for e in app.main.expander} == {"[1] lecture1.pdf — page 1"}

    app.run()  # simple réexécution : rien n'est perdu ni dupliqué
    assert questions(app) == ["Que rejette un volcan ?"]
    assert assistant_count(app) == 1


def test_all_documents_has_its_own_conversation(pipeline: Pipeline, tmp_path: Path) -> None:
    app = start()
    guide, _ = guide_and_lecture(app, tmp_path, pipeline)

    ask(app, "Question globale ?")
    select(app, guide)
    assert len(app.chat_message) == 0
    ask(app, "Question sur Guide ?")

    select(app, "__all__")
    assert questions(app) == ["Question globale ?"]
    assert any("Tous les documents" in md.value for md in app.chat_message[1].markdown)


def test_new_conversation_only_clears_the_displayed_document(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()
    guide, lecture = guide_and_lecture(app, tmp_path, pipeline)
    select(app, guide)
    ask(app, "Question sur Guide ?")
    select(app, lecture)
    ask(app, "Question sur lecture1 ?")

    next(b for b in app.sidebar.button if b.label == "Nouvelle conversation").click().run()

    assert len(app.chat_message) == 0
    select(app, guide)
    assert questions(app) == ["Question sur Guide ?"]


def test_deleted_document_history_is_never_shown_elsewhere(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()
    guide, lecture = guide_and_lecture(app, tmp_path, pipeline)
    select(app, guide)
    ask(app, "Question sur Guide ?")
    select(app, lecture)
    ask(app, "Question sur lecture1 ?")
    select(app, guide)

    app.sidebar.button(key=f"delete-{guide}").click().run()

    assert app.selectbox(key="scope").value == "__all__"
    assert "Question sur Guide ?" not in questions(app)  # « Tous les documents » : vide
    select(app, lecture)
    assert questions(app) == ["Question sur lecture1 ?"]
    # Même contenu renvoyé (même doc_id) : conversation vide, l'ancienne a été supprimée.
    upload(app, [("Guide.pdf", pdf(tmp_path, "guide.pdf", OTTERS))])
    select(app, guide)
    assert len(app.chat_message) == 0


def test_duplicate_filenames_keep_separate_conversations(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    app = start()
    upload(
        app,
        [
            ("notes.pdf", pdf(tmp_path, "a.pdf", OTTERS)),
            ("notes.pdf", pdf(tmp_path, "b.pdf", VOLCANOES)),
        ],
    )
    first, second = (d.doc_id for d in pipeline.store.list_documents() if d.filename == "notes.pdf")

    select(app, first)
    ask(app, "Première ?")
    select(app, second)

    assert len(app.chat_message) == 0


# ---------------------------------------------------------------- défilement automatique


def scroll_tokens(app: AppTest) -> list[str]:
    return [h.proto.body.split("data-token='")[1].split("'")[0] for h in app.html]


def test_submitting_a_question_scrolls_to_it_and_to_the_answer(pipeline: Pipeline) -> None:
    app = start()
    assert scroll_tokens(app) == []  # aucun défilement à l'ouverture

    ask(app, "Que font les chats ?")

    # Un défilement à l'envoi (pendant la génération), un autre une fois la réponse affichée.
    assert scroll_tokens(app) == ["1-question", "1-answer"]
    assert all(h.proto.unsafe_allow_javascript for h in app.html)
    assert all("scrollIntoView" in h.proto.body for h in app.html)

    ask(app, "Inconnu ?")
    assert scroll_tokens(app) == ["2-question", "2-answer"]  # nouveau jeton : script relancé


def test_reading_older_messages_is_never_interrupted(pipeline: Pipeline) -> None:
    app = start()
    ask(app, "Que font les chats ?")

    app.run()  # toute autre interaction : aucun défilement forcé

    assert scroll_tokens(app) == []
    assert len(app.chat_message) == 2


def test_scroll_helper_is_hidden_from_the_layout() -> None:
    source = APP.read_text(encoding="utf-8")

    assert '[data-testid="stElementContainer"]:has(.docassist-scroll) { display: none; }' in source


# ---------------------------------------------------------------- déploiement : démonstration


def demo_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "demo"
    folder.mkdir()
    (folder / "Loutres.pdf").write_bytes(pdf(tmp_path, "demo-a.pdf", OTTERS))
    (folder / "Volcans.pdf").write_bytes(pdf(tmp_path, "demo-b.pdf", VOLCANOES))
    return folder


def test_demo_mode_indexes_its_documents_once_and_hides_management(
    pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(demo_folder(tmp_path)))

    app = start()

    text = sidebar_text(app)
    assert "Loutres.pdf" in text and "Volcans.pdf" in text
    assert len(app.sidebar.file_uploader) == 0
    assert not any(b.key and b.key.startswith("delete-") for b in app.sidebar.button)
    chunks = pipeline.store.count()
    app.run()
    restart()  # nouveau processus : les documents déjà indexés ne sont pas réindexés
    assert pipeline.store.count() == chunks
    assert len(DocumentRegistry.for_data_dir(Path("data")).records()) == 2


def test_demo_mode_starts_from_an_empty_cloud_data_folder(
    pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "data" / "chroma" / "chroma.sqlite3").unlink()  # disque du cloud : vide
    pipeline.store.delete_document(CLI_DOC)
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(demo_folder(tmp_path)))

    app = start()

    assert not app.chat_input[0].disabled
    ask(app, "Comment dorment les loutres de mer ?")
    assert app.main.expander[0].label == "[1] Loutres.pdf — page 1"


def test_demo_mode_keeps_management_disabled_even_if_uploads_are_enabled(
    pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(demo_folder(tmp_path)))
    monkeypatch.setenv("DOCASSIST_APP_UPLOADS", "true")

    app = start()

    assert len(app.sidebar.file_uploader) == 0
    assert not any(b.key and b.key.startswith("delete-") for b in app.sidebar.button)


def test_missing_demo_folder_does_not_break_the_page(
    pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(tmp_path / "absent"))

    app = start()

    assert "cours.pdf" in sidebar_text(app)
    assert len(app.sidebar.file_uploader) == 0


def test_local_default_keeps_uploads_and_demo_off(pipeline: Pipeline) -> None:
    app = start()

    assert len(app.sidebar.file_uploader) == 1
    assert pipeline.store.count() == 1  # seul le document existant : rien n'est indexé


# ---------------------------------------------------------------- envois privés des visiteurs

CAKES = "Un gâteau au chocolat cuit trente minutes au four."
SESSIONS = Path("data") / "sessions"
EXPIRED_AGE_S = 3 * 3600  # au-delà des 2 h d'inactivité tolérées


def session_folders() -> list[Path]:
    return sorted(p for p in SESSIONS.iterdir() if p.is_dir()) if SESSIONS.is_dir() else []


def age(folder: Path, seconds: float) -> None:
    """Dernière activité de la bibliothèque ``folder`` il y a ``seconds`` secondes."""
    past = time.time() - seconds
    os.utime(folder, (past, past))


def doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def delete_buttons(app: AppTest) -> list[str]:
    """Boutons de suppression d'un document (pas celui du compte)."""
    return [
        b.key for b in app.sidebar.button if b.key and re.fullmatch(r"delete-[0-9a-f]{16}", b.key)
    ]


def sent_filenames(pipeline: Pipeline) -> list[str]:
    """Fichiers des passages transmis au LLM lors de la dernière question, quel que soit le
    générateur (un par session ou par utilisateur) qui l'a reçue."""
    assert pipeline.sent
    return [p.chunk.filename for p in pipeline.sent[-1]]


@pytest.fixture
def sessions_on(pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """Mode démonstration avec envois privés, sur le faux index partagé (aucun fichier ouvert :
    les dossiers de session peuvent être supprimés, même sous Windows)."""
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(demo_folder(tmp_path)))
    monkeypatch.setenv("DOCASSIST_SESSION_UPLOADS", "true")
    return pipeline


@pytest.fixture
def visitors(sessions_on: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """Comme ``sessions_on``, mais avec les vrais index ChromaDB sur disque : chaque session
    travaille sur sa propre copie de l'index de démonstration (seuls l'embedder et le LLM sont
    faux).

    Chemin absolu : ChromaDB garde ses clients en mémoire par chemin ; avec ``data`` relatif,
    un test suivant (autre dossier courant) retrouverait l'index du précédent."""
    shutil.rmtree(tmp_path / "data" / "chroma")
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(bootstrap, "build_document_library", REAL_BUILD_LIBRARY)
    monkeypatch.setattr(bootstrap, "build_qa_service", REAL_BUILD_QA_SERVICE)
    monkeypatch.setattr(bootstrap, "build_store", REAL_BUILD_STORE)
    return sessions_on


@pytest.mark.parametrize(
    "extra_env",
    [
        {},
        {"DOCASSIST_SESSION_UPLOADS": "false"},
        # Secrets du déploiement documentés dans le README.
        {"DOCASSIST_APP_UPLOADS": "false", "DOCASSIST_LLM_PROVIDER": "gemini"},
    ],
    ids=["default", "explicitly-off", "deployed-secrets"],
)
def test_visitor_uploads_stay_disabled_unless_explicitly_enabled(
    pipeline: Pipeline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_env: dict[str, str],
) -> None:
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(demo_folder(tmp_path)))
    for name, value in extra_env.items():
        monkeypatch.setenv(name, value)

    app = start()

    assert "Loutres.pdf" in sidebar_text(app)
    assert len(app.sidebar.file_uploader) == 0
    assert delete_buttons(app) == []
    assert not SESSIONS.exists()  # aucune bibliothèque de visiteur créée


def test_visitor_uploads_require_demo_mode(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_SESSION_UPLOADS", "true")

    app = start()

    assert "DOCASSIST_SESSION_UPLOADS" in app.error[0].value
    assert "DOCASSIST_DEMO_DOCS_DIR" in app.error[0].value
    assert pipeline.libraries_built == 0
    assert len(app.sidebar.file_uploader) == 0
    assert not SESSIONS.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DOCASSIST_SESSION_UPLOADS", "parfois"),
        ("DOCASSIST_UPLOAD_MAX_MB", "0"),
        ("DOCASSIST_UPLOAD_MAX_MB", "2.5"),
        ("DOCASSIST_UPLOAD_MAX_PAGES", "-5"),
        ("DOCASSIST_UPLOAD_MAX_PAGES", ""),
        ("DOCASSIST_UPLOAD_MAX_FILES", "trois"),
        # Régression : chiffres non ASCII acceptés par str.isdigit (« ² » faisait planter int()).
        ("DOCASSIST_UPLOAD_MAX_MB", "²"),
        ("DOCASSIST_UPLOAD_MAX_PAGES", "\u0661\u0660"),
        ("DOCASSIST_MAX_SESSIONS", "0"),
        # Au-delà de server.maxUploadSize (200 Mo par défaut, sans le
        # .streamlit/config.toml du dépôt) : refusé au démarrage.
        ("DOCASSIST_UPLOAD_MAX_MB", "500"),
    ],
)
def test_invalid_visitor_upload_configuration_is_reported(
    sessions_on: Pipeline, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    app = start()

    assert name in app.error[0].value
    assert sessions_on.libraries_built == 0
    assert not SESSIONS.exists()


def test_visitor_sees_the_configured_limits_and_a_privacy_warning(
    sessions_on: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_MB", "5")
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_PAGES", "7")
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_FILES", "2")

    app = start()

    uploader = app.sidebar.file_uploader[0]
    assert "5 Mo et 7 pages au plus, 2 documents par session" in uploader.help
    captions = "\n".join(c.value for c in app.sidebar.caption)
    assert "supprimés après 2 heures d'inactivité" in captions
    assert "n'ajoutez pas de document confidentiel" in captions


def test_each_visitor_session_has_its_own_private_library(
    visitors: Pipeline, tmp_path: Path
) -> None:
    first, second = start(), start()  # deux onglets : deux sessions du même processus
    cakes = pdf(tmp_path, "c.pdf", CAKES)

    upload(first, [("Gateaux.pdf", cakes)])
    second.run()
    upload(second, [("Volcans-bis.pdf", pdf(tmp_path, "v.pdf", "Lave et cendres du volcan."))])

    assert "Gateaux.pdf" in sidebar_text(first)
    assert "Gateaux.pdf" not in sidebar_text(second)
    assert "Loutres.pdf" in sidebar_text(second)  # documents de démonstration pour chacun
    ask(second, "Combien de temps cuit un gâteau au chocolat ?")
    assert sent_filenames(visitors)  # des passages ont bien été envoyés…
    assert "Gateaux.pdf" not in sent_filenames(visitors)  # … jamais ceux de l'autre session
    assert "Volcans-bis.pdf" not in sidebar_text(first)

    folders = session_folders()
    assert len(folders) == 2
    assert all(len(f.name) == 32 and int(f.name, 16) >= 0 for f in folders)  # noms aléatoires
    owners = [f for f in folders if (f / "uploads" / doc_id(cakes) / "Gateaux.pdf").is_file()]
    assert len(owners) == 1
    # L'index de démonstration partagé n'est jamais modifié par un visiteur.
    assert not (tmp_path / "data" / "uploads" / doc_id(cakes)).exists()
    assert len(DocumentRegistry.for_data_dir(Path("data")).records()) == 2
    assert "Gateaux.pdf" not in sidebar_text(start())  # nouvelle visite : rien d'hérité


def test_demo_documents_are_protected_but_own_uploads_can_be_deleted(
    visitors: Pipeline, tmp_path: Path
) -> None:
    app = start()
    cakes = pdf(tmp_path, "c.pdf", CAKES)
    upload(app, [("Gateaux.pdf", cakes)])

    assert sidebar_text(app).count("badge-demo") == 2  # Loutres.pdf et Volcans.pdf
    assert delete_buttons(app) == [f"delete-{doc_id(cakes)}"]

    app.sidebar.button(key=f"delete-{doc_id(cakes)}").click().run()

    assert [t.value for t in app.toast] == ["« Gateaux.pdf » supprimé de vos documents."]
    text = sidebar_text(app)
    assert "Gateaux.pdf" not in text
    assert "Loutres.pdf" in text and "Volcans.pdf" in text


def test_visitor_limits_are_enforced_by_the_server(
    visitors: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Les fichiers sont injectés directement : la limite de taille du composant d'envoi
    (navigateur) est contournée, seule la vérification côté serveur s'applique."""
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_MB", "1")
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_PAGES", "1")
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_FILES", "1")
    app = start()
    two_pages = write_pdf(tmp_path / "long.pdf", [[OTTERS], [VOLCANOES]]).read_bytes()

    upload(app, [("gros.pdf", b"%PDF-1.4\n" + b"0" * 1_200_000)])
    assert "gros.pdf : fichier trop volumineux" in app.sidebar.error[0].value
    upload(app, [("long.pdf", two_pages)])
    assert "long.pdf : trop de pages (2 ; maximum 1" in app.sidebar.error[0].value

    upload(app, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    upload(app, [("Autre.pdf", pdf(tmp_path, "d.pdf", "Un autre document."))])
    assert "limite de 1 documents ajoutés atteinte" in app.sidebar.error[0].value

    text = sidebar_text(app)
    assert "Gateaux.pdf" in text
    assert not any(name in text for name in ("gros.pdf", "long.pdf", "Autre.pdf"))


def padded_pdf(tmp_path: Path, line: str, size: int) -> bytes:
    """PDF valide d'environ ``size`` octets (une métadonnée sert de remplissage)."""
    source = write_pdf(tmp_path / "padded-src.pdf", [[line]])
    writer = PdfWriter(clone_from=source)
    writer.add_metadata({"/Padding": "x" * (size - source.stat().st_size)})
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_browser_and_server_size_limits_use_the_same_unit(
    visitors: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Régression : le serveur comptait 1 Mo = 1 000 000 octets, le composant d'envoi
    1024 * 1024 dans un navigateur sous Windows ; un fichier entre les deux passait le
    navigateur puis était refusé."""
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_MB", "1")
    app = start()
    assert app.sidebar.file_uploader[0].proto.max_upload_size_mb == 1
    between = padded_pdf(tmp_path, CAKES, 1_030_000)
    assert 1_000_000 < len(between) <= MEGABYTE

    upload(app, [("Gateaux.pdf", between)])
    assert len(app.sidebar.error) == 0
    assert "Gateaux.pdf" in sidebar_text(app)

    upload(app, [("gros.pdf", b"%PDF-1.4\n" + b"0" * MEGABYTE)])  # 1 Mio + 9 octets
    assert "gros.pdf : fichier trop volumineux (1.0 Mo ; maximum 1 Mo)" in (
        app.sidebar.error[0].value
    )


def test_expired_session_with_an_open_index_is_deleted(visitors: Pipeline, tmp_path: Path) -> None:
    """Régression (Windows) : l'index ChromaDB encore ouvert d'une session expirée empêchait
    la suppression de son dossier ; seuls ses PDF et son registre disparaissaient."""
    first = start()
    upload(first, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    ask(first, "Combien de temps cuit un gâteau ?")  # index ouvert : bibliothèque et réponses
    (old,) = session_folders()
    age(old, EXPIRED_AGE_S)

    # La création d'une autre bibliothèque privée déclenche le nettoyage.
    upload(start(), [("Autre.pdf", pdf(tmp_path, "d.pdf", "Un autre document."))])

    assert not old.exists()
    first.run()
    assert not first.exception
    assert "Votre session a expiré" in first.sidebar.info[0].value
    text = sidebar_text(first)
    assert "Gateaux.pdf" not in text and "Loutres.pdf" in text
    ask(first, "Comment dorment les loutres de mer ?")  # démonstration toujours utilisable
    assert [m.name for m in first.chat_message] == ["user", "assistant"]


def test_inactive_session_libraries_are_deleted_and_recent_ones_kept(
    sessions_on: Pipeline,
) -> None:
    expired, recent = SESSIONS / ("0" * 32), SESSIONS / ("1" * 32)
    for folder in (expired, recent):
        (folder / "uploads").mkdir(parents=True)
        (folder / "documents.json").write_text("{}", encoding="utf-8")
    age(expired, EXPIRED_AGE_S)
    age(recent, 60)

    app = start()

    assert not expired.exists()
    assert session_folders() == [recent]  # et aucune bibliothèque créée pour une simple visite
    assert "Loutres.pdf" in sidebar_text(app)


def test_session_cleanup_runs_at_most_once_per_interval(sessions_on: Pipeline) -> None:
    app = start()
    late = SESSIONS / ("2" * 32)
    late.mkdir(parents=True)  # une simple visite ne crée plus le dossier des sessions
    age(late, EXPIRED_AGE_S)

    app.run()
    assert late.is_dir()  # nettoyage récent : pas de nouveau parcours avant 10 min

    restart()  # nouveau processus : l'intervalle repart de zéro
    assert not late.exists()


def test_activity_keeps_the_current_session_alive(sessions_on: Pipeline, tmp_path: Path) -> None:
    app = start()
    upload(app, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    (own,) = session_folders()
    age(own, EXPIRED_AGE_S)

    app.run()

    assert time.time() - own.stat().st_mtime < 60
    restart()  # le nettoyage d'une autre session ne la supprime donc pas
    assert own.is_dir()


def test_expired_session_returns_to_the_demo_with_an_empty_conversation(
    sessions_on: Pipeline, tmp_path: Path
) -> None:
    app = start()
    upload(app, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    ask(app, "Que font les chats ?")
    assert len(app.chat_message) == 2
    (old,) = session_folders()

    shutil.rmtree(old)  # supprimée par le nettoyage d'une autre session
    app.run()

    assert not app.exception
    assert "Votre session a expiré" in app.sidebar.info[0].value
    assert len(app.chat_message) == 0
    assert session_folders() == []  # pas de nouvelle bibliothèque avant un nouvel envoi
    assert "Loutres.pdf" in sidebar_text(app)
    assert len(app.sidebar.file_uploader) == 1

    upload(app, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    (new,) = session_folders()
    assert new != old


# ---------------------------------------------------------------- création à la demande, plafonds


def test_viewing_the_demo_never_creates_a_private_library(visitors: Pipeline) -> None:
    """Régression (M1) : chaque affichage de la page copiait l'index de démonstration et
    ouvrait un index ChromaDB, gardés 2 h, même pour un visiteur qui n'envoyait rien."""
    first = start()
    ask(first, "Comment dorment les loutres de mer ?")
    second = start()

    assert not SESSIONS.exists()
    assert first.main.expander[0].label == "[1] Loutres.pdf — page 1"  # réponses disponibles
    for app in (first, second):
        assert sidebar_text(app).count("badge-demo") == 2  # tout est en lecture seule
        assert delete_buttons(app) == []
        assert len(app.sidebar.file_uploader) == 1


def test_private_libraries_are_capped_and_freed_by_cleanup(
    visitors: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_MAX_SESSIONS", "1")
    first, second = start(), start()
    upload(first, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    (taken,) = session_folders()

    upload(second, [("Autre.pdf", pdf(tmp_path, "d.pdf", "Un autre document."))])

    assert "Trop de visiteurs ont ajouté des documents" in second.sidebar.error[0].value
    assert session_folders() == [taken]
    text = sidebar_text(second)
    assert "Autre.pdf" not in text and "Loutres.pdf" in text  # la démonstration reste utilisable

    age(taken, EXPIRED_AGE_S)  # la première session devient inactive : sa place est libérée
    upload(second, [("Autre.pdf", pdf(tmp_path, "d.pdf", "Un autre document."))])

    assert len(second.sidebar.error) == 0
    assert "Autre.pdf" in sidebar_text(second)
    assert taken not in session_folders()
    assert len(session_folders()) == 1


def test_files_beyond_the_per_session_count_are_refused_by_the_server(
    visitors: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Régression (M2) : un envoi de nombreux fichiers était traité fichier par fichier ;
    seuls les ``DOCASSIST_UPLOAD_MAX_FILES`` premiers passent désormais par le pipeline."""
    monkeypatch.setenv("DOCASSIST_UPLOAD_MAX_FILES", "2")
    app = start()
    names = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]

    upload(app, [(n, pdf(tmp_path, n, f"Document numéro {n}.")) for n in names])

    assert "2 fichiers non envoyés (« c.pdf », « d.pdf ») : au plus 2 documents" in (
        app.sidebar.error[0].value
    )
    text = sidebar_text(app)
    assert "a.pdf" in text and "b.pdf" in text
    assert "c.pdf" not in text and "d.pdf" not in text
    (own,) = session_folders()
    assert sorted(p.name for p in (own / "uploads").rglob("*.pdf")) == ["a.pdf", "b.pdf"]


def test_server_upload_size_cap_matches_the_largest_app_limit() -> None:
    """Régression (M2) : sans ``server.maxUploadSize``, Streamlit recevait jusqu'à 200 Mo par
    fichier avant que l'application ne le refuse."""
    config = tomllib.loads((APP.parent / ".streamlit" / "config.toml").read_text("utf-8"))

    assert config["server"]["maxUploadSize"] == MAX_UPLOAD_BYTES // MEGABYTE == 20


# ---------------------------------------------------------------- bibliothèques personnelles

GOOGLE = "https://accounts.google.com"
PRIVACY_URL = "https://example.org/confidentialite"  # adresse factice de test
TERMS_VERSION = "2026-10-21"  # celle d'app.py : la changer impose une nouvelle acceptation
# Valeurs factices de test (aucun vrai identifiant) : la connexion n'est jamais lancée.
AUTH_SECRETS: dict[str, Any] = {
    "auth": {
        "redirect_uri": "http://localhost:8501/oauth2callback",
        "cookie_secret": "test-cookie-secret-" + "x" * 32,
        "client_id": "test-client-id.apps.googleusercontent.com",
        "client_secret": "test-client-secret",
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    }
}


def google_claims(sub: str, name: str) -> dict[str, object]:
    return {"is_logged_in": True, "iss": GOOGLE, "sub": sub, "name": name}


ALICE = google_claims("1001", "Alice")
BOB = google_claims("1002", "Bob")


class SignedIn:
    """Remplace la connexion Google : les informations que le serveur aurait vérifiées pour
    la prochaine exécution de la page (``None`` : personne n'est connecté)."""

    def __init__(self) -> None:
        self.claims: dict[str, object] | None = None


@pytest.fixture
def accounts(pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SignedIn:
    """Bibliothèques personnelles activées sur une base ChromaDB locale (même API que Chroma
    Cloud), démonstration pour les visiteurs déconnectés ; embedder et LLM factices."""
    shutil.rmtree(tmp_path / "data" / "chroma")
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "data"))  # chemin absolu (Chroma)
    monkeypatch.setattr(bootstrap, "build_document_library", REAL_BUILD_LIBRARY)
    monkeypatch.setattr(bootstrap, "build_qa_service", REAL_BUILD_QA_SERVICE)
    monkeypatch.setattr(bootstrap, "build_store", REAL_BUILD_STORE)
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(demo_folder(tmp_path)))
    monkeypatch.setenv("DOCASSIST_USER_LIBRARIES", "true")
    monkeypatch.setenv("DOCASSIST_USER_STORE", "local")
    monkeypatch.setenv("DOCASSIST_PRIVACY_URL", PRIVACY_URL)
    monkeypatch.setattr(web_identity, "login_dependency_available", lambda: True)
    signed_in = SignedIn()
    monkeypatch.setattr(web_identity, "signed_in_claims", lambda: signed_in.claims)
    return signed_in


def open_app(secrets: dict[str, Any] = AUTH_SECRETS) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    app.secrets.update(secrets)
    app.run()
    assert not app.exception
    return app


def visit_as(
    signed_in: SignedIn,
    who: dict[str, object] | None,
    app: AppTest | None = None,
    *,
    accept: bool = True,
) -> AppTest:
    """Prochaine exécution de ``app`` (ou nouvelle session) avec l'identité ``who``. Avec
    ``accept``, les conditions sont acceptées si elles sont demandées (les tests des conditions
    elles-mêmes passent ``accept=False``)."""
    signed_in.claims = who
    if app is None:
        app = open_app()
    else:
        app.run()
        assert not app.exception
    if accept and has_button(app, "accept-terms"):
        accept_terms(app)
    return app


def accept_terms(app: AppTest) -> None:
    app.sidebar.checkbox(key=f"terms-{TERMS_VERSION}").check().run()
    app.sidebar.button(key="accept-terms").click().run()
    assert not app.exception
    assert not has_button(app, "accept-terms")


def users_db_collections(tmp_path: Path, who: dict[str, object]) -> list[str]:
    """Collections de la base locale de ``who`` (accès indépendant de l'application)."""
    folder = tmp_path / "data" / "users" / key_of(who)
    if not folder.is_dir():
        return []
    client = bootstrap.build_user_store("local", folder.parent).client_for(
        UserIdentity.from_claims(who)
    )
    return sorted(c.name for c in client.list_collections())


def has_button(app: AppTest, key: str) -> bool:
    return any(b.key == key for b in app.sidebar.button)


def key_of(who: dict[str, object]) -> str:
    return UserIdentity.from_claims(who).key


def test_personal_libraries_are_off_by_default(
    pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_DEMO_DOCS_DIR", str(demo_folder(tmp_path)))

    app = open_app()

    assert not has_button(app, "login")
    assert len(app.sidebar.file_uploader) == 0
    assert not (tmp_path / "data" / "users").exists()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"secrets": {}}, "section [auth] des Secrets incomplète (redirect_uri, cookie_secret"),
        (
            {"secrets": {"auth": AUTH_SECRETS["auth"] | {"cookie_secret": "court-et-faible"}}},
            "cookie_secret trop court",
        ),
        ({"env": {"DOCASSIST_USER_STORE": "s3"}}, "DOCASSIST_USER_STORE doit valoir"),
        (
            {
                "env": {
                    "DOCASSIST_USER_STORE": "chroma-cloud",
                    "CHROMA_API_KEY": "ck-not-a-real-key",
                }
            },
            "Chroma Cloud : variable(s) absente(s) : CHROMA_TENANT, CHROMA_DATABASE",
        ),
        ({"env": {"DOCASSIST_SESSION_UPLOADS": "true"}}, "sont incompatibles"),
        ({"no_authlib": True}, "installez streamlit[auth]"),
    ],
    ids=["no-auth", "weak-cookie-secret", "bad-store", "cloud-ids", "with-sessions", "no-authlib"],
)
def test_incomplete_account_configuration_enables_nothing(
    accounts: SignedIn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, Any],
    expected: str,
) -> None:
    for name, value in change.get("env", {}).items():
        monkeypatch.setenv(name, value)
    if change.get("no_authlib"):
        monkeypatch.setattr(web_identity, "login_dependency_available", lambda: False)
    accounts.claims = ALICE

    app = open_app(change.get("secrets", AUTH_SECRETS))

    message = app.error[0].value
    assert expected in message
    for secret in ("court-et-faible", "ck-not-a-real-key", "test-client-secret"):
        assert secret not in message  # jamais une valeur secrète
    assert len(app.sidebar.file_uploader) == 0
    assert not (tmp_path / "data" / "users").exists()


def test_signed_out_visitors_get_the_read_only_demo(accounts: SignedIn, tmp_path: Path) -> None:
    app = visit_as(accounts, None)

    assert has_button(app, "login")
    assert sidebar_text(app).count("badge-demo") == 2
    assert len(app.sidebar.file_uploader) == 0
    assert delete_buttons(app) == []
    assert not (tmp_path / "data" / "users").exists()  # aucune base ouverte pour une visite
    ask(app, "Comment dorment les loutres de mer ?")
    assert app.main.expander[0].label == "[1] Loutres.pdf — page 1"


def test_signed_in_user_starts_with_their_own_empty_library(accounts: SignedIn) -> None:
    app = visit_as(accounts, ALICE)

    text = sidebar_text(app)
    assert "Connecté : Alice" in text
    assert "Loutres.pdf" not in text and "badge-demo" not in text  # pas la démonstration
    assert len(app.sidebar.file_uploader) == 1
    assert has_button(app, "logout") and has_button(app, "delete-account")
    assert not has_button(app, "login")
    assert app.chat_input[0].disabled  # rien à interroger avant le premier envoi


def test_users_never_see_retrieve_or_delete_each_others_documents(
    accounts: SignedIn, pipeline: Pipeline, tmp_path: Path
) -> None:
    alice = visit_as(accounts, ALICE)
    cakes = pdf(tmp_path, "c.pdf", CAKES)
    upload(alice, [("Gateaux.pdf", cakes)])
    bob = visit_as(accounts, BOB)

    assert "Gateaux.pdf" not in sidebar_text(bob)
    upload(bob, [("Volcans-bis.pdf", pdf(tmp_path, "v.pdf", "Lave et cendres du volcan."))])
    ask(bob, "Combien de temps cuit un gâteau au chocolat ?")  # mots du document d'Alice
    assert sent_filenames(pipeline) == ["Volcans-bis.pdf"]
    assert delete_buttons(bob) == [
        f"delete-{doc_id(pdf(tmp_path, 'v.pdf', 'Lave et cendres du volcan.'))}"
    ]

    visit_as(accounts, ALICE, alice)
    text = sidebar_text(alice)
    assert "Gateaux.pdf" in text and "Volcans-bis.pdf" not in text
    assert delete_buttons(alice) == [f"delete-{doc_id(cakes)}"]
    ask(alice, "Lave et cendres du volcan ?")  # mots du document de Bob
    assert sent_filenames(pipeline) == ["Gateaux.pdf"]


def test_identity_is_never_taken_from_the_session_state(accounts: SignedIn, tmp_path: Path) -> None:
    alice = visit_as(accounts, ALICE)
    upload(alice, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    bob = visit_as(accounts, BOB)
    # Le navigateur ne peut pas écrire st.session_state, mais même une valeur forgée côté
    # serveur ne change pas d'utilisateur : seule l'identité vérifiée compte.
    for name in ("history_owner", "user", "user_key", "identity", "account"):
        bob.session_state[name] = key_of(ALICE)

    visit_as(accounts, BOB, bob)

    assert "Gateaux.pdf" not in sidebar_text(bob)
    assert "Connecté : Bob" in sidebar_text(bob)


def test_switching_account_in_the_same_browser_never_shows_the_previous_conversation(
    accounts: SignedIn, tmp_path: Path
) -> None:
    app = visit_as(accounts, ALICE)
    upload(app, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    ask(app, "Combien de temps cuit le gâteau ?")
    assert len(app.chat_message) == 2

    visit_as(accounts, None, app)  # déconnexion
    assert len(app.chat_message) == 0
    visit_as(accounts, BOB, app)
    assert len(app.chat_message) == 0
    assert "Gateaux.pdf" not in sidebar_text(app)


def test_library_persists_after_restart_and_on_another_device(
    accounts: SignedIn, tmp_path: Path
) -> None:
    laptop = visit_as(accounts, ALICE)
    upload(laptop, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])

    st.cache_resource.clear()  # redémarrage de l'application : rien n'est gardé en mémoire
    phone = visit_as(accounts, ALICE)  # autre appareil : nouvelle session, même compte

    assert "Gateaux.pdf" in sidebar_text(phone)
    assert "1 page · 1 passage · ajouté le" in sidebar_text(phone)
    upload(phone, [("Volcans.pdf", pdf(tmp_path, "v.pdf", VOLCANOES))])  # ajout plus tard
    visit_as(accounts, ALICE, laptop)
    text = sidebar_text(laptop)
    assert "Gateaux.pdf" in text and "Volcans.pdf" in text
    assert not list((tmp_path / "data").rglob("Gateaux.pdf"))  # aucun PDF gardé sur le disque


def test_deleting_a_document_leaves_the_same_document_of_another_user(
    accounts: SignedIn, tmp_path: Path
) -> None:
    cakes = pdf(tmp_path, "c.pdf", CAKES)
    alice = visit_as(accounts, ALICE)
    upload(alice, [("Gateaux.pdf", cakes)])
    bob = visit_as(accounts, BOB)
    upload(bob, [("Gateaux.pdf", cakes)])

    visit_as(accounts, ALICE, alice)
    alice.sidebar.button(key=f"delete-{doc_id(cakes)}").click().run()

    assert [t.value for t in alice.toast] == ["« Gateaux.pdf » supprimé de vos documents."]
    assert "Gateaux.pdf" not in sidebar_text(alice)
    assert "Gateaux.pdf" in sidebar_text(visit_as(accounts, BOB, bob))


def test_account_deletion_removes_only_that_users_library(
    accounts: SignedIn, tmp_path: Path
) -> None:
    alice = visit_as(accounts, ALICE)
    upload(alice, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    bob = visit_as(accounts, BOB)
    upload(bob, [("Volcans.pdf", pdf(tmp_path, "v.pdf", VOLCANOES))])

    visit_as(accounts, ALICE, alice)
    alice.sidebar.button(key="delete-account").click().run()

    assert [t.value for t in alice.toast] == [
        "Votre compte et vos documents (1 document) sont supprimés."
    ]
    assert "Gateaux.pdf" not in sidebar_text(alice)
    assert "Volcans.pdf" in sidebar_text(visit_as(accounts, BOB, bob))
    st.cache_resource.clear()
    assert "Gateaux.pdf" not in sidebar_text(visit_as(accounts, ALICE))  # rien n'est revenu


def test_incomplete_identity_opens_no_library(accounts: SignedIn) -> None:
    app = visit_as(accounts, {"is_logged_in": True, "iss": GOOGLE, "name": "Sans identifiant"})

    assert "Connexion incomplète" in app.sidebar.error[0].value
    assert has_button(app, "login")
    assert len(app.sidebar.file_uploader) == 0


# ---------------------------------------------------------------- conditions, limites, journaux


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("DOCASSIST_PRIVACY_URL", "", "DOCASSIST_PRIVACY_URL"),
        ("DOCASSIST_PRIVACY_URL", "http://example.org/confidentialite", "DOCASSIST_PRIVACY_URL"),
        ("DOCASSIST_PRIVACY_URL", "https://", "DOCASSIST_PRIVACY_URL"),
        ("DOCASSIST_MAX_ACCOUNTS", "0", "DOCASSIST_MAX_ACCOUNTS"),
        ("DOCASSIST_ACCOUNT_DAILY_QUESTIONS", "-1", "DOCASSIST_ACCOUNT_DAILY_QUESTIONS"),
        ("DOCASSIST_ACCOUNT_DAILY_UPLOADS", "beaucoup", "DOCASSIST_ACCOUNT_DAILY_UPLOADS"),
    ],
)
def test_privacy_notice_and_limits_must_be_configured(
    accounts: SignedIn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    expected: str,
) -> None:
    monkeypatch.setenv(name, value)
    accounts.claims = ALICE

    app = open_app()

    assert expected in app.error[0].value
    assert len(app.sidebar.file_uploader) == 0
    assert not (tmp_path / "data" / "users").exists()


def test_terms_must_be_accepted_before_any_upload_or_question(
    accounts: SignedIn, tmp_path: Path
) -> None:
    app = visit_as(accounts, ALICE, accept=False)

    text = sidebar_text(app)
    assert "18 ans ou plus" in text and "aucune donnée sensible" in text
    assert f"[Notice de confidentialité]({PRIVACY_URL})" in text
    assert app.sidebar.button(key="accept-terms").disabled  # case non cochée
    assert len(app.sidebar.file_uploader) == 0
    assert app.chat_input[0].disabled
    assert has_button(app, "delete-account")  # l'effacement reste toujours possible

    accept_terms(app)
    assert len(app.sidebar.file_uploader) == 1
    assert users_db_collections(tmp_path, ALICE) == []  # acceptation : rien d'écrit encore
    upload(app, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    assert users_db_collections(tmp_path, ALICE) == [f"user-{key_of(ALICE)}-chunks"]

    st.cache_resource.clear()  # redémarrage, autre appareil
    later = visit_as(accounts, ALICE, accept=False)
    assert not has_button(later, "accept-terms")  # enregistrée avec la bibliothèque
    assert len(later.sidebar.file_uploader) == 1


def test_terms_are_asked_again_until_the_first_upload(accounts: SignedIn, tmp_path: Path) -> None:
    accept_terms(visit_as(accounts, ALICE, accept=False))  # acceptées, rien envoyé

    again = visit_as(accounts, ALICE, accept=False)  # nouvelle session

    assert has_button(again, "accept-terms")
    assert users_db_collections(tmp_path, ALICE) == []


def test_daily_question_limit_is_checked_before_calling_the_llm(
    accounts: SignedIn, pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_ACCOUNT_DAILY_QUESTIONS", "1")
    alice = visit_as(accounts, ALICE)
    upload(alice, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])

    ask(alice, "Combien de temps cuit le gâteau ?")
    ask(alice, "Et à quelle température ?")

    assert len(pipeline.sent) == 1  # la seconde question n'a jamais atteint le LLM
    assert "limite de 1 questions par jour" in alice.chat_message[3].error[0].value
    bob = visit_as(accounts, BOB)  # le budget d'Alice ne touche pas Bob
    upload(bob, [("Volcans.pdf", pdf(tmp_path, "v.pdf", VOLCANOES))])
    ask(bob, "Que rejette un volcan ?")
    assert len(pipeline.sent) == 2


def test_daily_upload_limit_is_checked_before_indexing(
    accounts: SignedIn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_ACCOUNT_DAILY_UPLOADS", "1")
    app = visit_as(accounts, ALICE)
    upload(app, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])

    upload(app, [("Volcans.pdf", pdf(tmp_path, "v.pdf", VOLCANOES))])

    assert "Volcans.pdf : limite de 1 envois par jour atteinte" in app.sidebar.error[0].value
    text = sidebar_text(app)
    assert "Gateaux.pdf" in text and "Volcans.pdf" not in text


def test_account_cap_refuses_only_new_libraries(
    accounts: SignedIn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_MAX_ACCOUNTS", "1")
    alice = visit_as(accounts, ALICE)
    upload(alice, [("Gateaux.pdf", pdf(tmp_path, "c.pdf", CAKES))])
    bob = visit_as(accounts, BOB)

    upload(bob, [("Volcans.pdf", pdf(tmp_path, "v.pdf", VOLCANOES))])

    assert "nombre maximal de bibliothèques est atteint" in bob.sidebar.error[0].value
    assert users_db_collections(tmp_path, BOB) == []
    upload(visit_as(accounts, ALICE, alice), [("Volcans.pdf", pdf(tmp_path, "v.pdf", VOLCANOES))])
    assert "Volcans.pdf" in sidebar_text(alice)  # un compte existant continue normalement


def test_unexpected_indexing_error_is_logged_without_the_filename(
    accounts: SignedIn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = visit_as(accounts, ALICE)

    def broken(_library: DocumentLibrary, name: str, _data: bytes) -> None:
        raise RuntimeError(f"panne en lisant /tmp/{name}")

    monkeypatch.setattr(DocumentLibrary, "add", broken)
    with caplog.at_level(logging.INFO):
        upload(app, [("Dossier-medical-Dupont.pdf", pdf(tmp_path, "c.pdf", CAKES))])

    assert "échec de l'indexation d'un document envoyé (RuntimeError)" in caplog.text
    assert "Dupont" not in caplog.text
    error = app.sidebar.error[0].value  # l'utilisateur, lui, voit le nom de son fichier
    assert error == "Dossier-medical-Dupont.pdf : erreur inattendue lors de l'indexation"


def test_document_without_its_record_is_not_labelled_as_a_cli_document(
    accounts: SignedIn, tmp_path: Path
) -> None:
    """Envoi interrompu entre l'écriture des passages et celle de l'enregistrement : la
    bibliothèque d'un utilisateur ne contient jamais de document « indexé avec la CLI »."""
    app = visit_as(accounts, ALICE)
    cakes = pdf(tmp_path, "c.pdf", CAKES)
    upload(app, [("Gateaux.pdf", cakes)])
    client = bootstrap.build_user_store("local", tmp_path / "data" / "users").client_for(
        UserIdentity.from_claims(ALICE)
    )
    collection = client.get_collection(f"user-{key_of(ALICE)}-chunks", embedding_function=None)
    first = collection.get(
        ids=[f"{doc_id(cakes)}:0"], include=["documents", "metadatas", "embeddings"]
    )
    assert first["metadatas"] is not None
    metadata = dict(first["metadatas"][0])
    metadata.pop("document_record")
    collection.delete(ids=first["ids"])
    collection.add(
        ids=first["ids"],
        embeddings=first["embeddings"],
        documents=first["documents"],
        metadatas=[metadata],
    )

    app.run()

    meta = [
        html.unescape(m)
        for m in re.findall(r"<div class='doc-meta'>(.*?)</div>", sidebar_text(app))
    ]
    assert "1 passage · détails de l'envoi indisponibles" in meta
    assert not any("indexé avec la CLI" in m for m in meta)


# ---------------------------------------------------------------- déploiement : modèle


@pytest.fixture
def downloads(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Remplace le téléchargement réel (réseau) : enregistre le dossier visé."""
    calls: list[Path] = []

    def fake_download(settings: Settings) -> Path:
        calls.append(settings.embedding_model_dir)
        return settings.embedding_model_dir

    monkeypatch.setattr(model_download, "download_embedding_model", fake_download)
    return calls


def test_missing_model_is_downloaded_once_when_enabled(
    downloads: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_MODEL_AUTO_DOWNLOAD", "true")

    app = start()
    ask(app, "Que font les chats ?")
    app.run()

    assert downloads == [Path("models") / "multilingual-e5-small"]


def test_model_is_never_downloaded_by_default(downloads: list[Path]) -> None:
    app = start()
    ask(app, "Que font les chats ?")

    assert downloads == []


def test_present_model_is_not_downloaded_again(
    downloads: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "models" / "multilingual-e5-small" / "model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.touch()
    monkeypatch.setenv("DOCASSIST_MODEL_AUTO_DOWNLOAD", "true")

    start()

    assert downloads == []


def test_invalid_auto_download_value_is_reported(
    downloads: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCASSIST_MODEL_AUTO_DOWNLOAD", "parfois")

    app = start()

    assert "DOCASSIST_MODEL_AUTO_DOWNLOAD" in app.error[0].value
    assert downloads == []
