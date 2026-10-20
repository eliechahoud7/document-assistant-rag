"""Interface Streamlit (app.py) exécutée hors ligne : faux embedder, faux index, faux LLM.

Le dossier courant est un dossier temporaire : les vrais index (data/chroma,
data/courses/chroma) ne sont jamais ouverts. La bibliothèque, son registre, l'indexation, le
chargeur PDF et le découpeur sont les vrais ; seul l'index est en mémoire (il survit aux
« redémarrages » simulés, comme un index sur disque).
"""

from __future__ import annotations

import _socket
import json
import socket
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import NetworkAccessError
from doc_assistant import bootstrap
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
from fakes import FakeEmbedder, InMemoryVectorStore, ScriptedAnswerGenerator
from pdf_factory import write_pdf

APP = Path(__file__).resolve().parents[2] / "app.py"
# Constructeurs réels, capturés avant que la fixture ne les remplace.
REAL_BUILD_LIBRARY = bootstrap.build_document_library
REAL_BUILD_QA_SERVICE = bootstrap.build_qa_service
TIMEOUT_S = 60
CLI_DOC = "aaaaaaaaaaaaaaaa"  # document « indexé avec la CLI » présent au démarrage


class RecordingGenerator(ScriptedAnswerGenerator):
    """Générateur scripté qui garde les passages reçus : ce qui serait envoyé à Gemini."""

    def __init__(self, script: dict[str, str | Exception], model_name: str) -> None:
        super().__init__(script, model_name=model_name)
        self.passages: list[list[SearchResult]] = []

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer:
        self.passages.append(list(passages))
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
        self.generator = RecordingGenerator(self.script, model_name="fake-gemini")
        return self.generator

    def build_qa_service(
        self, settings: Settings, embedder: Embedder, generator: AnswerGenerator
    ) -> QAService:
        self.qa_built_for.append(settings.data_dir)
        return QAService(Retriever(embedder, self.store, default_top_k=3), generator)

    def build_document_library(self, settings: Settings, embedder: Embedder) -> DocumentLibrary:
        self.libraries_built += 1
        # Comme ChromaVectorStore : ouvrir la bibliothèque crée l'index s'il n'existe pas.
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        (settings.chroma_dir / "chroma.sqlite3").touch()
        indexing = IndexingService(PypdfLoader(), RecursiveChunker(800, 100), embedder, self.store)
        registry = DocumentRegistry.for_data_dir(settings.data_dir)
        return DocumentLibrary(indexing, self.store, registry, settings.uploads_dir)


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
    ):
        monkeypatch.delenv(name, raising=False)
    fake = Pipeline({"Que font les chats ?": "Ils dorment [1].", "Inconnu ?": "abstain"})
    monkeypatch.setattr(bootstrap, "build_embedder", fake.build_embedder)
    monkeypatch.setattr(bootstrap, "build_generator", fake.build_generator)
    monkeypatch.setattr(bootstrap, "build_qa_service", fake.build_qa_service)
    monkeypatch.setattr(bootstrap, "build_document_library", fake.build_document_library)
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
    import shutil

    shutil.rmtree(tmp_path / "data" / "chroma")  # vrai index ChromaDB à la place du faux
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
