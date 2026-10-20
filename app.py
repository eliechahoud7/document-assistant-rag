"""Interface web Streamlit : ``streamlit run app.py`` (depuis la racine du projet).

Réutilise le pipeline existant (``bootstrap``) sans le dupliquer. Une seule bibliothèque de
documents : l'index de ``DOCASSIST_DATA_DIR`` (``data`` par défaut), le même que celui de la
CLI (``index``, ``search``, ``ask``). On y ajoute des PDF, on les consulte et on les supprime ;
les questions portent sur l'ensemble de la bibliothèque, avec citations des sources.
"""

from __future__ import annotations

import html
import logging
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import streamlit as st

from doc_assistant import bootstrap
from doc_assistant.config import ConfigError, Settings
from doc_assistant.domain.errors import (
    DocumentNotFoundError,
    DocumentRegistryError,
    DocumentScopeError,
    EmbeddingModelMismatchError,
    EmbeddingModelNotFoundError,
    IngestionError,
    LLMConfigurationError,
    LLMError,
    LLMQuotaError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    TranslationModelNotFoundError,
)
from doc_assistant.domain.models import Answer
from doc_assistant.domain.ports import Embedder
from doc_assistant.indexing import model_download
from doc_assistant.services.library_service import DocumentLibrary, LibraryEntry
from doc_assistant.services.qa_service import QAService
from doc_assistant.services.upload_service import MAX_UPLOAD_BYTES

logger = logging.getLogger("doc_assistant.app")

# Gestion des documents (ajout et suppression) depuis l'interface : à désactiver pour un
# déploiement public, où tout visiteur pourrait sinon modifier la bibliothèque.
APP_UPLOADS_ENV = "DOCASSIST_APP_UPLOADS"
# Déploiement (Streamlit Community Cloud) ; inactifs par défaut en local.
# Téléchargement du modèle d'embeddings au premier démarrage s'il est absent.
MODEL_AUTO_DOWNLOAD_ENV = "DOCASSIST_MODEL_AUTO_DOWNLOAD"
# Mode démonstration : dossier de PDF publics indexés au démarrage ; ajout et suppression
# toujours désactivés.
DEMO_DOCS_ENV = "DOCASSIST_DEMO_DOCS_DIR"
_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")

# Messages destinés à l'utilisateur, sans détail technique (fournisseur, modèle, variables) :
# le message d'origine de l'exception est écrit dans le terminal.
# Ordre important : les sous-classes avant leur classe mère.
LLM_ERROR_MESSAGES: tuple[tuple[type[LLMError], str, str], ...] = (
    (
        LLMQuotaError,
        "Limite de requêtes atteinte",
        "Le service de réponses a atteint sa limite d'utilisation. Réessayez plus tard.",
    ),
    (
        LLMTimeoutError,
        "Délai de réponse dépassé",
        "La réponse a pris trop de temps. Réessayez dans quelques instants.",
    ),
    (
        LLMUnavailableError,
        "Service de réponses indisponible",
        "Le service de réponses est momentanément indisponible. Réessayez dans quelques instants.",
    ),
    (
        LLMConfigurationError,
        "Service de réponses mal configuré",
        "Le service de réponses n'est pas correctement configuré.",
    ),
    (
        LLMResponseError,
        "Réponse inexploitable",
        "La réponse obtenue n'a pas pu être exploitée. Reformulez la question ou réessayez.",
    ),
)
SETUP_ERROR_TEXT = "Le service de réponses n'est pas disponible pour le moment."
LIBRARY_ERROR_TEXT = "Vos documents ne peuvent pas être chargés pour le moment."


class SetupError(Exception):
    """Bibliothèque ou service de réponses inutilisable : configuration ou modèle manquant."""


@dataclass(frozen=True, slots=True)
class Source:
    number: int
    filename: str
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    kind: Literal["question", "answer", "abstention", "error"] = "question"
    model: str = ""
    detail: str = ""  # raison de l'abstention ou titre de l'erreur
    sources: tuple[Source, ...] = field(default_factory=tuple)
    scope: str = ""  # document interrogé (libellé) ; vide : tous les documents


@dataclass(frozen=True, slots=True)
class Notice:
    """Message affiché après un ajout ou une suppression (conservé jusqu'à l'action suivante)."""

    kind: Literal["success", "info", "error"]
    text: str


# ---------------------------------------------------------------- pipeline


def _flag(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name, "true" if default else "false").strip().lower()
    if raw not in _TRUE + _FALSE:
        raise ConfigError(f"{name} doit valoir true ou false, reçu {raw!r}")
    return raw in _TRUE


def demo_documents_dir(environ: Mapping[str, str]) -> Path | None:
    """Dossier des documents de démonstration ; ``None`` hors mode démonstration."""
    raw = environ.get(DEMO_DOCS_ENV, "").strip()
    return Path(raw) if raw else None


def uploads_enabled(environ: Mapping[str, str]) -> bool:
    """Ajout et suppression depuis l'interface ; toujours désactivés en mode démonstration."""
    enabled = _flag(environ, APP_UPLOADS_ENV, default=True)
    if demo_documents_dir(environ) is None:
        return enabled
    if APP_UPLOADS_ENV in environ and enabled:
        logger.warning("%s ignoré : mode démonstration (%s)", APP_UPLOADS_ENV, DEMO_DOCS_ENV)
    return False


def model_auto_download(environ: Mapping[str, str]) -> bool:
    return _flag(environ, MODEL_AUTO_DOWNLOAD_ENV, default=False)


def _embedding_key(settings: Settings) -> str:
    return repr(
        (
            settings.models_dir,
            settings.embedding_model,
            settings.embedding_model_revision,
            settings.embedding_device,
            settings.embedding_batch_size,
            # La traduction des requêtes enveloppe l'embedder : elle fait partie de la clé.
            settings.query_translation,
            settings.translation_model,
            settings.translation_model_revision,
        )
    )


@st.cache_resource(show_spinner=False)
def _cached_embedder(cache_key: str, _settings: Settings) -> Embedder:
    """Modèle E5 chargé une seule fois, partagé par la recherche et l'indexation.

    Avec ``DOCASSIST_MODEL_AUTO_DOWNLOAD=true`` (cloud), un modèle absent est d'abord téléchargé
    (révision figée, fichiers nécessaires seulement) ; l'embedder reste ensuite hors ligne.
    """
    del cache_key  # sert uniquement de clé de cache
    weights = _settings.embedding_model_dir / "model.safetensors"
    if model_auto_download(os.environ) and not weights.is_file():
        logger.info("modèle d'embeddings absent : téléchargement de %s", _settings.embedding_model)
        model_download.download_embedding_model(_settings)
    return bootstrap.build_embedder(_settings)


@st.cache_resource(show_spinner=False, max_entries=4)
def _cached_qa_service(cache_key: str, _settings: Settings) -> QAService:
    """Un service par configuration complète : la clé inclut ``data_dir``, donc l'index.

    Les exceptions ne sont pas mises en cache : une clé API ajoutée ensuite est prise en compte
    au rechargement suivant.
    """
    del cache_key
    # Configuration LLM vérifiée avant de charger le modèle d'embeddings, comme la CLI.
    generator = bootstrap.build_generator(_settings)
    embedder = _cached_embedder(_embedding_key(_settings), _settings)
    service = bootstrap.build_qa_service(_settings, embedder, generator)
    # Diagnostic pour l'exploitant, hors de l'interface.
    logger.info(
        "service de réponses prêt : fournisseur=%s, modèle=%s, index=%s",
        _settings.llm_provider,
        generator.model_name,
        _settings.chroma_dir,
    )
    return service


def index_exists(settings: Settings) -> bool:
    return (settings.chroma_dir / "chroma.sqlite3").is_file()


def load_service(settings: Settings) -> QAService:
    """Service de réponses, sans jamais créer d'index (Chroma en créerait un vide)."""
    if not index_exists(settings):
        raise SetupError(f"index introuvable : {settings.chroma_dir / 'chroma.sqlite3'}")
    try:
        return _cached_qa_service(repr(settings), settings)
    except (
        EmbeddingModelNotFoundError,
        EmbeddingModelMismatchError,
        TranslationModelNotFoundError,
        LLMError,
    ) as exc:
        raise SetupError(str(exc)) from exc


@st.cache_resource(show_spinner=False, max_entries=4)
def _cached_library(cache_key: str, _settings: Settings) -> DocumentLibrary:
    """Bibliothèque existante (indexation, liste, suppression) ; sans configuration LLM."""
    del cache_key
    embedder = _cached_embedder(_embedding_key(_settings), _settings)
    return bootstrap.build_document_library(_settings, embedder)


def get_library(settings: Settings, *, create: bool) -> DocumentLibrary | None:
    """Bibliothèque de l'index ; ``None`` si l'index n'existe pas encore et ``create`` est
    faux (l'ouvrir le créerait : seul un premier ajout de PDF le crée)."""
    if not create and not index_exists(settings):
        return None
    try:
        return _cached_library(repr(settings), settings)
    except (
        EmbeddingModelNotFoundError,
        EmbeddingModelMismatchError,
        TranslationModelNotFoundError,
    ) as exc:
        raise SetupError(str(exc)) from exc


@st.cache_resource(show_spinner=False, max_entries=4)
def _seed_demo_documents(cache_key: str, _settings: Settings, _folder: Path) -> int:
    """Indexe les PDF de démonstration, une fois par processus, avec le pipeline d'envoi
    (même déduplication par SHA-256 : un document déjà présent n'est pas réindexé)."""
    del cache_key
    pdfs = sorted(_folder.glob("*.pdf")) if _folder.is_dir() else []
    if not pdfs:
        logger.error("aucun PDF de démonstration dans %s", _folder)
        return 0
    library = get_library(_settings, create=True)
    assert library is not None
    indexed = 0
    for pdf in pdfs:
        try:
            outcome = library.add(pdf.name, pdf.read_bytes())
        except (IngestionError, DocumentRegistryError) as exc:
            logger.error("document de démonstration ignoré : %s", exc)
            continue
        indexed += outcome.kind == "indexed"
    logger.info("démonstration : %d PDF, %d nouvellement indexé(s)", len(pdfs), indexed)
    return indexed


def seed_demo_documents(settings: Settings, folder: Path) -> None:
    _seed_demo_documents(repr((settings, folder)), settings, folder)


def list_documents(library: DocumentLibrary) -> tuple[list[LibraryEntry], str]:
    """Documents et éventuel avertissement (registre illisible : liste issue de l'index)."""
    try:
        return library.documents(), ""
    except DocumentRegistryError as exc:
        return library.documents_from_index(), str(exc)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' if count > 1 else ''}"


def index_files(settings: Settings, files: Sequence[tuple[str, bytes]]) -> list[Notice]:
    """Indexe les PDF envoyés un par un (un échec n'empêche pas les suivants)."""
    try:
        library = get_library(settings, create=True)
    except SetupError as exc:
        return [Notice("error", str(exc))]
    assert library is not None
    notices: list[Notice] = []
    for name, data in files:
        try:
            outcome = library.add(name, data)
        except (IngestionError, DocumentRegistryError) as exc:
            notices.append(Notice("error", str(exc)))
            continue
        except Exception:
            logger.exception("échec de l'indexation de %s", name)
            notices.append(Notice("error", f"{name} : erreur inattendue (détail dans le terminal)"))
            continue
        if outcome.kind == "duplicate":
            same = (
                ""
                if outcome.duplicate_of == outcome.filename
                else f" (même contenu que « {outcome.duplicate_of} »)"
            )
            notices.append(
                Notice(
                    "info",
                    f"« {outcome.filename} » est déjà dans vos documents{same} : "
                    "il n'a pas été réindexé.",
                )
            )
        else:
            entry = outcome.entry
            notices.append(Notice("success", f"« {entry.filename} » ajouté à vos documents."))
    return notices


def delete_document(settings: Settings, doc_id: str) -> Notice:
    try:
        library = get_library(settings, create=False)
        if library is None:
            return Notice("error", "aucun index : rien à supprimer.")
        outcome = library.delete(doc_id)
    except (SetupError, DocumentNotFoundError, DocumentRegistryError) as exc:
        return Notice("error", str(exc))
    return Notice("success", f"« {outcome.filename} » supprimé de vos documents.")


def to_message(answer: Answer, scope: str = "") -> ChatMessage:
    if answer.abstained:
        return ChatMessage(
            "assistant",
            answer.text,
            kind="abstention",
            model=answer.model,
            detail=answer.abstain_reason or "",
            scope=scope,
        )
    sources = tuple(
        Source(c.number, c.filename, c.page_number, c.chunk.text) for c in answer.citations
    )
    return ChatMessage(
        "assistant", answer.text, kind="answer", model=answer.model, sources=sources, scope=scope
    )


def ask(
    service: QAService, question: str, *, doc_id: str | None = None, scope: str = ""
) -> ChatMessage:
    """Répond à partir de toute la bibliothèque, ou du seul document ``doc_id``.

    Le filtre est appliqué par l'index pendant la recherche ; ``QAService.retrieve`` vérifie
    en plus, avant l'appel au LLM, qu'aucun passage n'appartient à un autre document.
    """
    try:
        return to_message(service.ask(question, doc_id=doc_id).answer, scope)
    except DocumentScopeError as exc:
        logger.error("filtre par document non respecté : %s", exc)
        return ChatMessage("assistant", str(exc), kind="error", detail="Réponse refusée")
    except LLMError as exc:
        logger.warning("échec de la génération (%s) : %s", type(exc).__name__, exc)
        title, text = next((t, m) for cls, t, m in LLM_ERROR_MESSAGES if isinstance(exc, cls))
        return ChatMessage("assistant", text, kind="error", detail=title)
    except ValueError as exc:
        return ChatMessage("assistant", str(exc), kind="error", detail="Question invalide")
    except Exception:
        logger.exception("erreur inattendue pendant la réponse")
        return ChatMessage(
            "assistant",
            "Une erreur inattendue s'est produite ; le détail est dans le terminal.",
            kind="error",
            detail="Erreur inattendue",
        )


# ---------------------------------------------------------------- interface

CSS = """
<style>
:root { --accent: #2563eb; --accent-soft: #eff6ff; --border: #e2e8f0; --muted: #64748b; }
.block-container { padding-top: 2.5rem; padding-bottom: 7rem; max-width: 820px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
.brand { display: flex; align-items: center; gap: .6rem; margin-bottom: .25rem; }
.brand-logo {
  width: 2.1rem; height: 2.1rem; border-radius: .6rem; display: grid; place-items: center;
  background: linear-gradient(135deg, #2563eb, #60a5fa); color: #fff; font-weight: 700;
}
.brand-name { font-weight: 700; font-size: 1.1rem; }
.section-title { font-weight: 700; font-size: .95rem; margin: .25rem 0 .5rem; }
.count {
  display: inline-block; min-width: 1.4rem; padding: 0 .45rem; margin-left: .35rem;
  border-radius: 999px; font-size: .75rem; text-align: center;
  background: var(--accent-soft); color: var(--accent);
}
.doc-name { font-weight: 600; font-size: .9rem; word-break: break-word; }
.doc-meta { color: var(--muted); font-size: .78rem; line-height: 1.45; }
.badge {
  display: inline-block; padding: 0 .5rem; border-radius: 999px; font-size: .72rem;
  font-weight: 600; margin-top: .2rem;
}
.badge-ok { background: #dcfce7; color: #166534; }
.badge-warn { background: #fef3c7; color: #92400e; }
.hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero h1 { font-size: 2rem; font-weight: 700; margin-bottom: .35rem; }
.hero p { color: var(--muted); margin: 0; }
.pill {
  display: inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .8rem;
  font-weight: 600; background: var(--accent-soft); color: var(--accent);
}
[data-testid="stChatMessage"] {
  border-radius: 1rem; padding: .85rem 1rem; margin-bottom: .6rem;
  border: 1px solid var(--border); background: #fff;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--accent-soft); border-color: #bfdbfe; flex-direction: row-reverse;
  margin-left: 12%;
}
.meta { color: var(--muted); font-size: .8rem; margin-top: .25rem; }
[data-testid="stElementContainer"]:has(.docassist-scroll) { display: none; }
.source-text {
  white-space: pre-wrap; font-size: .85rem; line-height: 1.5; color: #334155;
  border-left: 3px solid var(--accent); padding: .25rem 0 .25rem .75rem;
}
@media (max-width: 640px) {
  .block-container { padding-top: 1.25rem; }
  .hero h1 { font-size: 1.5rem; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { margin-left: 0; }
}
</style>
"""

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"


def conversation(scope_key: str) -> list[ChatMessage]:
    """Historique de la session pour une portée : un ``doc_id``, ou ``ALL_DOCUMENTS``.

    Clé = identifiant du document (jamais son nom) : deux fichiers homonymes ont chacun le leur.
    """
    histories: dict[str, list[ChatMessage]] = st.session_state.histories
    return histories.setdefault(scope_key, [])


def reset_conversation() -> None:
    """« Nouvelle conversation » : vide seulement l'historique de la portée affichée."""
    st.session_state.histories.pop(st.session_state.get("scope", ALL_DOCUMENTS), None)


def scroll_to_latest_question(token: str) -> None:
    """Fait défiler la page jusqu'à la dernière question posée.

    Émis uniquement pendant l'exécution qui traite une question : la page n'est jamais
    ramenée en bas pendant la lecture d'un ancien message. Le navigateur réexécute le script
    dès que son contenu change, d'où le jeton ; son conteneur est masqué par la CSS.
    """
    st.html(
        f"<div class='docassist-scroll' data-token='{html.escape(token)}'></div>"
        "<script>setTimeout(() => {"
        "const users = [...document.querySelectorAll('[data-testid=\"stChatMessage\"]')]"
        ".filter((m) => m.querySelector('[data-testid=\"stChatMessageAvatarUser\"]'));"
        "const last = users[users.length - 1];"
        "if (last) last.scrollIntoView({behavior: 'smooth', block: 'start'});"
        "}, 50);</script>",
        unsafe_allow_javascript=True,
    )


def render_message(message: ChatMessage) -> None:
    if message.role == "user":
        st.markdown(message.content)
        return
    if message.kind == "error":
        st.error(f"**{message.detail}** — {message.content}", icon=":material/error:")
        return
    if message.kind == "abstention":
        st.info(
            f"Je n'ai pas trouvé la réponse dans « {message.scope} »."
            if message.scope
            else "Je n'ai pas trouvé la réponse dans vos documents.",
            icon=":material/search_off:",
        )
        if message.content:
            st.markdown(message.content)
        st.markdown(
            f"<div class='meta'>Raison : {html.escape(message.detail)}</div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(message.content)
    count = len(message.sources)
    scope = f"Document : {message.scope}" if message.scope else "Tous les documents"
    st.markdown(
        f"<div class='meta'>{count} source{'s' if count > 1 else ''} · {html.escape(scope)}</div>",
        unsafe_allow_html=True,
    )
    for source in message.sources:
        with st.expander(
            f"[{source.number}] {source.filename} — page {source.page_number}",
            icon=":material/description:",
        ):
            st.markdown(
                f"<div class='source-text'>{html.escape(source.text)}</div>",
                unsafe_allow_html=True,
            )


def render_notices() -> None:
    """Succès : notification discrète, une seule fois. Doublons et erreurs : restent affichés
    jusqu'à l'action suivante."""
    notices: list[Notice] = st.session_state.get("notices", [])
    for notice in notices:
        if notice.kind == "success":
            st.toast(notice.text, icon=":material/check_circle:")
    st.session_state.notices = [n for n in notices if n.kind != "success"]
    show = {"info": st.info, "error": st.error}
    for notice in st.session_state.notices:
        show[notice.kind](notice.text)


def document_meta(entry: LibraryEntry) -> str:
    parts = [] if entry.page_count is None else [plural(entry.page_count, "page")]
    parts.append(plural(entry.chunk_count, "passage"))
    if entry.added_at is not None:
        parts.append(f"ajouté le {entry.added_at.astimezone():%d/%m/%Y à %H:%M}")
    elif entry.source == "external":
        parts.append("indexé avec la CLI")
    return " · ".join(parts)


def render_documents(
    settings: Settings, entries: Sequence[LibraryEntry], warning: str, manage: bool
) -> None:
    st.markdown(
        f"<div class='section-title'>Mes documents<span class='count'>{len(entries)}</span></div>",
        unsafe_allow_html=True,
    )
    if warning:
        st.warning(warning)
    if not entries:
        st.caption(
            "Aucun document pour l'instant." + (" Ajoutez un PDF ci-dessous." if manage else "")
        )
        return
    for entry in entries:
        badge = (
            "<span class='badge badge-ok'>Indexé</span>"
            if entry.status == "indexed"
            else "<span class='badge badge-warn'>Absent de l'index</span>"
        )
        with st.container(border=True):
            info, action = st.columns([5, 1], vertical_alignment="center")
            info.markdown(
                f"<div class='doc-name'>{html.escape(entry.filename)}</div>"
                f"<div class='doc-meta'>{html.escape(document_meta(entry))}</div>{badge}",
                unsafe_allow_html=True,
            )
            if manage:
                with action.popover("", icon=":material/delete:", help="Supprimer ce document"):
                    st.markdown(
                        f"Supprimer **{html.escape(entry.filename)}** ? Ses passages seront "
                        "retirés de l'index ; les autres documents ne sont pas touchés."
                    )
                    if st.button(
                        "Supprimer définitivement",
                        key=f"delete-{entry.doc_id}",
                        type="primary",
                        width="stretch",
                    ):
                        notice = delete_document(settings, entry.doc_id)
                        if notice.kind == "success":
                            # Son historique disparaît avec lui (un nouvel envoi du même
                            # contenu, même doc_id, repart d'une conversation vide).
                            st.session_state.histories.pop(entry.doc_id, None)
                        st.session_state.notices = [notice]
                        st.rerun()


def render_upload(settings: Settings) -> None:
    st.markdown("<div class='section-title'>Ajouter des PDF</div>", unsafe_allow_html=True)
    files = st.file_uploader(
        "Fichiers PDF",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
        help="PDF à texte sélectionnable (pas d'OCR), 20 Mo maximum par fichier.",
        key=f"uploader-{st.session_state.uploader_round}",  # nouvelle clé = sélection vidée
        label_visibility="collapsed",
    )
    files = list(files or [])
    if files:
        names = ", ".join(f"« {f.name} »" for f in files)
        st.caption(f"Sélection : {names}")
    label = f"Indexer {len(files)} fichier{'s' if len(files) > 1 else ''}" if files else "Indexer"
    if st.button(
        label,
        key="index-files",
        icon=":material/library_add:",
        type="primary",
        disabled=not files,
        width="stretch",
    ):
        with st.spinner("Extraction, découpage et calcul des embeddings…"):
            st.session_state.notices = index_files(
                settings, [(f.name, f.getvalue()) for f in files]
            )
        st.session_state.uploader_round += 1
        st.rerun()  # liste à jour, et service de réponses disponible dès le premier document


ALL_DOCUMENTS = "__all__"
ALL_DOCUMENTS_LABEL = "Tous les documents"


def document_labels(entries: Sequence[LibraryEntry]) -> dict[str, str]:
    """Libellé par ``doc_id`` ; deux fichiers de même nom sont distingués par leur identifiant."""
    names = Counter(e.filename for e in entries)
    return {
        e.doc_id: e.filename if names[e.filename] == 1 else f"{e.filename} · {e.doc_id[:8]}"
        for e in entries
    }


def render_scope(entries: Sequence[LibraryEntry]) -> tuple[str | None, str]:
    """Choix du document interrogé ; renvoie (``doc_id`` ou ``None`` pour tous, libellé).

    La sélection est conservée entre les réexécutions ; si le document choisi n'existe plus
    (supprimé), elle revient à « Tous les documents ».
    """
    labels = document_labels([e for e in entries if e.status == "indexed"])
    options = [ALL_DOCUMENTS, *labels]
    if st.session_state.get("scope") not in options:
        st.session_state.scope = ALL_DOCUMENTS
    choice = st.selectbox(
        "Document interrogé",
        options,
        key="scope",
        format_func=lambda key: ALL_DOCUMENTS_LABEL if key == ALL_DOCUMENTS else labels[key],
        help="Les réponses ne s'appuient que sur le document choisi.",
        disabled=not labels,
    )
    if choice is None or choice == ALL_DOCUMENTS:
        return None, ""
    return choice, labels[choice]


def main() -> None:
    st.set_page_config(
        page_title="Doc Assistant", page_icon=":material/menu_book:", layout="centered"
    )
    logging.getLogger("doc_assistant").setLevel(logging.INFO)
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("histories", {})  # portée -> list[ChatMessage]
    st.session_state.setdefault("questions_asked", 0)  # jeton du défilement automatique
    st.session_state.setdefault("uploader_round", 0)

    try:
        settings, manage = Settings.from_env(), uploads_enabled(os.environ)
        model_auto_download(os.environ)  # valeur vérifiée dès le démarrage
    except ConfigError as exc:
        st.error(f"**Configuration invalide** — {exc}")
        return
    demo_dir = demo_documents_dir(os.environ)

    entries: list[LibraryEntry] = []
    warning = library_error = setup_error = ""
    service: QAService | None = None
    try:
        with st.spinner("Chargement de vos documents…"):
            if demo_dir is not None:
                seed_demo_documents(settings, demo_dir)
            library = get_library(settings, create=False)
            if library is not None:
                entries, warning = list_documents(library)
    except SetupError as exc:
        logger.error("bibliothèque indisponible : %s", exc)
        library_error = LIBRARY_ERROR_TEXT
    except Exception:
        logger.exception("échec du chargement de la bibliothèque")
        library_error = LIBRARY_ERROR_TEXT
    if entries:
        try:
            with st.spinner("Préparation du service de réponses…"):
                service = load_service(settings)
        except SetupError as exc:
            logger.error("réponses indisponibles : %s", exc)
            setup_error = SETUP_ERROR_TEXT

    with st.sidebar:
        st.markdown(
            "<div class='brand'><div class='brand-logo'>D</div>"
            "<div class='brand-name'>Doc Assistant</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("Réponses citées, fondées uniquement sur vos documents.")
        st.button(
            "Nouvelle conversation",
            icon=":material/add_comment:",
            on_click=reset_conversation,
            width="stretch",
        )
        st.divider()
        render_notices()
        render_documents(settings, entries, warning, manage)
        if manage:
            st.divider()
            render_upload(settings)

    doc_id, scope_label = render_scope(entries) if entries else (None, "")
    prompt = st.chat_input(
        f"Posez une question sur « {scope_label} »…"
        if scope_label
        else "Posez une question sur vos documents…",
        disabled=service is None,
        max_chars=2000,
    )
    question = prompt if isinstance(prompt, str) else None
    messages = conversation(doc_id or ALL_DOCUMENTS)

    if library_error:
        st.error(f"**Documents indisponibles** — {library_error}")
    elif not entries:
        st.info(
            "Votre bibliothèque est vide : ajoutez un premier PDF dans la barre latérale."
            if manage
            else "Aucun document n'est indexé pour le moment.",
            icon=":material/upload_file:",
        )
    elif setup_error:
        st.error(f"**Réponses indisponibles** — {setup_error}")

    if not messages and not question:
        count = len(entries)
        st.markdown(
            f"<div class='hero'><span class='pill'>"
            f"{html.escape(scope_label) if scope_label else plural(count, 'document')}</span>"
            "<h1>Que voulez-vous savoir ?</h1>"
            "<p>Chaque réponse cite le fichier et la page d'où elle provient.</p></div>",
            unsafe_allow_html=True,
        )

    for message in messages:
        with st.chat_message(
            message.role, avatar=USER_AVATAR if message.role == "user" else ASSISTANT_AVATAR
        ):
            render_message(message)

    if question and service is not None:
        user_message = ChatMessage("user", question)
        messages.append(user_message)
        st.session_state.questions_asked += 1
        token = str(st.session_state.questions_asked)
        with st.chat_message("user", avatar=USER_AVATAR):
            render_message(user_message)
        scroll_to_latest_question(f"{token}-question")  # pendant la génération
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Réflexion en cours…"):
                reply = ask(service, question, doc_id=doc_id, scope=scope_label)
            render_message(reply)
        messages.append(reply)
        scroll_to_latest_question(f"{token}-answer")  # réponse affichée sous la question


main()
