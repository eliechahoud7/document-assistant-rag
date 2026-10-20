"""Interface web Streamlit : ``streamlit run app.py`` (depuis la racine du projet).

Réutilise le pipeline existant (``bootstrap``) sans le dupliquer. Deux collections, chacune
avec son propre index Chroma, jamais modifié ici : l'application ne fait que des recherches.
"""

from __future__ import annotations

import html
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import streamlit as st

from doc_assistant import bootstrap
from doc_assistant.config import ConfigError, Settings
from doc_assistant.domain.errors import (
    EmbeddingModelMismatchError,
    EmbeddingModelNotFoundError,
    LLMConfigurationError,
    LLMError,
    LLMQuotaError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from doc_assistant.domain.models import Answer
from doc_assistant.domain.ports import Embedder
from doc_assistant.generation.gemini_generator import API_KEY_ENV
from doc_assistant.services.qa_service import QAService

logger = logging.getLogger("doc_assistant.app")


@dataclass(frozen=True, slots=True)
class Collection:
    key: str
    label: str
    caption: str
    data_dir: Path
    examples: tuple[str, ...]


COLLECTIONS: tuple[Collection, ...] = (
    Collection(
        "cv",
        "CV",
        "Parcours, compétences et expériences",
        Path("data"),
        ("Quelles sont les compétences techniques principales ?", "Quel est le parcours ?"),
    ),
    Collection(
        "courses",
        "Cours",
        "Supports de cours indexés",
        Path("data/courses"),
        ("Comment choisir k dans les k-moyennes ?", "Qu'est-ce que le surapprentissage ?"),
    ),
)
COLLECTIONS_BY_KEY = {collection.key: collection for collection in COLLECTIONS}

# Ordre important : les sous-classes avant leur classe mère.
LLM_ERROR_TITLES: tuple[tuple[type[LLMError], str], ...] = (
    (LLMQuotaError, "Quota Gemini atteint"),
    (LLMTimeoutError, "Délai de réponse dépassé"),
    (LLMUnavailableError, "Service Gemini indisponible"),
    (LLMConfigurationError, "Configuration Gemini refusée"),
    (LLMResponseError, "Réponse du modèle inexploitable"),
)


class SetupError(Exception):
    """Collection inutilisable : configuration, index ou modèle manquant."""


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


# ---------------------------------------------------------------- pipeline


def collection_settings(collection: Collection) -> Settings:
    """Configuration de l'environnement, avec le dossier de données propre à la collection."""
    return replace(Settings.from_env(), data_dir=collection.data_dir)


def _embedding_key(settings: Settings) -> str:
    return repr(
        (
            settings.models_dir,
            settings.embedding_model,
            settings.embedding_model_revision,
            settings.embedding_device,
            settings.embedding_batch_size,
        )
    )


@st.cache_resource(show_spinner=False)
def _cached_embedder(cache_key: str, _settings: Settings) -> Embedder:
    """Modèle E5 chargé une seule fois, partagé par les deux collections (même modèle)."""
    del cache_key  # sert uniquement de clé de cache
    return bootstrap.build_embedder(_settings)


@st.cache_resource(show_spinner=False, max_entries=len(COLLECTIONS))
def _cached_qa_service(cache_key: str, _settings: Settings) -> QAService:
    """Un service par configuration complète : la clé inclut ``data_dir``, donc l'index.

    Les exceptions ne sont pas mises en cache : une clé API ajoutée ensuite est prise en compte
    au rechargement suivant.
    """
    del cache_key
    # Configuration LLM vérifiée avant de charger le modèle d'embeddings, comme la CLI.
    generator = bootstrap.build_generator(_settings)
    embedder = _cached_embedder(_embedding_key(_settings), _settings)
    return bootstrap.build_qa_service(_settings, embedder, generator)


def load_service(settings: Settings) -> QAService:
    """Service de la collection, sans jamais créer d'index (Chroma en créerait un vide)."""
    index_file = settings.chroma_dir / "chroma.sqlite3"
    if not index_file.is_file():
        raise SetupError(
            f"index introuvable : {index_file}. Lancez l'application depuis la racine du projet "
            "ou indexez d'abord les documents avec la CLI."
        )
    try:
        return _cached_qa_service(repr(settings), settings)
    except (EmbeddingModelNotFoundError, EmbeddingModelMismatchError, LLMError) as exc:
        raise SetupError(str(exc)) from exc


def to_message(answer: Answer) -> ChatMessage:
    if answer.abstained:
        return ChatMessage(
            "assistant",
            answer.text,
            kind="abstention",
            model=answer.model,
            detail=answer.abstain_reason or "",
        )
    sources = tuple(
        Source(c.number, c.filename, c.page_number, c.chunk.text) for c in answer.citations
    )
    return ChatMessage("assistant", answer.text, kind="answer", model=answer.model, sources=sources)


def ask(service: QAService, question: str) -> ChatMessage:
    try:
        return to_message(service.ask(question).answer)
    except LLMError as exc:
        title = next(t for cls, t in LLM_ERROR_TITLES if isinstance(exc, cls))
        return ChatMessage("assistant", str(exc), kind="error", detail=title)
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
.source-text {
  white-space: pre-wrap; font-size: .85rem; line-height: 1.5; color: #334155;
  border-left: 3px solid var(--accent); padding: .25rem 0 .25rem .75rem;
}
.status { font-size: .85rem; line-height: 1.8; }
.status code { font-size: .78rem; }
@media (max-width: 640px) {
  .block-container { padding-top: 1.25rem; }
  .hero h1 { font-size: 1.5rem; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { margin-left: 0; }
}
</style>
"""

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)


def render_message(message: ChatMessage) -> None:
    if message.role == "user":
        st.markdown(message.content)
        return
    if message.kind == "error":
        st.error(f"**{message.detail}** — {message.content}", icon=":material/error:")
        return
    if message.kind == "abstention":
        st.info(
            "Je n'ai pas trouvé la réponse dans les documents de cette collection.",
            icon=":material/search_off:",
        )
        if message.content:
            st.markdown(message.content)
        st.markdown(
            f"<div class='meta'>Raison : {html.escape(message.detail)} · "
            f"{html.escape(message.model)}</div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(message.content)
    count = len(message.sources)
    st.markdown(
        f"<div class='meta'>{html.escape(message.model)} · "
        f"{count} source{'s' if count > 1 else ''}</div>",
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


def render_sidebar(collection_keys: Sequence[str]) -> Collection:
    with st.sidebar:
        st.markdown(
            "<div class='brand'><div class='brand-logo'>D</div>"
            "<div class='brand-name'>Doc Assistant</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("Réponses citées, fondées uniquement sur vos documents.")
        st.divider()
        key = st.radio(
            "Collection de documents",
            collection_keys,
            format_func=lambda k: COLLECTIONS_BY_KEY[k].label,
            captions=[COLLECTIONS_BY_KEY[k].caption for k in collection_keys],
            key="collection",
            on_change=reset_conversation,  # jamais d'historique mêlant deux collections
        )
        st.button(
            "Nouvelle conversation",
            icon=":material/add_comment:",
            on_click=reset_conversation,
            width="stretch",
        )
    return COLLECTIONS_BY_KEY[key or collection_keys[0]]


def render_status(settings: Settings | None, ready: bool) -> None:
    with st.sidebar:
        st.divider()
        st.markdown("**État**")
        if settings is None:
            st.markdown("<div class='status'>Configuration invalide</div>", unsafe_allow_html=True)
            return
        has_key = bool(os.environ.get(API_KEY_ENV, "").strip())
        rows = [
            f"Index : <code>{html.escape(settings.chroma_dir.as_posix())}</code>",
            f"Modèle : <code>{html.escape(settings.llm_model)}</code>",
            f"Clé {API_KEY_ENV} : {'configurée' if has_key else 'absente'}",
            f"Passages : top-{settings.retrieval_top_k} + {settings.context_neighbors} voisin(s)",
            f"Service : {'prêt' if ready else 'indisponible'}",
        ]
        st.markdown(f"<div class='status'>{'<br>'.join(rows)}</div>", unsafe_allow_html=True)


def queue_example(question: str) -> None:
    st.session_state.pending_question = question


def main() -> None:
    st.set_page_config(
        page_title="Doc Assistant", page_icon=":material/menu_book:", layout="centered"
    )
    logging.getLogger("doc_assistant").setLevel(logging.INFO)
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("messages", [])

    collection = render_sidebar([c.key for c in COLLECTIONS])

    settings: Settings | None = None
    service: QAService | None = None
    setup_error = ""
    try:
        settings = collection_settings(collection)
        with st.spinner("Chargement du modèle et de l'index…"):
            service = load_service(settings)
    except (ConfigError, SetupError) as exc:
        setup_error = str(exc)
    except Exception:
        logger.exception("échec de l'initialisation de la collection %s", collection.key)
        setup_error = "erreur inattendue à l'initialisation ; le détail est dans le terminal."
    render_status(settings, service is not None)

    prompt = st.chat_input(
        f"Posez une question sur « {collection.label} »…",
        disabled=service is None,
        max_chars=2000,
    )
    question = prompt if isinstance(prompt, str) else st.session_state.pop("pending_question", None)
    messages: list[ChatMessage] = st.session_state.messages

    if setup_error:
        st.error(f"**Collection « {collection.label} » indisponible** — {setup_error}")
        if API_KEY_ENV in setup_error or "DOCASSIST_LLM_PROVIDER" in setup_error:
            st.code(
                '$env:DOCASSIST_LLM_PROVIDER = "gemini"\n$env:GEMINI_API_KEY = "..."',
                language="powershell",
            )

    if not messages and not question:
        st.markdown(
            f"<div class='hero'><span class='pill'>{html.escape(collection.label)}</span>"
            "<h1>Que voulez-vous savoir ?</h1>"
            "<p>Chaque réponse cite le fichier et la page d'où elle provient.</p></div>",
            unsafe_allow_html=True,
        )
        if service is not None:
            columns = st.columns(len(collection.examples))
            for column, example in zip(columns, collection.examples, strict=True):
                column.button(
                    example,
                    on_click=queue_example,
                    args=(example,),
                    width="stretch",
                    key=f"example-{collection.key}-{example}",
                )

    for message in messages:
        with st.chat_message(
            message.role, avatar=USER_AVATAR if message.role == "user" else ASSISTANT_AVATAR
        ):
            render_message(message)

    if question and service is not None:
        user_message = ChatMessage("user", question)
        messages.append(user_message)
        with st.chat_message("user", avatar=USER_AVATAR):
            render_message(user_message)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Recherche des passages et rédaction de la réponse…"):
                reply = ask(service, question)
            render_message(reply)
        messages.append(reply)


main()
