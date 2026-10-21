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
import secrets
import shutil
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import streamlit as st

from doc_assistant import bootstrap, web_identity
from doc_assistant.config import ConfigError, Settings
from doc_assistant.domain.errors import (
    AuthenticationError,
    DocumentNotFoundError,
    DocumentProtectedError,
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
from doc_assistant.domain.ports import Embedder, VectorStore
from doc_assistant.indexing import model_download
from doc_assistant.indexing.document_registry import REGISTRY_FILENAME
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.services.library_service import DocumentLibrary, LibraryEntry
from doc_assistant.services.qa_service import QAService
from doc_assistant.services.upload_service import MAX_UPLOAD_BYTES, MEGABYTE, safe_filename
from doc_assistant.services.user_libraries import (
    UserIdentity,
    UserLibraries,
    UserLibrary,
    UserLimits,
)

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
# Envois privés des visiteurs (mode démonstration uniquement) : chaque session de navigateur a
# sa propre bibliothèque, copie de celle de démonstration ; jamais de bibliothèque partagée.
SESSION_UPLOADS_ENV = "DOCASSIST_SESSION_UPLOADS"
UPLOAD_MAX_MB_ENV = "DOCASSIST_UPLOAD_MAX_MB"
UPLOAD_MAX_PAGES_ENV = "DOCASSIST_UPLOAD_MAX_PAGES"
UPLOAD_MAX_FILES_ENV = "DOCASSIST_UPLOAD_MAX_FILES"
MAX_SESSIONS_ENV = "DOCASSIST_MAX_SESSIONS"
# Bibliothèques personnelles et persistantes (connexion Google) : désactivées par défaut.
USER_LIBRARIES_ENV = "DOCASSIST_USER_LIBRARIES"
USER_STORE_ENV = "DOCASSIST_USER_STORE"  # chroma-cloud (défaut) | local (développement)
ACCOUNT_MAX_DOCUMENTS_ENV = "DOCASSIST_ACCOUNT_MAX_DOCUMENTS"
ACCOUNT_DAILY_QUESTIONS_ENV = "DOCASSIST_ACCOUNT_DAILY_QUESTIONS"
ACCOUNT_DAILY_UPLOADS_ENV = "DOCASSIST_ACCOUNT_DAILY_UPLOADS"
MAX_ACCOUNTS_ENV = "DOCASSIST_MAX_ACCOUNTS"
# Adresse de la notice de confidentialité, rédigée et publiée par l'exploitant (obligatoire :
# l'application n'en rédige aucune et n'affirme aucune conformité).
PRIVACY_URL_ENV = "DOCASSIST_PRIVACY_URL"
# Version des conditions affichées avant le premier usage ; à changer à chaque modification du
# texte (TERMS_POINTS) : chaque utilisateur doit alors les accepter de nouveau.
TERMS_VERSION = "2026-10-21"
USERS_DIRNAME = "users"  # base locale des bibliothèques (DOCASSIST_USER_STORE=local)
# Clés de [auth] (Secrets) pour un seul fournisseur OpenID Connect (Google).
AUTH_KEYS = ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
MIN_COOKIE_SECRET_LENGTH = 32
SESSION_TTL_S = 2 * 3600  # bibliothèque d'une session inactive supprimée après 2 h
SESSION_CLEANUP_INTERVAL_S = 600
SESSIONS_DIRNAME = "sessions"
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
SESSION_EXPIRED_TEXT = "Votre session a expiré : vos documents ajoutés ont été supprimés."
SESSIONS_FULL_TEXT = (
    "Trop de visiteurs ont ajouté des documents en ce moment : réessayez dans quelques minutes. "
    "Les documents de démonstration restent disponibles."
)


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


@dataclass(frozen=True, slots=True)
class UploadLimits:
    """Limites des envois d'un visiteur, vérifiées par ``DocumentLibrary`` (côté serveur)."""

    max_bytes: int
    max_pages: int
    max_documents: int
    max_sessions: int  # bibliothèques privées simultanées dans le processus


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, str(default)).strip()
    # Chiffres ASCII uniquement : str.isdigit accepte aussi « ² », que int() refuse.
    if not (raw.isascii() and raw.isdigit()) or int(raw) <= 0:
        raise ConfigError(f"{name} doit être un entier strictement positif, reçu {raw!r}")
    return int(raw)


def session_upload_limits(environ: Mapping[str, str], *, server_max_mb: int) -> UploadLimits | None:
    """Limites des envois privés ; ``None`` si ``DOCASSIST_SESSION_UPLOADS`` est désactivé.

    ``server_max_mb`` : ``server.maxUploadSize`` de Streamlit, seule limite que le serveur
    applique pendant la réception d'un fichier (celle du composant d'envoi n'est vérifiée que
    par le navigateur). La limite des visiteurs ne peut pas la dépasser.
    """
    if not _flag(environ, SESSION_UPLOADS_ENV, default=False):
        return None
    if demo_documents_dir(environ) is None:
        raise ConfigError(f"{SESSION_UPLOADS_ENV} exige le mode démonstration ({DEMO_DOCS_ENV})")
    max_mb = _positive_int(environ, UPLOAD_MAX_MB_ENV, 10)
    if max_mb > server_max_mb:
        raise ConfigError(
            f"{UPLOAD_MAX_MB_ENV} ({max_mb} Mo) dépasse server.maxUploadSize de Streamlit "
            f"({server_max_mb} Mo, .streamlit/config.toml)"
        )
    return UploadLimits(
        max_bytes=max_mb * MEGABYTE,
        max_pages=_positive_int(environ, UPLOAD_MAX_PAGES_ENV, 50),
        max_documents=_positive_int(environ, UPLOAD_MAX_FILES_ENV, 3),
        max_sessions=_positive_int(environ, MAX_SESSIONS_ENV, 20),
    )


@dataclass(frozen=True, slots=True)
class AccountsConfig:
    """Bibliothèques personnelles : base et limites. Ne contient aucun secret (sert de clé de
    cache) : les identifiants Chroma Cloud et ceux de [auth] restent dans l'environnement et
    dans les Secrets."""

    store: bootstrap.UserStoreKind
    limits: UserLimits
    privacy_url: str


def _auth_section(secrets: Mapping[str, object]) -> Mapping[str, object] | None:
    try:
        section = secrets.get("auth")
    except Exception:  # pas de secrets.toml : Streamlit lève une erreur à la lecture
        return None
    return section if isinstance(section, Mapping) else None


def user_libraries_config(
    environ: Mapping[str, str], secrets: Mapping[str, object], *, server_max_mb: int
) -> AccountsConfig | None:
    """Configuration des bibliothèques personnelles ; ``None`` si elles sont désactivées.

    Tout manque est une erreur de configuration (rien n'est alors activé) : section [auth]
    complète, ``cookie_secret`` assez long, Authlib installé, identifiants Chroma Cloud.
    Aucune valeur secrète n'apparaît dans les messages.
    """
    if not _flag(environ, USER_LIBRARIES_ENV, default=False):
        return None
    if _flag(environ, SESSION_UPLOADS_ENV, default=False):
        raise ConfigError(f"{USER_LIBRARIES_ENV} et {SESSION_UPLOADS_ENV} sont incompatibles")
    auth = _auth_section(secrets)
    missing = [k for k in AUTH_KEYS if not str((auth or {}).get(k, "")).strip()]
    if missing:
        raise ConfigError(
            f"{USER_LIBRARIES_ENV} exige la connexion Google : section [auth] des Secrets "
            f"incomplète ({', '.join(missing)})"
        )
    assert auth is not None
    if len(str(auth["cookie_secret"]).strip()) < MIN_COOKIE_SECRET_LENGTH:
        raise ConfigError(
            f"[auth] cookie_secret trop court (au moins {MIN_COOKIE_SECRET_LENGTH} caractères "
            "aléatoires)"
        )
    if not web_identity.login_dependency_available():
        raise ConfigError("connexion indisponible : installez streamlit[auth] (Authlib)")
    store = environ.get(USER_STORE_ENV, "chroma-cloud").strip().lower()
    if store not in ("chroma-cloud", "local"):
        raise ConfigError(f"{USER_STORE_ENV} doit valoir chroma-cloud ou local, reçu {store!r}")
    if store == "chroma-cloud":
        absent = [n for n in bootstrap.CHROMA_CLOUD_ENV if not environ.get(n, "").strip()]
        if absent:
            raise ConfigError(f"Chroma Cloud : variable(s) absente(s) : {', '.join(absent)}")
    privacy_url = environ.get(PRIVACY_URL_ENV, "").strip()
    if not privacy_url.startswith("https://") or len(privacy_url) <= len("https://"):
        raise ConfigError(
            f"{USER_LIBRARIES_ENV} exige {PRIVACY_URL_ENV} : adresse https:// de votre notice "
            "de confidentialité (affichée avant tout envoi)"
        )
    max_mb = _positive_int(environ, UPLOAD_MAX_MB_ENV, 10)
    if max_mb > server_max_mb:
        raise ConfigError(
            f"{UPLOAD_MAX_MB_ENV} ({max_mb} Mo) dépasse server.maxUploadSize de Streamlit "
            f"({server_max_mb} Mo, .streamlit/config.toml)"
        )
    return AccountsConfig(
        store="local" if store == "local" else "chroma-cloud",
        limits=UserLimits(
            max_bytes=max_mb * MEGABYTE,
            max_pages=_positive_int(environ, UPLOAD_MAX_PAGES_ENV, 50),
            max_documents=_positive_int(environ, ACCOUNT_MAX_DOCUMENTS_ENV, 20),
            daily_questions=_positive_int(environ, ACCOUNT_DAILY_QUESTIONS_ENV, 50),
            daily_uploads=_positive_int(environ, ACCOUNT_DAILY_UPLOADS_ENV, 20),
            max_accounts=_positive_int(environ, MAX_ACCOUNTS_ENV, 100),
        ),
        privacy_url=privacy_url,
    )


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


@dataclass(slots=True)
class ProcessState:
    """État partagé par toutes les sessions du processus. Streamlit réexécute ce script à
    chaque interaction : seul ``st.cache_resource`` conserve un objet d'une exécution à
    l'autre (des variables de module seraient recréées)."""

    template_lock: threading.Lock = field(default_factory=threading.Lock)
    indexing_lock: threading.Lock = field(default_factory=threading.Lock)  # CPU : un à la fois
    # Activité et nettoyage des sessions : une session vérifiée active n'est pas fermée.
    sessions_lock: threading.Lock = field(default_factory=threading.Lock)
    # Sessions dont l'index est ouvert dans ce processus, par dossier : le nettoyage les ferme
    # avant de supprimer leur dossier.
    sessions: dict[Path, VisitorSession] = field(default_factory=dict)
    last_cleanup: float = 0.0


@st.cache_resource(show_spinner=False)
def process_state() -> ProcessState:
    return ProcessState()


@dataclass(slots=True)
class VisitorSession:
    """Bibliothèque privée d'une session de navigateur : un dossier à nom aléatoire, copie de
    l'index de démonstration, que personne d'autre ne lit ni ne modifie."""

    settings: Settings
    store: VectorStore  # un seul accès à l'index, partagé par la bibliothèque et les réponses
    library: DocumentLibrary
    service: QAService | None = None
    closed: bool = False

    def close(self) -> None:
        """Libère l'index de la session : sous Windows, un fichier ouvert ne peut pas être
        supprimé. La session n'est plus utilisable ensuite."""
        self.closed = True
        if isinstance(self.store, ChromaVectorStore):
            self.store.close()


def sessions_root(settings: Settings) -> Path:
    return settings.data_dir / SESSIONS_DIRNAME


def cleanup_sessions(
    root: Path,
    *,
    now: float,
    sessions: dict[Path, VisitorSession],
    keep: Path | None = None,
) -> int:
    """Supprime les bibliothèques des sessions inactives depuis plus de ``SESSION_TTL_S`` ;
    celles encore ouvertes dans ce processus (``sessions``) sont d'abord fermées."""
    if not root.is_dir():
        return 0
    removed = 0
    for folder in root.iterdir():
        if folder == keep or not folder.is_dir():
            continue
        if now - folder.stat().st_mtime <= SESSION_TTL_S:
            continue
        visitor = sessions.pop(folder, None)
        if visitor is not None:
            visitor.close()
        shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():
            logger.warning("bibliothèque de session non supprimée (fichier verrouillé ?)")
        else:
            removed += 1
    return removed


def _new_visitor_session(settings: Settings, limits: UploadLimits, folder: Path) -> VisitorSession:
    """Remplit ``folder`` (dossier réservé, vide) d'une copie de l'index de démonstration et
    ouvre sa bibliothèque."""
    with process_state().template_lock:  # l'index de démonstration n'est plus modifié ici
        if index_exists(settings):
            shutil.copytree(settings.chroma_dir, folder / "chroma")
            registry = settings.data_dir / REGISTRY_FILENAME
            if registry.is_file():
                shutil.copyfile(registry, folder / REGISTRY_FILENAME)
    template = get_library(settings, create=False)
    protected = (
        frozenset(d.doc_id for d in template.documents_from_index()) if template else frozenset()
    )
    own = replace(settings, data_dir=folder)
    try:
        embedder = _cached_embedder(_embedding_key(own), own)
        store = bootstrap.build_store(own, embedder)
        library = bootstrap.build_document_library(
            own,
            embedder,
            protected=protected,
            max_documents=limits.max_documents,
            max_bytes=limits.max_bytes,
            max_pages=limits.max_pages,
            store=store,
        )
    except (
        EmbeddingModelNotFoundError,
        EmbeddingModelMismatchError,
        TranslationModelNotFoundError,
    ) as exc:
        raise SetupError(str(exc)) from exc
    logger.info("nouvelle session de visiteur (%d document(s) de démonstration)", len(protected))
    return VisitorSession(own, store, library)


def _cleanup(settings: Settings, state: ProcessState, keep: Path | None) -> None:
    """Nettoyage des sessions inactives ; à appeler avec ``state.sessions_lock``."""
    state.last_cleanup = now = time.time()
    removed = cleanup_sessions(sessions_root(settings), now=now, sessions=state.sessions, keep=keep)
    if removed:
        logger.info("%d bibliothèque(s) de session inactive(s) supprimée(s)", removed)


def current_visitor_session(settings: Settings) -> VisitorSession | None:
    """Bibliothèque privée de la session courante ; ``None`` tant que le visiteur n'a rien
    envoyé (il consulte alors la bibliothèque de démonstration partagée, en lecture seule).

    Si elle a été supprimée (inactivité), le visiteur revient à la démonstration avec une
    conversation vide ; une nouvelle bibliothèque ne sera créée qu'à son prochain envoi. Les
    bibliothèques inactives sont nettoyées au plus toutes les ``SESSION_CLEANUP_INTERVAL_S``.
    """
    state = process_state()
    visitor: VisitorSession | None = st.session_state.get("visitor")
    with state.sessions_lock:
        if visitor is not None:
            if not visitor.closed and visitor.settings.data_dir.is_dir():
                os.utime(visitor.settings.data_dir)  # activité : repousse la suppression
                # Toujours connue du nettoyage, même après un vidage des caches du processus.
                state.sessions.setdefault(visitor.settings.data_dir, visitor)
            else:
                state.sessions.pop(visitor.settings.data_dir, None)
                visitor.close()  # bibliothèque disparue : libère ce qui reste ouvert…
                shutil.rmtree(visitor.settings.data_dir, ignore_errors=True)  # … et le reste
                del st.session_state["visitor"]
                st.session_state.histories = {}
                st.session_state.notices = [Notice("info", SESSION_EXPIRED_TEXT)]
                visitor = None
        if time.time() - state.last_cleanup >= SESSION_CLEANUP_INTERVAL_S:
            _cleanup(settings, state, keep=visitor.settings.data_dir if visitor else None)
    return visitor


def start_visitor_session(settings: Settings, limits: UploadLimits) -> VisitorSession | None:
    """Crée la bibliothèque privée de la session (premier envoi) ; ``None`` si
    ``limits.max_sessions`` bibliothèques existent déjà, après nettoyage des inactives."""
    state = process_state()
    root = sessions_root(settings)
    with state.sessions_lock:
        _cleanup(settings, state, keep=None)
        existing = sum(1 for p in root.iterdir() if p.is_dir()) if root.is_dir() else 0
        if existing >= limits.max_sessions:
            logger.warning("limite de %d sessions de visiteurs atteinte", limits.max_sessions)
            return None
        folder = root / secrets.token_hex(16)
        folder.mkdir(parents=True)  # place réservée : comptée par les créations concurrentes
    try:
        visitor = _new_visitor_session(settings, limits, folder)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)  # place libérée
        raise
    with state.sessions_lock:
        state.sessions[folder] = visitor
    st.session_state.visitor = visitor
    return visitor


def visitor_service(visitor: VisitorSession) -> QAService:
    """Service de réponses de la session, sur sa propre bibliothèque (construit une fois)."""
    if visitor.service is None:
        try:
            generator = bootstrap.build_generator(visitor.settings)
            embedder = _cached_embedder(_embedding_key(visitor.settings), visitor.settings)
            visitor.service = bootstrap.build_qa_service(
                visitor.settings, embedder, generator, store=visitor.store
            )
        except (
            EmbeddingModelNotFoundError,
            EmbeddingModelMismatchError,
            TranslationModelNotFoundError,
            LLMError,
        ) as exc:
            raise SetupError(str(exc)) from exc
    return visitor.service


@st.cache_resource(show_spinner=False)
def _cached_user_libraries(
    cache_key: str, _settings: Settings, _config: AccountsConfig
) -> UserLibraries:
    """Un seul client de la base des bibliothèques personnelles par processus."""
    del cache_key
    embedder = _cached_embedder(_embedding_key(_settings), _settings)
    store = bootstrap.build_user_store(_config.store, _settings.data_dir / USERS_DIRNAME)
    return bootstrap.build_user_libraries(_settings, embedder, store, _config.limits)


def get_user_libraries(settings: Settings, config: AccountsConfig) -> UserLibraries:
    try:
        return _cached_user_libraries(repr((settings, config)), settings, config)
    except (EmbeddingModelNotFoundError, TranslationModelNotFoundError) as exc:
        raise SetupError(str(exc)) from exc
    except Exception as exc:  # base injoignable, identifiants refusés…
        # Type seulement : le message d'un client distant pourrait reprendre la configuration.
        raise SetupError(f"base des bibliothèques indisponible ({type(exc).__name__})") from exc


def account_service(settings: Settings, account: UserLibrary) -> QAService:
    """Service de réponses de l'utilisateur, sur son seul index (construit une fois)."""
    if account.service is None:
        try:
            generator = bootstrap.build_generator(settings)
            embedder = _cached_embedder(_embedding_key(settings), settings)
            account.service = bootstrap.build_qa_service(
                settings, embedder, generator, store=account.store
            )
        except (
            EmbeddingModelNotFoundError,
            EmbeddingModelMismatchError,
            TranslationModelNotFoundError,
            LLMError,
        ) as exc:
            raise SetupError(str(exc)) from exc
    return account.service


def list_documents(library: DocumentLibrary) -> tuple[list[LibraryEntry], str]:
    """Documents et éventuel avertissement (registre illisible : liste issue de l'index)."""
    try:
        return library.documents(), ""
    except DocumentRegistryError as exc:
        return library.documents_from_index(), str(exc)


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' if count > 1 else ''}"


def index_files(
    settings: Settings,
    files: Sequence[tuple[str, bytes]],
    library: DocumentLibrary | None = None,
) -> list[Notice]:
    """Indexe les PDF envoyés un par un (un échec n'empêche pas les suivants), dans
    ``library`` (bibliothèque d'un visiteur) ou, à défaut, dans la bibliothèque locale."""
    if library is None:
        try:
            library = get_library(settings, create=True)
        except SetupError as exc:
            return [Notice("error", str(exc))]
    assert library is not None
    notices: list[Notice] = []
    for name, data in files:
        try:
            with process_state().indexing_lock:
                outcome = library.add(name, data)
        except (IngestionError, DocumentRegistryError) as exc:
            notices.append(Notice("error", str(exc)))
            continue
        except Exception as exc:
            # Ni nom de fichier ni trace : le message ou le chemin temporaire peuvent contenir
            # le nom envoyé par l'utilisateur (donnée personnelle possible). Le type suffit à
            # reproduire le problème en local.
            logger.error("échec de l'indexation d'un document envoyé (%s)", type(exc).__name__)
            notices.append(Notice("error", f"{name} : erreur inattendue lors de l'indexation"))
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


def _split_submission(
    files: Sequence[tuple[str, bytes]], limit: int, scope: str
) -> tuple[Sequence[tuple[str, bytes]], list[Notice]]:
    """Au plus ``limit`` fichiers d'un envoi ; les suivants sont refusés sans être lus par le
    pipeline (vérification côté serveur, quel que soit le navigateur)."""
    kept, refused = files[:limit], files[limit:]
    if not refused:
        return kept, []
    names = ", ".join(f"« {safe_filename(name)} »" for name, _ in refused)
    text = (
        f"{plural(len(refused), 'fichier')} non envoyé{'s' if len(refused) > 1 else ''} "
        f"({names}) : au plus {limit} documents {scope}."
    )
    return kept, [Notice("error", text)]


def index_account_files(
    settings: Settings,
    account: UserLibrary,
    libraries: UserLibraries,
    files: Sequence[tuple[str, bytes]],
) -> list[Notice]:
    """Envoi d'un utilisateur connecté : dans sa seule bibliothèque. Chaque fichier envoyé
    compte dans sa limite quotidienne (vérifiée avant le pipeline, réussite ou non)."""
    limits = libraries.limits
    kept, notices = _split_submission(files, limits.max_documents, "dans votre bibliothèque")
    allowed: list[tuple[str, bytes]] = []
    for name, data in kept:
        if libraries.usage.try_consume(account.identity_key, "upload"):
            allowed.append((name, data))
        else:
            notices.append(
                Notice(
                    "error",
                    f"{safe_filename(name)} : limite de {limits.daily_uploads} envois par jour "
                    "atteinte ; réessayez demain.",
                )
            )
    return notices + index_files(settings, allowed, account.library)


def delete_account(settings: Settings, config: AccountsConfig, user: UserIdentity) -> Notice:
    """Supprime toutes les données de ``user``, et seulement les siennes."""
    try:
        removed = get_user_libraries(settings, config).delete_account(user)
    except SetupError as exc:
        logger.error("suppression du compte impossible : %s", exc)
        return Notice("error", LIBRARY_ERROR_TEXT)
    except Exception:
        logger.exception("échec de la suppression du compte")
        return Notice("error", "La suppression a échoué ; réessayez dans quelques minutes.")
    return Notice(
        "success", f"Votre compte et vos documents ({plural(removed, 'document')}) sont supprimés."
    )


def index_visitor_files(
    settings: Settings, limits: UploadLimits, files: Sequence[tuple[str, bytes]]
) -> list[Notice]:
    """Envoi d'un visiteur : au plus ``limits.max_documents`` fichiers par envoi (les suivants
    sont refusés sans être lus par le pipeline), dans sa bibliothèque privée, créée au premier
    envoi s'il reste une place."""
    kept, notices = _split_submission(files, limits.max_documents, "par session")
    visitor: VisitorSession | None = st.session_state.get("visitor")
    if visitor is None:
        try:
            visitor = start_visitor_session(settings, limits)
        except SetupError as exc:
            logger.error("bibliothèque de visiteur indisponible : %s", exc)
            return [Notice("error", LIBRARY_ERROR_TEXT)]
        if visitor is None:
            return [Notice("error", SESSIONS_FULL_TEXT)]
    return notices + index_files(settings, kept, visitor.library)


def delete_document(
    settings: Settings, doc_id: str, library: DocumentLibrary | None = None
) -> Notice:
    """Supprime de ``library`` (bibliothèque d'un visiteur) ou de la bibliothèque locale ; un
    document de démonstration est toujours refusé par la bibliothèque elle-même."""
    try:
        if library is None:
            library = get_library(settings, create=False)
        if library is None:
            return Notice("error", "aucun index : rien à supprimer.")
        outcome = library.delete(doc_id)
    except (
        SetupError,
        DocumentNotFoundError,
        DocumentProtectedError,
        DocumentRegistryError,
    ) as exc:
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
.badge-demo { background: var(--accent-soft); color: var(--accent); }
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


def document_meta(entry: LibraryEntry, *, personal: bool = False) -> str:
    """``personal`` : bibliothèque d'un utilisateur connecté, où la CLI n'écrit jamais ; un
    document sans enregistrement y vient d'un envoi interrompu, pas de la CLI."""
    parts = [] if entry.page_count is None else [plural(entry.page_count, "page")]
    parts.append(plural(entry.chunk_count, "passage"))
    if entry.added_at is not None:
        parts.append(f"ajouté le {entry.added_at.astimezone():%d/%m/%Y à %H:%M}")
    elif entry.source == "external":
        parts.append("détails de l'envoi indisponibles" if personal else "indexé avec la CLI")
    return " · ".join(parts)


def render_documents(
    settings: Settings,
    entries: Sequence[LibraryEntry],
    warning: str,
    manage: bool,
    library: DocumentLibrary | None = None,
    protected: frozenset[str] = frozenset(),
    personal: bool = False,
) -> None:
    """Liste des documents ; ``library`` : bibliothèque d'un visiteur ou d'un utilisateur
    connecté (``personal``). Les documents ``protected`` (démonstration) n'ont pas de bouton de
    suppression, et la bibliothèque du visiteur refuse aussi de les supprimer."""
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
        if entry.doc_id in protected:
            badge += " <span class='badge badge-demo'>Démo</span>"
        with st.container(border=True):
            info, action = st.columns([5, 1], vertical_alignment="center")
            info.markdown(
                f"<div class='doc-name'>{html.escape(entry.filename)}</div>"
                f"<div class='doc-meta'>{html.escape(document_meta(entry, personal=personal))}"
                f"</div>{badge}",
                unsafe_allow_html=True,
            )
            if manage and entry.doc_id not in protected:
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
                        notice = delete_document(settings, entry.doc_id, library)
                        if notice.kind == "success":
                            # Son historique disparaît avec lui (un nouvel envoi du même
                            # contenu, même doc_id, repart d'une conversation vide).
                            st.session_state.histories.pop(entry.doc_id, None)
                        st.session_state.notices = [notice]
                        st.rerun()


# Conditions affichées avant le premier usage d'une bibliothèque personnelle (TERMS_VERSION).
# Faits vérifiables uniquement : ni texte juridique ni affirmation de conformité ; la notice de
# confidentialité est celle de l'exploitant (DOCASSIST_PRIVACY_URL).
TERMS_POINTS = (
    "Ce service est réservé aux personnes de **18 ans ou plus**.",
    "N'envoyez **aucune donnée sensible** (santé, origine, opinions politiques, convictions "
    "religieuses, appartenance syndicale, données génétiques ou biométriques, vie ou orientation "
    "sexuelle…) ni de document confidentiel.",
    "Les passages de vos documents sont conservés par Chroma (Chroma Cloud) jusqu'à ce que vous "
    "les supprimiez ; ceux qui servent à vos questions sont envoyés à Google (Gemini) pour "
    "rédiger les réponses. Le fichier PDF lui-même n'est pas conservé.",
)


def terms_accepted(user: UserIdentity, account: UserLibrary) -> bool:
    """Conditions acceptées dans cette session par cet utilisateur, ou enregistrées avec sa
    bibliothèque (premier envoi déjà fait)."""
    if st.session_state.get("terms_accepted_by") == (user.key, TERMS_VERSION):
        return True
    return account.terms_accepted(TERMS_VERSION)


def render_terms_gate(config: AccountsConfig, user: UserIdentity, account: UserLibrary) -> None:
    """Conditions à accepter avant tout envoi et toute question."""
    st.markdown("<div class='section-title'>Avant de commencer</div>", unsafe_allow_html=True)
    st.markdown("\n".join(f"- {point}" for point in TERMS_POINTS))
    st.markdown(f"[Notice de confidentialité]({config.privacy_url})")
    agreed = st.checkbox(
        "J'ai 18 ans ou plus et j'accepte ces conditions.", key=f"terms-{TERMS_VERSION}"
    )
    if st.button(
        "Continuer",
        key="accept-terms",
        type="primary",
        disabled=not agreed,
        width="stretch",
    ):
        account.accept_terms(TERMS_VERSION)
        st.session_state.terms_accepted_by = (user.key, TERMS_VERSION)
        st.rerun()


def render_account(settings: Settings, config: AccountsConfig, user: UserIdentity | None) -> None:
    """Connexion Google, déconnexion et suppression du compte. La connexion et la déconnexion
    sont faites par Streamlit (``st.login`` / ``st.logout``), jamais par l'application."""
    if user is None:
        st.caption("Connectez-vous pour retrouver vos propres documents sur tous vos appareils.")
        st.button(
            "Se connecter avec Google",
            key="login",
            icon=":material/login:",
            on_click=st.login,
            width="stretch",
        )
        return
    st.markdown(
        f"<div class='doc-meta'>Connecté : {html.escape(user.label)}</div>",
        unsafe_allow_html=True,
    )
    st.button(
        "Se déconnecter",
        key="logout",
        icon=":material/logout:",
        on_click=st.logout,
        width="stretch",
    )
    with st.popover("Supprimer mon compte", icon=":material/person_remove:"):
        st.markdown(
            "Tous **vos** documents et leurs passages seront supprimés définitivement. Ceux des "
            "autres utilisateurs ne sont pas touchés."
        )
        if st.button(
            "Supprimer définitivement mon compte",
            key="delete-account",
            type="primary",
            width="stretch",
        ):
            st.session_state.notices = [delete_account(settings, config, user)]
            st.session_state.histories = {}
            st.rerun()


def render_upload(
    settings: Settings,
    limits: UploadLimits | None = None,
    account: tuple[UserLibrary, UserLibraries] | None = None,
) -> None:
    """Envoi de PDF : dans la bibliothèque locale ; avec ``limits``, dans celle de la session
    (créée au premier envoi) ; avec ``account``, dans la bibliothèque personnelle de
    l'utilisateur connecté. Les limites sont vérifiées de nouveau côté serveur."""
    st.markdown("<div class='section-title'>Ajouter un PDF</div>", unsafe_allow_html=True)
    # Même unité que le serveur : le navigateur n'accepte jamais un fichier que le serveur refuse.
    if account is not None:
        max_bytes = account[1].limits.max_bytes
    else:
        max_bytes = MAX_UPLOAD_BYTES if limits is None else limits.max_bytes
    max_mb = max_bytes // MEGABYTE
    help_text = f"PDF à texte sélectionnable (pas d'OCR), {max_mb} Mo maximum par fichier."
    if account is not None:
        own = account[1].limits
        help_text = (
            f"PDF à texte sélectionnable (pas d'OCR) : {max_mb} Mo et {own.max_pages} pages au "
            f"plus, {own.max_documents} documents dans votre bibliothèque, {own.daily_uploads} "
            "envois par jour."
        )
        st.caption(
            "Vos documents sont conservés dans votre bibliothèque personnelle, visible de vous "
            "seul, jusqu'à ce que vous les supprimiez. Les passages utiles à vos questions sont "
            "envoyés au service de réponses (Google Gemini) : n'ajoutez pas de document "
            "confidentiel."
        )
    elif limits is not None:
        help_text = (
            f"PDF à texte sélectionnable (pas d'OCR) : {max_mb} Mo et {limits.max_pages} pages "
            f"au plus, {limits.max_documents} documents par session."
        )
        st.caption(
            "Vos documents ne sont visibles que dans cette session et sont supprimés après "
            "2 heures d'inactivité. Les passages utiles à vos questions sont envoyés au service "
            "de réponses : n'ajoutez pas de document confidentiel ou personnel."
        )
    files = st.file_uploader(
        "Fichiers PDF",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=max_mb,
        help=help_text,
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
            sent = [(f.name, f.getvalue()) for f in files]
            if account is not None:
                notices = index_account_files(settings, account[0], account[1], sent)
            elif limits is not None:
                notices = index_visitor_files(settings, limits, sent)
            else:
                notices = index_files(settings, sent)
            st.session_state.notices = notices
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
        server_max_mb = int(st.get_option("server.maxUploadSize"))
        limits = session_upload_limits(os.environ, server_max_mb=server_max_mb)
        accounts = user_libraries_config(os.environ, st.secrets, server_max_mb=server_max_mb)
    except ConfigError as exc:
        st.error(f"**Configuration invalide** — {exc}")
        return
    demo_dir = demo_documents_dir(os.environ)

    # Identité relue à chaque exécution, uniquement auprès du serveur (cookie signé) : rien
    # dans st.session_state ne peut la choisir.
    user: UserIdentity | None = None
    if accounts is not None:
        try:
            user = web_identity.current_user()
        except AuthenticationError as exc:
            logger.warning("connexion refusée : %s", exc)
            st.session_state.notices = [Notice("error", "Connexion incomplète : reconnectez-vous.")]
    owner = user.key if user else None
    if st.session_state.get("history_owner") != owner:  # jamais la conversation d'un autre
        st.session_state.histories = {}
        st.session_state.history_owner = owner

    entries: list[LibraryEntry] = []
    warning = library_error = setup_error = ""
    service: QAService | None = None
    visitor: VisitorSession | None = None
    account: UserLibrary | None = None
    libraries: UserLibraries | None = None
    gated = False  # conditions pas encore acceptées : ni envoi ni question
    library: DocumentLibrary | None = None
    try:
        with st.spinner("Chargement de vos documents…"):
            if demo_dir is not None:
                seed_demo_documents(settings, demo_dir)
            if accounts is not None and user is not None:  # sa seule bibliothèque
                libraries = get_user_libraries(settings, accounts)
                account = libraries.library(user)
                gated = not terms_accepted(user, account)
                library = account.library
            elif accounts is not None:
                # Déconnecté : la démonstration en lecture seule, jamais l'index local.
                library = get_library(settings, create=False) if demo_dir else None
            else:
                if limits is not None:  # bibliothèque privée de la session, jamais partagée
                    visitor = current_visitor_session(settings)
                # Sans bibliothèque privée : la démonstration partagée, jamais modifiée par un
                # visiteur (tous ses documents sont alors protégés, voir ``protected``).
                library = visitor.library if visitor else get_library(settings, create=False)
            if library is not None:
                entries, warning = list_documents(library)
    except SetupError as exc:
        logger.error("bibliothèque indisponible : %s", exc)
        library_error = LIBRARY_ERROR_TEXT
    except Exception:
        logger.exception("échec du chargement de la bibliothèque")
        library_error = LIBRARY_ERROR_TEXT
    # Avec les comptes, déconnecté = lecture seule, même hors démonstration.
    manage = account is not None if accounts is not None else manage or limits is not None
    if account is not None:
        protected: frozenset[str] = frozenset()
    elif visitor is not None:
        protected = visitor.library.protected
    elif limits is not None or accounts is not None:
        protected = frozenset(e.doc_id for e in entries)
    else:
        protected = frozenset()
    if entries and not gated:
        try:
            with st.spinner("Préparation du service de réponses…"):
                if account is not None:
                    service = account_service(settings, account)
                elif visitor is not None:
                    service = visitor_service(visitor)
                else:
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
        if accounts is not None:
            render_account(settings, accounts, user)
            st.divider()
            if gated and user is not None and account is not None:
                render_terms_gate(accounts, user, account)
                st.divider()
        render_notices()
        own = account.library if account else (visitor.library if visitor else None)
        # Suppression toujours possible, même conditions non acceptées (droit à l'effacement).
        render_documents(settings, entries, warning, manage, own, protected, account is not None)
        if manage and not gated:
            st.divider()
            render_upload(
                settings,
                limits,
                (account, libraries) if account and libraries else None,
            )

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
            if libraries is not None and user is not None:
                allowed = libraries.usage.try_consume(user.key, "question")
            else:
                allowed = True
            if not allowed:  # vérifié avant tout appel au LLM
                assert libraries is not None
                reply = ChatMessage(
                    "assistant",
                    f"Vous avez atteint la limite de {libraries.limits.daily_questions} questions "
                    "par jour. Réessayez demain.",
                    kind="error",
                    detail="Limite quotidienne atteinte",
                )
            else:
                with st.spinner("Réflexion en cours…"):
                    reply = ask(service, question, doc_id=doc_id, scope=scope_label)
            render_message(reply)
        messages.append(reply)
        scroll_to_latest_question(f"{token}-answer")  # réponse affichée sous la question


main()
