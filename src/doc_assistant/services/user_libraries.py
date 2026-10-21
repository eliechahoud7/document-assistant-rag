"""Cas d'usage : bibliothèques personnelles et persistantes des utilisateurs connectés.

L'identité vient uniquement des informations de connexion vérifiées par le serveur
(``UserIdentity.from_claims``) : l'émetteur (``iss``) et l'identifiant stable (``sub``) du
compte, jamais l'e-mail ni une valeur transmise par l'interface. Chaque utilisateur a une
collection ChromaDB, nommée d'après cette identité (``user-<clé>-chunks``) : ses passages et,
attaché au premier passage de chaque document, l'enregistrement de ce document.

Isolation par construction : une ``UserLibrary`` n'ouvre que la collection de son utilisateur,
et aucune méthode n'accepte un nom de collection ou une clé fournis par l'appelant. Ajout,
liste, recherche et suppression passent par elle ; un ``doc_id`` d'un autre utilisateur est
simplement introuvable dans la collection.

Base de chaque utilisateur (``client_for``) : la base Chroma Cloud partagée en production ; en
local, une base (un dossier) par utilisateur. Le moteur local de ChromaDB 1.5.9 peut rendre la
recherche d'une collection impossible quand une autre collection de la même base est écrite
entre-temps (``Nothing found on disk``, voir ``indexing/chroma_registry.py``) : avec une base par
utilisateur, l'envoi d'un utilisateur ne touche jamais la base d'un autre.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError

from doc_assistant.domain.errors import AuthenticationError, UploadRejectedError
from doc_assistant.domain.ports import Embedder, VectorStore
from doc_assistant.indexing.chroma_registry import ChunkRecordRegistry
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.library_service import DocumentLibrary
from doc_assistant.services.qa_service import QAService
from doc_assistant.services.usage_limits import DailyUsageLimiter

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"[0-9a-f]{32}")
MAX_OPEN_LIBRARIES = 64  # bibliothèques gardées ouvertes dans le processus (les plus récentes)
# Métadonnées de la collection : version des conditions acceptées et date d'acceptation.
TERMS_VERSION_KEY = "terms_version"
TERMS_ACCEPTED_AT_KEY = "terms_accepted_at"


class AccountLimitError(UploadRejectedError):
    """Nombre maximal de bibliothèques atteint : aucun nouveau compte n'est créé."""


@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Utilisateur connecté. ``key`` : début du SHA-256 de ``iss`` et ``sub`` ; e-mail et nom
    servent seulement à l'affichage."""

    key: str
    email: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not _KEY_RE.fullmatch(self.key):
            raise AuthenticationError("clé d'utilisateur invalide")

    @classmethod
    def from_claims(cls, claims: Mapping[str, object]) -> UserIdentity:
        """Identité tirée des informations de connexion vérifiées (``st.user``)."""
        if claims.get("is_logged_in") is not True:
            raise AuthenticationError("aucun utilisateur connecté")
        issuer, subject = claims.get("iss"), claims.get("sub")
        if not isinstance(issuer, str) or not issuer.strip():
            raise AuthenticationError("identité incomplète : émetteur (iss) absent")
        if not isinstance(subject, str) or not subject.strip():
            raise AuthenticationError("identité incomplète : identifiant (sub) absent")
        digest = hashlib.sha256(f"{issuer.strip()}\n{subject.strip()}".encode()).hexdigest()
        email, name = claims.get("email"), claims.get("name")
        return cls(
            digest[:32],
            email if isinstance(email, str) else None,
            name if isinstance(name, str) else None,
        )

    @property
    def label(self) -> str:
        return self.name or self.email or "compte connecté"


@dataclass(frozen=True, slots=True)
class UserLimits:
    """Limites des bibliothèques personnelles, vérifiées côté serveur.

    ``max_bytes``, ``max_pages``, ``max_documents`` : par document et par bibliothèque
    (``DocumentLibrary``) ; ``daily_questions``, ``daily_uploads`` : par utilisateur et par jour
    (``DailyUsageLimiter``) ; ``max_accounts`` : nombre de bibliothèques existantes dans la base.
    """

    max_bytes: int
    max_pages: int
    max_documents: int
    daily_questions: int = 50
    daily_uploads: int = 20
    max_accounts: int = 100


@dataclass(slots=True)
class UserLibrary:
    """Bibliothèque d'un utilisateur : son index, son registre, son service de réponses."""

    identity_key: str
    store: ChromaVectorStore
    library: DocumentLibrary
    service: QAService | None = None  # construit à la première question
    # Posé par ``delete_account`` sous le verrou de l'utilisateur : tout envoi encore en cours
    # sur cet objet (autre appareil) est alors refusé (``LibraryClosedError``).
    closed: threading.Event = field(default_factory=threading.Event)

    def terms_accepted(self, version: str) -> bool:
        """Conditions d'utilisation (``version``) acceptées et enregistrées avec la bibliothèque."""
        return self.store.collection_metadata().get(TERMS_VERSION_KEY) == version

    def accept_terms(self, version: str) -> None:
        """Enregistre l'acceptation avec la bibliothèque ; rien n'est écrit dans la base avant le
        premier envoi (la collection n'existe pas encore)."""
        accepted_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.store.set_collection_metadata(
            {TERMS_VERSION_KEY: version, TERMS_ACCEPTED_AT_KEY: accepted_at}
        )


def chunks_collection(identity: UserIdentity) -> str:
    return f"user-{identity.key}-chunks"


class UserLibraries:
    """Toutes les bibliothèques personnelles. ``client_for`` donne la base ChromaDB d'un
    utilisateur : Chroma Cloud (partagée) en production, un dossier local par utilisateur pour
    le développement et les tests."""

    def __init__(
        self,
        client_for: Callable[[UserIdentity], ClientAPI],
        embedder: Embedder,
        limits: UserLimits,
        indexing: Callable[[VectorStore], IndexingService],
        count_accounts: Callable[[], int],
    ) -> None:
        """``count_accounts`` : nombre de bibliothèques existant dans la base (plafond
        ``limits.max_accounts``, vérifié à la création d'une bibliothèque)."""
        self._client_for = client_for
        self._embedder = embedder
        self.limits = limits
        self._indexing = indexing
        self._count_accounts = count_accounts
        self._open: OrderedDict[str, UserLibrary] = OrderedDict()
        self._lock = threading.Lock()
        # Un verrou par utilisateur, gardé même quand sa bibliothèque quitte ``_open`` : deux
        # objets de la même bibliothèque (deux appareils) ne s'exécutent jamais en parallèle.
        self._user_locks: dict[str, threading.Lock] = {}
        self._creation_lock = threading.Lock()  # comptage et création des bibliothèques
        self.usage = DailyUsageLimiter(
            {"question": limits.daily_questions, "upload": limits.daily_uploads}
        )

    def library(self, identity: UserIdentity) -> UserLibrary:
        """Bibliothèque de ``identity``. Rien n'est créé dans la base avant son premier envoi."""
        with self._lock:
            opened = self._open.get(identity.key)
            if opened is None:
                opened = self._build(identity)
                self._open[identity.key] = opened
                while len(self._open) > MAX_OPEN_LIBRARIES:
                    self._open.popitem(last=False)
            self._open.move_to_end(identity.key)
            return opened

    def delete_account(self, identity: UserIdentity) -> int:
        """Supprime toutes les données de ``identity`` (sa collection) et seulement elles ;
        renvoie le nombre de documents supprimés. Attend la fin d'un envoi en cours sur le même
        compte (autre appareil) ; les envois suivants sur l'ancienne bibliothèque sont refusés."""
        with self._lock:
            opened = self._open.pop(identity.key, None) or self._build(identity)
            user_lock = self._user_lock(identity.key)
        with user_lock:
            opened.closed.set()
            removed = len(opened.store.list_documents())
            with contextlib.suppress(NotFoundError, ValueError):  # déjà absente
                self._client_for(identity).delete_collection(chunks_collection(identity))
        logger.info("compte supprimé (%d document(s))", removed)
        return removed

    def _user_lock(self, key: str) -> threading.Lock:
        """À appeler avec ``self._lock``."""
        return self._user_locks.setdefault(key, threading.Lock())

    @contextlib.contextmanager
    def _account_creation(self) -> Iterator[None]:
        """Création d'une bibliothèque : refusée au-delà de ``limits.max_accounts``. Comptage et
        création sous le même verrou (pas de dépassement par deux créations simultanées)."""
        with self._creation_lock:
            if self._count_accounts() >= self.limits.max_accounts:
                logger.warning("plafond de %d comptes atteint", self.limits.max_accounts)
                raise AccountLimitError(
                    "le nombre maximal de bibliothèques est atteint : aucun nouveau compte ne "
                    "peut être créé pour le moment"
                )
            yield

    def _build(self, identity: UserIdentity) -> UserLibrary:
        """À appeler avec ``self._lock``."""
        embedder = self._embedder
        store = ChromaVectorStore(
            None,
            chunks_collection(identity),
            embedder.model_name,
            lambda: embedder.dimension,
            client=self._client_for(identity),
            lazy=True,  # créée au premier envoi seulement
            creation_guard=self._account_creation,
        )
        closed = threading.Event()
        library = DocumentLibrary(
            self._indexing(store),
            store,
            ChunkRecordRegistry(store),  # même collection : un ajout n'écrit nulle part ailleurs
            None,  # aucun PDF conservé sur le disque du serveur
            max_documents=self.limits.max_documents,
            max_bytes=self.limits.max_bytes,
            max_pages=self.limits.max_pages,
            lock=self._user_lock(identity.key),
            is_open=lambda: not closed.is_set(),
        )
        return UserLibrary(identity.key, store, library, closed=closed)
