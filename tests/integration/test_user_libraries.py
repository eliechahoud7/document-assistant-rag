"""Bibliothèques personnelles des utilisateurs connectés, sur une vraie base ChromaDB locale
(même API que Chroma Cloud ; dossier temporaire, embedder factice, aucun appel réseau).

Couvre l'identité (dérivée de ``iss`` et ``sub`` seulement), l'isolation entre utilisateurs
(liste, recherche, suppression), la persistance après redémarrage et la suppression du compte.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from conftest import MakePdf
from doc_assistant import bootstrap
from doc_assistant.config import Settings
from doc_assistant.domain.errors import (
    AuthenticationError,
    DocumentNotFoundError,
    DocumentRegistryError,
    LibraryClosedError,
    UploadRejectedError,
)
from doc_assistant.domain.models import Chunk
from doc_assistant.indexing.chroma_registry import ChunkRecordRegistry
from doc_assistant.indexing.document_registry import DocumentRecord
from doc_assistant.indexing.vector_store import MAX_RECORDS_PER_CALL, ChromaVectorStore
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services import user_libraries
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.user_libraries import (
    AccountLimitError,
    UserIdentity,
    UserLibraries,
    UserLimits,
    chunks_collection,
)
from fakes import FakeEmbedder

GOOGLE = "https://accounts.google.com"
LIMITS = UserLimits(max_bytes=10 * 1024 * 1024, max_pages=50, max_documents=3)
OTTERS = [["Les loutres de mer dorment en se tenant la patte."]]
VOLCANOES = [["Un volcan rejette de la lave et des cendres lors d'une éruption."]]


def claims(sub: str, **extra: object) -> dict[str, object]:
    return {"is_logged_in": True, "iss": GOOGLE, "sub": sub} | extra


ALICE = UserIdentity.from_claims(claims("1001", email="alice@example.com", name="Alice"))
BOB = UserIdentity.from_claims(claims("1002", email="bob@example.com"))


def open_libraries(folder: Path, limits: UserLimits = LIMITS) -> UserLibraries:
    """Nouveau client sur ``folder`` : comme un nouveau processus (redémarrage de l'app)."""
    SharedSystemClient.clear_system_cache()  # sinon Chroma réutilise le client en mémoire
    settings = Settings(data_dir=folder.parent)
    embedder = FakeEmbedder()
    store = bootstrap.build_user_store("local", folder)
    return bootstrap.build_user_libraries(settings, embedder, store, limits)


def collections_of(users_dir: Path, user: UserIdentity) -> list[str]:
    """Collections de la base locale de ``user`` (un accès indépendant de la bibliothèque)."""
    client = bootstrap.build_user_store("local", users_dir).client_for(user)
    return sorted(c.name for c in client.list_collections())


@pytest.fixture
def users_dir(tmp_path: Path) -> Path:
    return tmp_path / "users"


def doc_ids(libraries: UserLibraries, user: UserIdentity) -> list[str]:
    return [e.filename for e in libraries.library(user).library.documents()]


def search(libraries: UserLibraries, user: UserIdentity, question: str) -> list[str]:
    store = libraries.library(user).store
    return [r.chunk.filename for r in Retriever(FakeEmbedder(), store).search(question, top_k=5)]


# ---------------------------------------------------------------- identité


def test_identity_depends_only_on_issuer_and_subject() -> None:
    same = UserIdentity.from_claims(claims("1001", email="new-address@example.com", name="A."))
    other_issuer = UserIdentity.from_claims(claims("1001") | {"iss": "https://login.example"})

    assert same.key == ALICE.key  # changer d'e-mail ne change pas de bibliothèque
    assert ALICE.key != BOB.key
    assert other_issuer.key != ALICE.key  # même « sub » chez un autre fournisseur : autre compte
    assert "1001" not in ALICE.key and "alice" not in ALICE.key  # aucune donnée lisible
    assert ALICE.label == "Alice" and BOB.label == "bob@example.com"


@pytest.mark.parametrize(
    "raw",
    [
        {"is_logged_in": False, "iss": GOOGLE, "sub": "1001"},
        {"iss": GOOGLE, "sub": "1001"},
        {"is_logged_in": "true", "iss": GOOGLE, "sub": "1001"},  # pas un vrai booléen
        claims(""),
        claims("   "),
        {"is_logged_in": True, "sub": "1001"},
        claims("1001") | {"sub": 1001},
    ],
)
def test_incomplete_identity_is_refused(raw: dict[str, object]) -> None:
    with pytest.raises(AuthenticationError):
        UserIdentity.from_claims(raw)


@pytest.mark.parametrize("key", ["", "../etc", "A" * 32, "0" * 31, "0" * 32 + "-chunks"])
def test_collection_names_can_only_come_from_a_valid_key(key: str) -> None:
    with pytest.raises(AuthenticationError):
        UserIdentity(key)


def test_collection_names_are_derived_from_the_identity() -> None:
    assert chunks_collection(ALICE) == f"user-{ALICE.key}-chunks"
    assert chunks_collection(ALICE) != chunks_collection(BOB)


# ---------------------------------------------------------------- isolation


def test_each_user_lists_and_searches_only_their_own_documents(
    users_dir: Path, make_pdf: MakePdf
) -> None:
    libraries = open_libraries(users_dir)
    libraries.library(ALICE).library.add("Loutres.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())
    libraries.library(BOB).library.add("Volcans.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())

    assert doc_ids(libraries, ALICE) == ["Loutres.pdf"]
    assert doc_ids(libraries, BOB) == ["Volcans.pdf"]
    # Même question, mots de l'autre document : jamais ses passages.
    assert search(libraries, BOB, "loutres de mer dorment patte") == ["Volcans.pdf"]
    assert search(libraries, ALICE, "volcan lave cendres éruption") == ["Loutres.pdf"]


def test_a_user_cannot_delete_another_users_document(users_dir: Path, make_pdf: MakePdf) -> None:
    libraries = open_libraries(users_dir)
    otters = libraries.library(ALICE).library.add(
        "Loutres.pdf", make_pdf(OTTERS, "a.pdf").read_bytes()
    )

    with pytest.raises(DocumentNotFoundError):
        libraries.library(BOB).library.delete(otters.entry.doc_id)  # identifiant connu de Bob

    assert doc_ids(libraries, ALICE) == ["Loutres.pdf"]


def test_same_pdf_in_two_libraries_is_deleted_independently(
    users_dir: Path, make_pdf: MakePdf
) -> None:
    libraries = open_libraries(users_dir)
    data = make_pdf(OTTERS, "a.pdf").read_bytes()
    alice = libraries.library(ALICE).library.add("Loutres.pdf", data)
    bob = libraries.library(BOB).library.add("Mes-loutres.pdf", data)
    assert alice.kind == bob.kind == "indexed"  # pas un doublon : bibliothèques séparées

    libraries.library(ALICE).library.delete(alice.entry.doc_id)

    assert doc_ids(libraries, ALICE) == []
    assert doc_ids(libraries, BOB) == ["Mes-loutres.pdf"]
    assert search(libraries, BOB, "loutres") == ["Mes-loutres.pdf"]


def test_document_limit_is_per_user(users_dir: Path, make_pdf: MakePdf) -> None:
    libraries = open_libraries(users_dir, UserLimits(10 * 1024 * 1024, 50, max_documents=1))
    libraries.library(ALICE).library.add("a.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())

    with pytest.raises(UploadRejectedError, match="limite de 1 documents"):
        libraries.library(ALICE).library.add("b.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())
    libraries.library(BOB).library.add("b.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())

    assert doc_ids(libraries, BOB) == ["b.pdf"]


# ---------------------------------------------------------------- persistance


def test_library_survives_a_restart_with_its_metadata(
    users_dir: Path, tmp_path: Path, make_pdf: MakePdf
) -> None:
    libraries = open_libraries(users_dir)
    first = libraries.library(ALICE).library.add(
        "Loutres.pdf", make_pdf([*OTTERS, ["Deuxième page."]], "a.pdf").read_bytes()
    )

    restarted = open_libraries(users_dir)  # nouveau processus, nouveau client
    (entry,) = restarted.library(ALICE).library.documents()

    assert entry.doc_id == first.entry.doc_id
    assert (entry.filename, entry.page_count, entry.status) == ("Loutres.pdf", 2, "indexed")
    assert entry.added_at is not None and entry.source == "app"
    restarted.library(ALICE).library.add("Volcans.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())
    assert doc_ids(open_libraries(users_dir), ALICE) == ["Loutres.pdf", "Volcans.pdf"]
    # Aucun PDF n'est conservé sur le disque du serveur : seuls les passages sont indexés.
    stored = [p for p in tmp_path.rglob("*.pdf") if p.parent == tmp_path]
    assert sorted(p.name for p in stored) == ["a.pdf", "b.pdf"]  # les fichiers source du test
    assert not list(users_dir.rglob("*.pdf"))


# ---------------------------------------------------------------- suppression du compte


def test_account_deletion_removes_only_that_users_data(users_dir: Path, make_pdf: MakePdf) -> None:
    libraries = open_libraries(users_dir)
    data = make_pdf(OTTERS, "a.pdf").read_bytes()
    libraries.library(ALICE).library.add("Loutres.pdf", data)
    libraries.library(ALICE).library.add("Volcans.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())
    libraries.library(BOB).library.add("Loutres.pdf", data)

    assert libraries.delete_account(ALICE) == 2

    assert collections_of(users_dir, ALICE) == []
    assert collections_of(users_dir, BOB) == [chunks_collection(BOB)]
    assert doc_ids(libraries, BOB) == ["Loutres.pdf"]
    assert doc_ids(open_libraries(users_dir), ALICE) == []  # reconnexion : bibliothèque vide


# ---------------------------------------------------------------- régression « Nothing found »
#
# ChromaDB 1.5.9 (moteur local) : écrire dans une collection, puis dans une autre collection de
# la même base, peut rendre la recherche de la première impossible (« InternalError : Error
# creating hnsw segment reader: Nothing found on disk ») jusqu'à sa prochaine écriture. Mesuré
# avant correction : 8 échecs sur 120 répétitions des deux scénarios ci-dessous (registre dans
# une seconde collection), et 2 sur 150 avec des envois d'utilisateurs entrelacés dans une base
# commune ; reproduit aussi sans le code du projet. Correction : un ajout n'écrit que dans la
# collection de l'utilisateur (registre attaché à ses passages), et en local chaque utilisateur
# a sa propre base.

STRESS_RUNS = 40  # avec l'ancien schéma, au moins un échec dans ~95 % des exécutions du test


def test_an_upload_writes_only_to_its_users_single_collection(
    users_dir: Path, make_pdf: MakePdf
) -> None:
    """Invariant qui écarte la séquence fautive, vérifié de façon déterministe."""
    libraries = open_libraries(users_dir)
    libraries.library(ALICE).library.add("Loutres.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())
    libraries.library(BOB).library.add("Volcans.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())

    assert collections_of(users_dir, ALICE) == [chunks_collection(ALICE)]  # pas de registre à part
    assert collections_of(users_dir, BOB) == [chunks_collection(BOB)]
    assert sorted(p.name for p in users_dir.iterdir()) == sorted(
        [ALICE.key, BOB.key]
    )  # 1 base/compte


def test_interleaved_uploads_never_break_search(tmp_path: Path, make_pdf: MakePdf) -> None:
    """Les deux scénarios des échecs intermittents, répétés sur des bases neuves, avec les envois
    de deux utilisateurs entrelacés avant chaque recherche."""
    otters = make_pdf(OTTERS, "a.pdf").read_bytes()
    volcanoes = make_pdf(VOLCANOES, "b.pdf").read_bytes()
    for run in range(STRESS_RUNS):
        libraries = open_libraries(tmp_path / f"run-{run}" / "users")
        alice, bob = libraries.library(ALICE).library, libraries.library(BOB).library
        first = alice.add("Loutres.pdf", otters)
        bob.add("Volcans.pdf", volcanoes)
        bob.add("Mes-loutres.pdf", otters)  # même contenu qu'Alice, autre bibliothèque

        assert search(libraries, ALICE, "loutres volcan") == ["Loutres.pdf"], run
        assert sorted(search(libraries, BOB, "loutres volcan")) == [
            "Mes-loutres.pdf",
            "Volcans.pdf",
        ]
        alice.delete(first.entry.doc_id)
        assert doc_ids(libraries, ALICE) == [], run
        assert search(libraries, BOB, "loutres")[0] == "Mes-loutres.pdf", run


def test_document_record_lives_on_the_first_passage(tmp_path: Path, make_pdf: MakePdf) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "chunks", "fake-hash", 256)
    registry = ChunkRecordRegistry(store)
    record = DocumentRecord(
        "0" * 16, "a.pdf", "f" * 64, 2, 1, (2,), "2026-10-21T10:00:00+00:00", None
    )

    with pytest.raises(DocumentRegistryError, match="aucun passage"):
        registry.put(record)  # rien d'indexé : pas d'enregistrement orphelin
    chunk = Chunk(f"{record.doc_id}:0", record.doc_id, "a.pdf", 1, 0, 0, "Texte de la page.")
    store.add([chunk], FakeEmbedder().embed_documents([chunk.text]))
    registry.put(record)

    assert registry.records() == {record.doc_id: record}
    assert [c.text for c in store.get_chunks(record.doc_id, [0])] == ["Texte de la page."]
    store.delete_document(record.doc_id)
    assert registry.remove(record.doc_id) is True
    assert registry.records() == {}


# ---------------------------------------------------------------- création paresseuse, plafonds


def test_signing_in_stores_nothing_until_the_first_upload(
    users_dir: Path, make_pdf: MakePdf
) -> None:
    libraries = open_libraries(users_dir)

    assert doc_ids(libraries, ALICE) == [] and search(libraries, ALICE, "loutres") == []
    assert collections_of(users_dir, ALICE) == []  # rien d'écrit pour une simple connexion
    libraries.library(ALICE).library.add("Loutres.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())
    assert collections_of(users_dir, ALICE) == [chunks_collection(ALICE)]

    libraries.delete_account(ALICE)
    assert doc_ids(libraries, ALICE) == []  # reconnexion après suppression…
    assert collections_of(users_dir, ALICE) == []  # … sans rien recréer


def test_account_cap_applies_to_new_libraries_only(users_dir: Path, make_pdf: MakePdf) -> None:
    libraries = open_libraries(users_dir, UserLimits(10 * 1024 * 1024, 50, 3, max_accounts=1))
    alice = libraries.library(ALICE).library
    alice.add("Loutres.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())

    with pytest.raises(AccountLimitError, match="nombre maximal de bibliothèques"):
        libraries.library(BOB).library.add("b.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())
    assert collections_of(users_dir, BOB) == []
    alice.add("Volcans.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())  # compte existant

    libraries.delete_account(ALICE)  # une place se libère
    libraries.library(BOB).library.add("b.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())
    assert doc_ids(libraries, BOB) == ["b.pdf"]


class PausedIndexing:
    """Suspend l'indexation juste après l'écriture des passages, avant l'enregistrement du
    document : un envoi « en cours » sur un appareil, pendant qu'un autre agit."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.written, self.resume = threading.Event(), threading.Event()
        original = IndexingService.index_pdf

        def paused(service: IndexingService, path: Path) -> Any:
            report = original(service, path)
            self.written.set()
            assert self.resume.wait(10)
            return report

        monkeypatch.setattr(IndexingService, "index_pdf", paused)


def run_in_thread(target: Callable[[], object]) -> tuple[threading.Thread, dict[str, object]]:
    result: dict[str, object] = {}

    def run() -> None:
        try:
            result["value"] = target()
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def test_account_deletion_waits_for_an_upload_in_progress(
    users_dir: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Régression : la suppression du compte n'attendait pas un envoi en cours (autre
    appareil) ; l'envoi échouait ensuite sur une erreur brute de ChromaDB."""
    libraries = open_libraries(users_dir)
    stale = libraries.library(ALICE).library  # tenu par l'autre appareil
    stale.add("Ancien.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())
    paused = PausedIndexing(monkeypatch)
    upload, uploaded = run_in_thread(
        lambda: stale.add("Nouveau.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes()).kind
    )
    assert paused.written.wait(10)

    deletion, deleted = run_in_thread(lambda: libraries.delete_account(ALICE))
    deletion.join(1.0)
    assert deletion.is_alive()  # attend la fin de l'envoi en cours
    paused.resume.set()
    upload.join(10)
    deletion.join(10)

    assert uploaded == {"value": "indexed"}  # l'envoi s'est terminé proprement…
    assert deleted == {"value": 2}  # … puis la suppression l'a emporté avec le reste
    assert collections_of(users_dir, ALICE) == []
    with pytest.raises(LibraryClosedError, match="votre compte vient d'être supprimé"):
        stale.add("Tardif.pdf", make_pdf([["Envoi tardif."]], "c.pdf").read_bytes())
    assert collections_of(users_dir, ALICE) == []  # rien n'a été recréé


def test_one_lock_per_user_survives_library_eviction(
    users_dir: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deux objets de la même bibliothèque (le premier gardé par un appareil, le second ouvert
    après éviction du cache) ne travaillent jamais en même temps."""
    monkeypatch.setattr(user_libraries, "MAX_OPEN_LIBRARIES", 1)
    libraries = open_libraries(users_dir)
    first = libraries.library(ALICE).library
    otters = first.add("Loutres.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())
    libraries.library(BOB)  # évince la bibliothèque d'Alice
    second = libraries.library(ALICE).library
    assert second is not first
    paused = PausedIndexing(monkeypatch)
    upload, _ = run_in_thread(
        lambda: first.add("Volcans.pdf", make_pdf(VOLCANOES, "b.pdf").read_bytes())
    )
    assert paused.written.wait(10)

    deletion, deleted = run_in_thread(lambda: second.delete(otters.entry.doc_id))
    deletion.join(1.0)
    assert deletion.is_alive()  # même verrou : attend l'envoi du premier objet
    paused.resume.set()
    upload.join(10)
    deletion.join(10)

    assert "error" not in deleted
    assert doc_ids(libraries, ALICE) == ["Volcans.pdf"]


def test_terms_acceptance_is_stored_with_the_library_from_the_first_upload(
    users_dir: Path, make_pdf: MakePdf
) -> None:
    libraries = open_libraries(users_dir)
    alice = libraries.library(ALICE)
    assert not alice.terms_accepted("v1")

    alice.accept_terms("v1")
    assert collections_of(users_dir, ALICE) == []  # rien n'est écrit avant le premier envoi
    alice.library.add("Loutres.pdf", make_pdf(OTTERS, "a.pdf").read_bytes())

    reopened = open_libraries(users_dir).library(ALICE)  # redémarrage, autre appareil
    assert reopened.terms_accepted("v1")
    assert not reopened.terms_accepted("v2")  # nouvelle version : à accepter de nouveau
    reopened.accept_terms("v2")  # collection existante : métadonnées modifiées
    assert open_libraries(users_dir).library(ALICE).terms_accepted("v2")
    assert doc_ids(open_libraries(users_dir), ALICE) == ["Loutres.pdf"]
    assert not open_libraries(users_dir).library(BOB).terms_accepted("v2")


# ---------------------------------------------------------------- quotas de Chroma Cloud


class CallSizes:
    """Enveloppe une collection et note la taille de chaque lecture et écriture."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection
        self.sizes: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._collection, name)

    def get(self, **kwargs: Any) -> Any:
        assert kwargs.get("limit") is not None or kwargs.get("ids") is not None
        result = self._collection.get(**kwargs)
        self.sizes.append(len(result["ids"]))
        return result

    def upsert(self, **kwargs: Any) -> Any:
        self.sizes.append(len(kwargs["ids"]))
        return self._collection.upsert(**kwargs)

    def delete(self, **kwargs: Any) -> Any:
        self.sizes.append(len(kwargs["ids"]))
        return self._collection.delete(**kwargs)


def test_large_documents_respect_the_records_per_call_quota(tmp_path: Path) -> None:
    """Chroma Cloud refuse plus de 300 enregistrements par écriture et par lecture."""
    store = ChromaVectorStore(tmp_path / "chroma", "chunks", "fake-hash", 8)
    spy = CallSizes(store._collection)
    store._collection = spy  # espion au contrat de Collection
    total = 2 * MAX_RECORDS_PER_CALL + 50
    chunks = [Chunk(f"d1:{i}", "d1", "gros.pdf", 1 + i // 10, i, 0, f"t{i}") for i in range(total)]

    store.add(chunks, [[1.0] + [0.0] * 7] * total)
    (document,) = store.list_documents()
    removed = store.delete_document("d1")

    assert document.chunk_count == total and removed == total
    assert store.count() == 0
    assert max(spy.sizes) <= MAX_RECORDS_PER_CALL
