"""Adaptateur ChromaDB : stockage vectoriel persistant, local ou Chroma Cloud."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Include, Where
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from doc_assistant.config import Settings
from doc_assistant.domain.errors import EmbeddingModelMismatchError
from doc_assistant.domain.models import Chunk, IndexedDocument, SearchResult
from doc_assistant.domain.ports import Embedder

if TYPE_CHECKING:
    from chromadb.api.client import Client

# Quotas de Chroma Cloud (enregistrements par écriture, résultats par lecture) ; respectés aussi
# en local, où ils ne coûtent qu'un découpage en lots.
MAX_RECORDS_PER_CALL = 300
# Métadonnée du premier passage (``chunk_index`` 0) d'un document : son enregistrement (JSON).
DOCUMENT_RECORD_KEY = "document_record"


def open_persistent_client(path: Path) -> ClientAPI:
    """Client ChromaDB local (dossier ``path``), sans télémétrie."""
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def get_all(
    collection: Any, *, where: Where | None = None, include: Include | None = None
) -> Iterator[tuple[str, str | None, Mapping[str, Any] | None]]:
    """Tous les enregistrements de ``collection`` (``(id, document, métadonnées)``), lus par
    pages de ``MAX_RECORDS_PER_CALL`` : Chroma Cloud refuse de renvoyer davantage en une fois."""
    offset = 0
    while True:
        page = collection.get(
            where=where, include=include or [], limit=MAX_RECORDS_PER_CALL, offset=offset
        )
        ids = page["ids"]
        documents = page.get("documents") or [None] * len(ids)
        metadatas = page.get("metadatas") or [None] * len(ids)
        yield from zip(ids, documents, metadatas, strict=True)
        if len(ids) < MAX_RECORDS_PER_CALL:
            return
        offset += len(ids)


class ChromaVectorStore:
    """Collection Chroma liée à un unique modèle d'embeddings.

    Le nom et la dimension du modèle sont enregistrés dans les métadonnées de la
    collection à sa création ; rouvrir la collection avec un autre modèle lève
    ``EmbeddingModelMismatchError``. Les vecteurs sont toujours fournis par l'appelant :
    Chroma n'utilise jamais sa fonction d'embedding par défaut (qui téléchargerait un modèle).

    ``dimension`` peut être une fonction : elle n'est alors appelée que pour créer une nouvelle
    collection. Une collection existante fournit sa dimension enregistrée, et seul le nom du
    modèle est vérifié : ouvrir l'index (lister, supprimer, rechercher) ne charge pas le modèle.

    ``client`` : client déjà ouvert et partagé (Chroma Cloud, bibliothèques des utilisateurs) ;
    le store ne le ferme alors jamais. Sinon, un client local est ouvert sur ``path``.

    ``lazy`` : la collection n'est créée qu'à la première écriture (``add``) ; d'ici là, les
    lectures renvoient un index vide et rien n'est enregistré dans la base. La création se fait
    dans ``creation_guard`` (par exemple : plafond du nombre de comptes, vérifié sous verrou).
    """

    def __init__(
        self,
        path: Path | None,
        collection_name: str,
        embedding_model: str,
        dimension: int | Callable[[], int],
        *,
        client: ClientAPI | None = None,
        lazy: bool = False,
        creation_guard: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._owns_client = client is None
        if client is None:
            if path is None:
                raise ValueError("un dossier ou un client ChromaDB est nécessaire")
            client = open_persistent_client(path)
        self._client = client
        self._name = collection_name
        self._dimension_source = dimension
        # Dimension connue d'avance : vérifiée ; sinon, celle de la collection (même modèle).
        self._dimension = dimension if isinstance(dimension, int) else 0
        self._creation_guard = creation_guard or contextlib.nullcontext
        self._pending_metadata: dict[str, str] = {}  # métadonnées à poser à la création
        self._collection: Any | None = self._find()
        if self._collection is None and not lazy:
            self._collection = self._create()

    def _find(self) -> Any | None:
        """Collection existante (adoptée et vérifiée), ou ``None``."""
        try:
            collection = self._client.get_collection(self._name, embedding_function=None)
        except NotFoundError:
            return None
        self._adopt(collection)
        return collection

    def _create(self) -> Any:
        source = self._dimension_source
        size = source() if callable(source) else source
        collection = self._client.get_or_create_collection(
            name=self._name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self._embedding_model, "dimension": size}
            | self._pending_metadata,
            embedding_function=None,
        )
        self._adopt(collection)
        return collection

    def _adopt(self, collection: Any) -> None:
        stored = collection.metadata or {}
        if not isinstance(self._dimension_source, int):
            self._dimension = int(stored.get("dimension", 0))
        self._check_compatibility(stored)

    def _existing(self) -> Any | None:
        """Collection à lire ; ``None`` tant qu'elle n'existe pas (store paresseux). Une autre
        instance a pu la créer entre-temps : elle est alors retrouvée."""
        if self._collection is None:
            self._collection = self._find()
        return self._collection

    def _writable(self) -> Any:
        """Collection à écrire, créée au besoin (dans ``creation_guard``)."""
        if self._existing() is None:
            with self._creation_guard():
                self._collection = self._find() or self._create()
        return self._collection

    @property
    def exists(self) -> bool:
        """La collection existe dans la base (toujours vrai pour un store non paresseux)."""
        return self._existing() is not None

    def collection_metadata(self) -> dict[str, Any]:
        collection = self._existing()
        return dict(collection.metadata or {}) if collection is not None else {}

    def set_collection_metadata(self, values: Mapping[str, str]) -> None:
        """Ajoute ``values`` aux métadonnées de la collection ; si elle n'existe pas encore,
        elles seront posées à sa création (rien n'est écrit dans la base d'ici là)."""
        collection = self._existing()
        if collection is None:
            self._pending_metadata.update(values)
            return
        collection.modify(metadata=dict(collection.metadata or {}) | dict(values))

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder) -> ChromaVectorStore:
        """Collection liée au modèle qui produit réellement les vecteurs : celui de l'embedder.

        La dimension n'est demandée à l'embedder (ce qui charge le modèle) que si la
        collection doit être créée.
        """
        return cls(
            settings.chroma_dir,
            settings.chroma_collection,
            embedder.model_name,
            lambda: embedder.dimension,
        )

    def close(self) -> None:
        """Libère les fichiers de l'index (sous Windows, un fichier ouvert ne peut pas être
        supprimé) ; idempotent. Le store n'est plus utilisable ensuite. Un client partagé
        (``client``) n'est jamais fermé ici."""
        if self._owns_client:
            # ``PersistentClient`` renvoie un ``Client`` (qui a ``close``), annoncé ``ClientAPI``.
            cast("Client", self._client).close()

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks mais {len(embeddings)} vecteurs")
        if not chunks:
            return
        collection = self._writable()
        if any(len(vector) != self._dimension for vector in embeddings):
            raise ValueError(f"tous les vecteurs doivent avoir la dimension {self._dimension}")
        batch = min(self._client.get_max_batch_size(), MAX_RECORDS_PER_CALL)
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors: list[Sequence[float]] = list(embeddings[start : start + batch])
            collection.upsert(
                ids=[chunk.chunk_id for chunk in part],
                embeddings=vectors,
                documents=[chunk.text for chunk in part],
                metadatas=[self._to_metadata(chunk) for chunk in part],
            )

    def search(
        self, query_embedding: Sequence[float], top_k: int, *, doc_id: str | None = None
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif")
        collection = self._existing()
        if collection is None:
            return []
        available = collection.count()
        if available == 0:
            return []
        queries: list[Sequence[float]] = [query_embedding]
        # Filtre sur les métadonnées de chaque chunk : jamais de chunk d'un autre document.
        where: Where | None = {"doc_id": doc_id} if doc_id is not None else None
        result = collection.query(
            query_embeddings=queries,
            n_results=min(top_k, available),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            # Distance cosinus de Chroma = 1 - similarité cosinus.
            SearchResult(chunk=self._to_chunk(chunk_id, text, meta), score=1.0 - distance)
            for chunk_id, text, meta, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def get_chunks(self, doc_id: str, chunk_indices: Sequence[int]) -> list[Chunk]:
        collection = self._existing()
        if not chunk_indices or collection is None:
            return []
        membership: dict[Literal["$in", "$nin"], list[str | int | float | bool]] = {
            "$in": list(chunk_indices)
        }
        where: Where = {"$and": [{"doc_id": doc_id}, {"chunk_index": membership}]}
        chunks = [
            self._to_chunk(chunk_id, text or "", meta or {})
            for chunk_id, text, meta in get_all(
                collection, where=where, include=["documents", "metadatas"]
            )
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def has_document(self, doc_id: str) -> bool:
        collection = self._existing()
        if collection is None:
            return False
        return bool(collection.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def delete_document(self, doc_id: str) -> int:
        collection = self._existing()
        if collection is None:
            return 0
        ids = [chunk_id for chunk_id, _, _ in get_all(collection, where={"doc_id": doc_id})]
        for start in range(0, len(ids), MAX_RECORDS_PER_CALL):
            collection.delete(ids=ids[start : start + MAX_RECORDS_PER_CALL])
        return len(ids)

    def count(self) -> int:
        collection = self._existing()
        return collection.count() if collection is not None else 0

    def put_document_record(self, doc_id: str, record: str) -> bool:
        """Attache ``record`` au premier passage du document, dans cette même collection ;
        ``False`` si le document n'a aucun passage. Le registre ne demande ainsi jamais
        d'écrire dans une autre collection (voir ``ChunkRecordRegistry``)."""
        collection = self._existing()
        if collection is None:
            return False
        where: Where = {"$and": [{"doc_id": doc_id}, {"chunk_index": 0}]}
        first = collection.get(where=where, include=["metadatas"], limit=1)
        if not first["ids"]:
            return False
        metadata = dict((first["metadatas"] or [{}])[0] or {})
        metadata[DOCUMENT_RECORD_KEY] = record
        collection.update(ids=first["ids"], metadatas=[metadata])
        return True

    def document_records(self) -> dict[str, str]:
        """Enregistrements attachés par ``put_document_record``, par ``doc_id``."""
        collection = self._existing()
        if collection is None:
            return {}
        return {
            str(meta["doc_id"]): str(meta[DOCUMENT_RECORD_KEY])
            for _, _, meta in get_all(collection, where={"chunk_index": 0}, include=["metadatas"])
            if meta and DOCUMENT_RECORD_KEY in meta
        }

    def list_documents(self) -> list[IndexedDocument]:
        """Regroupe les métadonnées de tous les chunks par ``doc_id`` (lecture seule)."""
        stats: dict[str, IndexedDocument] = {}
        collection = self._existing()
        if collection is None:
            return []
        metadatas = [meta or {} for _, _, meta in get_all(collection, include=["metadatas"])]
        for meta in metadatas:
            doc_id, page = str(meta["doc_id"]), int(meta["page_number"])
            known = stats.get(doc_id)
            stats[doc_id] = IndexedDocument(
                doc_id,
                known.filename if known else str(meta["filename"]),
                (known.chunk_count if known else 0) + 1,
                max(known.last_page if known else 0, page),
            )
        return sorted(stats.values(), key=lambda d: (d.filename.lower(), d.doc_id))

    def _check_compatibility(self, stored: Mapping[str, Any]) -> None:
        stored_model = stored.get("embedding_model")
        stored_dimension = stored.get("dimension")
        if stored_model != self._embedding_model or stored_dimension != self._dimension:
            raise EmbeddingModelMismatchError(
                f"la collection {self._name!r} a été construite avec "
                f"{stored_model!r} (dimension {stored_dimension}) ; impossible de l'utiliser avec "
                f"{self._embedding_model!r} (dimension {self._dimension}). Utilisez une autre "
                "collection (DOCASSIST_CHROMA_COLLECTION) ou réindexez dans une collection vide."
            )

    def _to_metadata(self, chunk: Chunk) -> dict[str, str | int]:
        return {
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "embedding_model": self._embedding_model,
        }

    @staticmethod
    def _to_chunk(chunk_id: str, text: str, meta: Mapping[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=str(meta["doc_id"]),
            filename=str(meta["filename"]),
            page_number=int(meta["page_number"]),
            chunk_index=int(meta["chunk_index"]),
            start_char=int(meta["start_char"]),
            text=text,
        )
