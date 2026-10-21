"""Registre des documents conservé avec leurs passages, dans la collection ChromaDB de
l'utilisateur (bibliothèques des utilisateurs connectés).

Même contrat que ``DocumentRegistry`` (fichier JSON local), mais persistant là où l'index l'est :
le disque du serveur de Streamlit Community Cloud est effacé à chaque redémarrage.
L'enregistrement de chaque document (JSON) est une métadonnée de son premier passage.

Pourquoi pas une collection à part : avec le moteur local de ChromaDB 1.5.9, écrire dans une
collection puis dans une autre de la même base peut rendre la recherche de la première
impossible (``InternalError : Error creating hnsw segment reader: Nothing found on disk``)
jusqu'à sa prochaine écriture ; reproduit sans le code du projet. Un ajout n'écrit donc que
dans la collection de l'utilisateur.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from doc_assistant.domain.errors import DocumentRegistryError
from doc_assistant.indexing.document_registry import DocumentRecord
from doc_assistant.indexing.vector_store import ChromaVectorStore


class ChunkRecordRegistry:
    def __init__(self, store: ChromaVectorStore) -> None:
        self._store = store

    def records(self) -> dict[str, DocumentRecord]:
        try:
            items = [
                DocumentRecord.from_dict(json.loads(raw))
                for raw in self._store.document_records().values()
            ]
        except (ValueError, KeyError, TypeError) as exc:
            raise DocumentRegistryError(f"registre des documents illisible : {exc}") from exc
        return {item.doc_id: item for item in items}

    def put(self, record: DocumentRecord) -> None:
        payload = asdict(record) | {"empty_pages": list(record.empty_pages)}
        if not self._store.put_document_record(
            record.doc_id, json.dumps(payload, ensure_ascii=False)
        ):
            raise DocumentRegistryError(f"aucun passage indexé pour le document {record.doc_id}")

    def remove(self, doc_id: str) -> bool:
        """L'enregistrement disparaît avec les passages du document (supprimés juste avant par
        ``DocumentLibrary.delete``) ; ``True`` s'il n'en reste rien."""
        return not self._store.has_document(doc_id)
