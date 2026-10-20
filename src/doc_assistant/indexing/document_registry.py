"""Registre persistant des documents ajoutés par l'application : ``<data_dir>/documents.json``.

L'index ChromaDB reste la référence de ce qui est indexé ; le registre conserve ce que les
chunks ne contiennent pas (date d'ajout, nombre de pages, fichier conservé). Écriture
atomique (fichier temporaire puis remplacement) ; un registre illisible n'est jamais écrasé.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from doc_assistant.domain.errors import DocumentRegistryError

REGISTRY_VERSION = 1
REGISTRY_FILENAME = "documents.json"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    doc_id: str
    filename: str
    sha256: str
    page_count: int
    chunk_count: int
    empty_pages: tuple[int, ...]
    added_at: str  # ISO 8601, UTC
    stored_file: str | None  # chemin relatif au dossier des envois, ou None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DocumentRecord:
        return cls(
            doc_id=str(raw["doc_id"]),
            filename=str(raw["filename"]),
            sha256=str(raw["sha256"]),
            page_count=int(raw["page_count"]),
            chunk_count=int(raw["chunk_count"]),
            empty_pages=tuple(int(p) for p in raw.get("empty_pages", [])),
            added_at=str(raw["added_at"]),
            stored_file=None if raw.get("stored_file") is None else str(raw["stored_file"]),
        )


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    @classmethod
    def for_data_dir(cls, data_dir: Path) -> DocumentRegistry:
        return cls(data_dir / REGISTRY_FILENAME)

    def records(self) -> dict[str, DocumentRecord]:
        with self._lock:
            return self._read()

    def put(self, record: DocumentRecord) -> None:
        with self._lock:
            records = self._read()
            records[record.doc_id] = record
            self._write(records)

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            records = self._read()
            if records.pop(doc_id, None) is None:
                return False
            self._write(records)
            return True

    def _read(self) -> dict[str, DocumentRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("version") != REGISTRY_VERSION:
                raise ValueError(f"version {raw.get('version')!r} non prise en charge")
            items = [DocumentRecord.from_dict(item) for item in raw["documents"]]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise DocumentRegistryError(
                f"registre des documents illisible ({self.path}) : {exc}. Il n'a pas été "
                "modifié ; corrigez-le ou déplacez-le pour repartir d'un registre vide."
            ) from exc
        return {item.doc_id: item for item in items}

    def _write(self, records: dict[str, DocumentRecord]) -> None:
        payload = {
            "version": REGISTRY_VERSION,
            "documents": [
                asdict(r) | {"empty_pages": list(r.empty_pages)}
                for r in sorted(records.values(), key=lambda r: (r.filename.lower(), r.doc_id))
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".documents-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(tmp, self.path)  # atomique : jamais de registre à moitié écrit
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
