"""Registre persistant des documents : relecture, écriture atomique, registre illisible."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.domain.errors import DocumentRegistryError
from doc_assistant.indexing.document_registry import DocumentRecord, DocumentRegistry


def record(doc_id: str = "0123456789abcdef", filename: str = "cours.pdf") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        filename=filename,
        sha256=doc_id * 4,
        page_count=3,
        chunk_count=5,
        empty_pages=(2,),
        added_at="2026-10-20T10:00:00+00:00",
        stored_file=f"{doc_id}/{filename}",
    )


def test_records_survive_a_new_registry_instance(tmp_path: Path) -> None:
    DocumentRegistry.for_data_dir(tmp_path).put(record())

    reloaded = DocumentRegistry.for_data_dir(tmp_path).records()  # « redémarrage »

    assert reloaded == {"0123456789abcdef": record()}


def test_missing_registry_is_empty_and_remove_reports_absence(tmp_path: Path) -> None:
    registry = DocumentRegistry.for_data_dir(tmp_path)

    assert registry.records() == {}
    assert registry.remove("0123456789abcdef") is False
    assert not (tmp_path / "documents.json").exists()  # rien d'écrit sans modification


def test_remove_only_drops_the_given_document(tmp_path: Path) -> None:
    registry = DocumentRegistry.for_data_dir(tmp_path)
    registry.put(record("aaaaaaaaaaaaaaaa", "a.pdf"))
    registry.put(record("bbbbbbbbbbbbbbbb", "b.pdf"))

    assert registry.remove("aaaaaaaaaaaaaaaa") is True

    assert list(DocumentRegistry.for_data_dir(tmp_path).records()) == ["bbbbbbbbbbbbbbbb"]


def test_write_is_atomic_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    registry = DocumentRegistry.for_data_dir(tmp_path)
    registry.put(record())

    assert sorted(p.name for p in tmp_path.iterdir()) == ["documents.json"]


@pytest.mark.parametrize("content", ["{pas du json", '{"version": 99, "documents": []}', "[]"])
def test_unreadable_registry_is_reported_and_never_overwritten(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "documents.json"
    path.write_text(content, encoding="utf-8")
    registry = DocumentRegistry(path)

    with pytest.raises(DocumentRegistryError, match="illisible"):
        registry.records()
    with pytest.raises(DocumentRegistryError):
        registry.put(record())

    assert path.read_text(encoding="utf-8") == content
