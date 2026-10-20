"""Envoi d'un PDF puis question immédiate, avec un vrai index ChromaDB (dossier temporaire).

Reproduit l'enchaînement de l'interface : le service de questions est construit d'abord (il
est mis en cache par l'application), l'indexation passe par un second accès au même index.
"""

from __future__ import annotations

from pathlib import Path

from conftest import MakePdf
from doc_assistant import bootstrap
from doc_assistant.config import Settings
from doc_assistant.services.upload_service import index_uploaded_pdf
from fakes import FakeEmbedder, ScriptedAnswerGenerator


def test_uploaded_pdf_is_immediately_answerable_with_citation(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path / "courses")
    embedder = FakeEmbedder()
    generator = ScriptedAnswerGenerator({}, model_name="fake-gemini")
    first = make_pdf([["Les chats dorment beaucoup pendant la journée."]], name="chats.pdf")
    index_uploaded_pdf(
        "chats.pdf",
        first.read_bytes(),
        bootstrap.build_indexing_service(settings, embedder),
        settings.uploads_dir,
    )
    qa = bootstrap.build_qa_service(settings, embedder, generator)  # comme le cache de l'app

    new = make_pdf([["Les loutres de mer dorment en se tenant la patte."]], name="src.pdf")
    result = index_uploaded_pdf(
        "Loutres.pdf",
        new.read_bytes(),
        bootstrap.build_indexing_service(settings, embedder),
        settings.uploads_dir,
    )
    answer = qa.ask("Comment dorment les loutres de mer ?", top_k=1)

    assert not result.report.skipped
    assert [c.filename for c in answer.answer.citations] == ["Loutres.pdf"]
    assert answer.answer.citations[0].page_number == 1
    assert generator.calls == ["Comment dorment les loutres de mer ?"]
    assert (settings.chroma_dir / "chroma.sqlite3").is_file()
    assert result.stored_path == settings.uploads_dir / result.report.doc_id / "Loutres.pdf"
