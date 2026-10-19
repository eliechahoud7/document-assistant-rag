"""PDF réel (généré) -> Document -> chunks : la chaîne complète de la phase 1."""

from collections.abc import Callable
from pathlib import Path

from doc_assistant.config import Settings
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader

MakePdf = Callable[..., Path]

PARAGRAPH = [
    "La génération augmentée par la recherche (RAG) combine un moteur de recherche",
    "et un modèle de langage. Les passages retrouvés servent de contexte à la réponse,",
    "ce qui permet de citer précisément les sources utilisées.",
]


def test_pdf_to_chunks_keeps_citation_metadata(make_pdf: MakePdf) -> None:
    path = make_pdf([PARAGRAPH * 4, [], PARAGRAPH * 2], name="rag.pdf")
    settings = Settings(chunk_size=300, chunk_overlap=50)

    document = PypdfLoader().load(path)
    chunks = RecursiveChunker.from_settings(settings).split(document)

    assert document.empty_pages == (2,)
    assert len(chunks) > 2
    assert {chunk.page_number for chunk in chunks} == {1, 3}
    for chunk in chunks:
        assert chunk.filename == "rag.pdf"
        assert chunk.doc_id == document.doc_id
        assert len(chunk.text) <= settings.chunk_size
        page_text = document.pages[chunk.page_number - 1].text
        assert page_text[chunk.start_char : chunk.start_char + len(chunk.text)] == chunk.text
