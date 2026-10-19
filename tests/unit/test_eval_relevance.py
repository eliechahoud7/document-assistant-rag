"""Règle de pertinence par citation : cas limites du découpage."""

import pytest

from doc_assistant.domain.models import Chunk, Document, PageContent
from doc_assistant.evaluation.dataset import Reference
from doc_assistant.ingestion.chunking import RecursiveChunker

PAGE = "abcdefghij" * 10  # 100 caractères sans blanc : positions faciles à raisonner


def chunk(start: int, end: int, page: int = 1, filename: str = "f.pdf") -> Chunk:
    return Chunk("c", "d", filename, page, 0, start, PAGE[start:end])


def ref(start: int, end: int) -> Reference:
    return Reference("f.pdf", 1, PAGE[start:end], start)


@pytest.mark.parametrize(
    ("chunk_span", "expected"),
    [
        ((0, 100), 1.0),  # citation entièrement dans le chunk
        ((20, 30), 1.0),  # chunk qui commence et finit exactement avec la citation
        ((25, 60), 0.5),  # exactement la moitié
        ((26, 60), 0.4),  # un peu moins que la moitié
        ((0, 20), 0.0),  # chunk qui s'arrête juste avant la citation
        ((30, 60), 0.0),  # chunk qui commence juste après
        ((22, 27), 0.5),  # chunk plus petit que la citation, à l'intérieur
    ],
)
def test_overlap_ratio(chunk_span: tuple[int, int], expected: float) -> None:
    assert ref(20, 30).overlap_ratio(chunk(*chunk_span)) == pytest.approx(expected)


def test_exactly_half_matches_and_just_below_does_not() -> None:
    reference = ref(20, 30)

    assert reference.matches(chunk(25, 60))
    assert not reference.matches(chunk(26, 60))


def test_other_page_or_file_never_matches() -> None:
    reference = ref(20, 30)

    assert reference.overlap_ratio(chunk(0, 100, page=2)) == 0.0
    assert reference.overlap_ratio(chunk(0, 100, filename="autre.pdf")) == 0.0


def test_quote_at_page_boundaries() -> None:
    assert ref(0, 10).matches(chunk(0, 8))
    assert ref(90, 100).matches(chunk(92, 100))


def test_blank_characters_are_not_counted() -> None:
    page = "mot5 suivan"  # 11 caractères dont 1 espace
    reference = Reference("f.pdf", 1, page, 0)
    # Le découpeur a retiré l'espace entre deux chunks : "mot5" | "suivan".
    left = Chunk("a", "d", "f.pdf", 1, 0, 0, "mot5")
    right = Chunk("b", "d", "f.pdf", 1, 1, 5, "suivan")

    assert reference.overlap_ratio(left) == pytest.approx(4 / 10)
    assert reference.overlap_ratio(right) == pytest.approx(6 / 10)
    assert reference.matches(right)


@pytest.mark.parametrize(("size", "overlap"), [(100, 20), (100, 10), (100, 0), (60, 0)])
@pytest.mark.parametrize("length", [10, 11, 21, 40, 60])
def test_every_quote_is_matched_by_some_real_chunk(size: int, overlap: int, length: int) -> None:
    """Propriété : où que soit la citation (longueur <= chunk_size), un chunk en couvre >= 50 %.

    Avec un décompte brut des caractères, ce test échouait pour des longueurs impaires
    coupées pile à une frontière de chunks (ex. 5/11 = 45 %).
    """
    text = " ".join(f"mot{i} suivant{i}." for i in range(60))
    document = Document("d", "f.pdf", "a" * 64, (PageContent(1, text),))
    chunks = RecursiveChunker(size, overlap).split(document)

    for start in range(len(text) - length):
        quote = text[start : start + length]
        if not quote.strip():
            continue
        reference = Reference("f.pdf", 1, quote, start)
        assert any(reference.matches(c) for c in chunks), (start, quote)
