"""Modèles du domaine : objets immuables et validés à la construction."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PageContent:
    """Texte nettoyé d'une page. ``page_number`` commence à 1, comme dans un lecteur PDF."""

    page_number: int
    text: str

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"page_number doit être >= 1, reçu {self.page_number}")

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Document:
    """Un PDF importé : identité, empreinte et contenu page par page."""

    doc_id: str
    filename: str
    sha256: str
    pages: tuple[PageContent, ...]

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id ne peut pas être vide")
        if not self.filename:
            raise ValueError("filename ne peut pas être vide")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 doit contenir 64 caractères hexadécimaux en minuscules")
        numbers = [page.page_number for page in self.pages]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("les pages doivent être numérotées 1, 2, ..., n dans l'ordre")

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def empty_pages(self) -> tuple[int, ...]:
        """Numéros des pages sans texte extractible (souvent des images ou des scans)."""
        return tuple(page.page_number for page in self.pages if page.is_empty)


@dataclass(frozen=True, slots=True)
class Chunk:
    """Passage indexable, avec tout ce qu'il faut pour le citer.

    ``chunk_index`` est la position du chunk dans le document (0, 1, 2...) ;
    ``start_char`` est sa position dans le texte de sa page.
    """

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    start_char: int
    text: str

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id ne peut pas être vide")
        if not self.doc_id:
            raise ValueError("doc_id ne peut pas être vide")
        if not self.filename:
            raise ValueError("filename ne peut pas être vide")
        if self.page_number < 1:
            raise ValueError(f"page_number doit être >= 1, reçu {self.page_number}")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index doit être >= 0, reçu {self.chunk_index}")
        if self.start_char < 0:
            raise ValueError(f"start_char doit être >= 0, reçu {self.start_char}")
        if not self.text.strip():
            raise ValueError("le texte d'un chunk ne peut pas être vide")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Chunk retrouvé et sa similarité cosinus avec la question (entre -1 et 1)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    """Source d'une réponse, construite à partir du chunk retrouvé (jamais du texte du LLM).

    ``number`` est la position (1, 2, ...) du passage dans le contexte fourni au LLM.
    """

    number: int
    chunk: Chunk

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"number doit être >= 1, reçu {self.number}")

    @property
    def filename(self) -> str:
        return self.chunk.filename

    @property
    def page_number(self) -> int:
        return self.chunk.page_number


@dataclass(frozen=True, slots=True)
class Answer:
    """Réponse générée. Une réponse non abstenue cite toujours au moins un passage."""

    text: str
    citations: tuple[Citation, ...]
    abstained: bool
    model: str
    abstain_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.abstained:
            if self.citations:
                raise ValueError("une abstention ne cite aucun passage")
            if not self.abstain_reason:
                raise ValueError("une abstention doit indiquer sa raison")
        else:
            if not self.text.strip():
                raise ValueError("une réponse non abstenue ne peut pas être vide")
            if not self.citations:
                raise ValueError("une réponse non abstenue doit citer au moins un passage")
