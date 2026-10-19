"""Jeux d'évaluation : chargement, validation et règle de pertinence.

Une référence n'est pas un ``chunk_id`` (qui dépend du découpage) mais une citation exacte
d'une page. Un chunk retrouvé est pertinent s'il est sur la même page et couvre au moins
``min_overlap`` (50 % par défaut) des caractères de la citation. Le même jeu reste donc
valable quand on change ``chunk_size`` ou ``chunk_overlap``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from doc_assistant.domain.models import Chunk, Document, PageContent
from doc_assistant.domain.ports import DocumentLoader
from doc_assistant.ingestion.cleaning import clean_text

DatasetKind = Literal["synthetic", "real"]
Language = Literal["fr", "en"]
_KINDS: tuple[str, ...] = ("synthetic", "real")
_LANGUAGES: tuple[str, ...] = ("fr", "en")
DEFAULT_MIN_OVERLAP = 0.5
MAX_QUOTE_LENGTH = 300


class DatasetError(ValueError):
    """Jeu d'évaluation invalide."""


def _non_blank(text: str) -> int:
    return sum(not char.isspace() for char in text)


@dataclass(frozen=True, slots=True)
class Reference:
    """Citation exacte attendue, localisée dans le texte nettoyé de sa page."""

    filename: str
    page: int
    quote: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.quote)

    def overlap_ratio(self, chunk: Chunk) -> float:
        """Part des caractères non blancs de la citation couverts par le chunk (0 si autre page).

        Les blancs sont exclus car le découpeur supprime l'espace situé entre deux chunks
        consécutifs : avec un décompte brut, une citation de longueur impaire coupée pile à
        la frontière n'atteindrait 50 % dans aucun des deux chunks. Sur les seuls caractères
        non blancs, les deux chunks se partagent toute la citation : l'un en couvre au moins
        la moitié.
        """
        if chunk.filename != self.filename or chunk.page_number != self.page:
            return 0.0
        first = max(self.start, chunk.start_char) - self.start
        last = min(self.end, chunk.start_char + len(chunk.text)) - self.start
        if last <= first:
            return 0.0
        return _non_blank(self.quote[first:last]) / _non_blank(self.quote)

    def matches(self, chunk: Chunk, min_overlap: float = DEFAULT_MIN_OVERLAP) -> bool:
        return self.overlap_ratio(chunk) >= min_overlap


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    question: str
    lang: Language
    category: str
    references: tuple[Reference, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.references)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    name: str
    kind: DatasetKind
    description: str
    sha256: str
    documents: tuple[Document, ...]
    document_langs: dict[str, Language]
    questions: tuple[EvalQuestion, ...]

    @property
    def answerable(self) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if q.answerable)

    @property
    def unanswerable(self) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if not q.answerable)

    @property
    def longest_quote(self) -> int:
        return max((len(r.quote) for q in self.questions for r in q.references), default=0)

    def language_pair(self, question: EvalQuestion) -> str:
        """Ex. ``"fr->en"`` ; ``"fr->mixed"`` si les références sont dans les deux langues."""
        langs = {self.document_langs[ref.filename] for ref in question.references}
        target = langs.pop() if len(langs) == 1 else "mixed"
        return f"{question.lang}->{target}"


def _synthetic_document(filename: str, pages: list[str]) -> Document:
    """Document construit depuis du texte (même nettoyage que les PDF, sans extraction)."""
    texts = [clean_text(page) for page in pages]
    sha256 = hashlib.sha256("\f".join(texts).encode("utf-8")).hexdigest()
    return Document(
        doc_id=sha256[:16],
        filename=filename,
        sha256=sha256,
        pages=tuple(PageContent(i, text) for i, text in enumerate(texts, start=1)),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def _load_documents(
    raw: list[dict[str, Any]], kind: str, base_dir: Path, loader: DocumentLoader | None
) -> tuple[tuple[Document, ...], dict[str, Language]]:
    documents: list[Document] = []
    langs: dict[str, Language] = {}
    for entry in raw:
        lang = entry.get("lang")
        if lang not in _LANGUAGES:
            raise DatasetError(f"document {entry!r} : 'lang' doit valoir 'fr' ou 'en'")
        if kind == "synthetic":
            _require(
                "filename" in entry and "pages" in entry,
                "document synthétique sans 'filename' ou 'pages'",
            )
            document = _synthetic_document(entry["filename"], entry["pages"])
        else:
            _require("path" in entry, "document réel sans 'path'")
            _require(loader is not None, "un DocumentLoader est requis pour un jeu réel")
            assert loader is not None
            document = loader.load(base_dir / entry["path"])
        _require(document.filename not in langs, f"document en double : {document.filename}")
        documents.append(document)
        langs[document.filename] = cast(Language, lang)
    return tuple(documents), langs


def _load_reference(raw: dict[str, Any], qid: str, pages: dict[str, Document]) -> Reference:
    filename, page, quote = raw.get("filename"), raw.get("page"), raw.get("quote")
    if not isinstance(filename, str) or filename not in pages:
        raise DatasetError(f"{qid} : document inconnu {filename!r}")
    document = pages[filename]
    if not isinstance(page, int) or not 1 <= page <= document.page_count:
        raise DatasetError(
            f"{qid} : page {page!r} hors de {filename} ({document.page_count} pages)"
        )
    if not isinstance(quote, str) or not quote.strip() or len(quote) > MAX_QUOTE_LENGTH:
        raise DatasetError(
            f"{qid} : citation vide ou plus longue que {MAX_QUOTE_LENGTH} caractères"
        )
    text = document.pages[page - 1].text
    occurrences = text.count(quote)
    _require(
        occurrences == 1,
        f"{qid} : citation trouvée {occurrences} fois dans {filename} p. {page} "
        f"(exactement 1 attendue) : {quote!r}",
    )
    return Reference(filename=filename, page=page, quote=quote, start=text.index(quote))


def load_dataset(path: Path, loader: DocumentLoader | None = None) -> EvalDataset:
    """Charge et valide un jeu d'évaluation JSON. Toute incohérence lève ``DatasetError``."""
    content = path.read_bytes()
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{path.name} : JSON invalide ({exc})") from exc

    kind = raw.get("kind")
    _require(kind in _KINDS, f"'kind' doit valoir 'synthetic' ou 'real', reçu {kind!r}")
    _require(bool(raw.get("name")), "'name' manquant")
    documents, langs = _load_documents(raw.get("documents", []), kind, path.parent, loader)
    _require(bool(documents), "le jeu ne contient aucun document")
    by_name = {document.filename: document for document in documents}

    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for item in raw.get("questions", []):
        qid = item.get("id")
        if not isinstance(qid, str) or qid in seen:
            raise DatasetError(f"identifiant absent ou dupliqué : {qid}")
        seen.add(qid)
        _require(bool(str(item.get("question", "")).strip()), f"{qid} : question vide")
        _require(item.get("lang") in _LANGUAGES, f"{qid} : 'lang' doit valoir 'fr' ou 'en'")
        answerable = item.get("answerable")
        _require(isinstance(answerable, bool), f"{qid} : 'answerable' doit être un booléen")
        raw_refs = item.get("references", [])
        _require(
            bool(raw_refs) == answerable,
            f"{qid} : une question répondable a au moins une référence, une question sans "
            "réponse n'en a aucune",
        )
        questions.append(
            EvalQuestion(
                id=qid,
                question=item["question"],
                lang=item["lang"],
                category=str(item.get("category", "uncategorized")),
                references=tuple(_load_reference(r, qid, by_name) for r in raw_refs),
            )
        )
    _require(bool(questions), "le jeu ne contient aucune question")

    return EvalDataset(
        name=raw["name"],
        kind=kind,
        description=str(raw.get("description", "")),
        sha256=hashlib.sha256(content).hexdigest(),
        documents=documents,
        document_langs=langs,
        questions=tuple(questions),
    )
