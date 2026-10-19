"""Exécution d'une évaluation et production d'un rapport reproductible."""

from __future__ import annotations

import platform
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from doc_assistant.domain.models import SearchResult
from doc_assistant.domain.ports import Chunker, Embedder, VectorStore
from doc_assistant.evaluation.dataset import DEFAULT_MIN_OVERLAP, EvalDataset, EvalQuestion
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.retrieval.metrics import (
    auroc,
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
)
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.indexing_service import IndexingService

KS = (1, 3, 5)
DECIMALS = 6
_PACKAGES = ("pypdf", "langchain-text-splitters", "chromadb", "sentence-transformers", "torch")


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question: EvalQuestion
    retrieved: tuple[SearchResult, ...]
    reference_ranks: tuple[int | None, ...]

    @property
    def first_relevant_rank(self) -> int | None:
        found = [rank for rank in self.reference_ranks if rank is not None]
        return min(found) if found else None

    @property
    def top1_score(self) -> float:
        return self.retrieved[0].score


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    dataset: EvalDataset
    results: tuple[QuestionResult, ...]
    chunk_count: int
    top_k: int
    min_overlap: float


def index_dataset(
    dataset: EvalDataset, embedder: Embedder, chunker: Chunker, store: VectorStore
) -> Retriever:
    """Indexe les documents du jeu dans un index dédié et vide, puis renvoie le retriever.

    Partagé par l'évaluation de la recherche et celle de la génération : les deux
    interrogent exactement le même index temporaire, construit de la même façon.
    """
    if store.count() != 0:
        raise ValueError("l'évaluation exige un index vide (index temporaire dédié)")
    service = IndexingService(PypdfLoader(), chunker, embedder, store)
    for document in dataset.documents:
        service.index_document(document)
    return Retriever(embedder, store)


def run_evaluation(
    dataset: EvalDataset,
    embedder: Embedder,
    chunker: Chunker,
    store: VectorStore,
    *,
    top_k: int = max(KS),
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    chunk_size: int | None = None,
) -> EvaluationRun:
    """Indexe les documents du jeu dans ``store`` (qui doit être vide), puis interroge.

    Le store doit être un index dédié : l'évaluation refuse d'écrire dans un index existant.
    """
    if store.count() != 0:
        raise ValueError("l'évaluation exige un index vide (index temporaire dédié)")
    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif")
    if chunk_size is not None and dataset.longest_quote > chunk_size:
        raise ValueError(
            f"une citation fait {dataset.longest_quote} caractères, plus que chunk_size="
            f"{chunk_size} : aucun chunk ne pourrait en couvrir la moitié de façon fiable"
        )
    retriever = index_dataset(dataset, embedder, chunker, store)
    results = []
    for question in dataset.questions:
        retrieved = tuple(retriever.search(question.question, top_k=top_k))
        ranks = tuple(
            next(
                (i for i, r in enumerate(retrieved, start=1) if ref.matches(r.chunk, min_overlap)),
                None,
            )
            for ref in question.references
        )
        results.append(QuestionResult(question, retrieved, ranks))
    return EvaluationRun(dataset, tuple(results), store.count(), top_k, min_overlap)


def _round(value: float) -> float:
    return round(value, DECIMALS)


def _retrieval_metrics(results: Sequence[QuestionResult], top_k: int) -> dict[str, Any]:
    ks = [k for k in KS if k <= top_k]
    first = [r.first_relevant_rank for r in results]
    refs = [list(r.reference_ranks) for r in results]
    metrics: dict[str, Any] = {"n": len(results)}
    metrics |= {f"recall@{k}": _round(recall_at_k(refs, k)) for k in ks}
    metrics |= {f"hit@{k}": _round(hit_rate_at_k(first, k)) for k in ks}
    metrics["mrr"] = _round(mean_reciprocal_rank(first))
    return metrics


def _grouped(
    results: Sequence[QuestionResult], key: Callable[[QuestionResult], str], top_k: int
) -> dict[str, Any]:
    groups: dict[str, list[QuestionResult]] = {}
    for result in results:
        groups.setdefault(key(result), []).append(result)
    return {name: _retrieval_metrics(group, top_k) for name, group in sorted(groups.items())}


def _stats(scores: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(scores),
        "mean": _round(statistics.fmean(scores)),
        "median": _round(statistics.median(scores)),
        "min": _round(min(scores)),
        "max": _round(max(scores)),
    }


def _absence_analysis(run: EvaluationRun) -> dict[str, Any] | None:
    """Le score du meilleur passage distingue-t-il les questions sans réponse ?"""
    positives = [r.top1_score for r in run.results if r.question.answerable]
    negatives = [r.top1_score for r in run.results if not r.question.answerable]
    if not positives or not negatives:
        return None
    max_negative, min_positive = max(negatives), min(positives)
    return {
        "signal": "score cosinus du meilleur passage (plus élevé = plus probablement répondable)",
        "auroc": _round(auroc(positives, negatives)),
        "answerable_top1": _stats(positives),
        "unanswerable_top1": _stats(negatives),
        "separable": min_positive > max_negative,
        "answerable_at_or_below_max_unanswerable": sum(p <= max_negative for p in positives),
        "unanswerable_at_or_above_min_answerable": sum(n >= min_positive for n in negatives),
    }


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "absent"
    return versions


def build_report(run: EvaluationRun, config: dict[str, Any]) -> dict[str, Any]:
    """Rapport JSON. Seul ``generated_at`` change d'une exécution à l'autre."""
    answerable = [r for r in run.results if r.question.answerable]
    dataset = run.dataset
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "name": dataset.name,
            "kind": dataset.kind,
            "sha256": dataset.sha256,
            "documents": len(dataset.documents),
            "questions": len(dataset.questions),
            "answerable": len(dataset.answerable),
            "unanswerable": len(dataset.unanswerable),
        },
        "config": config
        | {"top_k": run.top_k, "min_overlap": run.min_overlap, "chunk_count": run.chunk_count},
        "environment": _versions(),
        "retrieval": {
            "overall": _retrieval_metrics(answerable, run.top_k) if answerable else None,
            "by_category": _grouped(answerable, lambda r: r.question.category, run.top_k),
            "by_language_pair": _grouped(
                answerable, lambda r: dataset.language_pair(r.question), run.top_k
            ),
        },
        "absence_detection": _absence_analysis(run),
        "questions": [
            {
                "id": r.question.id,
                "lang": r.question.lang,
                "category": r.question.category,
                "answerable": r.question.answerable,
                "reference_ranks": list(r.reference_ranks),
                "first_relevant_rank": r.first_relevant_rank,
                "top1_score": _round(r.top1_score),
                "retrieved": [
                    {
                        "filename": s.chunk.filename,
                        "page": s.chunk.page_number,
                        "chunk_index": s.chunk.chunk_index,
                        "score": _round(s.score),
                    }
                    for s in r.retrieved
                ],
            }
            for r in run.results
        ],
    }


def format_summary(report: dict[str, Any]) -> str:
    dataset, config = report["dataset"], report["config"]
    kind = {"synthetic": "SYNTHÉTIQUE", "real": "RÉEL"}[dataset["kind"]]
    lines = [
        f"Jeu : {dataset['name']} ({kind}) — {dataset['questions']} questions "
        f"({dataset['answerable']} répondables, {dataset['unanswerable']} sans réponse), "
        f"{dataset['documents']} documents, {config['chunk_count']} chunks",
        f"Config : {config['embedding_model']}, chunk_size={config['chunk_size']}, "
        f"chunk_overlap={config['chunk_overlap']}, top_k={config['top_k']}, "
        f"chevauchement minimal={config['min_overlap']:.0%}",
    ]
    if dataset["kind"] == "synthetic":
        lines.append(
            "ATTENTION : jeu SYNTHÉTIQUE — ces chiffres ne mesurent pas la qualité sur de vrais "
            "documents."
        )

    overall = report["retrieval"]["overall"]
    if overall:
        metric_names = [name for name in overall if name != "n"]
        lines += ["", "Recherche (questions répondables)"]
        header = f"  {'groupe':<22} {'n':>3} " + " ".join(f"{m:>9}" for m in metric_names)
        lines.append(header)

        def row(label: str, values: dict[str, Any]) -> str:
            cells = " ".join(f"{values[m]:>9.3f}" for m in metric_names)
            return f"  {label:<22} {values['n']:>3} {cells}"

        lines.append(row("global", overall))
        for section in ("by_category", "by_language_pair"):
            for name, values in report["retrieval"][section].items():
                lines.append(row(name, values))

    absence = report["absence_detection"]
    if absence:
        pos, neg = absence["answerable_top1"], absence["unanswerable_top1"]
        lines += [
            "",
            "Détection d'absence (score du meilleur passage ; plus élevé = répondable)",
            f"  AUROC = {absence['auroc']:.3f}  (1 = séparation parfaite, 0,5 = hasard)",
            f"  répondables   : moyenne {pos['mean']:.3f}  [{pos['min']:.3f} ; {pos['max']:.3f}]",
            f"  sans réponse  : moyenne {neg['mean']:.3f}  [{neg['min']:.3f} ; {neg['max']:.3f}]",
        ]
        if absence["separable"]:
            lines.append("  Distributions séparées sur ce jeu (à confirmer sur d'autres données).")
        else:
            below = absence["answerable_at_or_below_max_unanswerable"]
            lines.append(
                f"  Distributions qui se chevauchent : {below} question(s) répondable(s) sous le "
                "score maximal d'une question sans réponse ; aucun seuil ne sépare parfaitement "
                "les deux groupes."
            )
    return "\n".join(lines)
