"""Interface en ligne de commande : ``python -m doc_assistant <commande>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from doc_assistant import bootstrap
from doc_assistant.config import ConfigError, Settings
from doc_assistant.domain.errors import (
    EmbeddingModelMismatchError,
    EmbeddingModelNotFoundError,
    IngestionError,
    LLMError,
)
from doc_assistant.domain.ports import Embedder
from doc_assistant.evaluation.dataset import load_dataset
from doc_assistant.evaluation.generation import (
    format_generation_summary,
    format_review_sheet,
    load_gold,
    load_overrides,
    read_records,
    record_answers,
    score_run,
)
from doc_assistant.evaluation.runner import (
    build_report,
    format_summary,
    index_dataset,
    run_evaluation,
)
from doc_assistant.indexing.model_download import download_embedding_model
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader

EXCERPT_LENGTH = 300


def _collect_pdfs(paths: Sequence[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for path in paths:
        if path.is_dir():
            pdfs.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() == ".pdf"))
        else:
            pdfs.append(path)
    return pdfs


def _download(settings: Settings, _args: argparse.Namespace) -> int:
    print(f"Téléchargement de {settings.embedding_model} ({settings.embedding_model_revision[:8]})")
    print(f"vers {settings.embedding_model_dir} ...")
    target = download_embedding_model(settings)
    print(f"Modèle prêt : {target}")
    return 0


def _index(settings: Settings, args: argparse.Namespace) -> int:
    pdfs = _collect_pdfs(args.paths)
    if not pdfs:
        print("Aucun PDF trouvé.", file=sys.stderr)
        return 1
    service = bootstrap.build_indexing_service(settings, bootstrap.build_embedder(settings))
    failures = 0
    for pdf in pdfs:
        try:
            report = service.index_pdf(pdf, force=args.force)
        except (IngestionError, FileNotFoundError) as exc:
            failures += 1
            print(f"ÉCHEC   {pdf.name} : {exc}", file=sys.stderr)
            continue
        if report.skipped:
            print(f"IGNORÉ  {report.filename} : déjà indexé (--force pour réindexer)")
            continue
        warning = f", pages sans texte : {list(report.empty_pages)}" if report.empty_pages else ""
        print(
            f"INDEXÉ  {report.filename} : {report.page_count} pages, "
            f"{report.chunk_count} chunks en {report.seconds:.1f} s{warning}"
        )
    return 1 if failures else 0


def _search(settings: Settings, args: argparse.Namespace) -> int:
    retriever = bootstrap.build_retriever(settings, bootstrap.build_embedder(settings))
    results = retriever.search(args.question, top_k=args.top_k)
    if not results:
        print("Aucun passage indexé. Lancez d'abord : python -m doc_assistant index <pdf>")
        return 1
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"[{rank}] score={result.score:.3f}  {chunk.filename}, page {chunk.page_number}")
        print(f"    {_excerpt(chunk.text)}")
    return 0


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:EXCERPT_LENGTH] + "…" if len(flat) > EXCERPT_LENGTH else flat


def _ask(settings: Settings, args: argparse.Namespace) -> int:
    # Configuration LLM vérifiée avant de charger le modèle d'embeddings (~13 s).
    generator = bootstrap.build_generator(settings)
    service = bootstrap.build_qa_service(settings, bootstrap.build_embedder(settings), generator)
    passages = service.retrieve(args.question, top_k=args.top_k)
    if not passages:
        print("Aucun passage indexé. Lancez d'abord : python -m doc_assistant index <pdf>")
        return 1

    pages: dict[str, list[int]] = {}
    for result in passages:
        pages.setdefault(result.chunk.filename, []).append(result.chunk.page_number)
    sent = "; ".join(
        f"{name} p. {', '.join(map(str, sorted(set(p))))}" for name, p in pages.items()
    )
    print(f"Index : {settings.chroma_dir}")
    target = f"{settings.llm_provider} ({generator.model_name})"
    print(f"Envoi de {len(passages)} passages à {target} : {sent}")
    if args.show_passages:
        for rank, result in enumerate(passages, start=1):
            chunk = result.chunk
            print(
                f"  [{rank}] score={result.score:.3f}  {chunk.filename}, page {chunk.page_number}"
            )
            print(f"      {_excerpt(chunk.text)}")

    answer = service.answer(args.question, passages).answer
    print()
    if answer.abstained:
        print("Pas de réponse trouvée dans les documents.")
        print(f"(raison : {answer.abstain_reason})")
        if answer.text:
            print(answer.text)
        return 0
    print(answer.text)
    print("\nSources :")
    for citation in answer.citations:
        print(f"  [{citation.number}] {citation.filename}, page {citation.page_number}")
        print(f"      {_excerpt(citation.chunk.text)}")
    return 0


@contextmanager
def _temporary_store(embedder: Embedder) -> Iterator[ChromaVectorStore]:
    """Index jetable : les évaluations ne lisent ni n'écrivent jamais un index de l'utilisateur."""
    with tempfile.TemporaryDirectory(prefix="docassist-eval-", ignore_cleanup_errors=True) as tmp:
        yield ChromaVectorStore(Path(tmp), "evaluation", embedder.model_name, embedder.dimension)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evaluate(settings: Settings, args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset, loader=PypdfLoader())
    embedder = bootstrap.build_embedder(settings)
    with _temporary_store(embedder) as store:
        run = run_evaluation(
            dataset,
            embedder,
            RecursiveChunker.from_settings(settings),
            store,
            top_k=args.top_k,
            chunk_size=settings.chunk_size,
        )
    report = build_report(
        run,
        {
            "embedding_model": embedder.model_name,
            "embedding_model_revision": settings.embedding_model_revision,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )
    print(format_summary(report))
    if args.output:
        _write_json(args.output, report)
        print(f"\nRapport écrit : {args.output}")
    return 0


QUOTA_EXIT_CODE = 3


def _generate_answers(settings: Settings, args: argparse.Namespace) -> int:
    if not args.live:
        print(
            "Erreur : generate-answers envoie les passages retrouvés à l'API LLM et consomme du "
            "quota ; relancez avec --live pour confirmer.",
            file=sys.stderr,
        )
        return 2
    dataset = load_dataset(args.dataset, loader=PypdfLoader())
    generator = bootstrap.build_generator(settings)  # vérifié avant de charger le modèle
    embedder = bootstrap.build_embedder(settings)
    run_config = {
        "provider": settings.llm_provider,
        "model": generator.model_name,
        "top_k": args.top_k,
        "embedding_model": embedder.model_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    if settings.context_neighbors:
        # Absent à 0 : les enregistrements d'origine gardent une configuration comparable.
        run_config["context_neighbors"] = settings.context_neighbors
    target = f"{settings.llm_provider} ({generator.model_name})"
    print(f"Envoi des passages à {target} ; sortie : {args.output}")
    with _temporary_store(embedder) as store:
        retriever = index_dataset(
            dataset, embedder, RecursiveChunker.from_settings(settings), store
        )
        summary = record_answers(
            dataset,
            retriever,
            generator,
            args.output,
            run_config=run_config,
            top_k=args.top_k,
            context_neighbors=settings.context_neighbors,
            delay_s=args.delay,
            progress=print,
        )
    print(
        f"\n{summary.recorded} enregistrées, {summary.skipped} déjà présentes, "
        f"{summary.errors} en erreur, {summary.api_calls} appels à l'API"
    )
    if summary.stopped_on_quota:
        print(
            f"Arrêt sur quota : {len(summary.remaining)} question(s) restante(s). Relancez la même "
            "commande plus tard pour reprendre.",
            file=sys.stderr,
        )
        return QUOTA_EXIT_CODE
    return 1 if summary.errors else 0


def _score_answers(settings: Settings, args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset, loader=PypdfLoader())
    gold = load_gold(args.gold, dataset)
    records = read_records(args.run)
    if not records:
        print(f"Erreur : aucun enregistrement dans {args.run}", file=sys.stderr)
        return 2
    overrides = load_overrides(args.overrides, dataset) if args.overrides else None
    report = score_run(records, dataset, gold, overrides)
    print(format_generation_summary(report))
    if args.output:
        _write_json(args.output, report)
        print(f"\nRapport écrit : {args.output}")
    if args.review:
        args.review.parent.mkdir(parents=True, exist_ok=True)
        args.review.write_text(format_review_sheet(report, records, gold), encoding="utf-8")
        print(f"Fiche de relecture écrite : {args.review}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m doc_assistant", description="Assistant documentaire RAG."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download-model", help="télécharge le modèle d'embeddings")
    download.set_defaults(handler=_download)

    index = commands.add_parser("index", help="indexe des PDF (fichiers ou dossiers)")
    index.add_argument("paths", nargs="+", type=Path)
    index.add_argument("--force", action="store_true", help="réindexe les PDF déjà présents")
    index.set_defaults(handler=_index)

    search = commands.add_parser("search", help="recherche les passages les plus pertinents")
    search.add_argument("question")
    search.add_argument("-k", "--top-k", type=int, default=None, help="nombre de résultats")
    search.set_defaults(handler=_search)

    ask = commands.add_parser(
        "ask", help="répond à une question en citant les passages (LLM, envoie les passages)"
    )
    ask.add_argument("question")
    ask.add_argument("-k", "--top-k", type=int, default=None, help="passages transmis au LLM")
    ask.add_argument(
        "--show-passages", action="store_true", help="affiche les passages envoyés au LLM"
    )
    ask.set_defaults(handler=_ask)

    evaluate = commands.add_parser(
        "evaluate", help="évalue la recherche sur un jeu annoté (index temporaire)"
    )
    evaluate.add_argument("dataset", type=Path, help="jeu d'évaluation JSON")
    evaluate.add_argument("-o", "--output", type=Path, default=None, help="rapport JSON")
    evaluate.add_argument("-k", "--top-k", type=int, default=5, help="résultats par question")
    evaluate.set_defaults(handler=_evaluate)

    generate = commands.add_parser(
        "generate-answers",
        help="enregistre les réponses du LLM sur un jeu annoté (APPELS API, reprise possible)",
    )
    generate.add_argument("dataset", type=Path, help="jeu d'évaluation JSON")
    generate.add_argument("-o", "--output", type=Path, required=True, help="fichier JSONL")
    generate.add_argument("-k", "--top-k", type=int, default=5, help="passages par question")
    generate.add_argument("--delay", type=float, default=5.0, help="pause entre appels (s)")
    generate.add_argument("--live", action="store_true", help="confirme les appels à l'API")
    generate.set_defaults(handler=_generate_answers)

    score = commands.add_parser(
        "score-answers", help="note un enregistrement de réponses (hors ligne, déterministe)"
    )
    score.add_argument("run", type=Path, help="fichier JSONL produit par generate-answers")
    score.add_argument("--dataset", type=Path, required=True, help="jeu d'évaluation JSON")
    score.add_argument("--gold", type=Path, required=True, help="annexe des réponses attendues")
    score.add_argument("--overrides", type=Path, default=None, help="verdicts humains (JSON)")
    score.add_argument("-o", "--output", type=Path, default=None, help="rapport JSON")
    score.add_argument("--review", type=Path, default=None, help="fiche de relecture Markdown")
    score.set_defaults(handler=_score_answers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        # Bibliothèques tierces en WARNING ; notre paquet au niveau configuré.
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        logging.getLogger("doc_assistant").setLevel(settings.log_level)
        code: int = args.handler(settings, args)
    except (
        ConfigError,
        EmbeddingModelNotFoundError,
        EmbeddingModelMismatchError,
        IngestionError,
        LLMError,
        FileNotFoundError,
        ValueError,  # inclut DatasetError
    ) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    return code
