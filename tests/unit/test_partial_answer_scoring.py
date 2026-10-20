"""Questions en partie répondables : champ facultatif ``unsupported`` de l'annexe.

Hors ligne : petit jeu synthétique, enregistrements écrits à la main. Les heuristiques testées
(marqueurs d'absence, faits interdits, nombres absents des passages) sont approximatives : ces
tests fixent leur comportement, ils ne prouvent pas qu'elles repèrent toute invention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc_assistant.evaluation.dataset import DatasetError, EvalDataset, load_dataset
from doc_assistant.evaluation.generation import (
    DEFAULT_ABSENCE_MARKERS,
    format_generation_summary,
    format_review_sheet,
    load_gold,
    load_overrides,
    score_run,
)
from doc_assistant.ingestion.chunking import RecursiveChunker

ROOT = Path(__file__).resolve().parents[2]
PAGE = (
    "Random forests consider a random subset of features at each split. "
    "Bagging draws 63.2% unique samples."
)
DATASET = {
    "name": "mini_partial",
    "kind": "synthetic",
    "documents": [{"filename": "trees.pdf", "lang": "en", "pages": [PAGE]}],
    "questions": [
        {
            "id": "q1",
            "lang": "en",
            "category": "paraphrase",
            "answerable": True,
            "question": "What share of unique samples does bagging draw?",
            "references": [{"filename": "trees.pdf", "page": 1, "quote": "63.2% unique samples"}],
        },
        {
            "id": "p1",
            "lang": "en",
            "category": "multi_part_partial",
            "answerable": True,
            "question": "How do random forests diversify trees, and how many trees should be used?",
            "references": [
                {"filename": "trees.pdf", "page": 1, "quote": "random subset of features"}
            ],
        },
        {
            "id": "u1",
            "lang": "en",
            "category": "unanswerable_on_topic",
            "answerable": False,
            "question": "Who invented bagging?",
        },
    ],
}
CONFIG = {"provider": "fake", "model": "scripted", "top_k": 1}
SUPPORTED = "Random forests use a random subset of features at each split [1]."
UNSUPPORTED = {"part": "number of trees", "forbidden_facts": [["500 trees", "500 arbres"]]}


@pytest.fixture
def dataset(tmp_path: Path) -> EvalDataset:
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(DATASET), encoding="utf-8")
    return load_dataset(path)


def gold_file(
    tmp_path: Path, dataset: EvalDataset, unsupported: dict[str, Any] | None = UNSUPPORTED
) -> Path:
    p1: dict[str, Any] = {
        "expected_answer": "A random subset of features at each split; tree count not given.",
        "required_facts": [["subset"]],
        "support": "weak",
        "flags": ["partie sans réponse"],
    }
    if unsupported is not None:
        p1["unsupported"] = unsupported
    answers = {
        "q1": {
            "expected_answer": "63.2%.",
            "required_facts": [["63.2"]],
            "support": "strong",
            "flags": [],
        },
        "p1": p1,
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"dataset_sha256": dataset.sha256, "answers": answers}), "utf-8")
    return path


def record(dataset: EvalDataset, qid: str, text: str, *, abstained: bool = False) -> dict[str, Any]:
    chunk = RecursiveChunker(800, 100).split(dataset.documents[0])[0]
    retrieved = [
        {
            "rank": 1,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "score": 0.9,
            "text": chunk.text,
        }
    ]
    citations = (
        []
        if abstained
        else [{"number": 1, "chunk_id": chunk.chunk_id, "filename": chunk.filename, "page": 1}]
    )
    return {
        "version": 1,
        "question_id": qid,
        "dataset": dataset.name,
        "dataset_sha256": dataset.sha256,
        "config": CONFIG,
        "status": "ok",
        "retrieved": retrieved,
        "answer": {
            "abstained": abstained,
            "abstain_reason": "r" if abstained else None,
            "text": text,
            "model": "scripted",
            "input_tokens": 1,
            "output_tokens": 1,
            "citations": citations,
        },
    }


def score(
    dataset: EvalDataset,
    tmp_path: Path,
    records: list[dict[str, Any]],
    unsupported: dict[str, Any] | None = UNSUPPORTED,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold = load_gold(gold_file(tmp_path, dataset, unsupported), dataset)
    return score_run(records, dataset, gold, overrides)


def row(report: dict[str, Any], qid: str) -> dict[str, Any]:
    return next(r for r in report["questions"] if r["id"] == qid)


# ---------------------------------------------------------------- comportements


@pytest.mark.parametrize(
    "absence",
    [
        "The documents do not specify how many trees to use.",
        "The course doesn’t say how many trees to use.",  # noqa: RUF001 (apostrophe typographique)
        "Le nombre d'arbres n'est pas précisé dans les documents.",
        "Les documents ne le précisent pas.",
    ],
)
def test_supported_answer_and_acknowledged_absence_is_success(
    tmp_path: Path, dataset: EvalDataset, absence: str
) -> None:
    report = score(dataset, tmp_path, [record(dataset, "p1", f"{SUPPORTED} {absence}")])

    p1 = row(report, "p1")
    assert p1["correctness"] == "correct"
    assert p1["partial"]["behavior"] == "complete"
    assert p1["partial"]["absence_acknowledged"] is True
    assert p1["partial"]["invented"] is False
    assert p1["outcome"] == "success"


def test_supported_answer_without_acknowledging_absence(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    report = score(dataset, tmp_path, [record(dataset, "p1", SUPPORTED)])

    p1 = row(report, "p1")
    assert p1["correctness"] == "correct"  # la partie répondable reste correcte
    assert p1["partial"]["behavior"] == "absence_not_acknowledged"
    assert p1["outcome"] == "absence_not_acknowledged"


def test_omission_accepted_when_expected_behavior_is_omit_ok(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    unsupported = UNSUPPORTED | {"expected_behavior": "omit_ok"}

    report = score(dataset, tmp_path, [record(dataset, "p1", SUPPORTED)], unsupported)

    assert row(report, "p1")["partial"]["behavior"] == "complete"
    assert row(report, "p1")["outcome"] == "success"


def test_abstaining_on_everything_omits_the_supported_answer(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    rec = record(dataset, "p1", "The documents do not say how many trees.", abstained=True)

    p1 = row(score(dataset, tmp_path, [rec]), "p1")

    assert p1["partial"]["behavior"] == "abstained_entirely"
    assert p1["outcome"] == "unjustified_abstention"  # issue existante, inchangée
    assert p1["correctness"] == "abstained"


def test_supported_part_incomplete_but_absence_acknowledged(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    rec = record(dataset, "p1", "Trees differ [1]. The documents do not specify the tree count.")

    p1 = row(score(dataset, tmp_path, [rec]), "p1")

    assert p1["partial"]["behavior"] == "supported_part_incomplete"
    assert p1["outcome"] == "generation_failure"


@pytest.mark.parametrize(
    ("invention", "forbidden", "numbers"),
    [
        ("Use 500 trees.", [["500 trees", "500 arbres"]], ["500"]),  # fait interdit et nombre
        ("About 300 of them are typical.", [], ["300"]),  # nombre absent des passages
    ],
)
def test_invented_unsupported_part_is_flagged(
    tmp_path: Path,
    dataset: EvalDataset,
    invention: str,
    forbidden: list[list[str]],
    numbers: list[str],
) -> None:
    # Même accompagnée d'un marqueur d'absence, une invention l'emporte.
    text = f"{SUPPORTED} The documents do not specify it, but {invention}"

    p1 = row(score(dataset, tmp_path, [record(dataset, "p1", text)]), "p1")

    assert p1["partial"]["forbidden_facts_found"] == forbidden
    assert p1["partial"]["unsupported_numbers"] == numbers
    assert p1["partial"]["behavior"] == "invented_unsupported_part"
    assert p1["outcome"] == "invented_unsupported_part"


def test_numbers_from_passages_question_or_citations_are_not_inventions(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    text = f"{SUPPORTED} Bagging keeps 63,2 % unique samples [1]. Tree count: not specified."

    p1 = row(score(dataset, tmp_path, [record(dataset, "p1", text)]), "p1")

    assert p1["partial"]["unsupported_numbers"] == []  # « 63,2 » = « 63.2 », « [1] » ignoré
    assert p1["partial"]["behavior"] == "complete"


def test_paraphrased_invention_is_not_detected(tmp_path: Path, dataset: EvalDataset) -> None:
    """Limite connue : une invention sans fait interdit ni nombre passe inaperçue."""
    text = f"{SUPPORTED} You should use several hundred trees."

    p1 = row(score(dataset, tmp_path, [record(dataset, "p1", text)]), "p1")

    assert p1["partial"]["invented"] is False
    assert p1["partial"]["behavior"] == "absence_not_acknowledged"


def test_number_check_can_be_disabled(tmp_path: Path, dataset: EvalDataset) -> None:
    text = f"{SUPPORTED} About 300 are typical, but the documents do not specify it."
    unsupported = UNSUPPORTED | {"check_numbers": False}

    p1 = row(score(dataset, tmp_path, [record(dataset, "p1", text)], unsupported), "p1")

    assert p1["partial"]["unsupported_numbers"] == []
    assert p1["partial"]["behavior"] == "complete"


def test_human_review_overrides_partial_heuristics(tmp_path: Path, dataset: EvalDataset) -> None:
    overrides = {"p1": {"absence_acknowledged": True, "invented": False}}

    p1 = row(
        score(dataset, tmp_path, [record(dataset, "p1", SUPPORTED)], overrides=overrides), "p1"
    )

    assert p1["partial"]["behavior"] == "complete"
    assert p1["outcome"] == "success"


def test_override_partial_keys_must_be_boolean(tmp_path: Path, dataset: EvalDataset) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"p1": {"invented": "yes"}}), encoding="utf-8")

    with pytest.raises(DatasetError, match="booléen"):
        load_overrides(path, dataset)


@pytest.mark.parametrize(
    ("text", "abstained", "outcome"),
    [
        ("The documents do not say who invented bagging.", True, "correct_abstention"),
        ("Bagging was invented by Leo Breiman in 1996.", False, "unsupported_answer"),
    ],
)
def test_fully_unanswerable_questions_are_unchanged(
    tmp_path: Path, dataset: EvalDataset, text: str, abstained: bool, outcome: str
) -> None:
    report = score(dataset, tmp_path, [record(dataset, "u1", text, abstained=abstained)])

    u1 = row(report, "u1")
    assert u1["outcome"] == outcome
    assert "partial" not in u1
    assert report["unanswerable"]["correct_abstentions"] == int(abstained)


# ---------------------------------------------------------------- rapport et fiche


def test_report_summary_and_review_sheet_show_partial_block(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    records = [
        record(dataset, "q1", "63.2% [1]."),
        record(dataset, "p1", f"{SUPPORTED} Use 500 trees."),
    ]
    gold = load_gold(gold_file(tmp_path, dataset), dataset)

    report = score_run(records, dataset, gold)

    assert report["partial_questions"] == {
        "n": 1,
        "behaviors": {"invented_unsupported_part": 1},
        "absence_acknowledged": 0,
        "invented": ["p1"],
    }
    assert "Questions en partie répondables (n=1)" in format_generation_summary(report)
    assert "**Partie sans réponse** : number of trees" in format_review_sheet(report, records, gold)


# ---------------------------------------------------------------- annexe : validation


def test_default_absence_markers_use_star_only_as_suffix() -> None:
    """Le « * » n'est reconnu qu'en fin de variante : ailleurs, il serait pris littéralement."""
    assert all("*" not in marker.rstrip("*") for marker in DEFAULT_ABSENCE_MARKERS)


@pytest.mark.parametrize(
    ("unsupported", "message"),
    [
        ({"forbidden_facts": [["500 trees"]]}, "décrire la partie"),
        (UNSUPPORTED | {"expected_behavior": "abstain"}, "expected_behavior"),
        (UNSUPPORTED | {"absence_markers": []}, "absence_markers"),
        (UNSUPPORTED | {"forbidden_facts": [[]]}, "forbidden_facts"),
        (UNSUPPORTED | {"check_numbers": "yes"}, "check_numbers"),
        # un fait « interdit » présent dans les documents ne prouverait aucune invention
        (UNSUPPORTED | {"forbidden_facts": [["random subset"]]}, "figure dans les documents"),
    ],
)
def test_invalid_unsupported_field_is_rejected(
    tmp_path: Path, dataset: EvalDataset, unsupported: dict[str, Any], message: str
) -> None:
    with pytest.raises(DatasetError, match=message):
        load_gold(gold_file(tmp_path, dataset, unsupported), dataset)


# ---------------------------------------------------------------- compatibilité


def test_gold_without_unsupported_field_scores_as_before(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    records = [record(dataset, "q1", "63.2% [1]."), record(dataset, "p1", SUPPORTED)]

    report = score(dataset, tmp_path, records, unsupported=None)

    assert "partial_questions" not in report
    assert all("partial" not in r for r in report["questions"])
    assert row(report, "p1")["outcome"] == "success"  # ancien comportement : absence ignorée
    assert "Questions en partie répondables" not in format_generation_summary(report)


@pytest.mark.parametrize("name", ["real_v1", "real_v2"])
def test_real_gold_annexes_load(name: str) -> None:
    documents = ROOT / "evaluation" / "documents"
    if not all((documents / f"lecture{i}.pdf").is_file() for i in (1, 2, 3)):
        pytest.skip("PDF des jeux réels absents (evaluation/documents, non versionné)")
    from doc_assistant.ingestion.pdf_loader import PypdfLoader

    datasets = ROOT / "evaluation" / "datasets"
    dataset = load_dataset(datasets / f"{name}.json", loader=PypdfLoader())
    gold = load_gold(datasets / f"{name}_answers.json", dataset)

    partial = {q.id for q in dataset.answerable if q.category == "multi_part_partial"}
    assert set(gold) == {q.id for q in dataset.answerable}
    assert {qid for qid, g in gold.items() if g.unsupported is not None} == partial
