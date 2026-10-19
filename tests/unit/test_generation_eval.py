"""Évaluation de la génération : annexe, notation hors ligne, enregistrement reprenable.

Aucun appel réseau : générateurs scriptés, embeddings factices, index en mémoire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc_assistant.domain.errors import (
    LLMConfigurationError,
    LLMQuotaError,
    LLMResponseError,
)
from doc_assistant.evaluation.dataset import DatasetError, EvalDataset, load_dataset
from doc_assistant.evaluation.generation import (
    PROXY_WARNING,
    fact_found,
    format_generation_summary,
    format_review_sheet,
    latest_records,
    load_gold,
    load_overrides,
    normalize,
    read_records,
    record_answers,
    score_run,
)
from doc_assistant.evaluation.runner import index_dataset
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.retrieval.retriever import Retriever
from fakes import FakeEmbedder, InMemoryVectorStore, ScriptedAnswerGenerator

ROOT = Path(__file__).resolve().parents[2]
Q1 = "Which method selects features by zeroing coefficients?"
Q2 = "Quelle pénalité ajoute la régression ridge ?"
Q3 = "What are the points lying on the margin called?"
U1 = "What is the default value of C?"
DATASET = {
    "name": "mini_gen",
    "kind": "synthetic",
    "documents": [
        {
            "filename": "ml.pdf",
            "lang": "en",
            "pages": [
                "Lasso sets some coefficients exactly to zero, which performs feature selection.",
                "Ridge adds a squared L2 penalty to the loss function.",
            ],
        },
        {
            "filename": "svm.pdf",
            "lang": "en",
            "pages": ["Examples that lie exactly on the margin are called support vectors."],
        },
    ],
    "questions": [
        {
            "id": "q1",
            "lang": "en",
            "category": "paraphrase",
            "answerable": True,
            "question": Q1,
            "references": [
                {"filename": "ml.pdf", "page": 1, "quote": "sets some coefficients exactly to zero"}
            ],
        },
        {
            "id": "q2",
            "lang": "fr",
            "category": "multi_passage",
            "answerable": True,
            "question": Q2,
            "references": [{"filename": "ml.pdf", "page": 2, "quote": "squared L2 penalty"}],
        },
        {
            "id": "q3",
            "lang": "en",
            "category": "paraphrase",
            "answerable": True,
            "question": Q3,
            "references": [{"filename": "svm.pdf", "page": 1, "quote": "called support vectors"}],
        },
        {"id": "u1", "lang": "en", "answerable": False, "question": U1},
    ],
}
CONFIG = {"provider": "fake", "model": "scripted", "top_k": 2}


@pytest.fixture
def dataset(tmp_path: Path) -> EvalDataset:
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(DATASET, ensure_ascii=False), encoding="utf-8")
    return load_dataset(path)


def gold_file(tmp_path: Path, dataset: EvalDataset, **changes: Any) -> Path:
    answers: dict[str, Any] = {
        "q1": {
            "expected_answer": "Lasso.",
            "required_facts": [["lasso"], ["select*"]],
            "support": "strong",
            "flags": [],
        },
        "q2": {
            "expected_answer": "L2 penalty.",
            "required_facts": [["l2", "carre", "squared"]],
            "support": "strong",
            "flags": [],
        },
        "q3": {
            "expected_answer": "Support vectors.",
            "required_facts": [["support vector*", "vecteurs de support"]],
            "support": "weak",
            "flags": ["exemple de signalement"],
        },
    }
    answers.update(changes)
    data = {"dataset_sha256": dataset.sha256, "answers": {k: v for k, v in answers.items() if v}}
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def passage(dataset: EvalDataset, filename: str, page: int, rank: int = 1) -> dict[str, Any]:
    document = next(d for d in dataset.documents if d.filename == filename)
    chunk = next(c for c in RecursiveChunker(800, 100).split(document) if c.page_number == page)
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "filename": filename,
        "page": page,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "score": 0.9,
        "text": chunk.text,
    }


def record(
    dataset: EvalDataset,
    qid: str,
    retrieved: list[dict[str, Any]],
    text: str = "",
    cite: list[int] | None = None,
    *,
    abstained: bool = False,
    status: str = "ok",
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": 1,
        "question_id": qid,
        "dataset": dataset.name,
        "dataset_sha256": dataset.sha256,
        "config": CONFIG,
        "status": status,
        "retrieved": retrieved,
    }
    if status != "ok":
        return base | {"error": {"type": "LLMResponseError", "message": "x"}}
    citations = [
        {
            "number": n,
            "chunk_id": retrieved[n - 1]["chunk_id"],
            "filename": retrieved[n - 1]["filename"],
            "page": retrieved[n - 1]["page"],
        }
        for n in (cite or [])
    ]
    return base | {
        "answer": {
            "abstained": abstained,
            "abstain_reason": "r" if abstained else None,
            "text": text,
            "model": "scripted",
            "input_tokens": 1,
            "output_tokens": 1,
            "citations": citations,
        }
    }


def score(
    dataset: EvalDataset, tmp_path: Path, records: list[dict[str, Any]], **kw: Any
) -> dict[str, Any]:
    return score_run(records, dataset, load_gold(gold_file(tmp_path, dataset), dataset), **kw)


def row(report: dict[str, Any], qid: str) -> dict[str, Any]:
    return next(r for r in report["questions"] if r["id"] == qid)


# ---------------------------------------------------------------- normalisation


def test_normalize_and_fact_matching() -> None:
    assert normalize("  Zéro\n CARRÉ  w² ") == "zero carre w2"
    assert fact_found(["vecteurs de support"], normalize("Les Vecteurs de Support."))
    assert not fact_found(["lasso"], normalize("ridge"))


@pytest.mark.parametrize(
    ("variant", "text", "expected"),
    [
        ("zero", "pulls weights to zero.", True),
        ("zero", "keeps nonzero weights", False),  # mot entier : pas de correspondance partielle
        ("60", "60% for training", True),
        ("60", "about 600 points or 160 items", False),
        ("63.2", "63.25 percent", False),
        ("63.2", "≈ 63.2% unique", True),
        ("about 63", "about 63.25 percent", False),  # le nombre continue après le point
        ("about 63", "about 63% unique", True),
        ("60", "a ratio of 1.60", False),
        ("60", "split 60/40", True),
        ("support vector", "support vectors", False),  # sans * : pas de pluriel implicite
        ("support vector*", "support vectors", True),  # avec * : préfixe explicite
        ("polynom*", "a Polynomial kernel", True),
        ("w2", "a penalty λw2", True),  # une lettre grecque n'est pas une lettre latine collée
        ("n × n", "inverting a n × n matrix", True),  # noqa: RUF001 (signe réel du PDF)
        ("far from the training", "so far, the training data", False),
    ],
)
def test_whole_word_matching(variant: str, text: str, expected: bool) -> None:
    assert fact_found([variant], normalize(text)) is expected


def test_bare_star_variant_is_rejected(tmp_path: Path, dataset: EvalDataset) -> None:
    bad = {"expected_answer": "x", "required_facts": [["*"]], "support": "strong"}

    with pytest.raises(DatasetError, match="listes"):
        load_gold(gold_file(tmp_path, dataset, q1=bad), dataset)


# ---------------------------------------------------------------- annexe


def test_gold_loads_and_checks_facts_against_references(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    gold = load_gold(gold_file(tmp_path, dataset), dataset)

    assert set(gold) == {"q1", "q2", "q3"}
    assert gold["q3"].flags == ("exemple de signalement",)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"q3": None}, "manquantes"),
        (
            {"q9": {"expected_answer": "x", "required_facts": [["x"]], "support": "strong"}},
            "en trop",
        ),
        (
            {"q1": {"expected_answer": "x", "required_facts": [["ridge"]], "support": "strong"}},
            "aucun passage de référence",
        ),
        (
            {"q1": {"expected_answer": "x", "required_facts": [["lasso"]], "support": "weak"}},
            "flags",
        ),
        (
            {"q1": {"expected_answer": "x", "required_facts": [["lasso"]], "support": "high"}},
            "support",
        ),
        ({"q1": {"expected_answer": "x", "required_facts": [[]], "support": "strong"}}, "listes"),
        (
            {"q1": {"expected_answer": " ", "required_facts": [["lasso"]], "support": "strong"}},
            "vide",
        ),
    ],
)
def test_invalid_gold_is_rejected(
    tmp_path: Path, dataset: EvalDataset, changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(DatasetError, match=message):
        load_gold(gold_file(tmp_path, dataset, **changes), dataset)


def test_gold_bound_to_dataset_version(tmp_path: Path, dataset: EvalDataset) -> None:
    path = gold_file(tmp_path, dataset)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dataset_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DatasetError, match="autre version"):
        load_gold(path, dataset)


def test_real_v1_gold_annex_is_valid() -> None:
    documents = ROOT / "evaluation" / "documents"
    if not all((documents / f"lecture{i}.pdf").is_file() for i in (1, 2, 3)):
        pytest.skip("PDF de real_v1 absents (evaluation/documents, non versionné)")
    from doc_assistant.ingestion.pdf_loader import PypdfLoader

    dataset = load_dataset(ROOT / "evaluation" / "datasets" / "real_v1.json", loader=PypdfLoader())
    gold = load_gold(ROOT / "evaluation" / "datasets" / "real_v1_answers.json", dataset)

    assert set(gold) == {q.id for q in dataset.answerable}
    assert all(g.flags for g in gold.values() if g.support == "weak")


# ---------------------------------------------------------------- issues (tableau d'attribution)


def test_success_with_supported_citation(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(
        dataset, "q1", [passage(dataset, "ml.pdf", 1)], "Lasso performs feature selection [1].", [1]
    )

    r = row(score(dataset, tmp_path, [rec]), "q1")

    assert (r["correctness"], r["outcome"]) == ("correct", "success")
    assert r["facts_supported_by_citations"] == [True, True]
    assert r["citations"] == [("ml.pdf", 1)]


def test_correct_answer_citing_non_reference_passage_is_citation_failure(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    retrieved = [passage(dataset, "ml.pdf", 2), passage(dataset, "ml.pdf", 1, rank=2)]
    rec = record(dataset, "q1", retrieved, "Lasso does feature selection [1].", [1])

    report = score(dataset, tmp_path, [rec])

    assert row(report, "q1")["outcome"] == "citation_failure"
    assert report["citations"]["reference_precision"] == 0.0
    assert report["citations"]["reference_recall"] == 0.0


def test_partial_answer_is_generation_failure(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(dataset, "q1", [passage(dataset, "ml.pdf", 1)], "Lasso [1].", [1])

    r = row(score(dataset, tmp_path, [rec]), "q1")

    assert (r["correctness"], r["outcome"]) == ("partial", "generation_failure")


def test_wrong_answer_is_generation_failure(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(dataset, "q1", [passage(dataset, "ml.pdf", 1)], "Ridge [1].", [1])

    assert row(score(dataset, tmp_path, [rec]), "q1")["correctness"] == "incorrect"


def test_abstention_despite_reference_is_unjustified(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(dataset, "q3", [passage(dataset, "svm.pdf", 1)], abstained=True)

    report = score(dataset, tmp_path, [rec])

    assert row(report, "q3")["outcome"] == "unjustified_abstention"
    assert report["answerable"]["abstained_unjustified"] == 1


def test_abstention_after_retrieval_miss_is_justified(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(dataset, "q3", [passage(dataset, "ml.pdf", 1)], abstained=True)

    report = score(dataset, tmp_path, [rec])

    assert row(report, "q3")["outcome"] == "retrieval_failure_abstained"
    assert report["answerable"]["abstained_justified"] == 1


def test_answer_after_retrieval_miss_is_flagged(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(dataset, "q3", [passage(dataset, "ml.pdf", 1)], "Support vectors [1].", [1])

    r = row(score(dataset, tmp_path, [rec]), "q3")

    assert r["outcome"] == "retrieval_failure_answered"
    assert r["facts_supported_by_citations"] == [False]  # le passage cité ne contient pas le fait


@pytest.mark.parametrize(
    ("abstained", "outcome"), [(True, "correct_abstention"), (False, "unsupported_answer")]
)
def test_unanswerable_questions(
    tmp_path: Path, dataset: EvalDataset, abstained: bool, outcome: str
) -> None:
    rec = record(
        dataset,
        "u1",
        [passage(dataset, "svm.pdf", 1)],
        "" if abstained else "C = 1 [1].",
        None if abstained else [1],
        abstained=abstained,
    )

    report = score(dataset, tmp_path, [rec])

    assert row(report, "u1")["outcome"] == outcome
    assert report["unanswerable"]["correct_abstentions"] == int(abstained)


def test_missing_and_errored_questions_are_reported(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(dataset, "q2", [passage(dataset, "ml.pdf", 2)], status="error")

    report = score(dataset, tmp_path, [rec])

    assert report["coverage"]["errors"] == ["q2"]
    assert report["coverage"]["missing"] == ["q1", "q3", "u1"]
    assert report["coverage"]["scored"] == 0


# ---------------------------------------------------------------- citations, langues, relecture


def test_invalid_citations_are_counted_and_ignored(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(dataset, "q1", [passage(dataset, "ml.pdf", 1)], "Lasso selection [1].", [1])
    rec["answer"]["citations"][0]["page"] = 99

    report = score(dataset, tmp_path, [rec])

    assert report["citations"]["invalid"] == 1
    assert row(report, "q1")["citations"] == []


def test_cross_lingual_fact_support(tmp_path: Path, dataset: EvalDataset) -> None:
    rec = record(
        dataset, "q2", [passage(dataset, "ml.pdf", 2)], "Une pénalité au carré (L2) [1].", [1]
    )

    report = score(dataset, tmp_path, [rec])

    assert row(report, "q2")["correctness"] == "correct"
    assert report["citations"]["fact_support"] == 1.0
    assert report["answerable"]["by_language"]["fr"]["correct"] == 1
    assert "q2" in report["multi_passage"]


def test_latest_record_wins(tmp_path: Path, dataset: EvalDataset) -> None:
    first = record(dataset, "q1", [passage(dataset, "ml.pdf", 1)], status="error")
    second = record(dataset, "q1", [passage(dataset, "ml.pdf", 1)], "Lasso selection [1].", [1])

    assert latest_records([first, second])["q1"]["status"] == "ok"
    assert row(score(dataset, tmp_path, [first, second]), "q1")["outcome"] == "success"


def test_human_overrides(tmp_path: Path, dataset: EvalDataset) -> None:
    partial = record(dataset, "q1", [passage(dataset, "ml.pdf", 1)], "Lasso [1].", [1])
    good = record(dataset, "q3", [passage(dataset, "svm.pdf", 1)], "Support vectors [1].", [1])
    overrides = {
        "q1": {"correctness": "correct", "note": "paraphrase"},
        "q3": {"citation_support": False},
    }

    report = score(dataset, tmp_path, [partial, good], overrides=overrides)

    q1 = row(report, "q1")
    assert (q1["correctness_auto"], q1["correctness"], q1["outcome"]) == (
        "partial",
        "correct",
        "success",
    )
    assert row(report, "q3")["outcome"] == "citation_failure"


def test_overrides_file_is_validated(tmp_path: Path, dataset: EvalDataset) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"q1": {"correctness": "maybe"}}), encoding="utf-8")
    with pytest.raises(DatasetError, match="correctness"):
        load_overrides(path, dataset)
    path.write_text(json.dumps({"zz": {}}), encoding="utf-8")
    with pytest.raises(DatasetError, match="inconnus"):
        load_overrides(path, dataset)


def test_scoring_is_deterministic_and_rejects_other_dataset(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    records = [record(dataset, "q1", [passage(dataset, "ml.pdf", 1)], "Lasso selection [1].", [1])]
    first, second = score(dataset, tmp_path, records), score(dataset, tmp_path, records)
    first.pop("generated_at"), second.pop("generated_at")
    assert first == second

    records[0]["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="version du jeu"):
        score(dataset, tmp_path, records)


def test_summary_and_review_sheet_state_proxy_warning(tmp_path: Path, dataset: EvalDataset) -> None:
    records = [record(dataset, "q3", [passage(dataset, "svm.pdf", 1)], "Support vectors [1].", [1])]
    gold = load_gold(gold_file(tmp_path, dataset), dataset)
    report = score_run(records, dataset, gold)

    assert PROXY_WARNING in format_generation_summary(report)
    sheet = format_review_sheet(report, records, gold)
    assert PROXY_WARNING in sheet
    assert "svm.pdf, page 1" in sheet
    assert "exemple de signalement" in sheet
    assert "**Relecture**" in sheet


# ---------------------------------------------------------------- enregistrement reprenable


def retriever_for(dataset: EvalDataset) -> Retriever:
    embedder = FakeEmbedder()
    return index_dataset(dataset, embedder, RecursiveChunker(800, 100), InMemoryVectorStore())


def run(dataset: EvalDataset, out: Path, generator: ScriptedAnswerGenerator, **kw: Any) -> Any:
    return record_answers(
        dataset, retriever_for(dataset), generator, out, run_config=CONFIG, top_k=2, **kw
    )


def test_records_one_line_per_question(tmp_path: Path, dataset: EvalDataset) -> None:
    out = tmp_path / "run.jsonl"
    generator = ScriptedAnswerGenerator({Q3: "abstain"})

    summary = run(dataset, out, generator)

    records = read_records(out)
    assert [r["question_id"] for r in records] == ["q1", "q2", "q3", "u1"]
    assert (summary.recorded, summary.api_calls, summary.errors) == (4, 4, 0)
    assert records[2]["answer"]["abstained"] is True
    assert all(len(r["retrieved"]) == 2 and r["retrieved"][0]["text"] for r in records)


def test_saved_after_each_question_even_if_interrupted(
    tmp_path: Path, dataset: EvalDataset
) -> None:
    out = tmp_path / "run.jsonl"

    with pytest.raises(RuntimeError):
        run(dataset, out, ScriptedAnswerGenerator({Q3: RuntimeError("crash")}))

    assert [r["question_id"] for r in read_records(out)] == ["q1", "q2"]


def test_quota_stops_cleanly_and_resume_completes(tmp_path: Path, dataset: EvalDataset) -> None:
    out = tmp_path / "run.jsonl"
    first = ScriptedAnswerGenerator({Q2: LLMQuotaError("429")})

    summary = run(dataset, out, first)

    assert summary.stopped_on_quota
    assert summary.remaining == ["q2", "q3", "u1"]
    assert first.calls == [Q1, Q2]  # pas de nouvelle tentative du 429
    assert [r["question_id"] for r in read_records(out)] == ["q1"]

    second = ScriptedAnswerGenerator({})
    resumed = run(dataset, out, second)

    assert second.calls == [Q2, Q3, U1]  # q1 n'est pas redemandée
    assert (resumed.skipped, resumed.recorded) == (1, 3)
    assert set(latest_records(read_records(out))) == {"q1", "q2", "q3", "u1"}


def test_errors_are_recorded_and_retried_on_resume(tmp_path: Path, dataset: EvalDataset) -> None:
    out = tmp_path / "run.jsonl"
    summary = run(dataset, out, ScriptedAnswerGenerator({Q1: LLMResponseError("tronquée")}))

    assert summary.errors == 1
    assert read_records(out)[0]["status"] == "error"

    retry = ScriptedAnswerGenerator({})
    run(dataset, out, retry)

    assert retry.calls == [Q1]
    assert latest_records(read_records(out))["q1"]["status"] == "ok"


def test_configuration_error_aborts(tmp_path: Path, dataset: EvalDataset) -> None:
    with pytest.raises(LLMConfigurationError):
        run(
            dataset,
            tmp_path / "run.jsonl",
            ScriptedAnswerGenerator({Q1: LLMConfigurationError("clé")}),
        )


def test_resume_refuses_another_configuration(tmp_path: Path, dataset: EvalDataset) -> None:
    out = tmp_path / "run.jsonl"
    run(dataset, out, ScriptedAnswerGenerator({}))

    with pytest.raises(ValueError, match="autre configuration"):
        record_answers(
            dataset,
            retriever_for(dataset),
            ScriptedAnswerGenerator({}),
            out,
            run_config=CONFIG | {"model": "other"},
        )


def test_delay_only_between_api_calls(tmp_path: Path, dataset: EvalDataset) -> None:
    pauses: list[float] = []

    run(
        dataset,
        tmp_path / "run.jsonl",
        ScriptedAnswerGenerator({}),
        delay_s=4.0,
        sleep=pauses.append,
    )

    assert pauses == [4.0, 4.0, 4.0]  # 4 appels -> 3 pauses, aucune avant le premier


def test_no_passages_means_no_call_and_no_pause(tmp_path: Path, dataset: EvalDataset) -> None:
    empty = Retriever(FakeEmbedder(), InMemoryVectorStore())
    generator = ScriptedAnswerGenerator({})
    pauses: list[float] = []

    summary = record_answers(
        dataset,
        empty,
        generator,
        tmp_path / "run.jsonl",
        run_config=CONFIG,
        delay_s=4.0,
        sleep=pauses.append,
    )

    assert generator.calls == [] and pauses == [] and summary.api_calls == 0
    assert all(r["answer"]["abstained"] for r in read_records(tmp_path / "run.jsonl"))


def test_shared_index_builder_refuses_non_empty_store(dataset: EvalDataset) -> None:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    index_dataset(dataset, embedder, RecursiveChunker(800, 100), store)

    with pytest.raises(ValueError, match="index vide"):
        index_dataset(dataset, embedder, RecursiveChunker(800, 100), store)


@pytest.mark.parametrize(
    "content",
    ["{pas du json", json.dumps({"answers": []}), json.dumps({"answers": "x"})],
)
def test_malformed_gold_files(tmp_path: Path, dataset: EvalDataset, content: str) -> None:
    path = tmp_path / "gold.json"
    if content.startswith('{"answers"'):
        data = json.loads(content) | {"dataset_sha256": dataset.sha256}
        content = json.dumps(data)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DatasetError):
        load_gold(path, dataset)


def test_gold_flags_must_be_strings(tmp_path: Path, dataset: EvalDataset) -> None:
    bad = {"expected_answer": "x", "required_facts": [["lasso"]], "support": "strong", "flags": "x"}

    with pytest.raises(DatasetError, match="flags"):
        load_gold(gold_file(tmp_path, dataset, q1=bad), dataset)


def test_override_citation_support_must_be_boolean(tmp_path: Path, dataset: EvalDataset) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"q1": {"citation_support": "oui"}}), encoding="utf-8")

    with pytest.raises(DatasetError, match="booléen"):
        load_overrides(path, dataset)
