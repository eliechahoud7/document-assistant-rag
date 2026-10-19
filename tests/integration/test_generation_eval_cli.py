"""generate-answers / score-answers de bout en bout : générateur scripté, aucun appel réseau."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import pytest

from doc_assistant import bootstrap, cli
from doc_assistant.config import Settings
from doc_assistant.domain.errors import LLMQuotaError
from doc_assistant.evaluation.dataset import load_dataset
from fakes import FakeEmbedder, ScriptedAnswerGenerator

QUESTION_1 = "Which method zeroes coefficients?"
QUESTION_2 = "What lies on the margin?"
DATASET: dict[str, Any] = {
    "name": "cli_gen",
    "kind": "synthetic",
    "documents": [
        {
            "filename": "ml.pdf",
            "lang": "en",
            "pages": [
                "Lasso sets some coefficients exactly to zero and selects features.",
                "Examples exactly on the margin are called support vectors.",
            ],
        },
    ],
    "questions": [
        {
            "id": "q1",
            "lang": "en",
            "category": "paraphrase",
            "answerable": True,
            "question": QUESTION_1,
            "references": [
                {"filename": "ml.pdf", "page": 1, "quote": "coefficients exactly to zero"}
            ],
        },
        {
            "id": "q2",
            "lang": "en",
            "category": "paraphrase",
            "answerable": True,
            "question": QUESTION_2,
            "references": [{"filename": "ml.pdf", "page": 2, "quote": "called support vectors"}],
        },
        {"id": "u1", "lang": "en", "answerable": False, "question": "Who invented SVMs?"},
    ],
}


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(DATASET, ensure_ascii=False), encoding="utf-8")
    sha = load_dataset(dataset).sha256
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "dataset_sha256": sha,
                "answers": {
                    "q1": {
                        "expected_answer": "Lasso.",
                        "required_facts": [["lasso"]],
                        "support": "strong",
                    },
                    "q2": {
                        "expected_answer": "Support vectors.",
                        "required_facts": [["support vector*"]],
                        "support": "strong",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    # Les index de l'utilisateur ne doivent jamais être utilisés : dossier de données jetable.
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "user-data"))
    monkeypatch.setattr(bootstrap, "build_embedder", lambda _settings: FakeEmbedder())
    return {
        "dataset": dataset,
        "gold": gold,
        "run": tmp_path / "out" / "run.jsonl",
        "root": tmp_path,
    }


def use(monkeypatch: pytest.MonkeyPatch, generator: ScriptedAnswerGenerator) -> None:
    monkeypatch.setattr(bootstrap, "build_generator", lambda _settings: generator)


def generate(files: dict[str, Path]) -> int:
    return cli.main(
        [
            "generate-answers",
            str(files["dataset"]),
            "--live",
            "--delay",
            "0",
            "-o",
            str(files["run"]),
        ]
    )


def test_requires_explicit_live_flag(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def must_not_build(_settings: Settings) -> NoReturn:
        raise AssertionError("aucun générateur ne doit être construit sans --live")

    monkeypatch.setattr(bootstrap, "build_generator", must_not_build)

    assert cli.main(["generate-answers", str(files["dataset"]), "-o", str(files["run"])]) == 2
    assert "--live" in capsys.readouterr().err
    assert not files["run"].exists()


def test_provider_none_fails_before_loading_anything(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DOCASSIST_LLM_PROVIDER", raising=False)

    def must_not_load(_settings: Settings) -> NoReturn:
        raise AssertionError("le modèle d'embeddings ne doit pas être chargé")

    monkeypatch.setattr(bootstrap, "build_embedder", must_not_load)

    assert generate(files) == 2
    assert "DOCASSIST_LLM_PROVIDER=gemini" in capsys.readouterr().err


def test_record_then_score_offline(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use(
        monkeypatch,
        ScriptedAnswerGenerator({QUESTION_1: "Lasso [1].", "Who invented SVMs?": "abstain"}),
    )

    assert generate(files) == 0
    assert len(files["run"].read_text(encoding="utf-8").splitlines()) == 3

    report_path, review_path = files["root"] / "scores.json", files["root"] / "review.md"
    code = cli.main(
        [
            "score-answers",
            str(files["run"]),
            "--dataset",
            str(files["dataset"]),
            "--gold",
            str(files["gold"]),
            "-o",
            str(report_path),
            "--review",
            str(review_path),
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "indicateur approximatif" in out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["coverage"]["scored"] == 3
    assert report["unanswerable"]["correct_abstentions"] == 1
    assert report["run_config"][0]["model"] == "scripted"
    assert "## q1" in review_path.read_text(encoding="utf-8")
    assert not (files["root"] / "user-data").exists()  # aucun index utilisateur créé


def test_quota_exit_code_and_resume(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use(monkeypatch, ScriptedAnswerGenerator({QUESTION_2: LLMQuotaError("429")}))

    assert generate(files) == cli.QUOTA_EXIT_CODE
    assert "Relancez la même commande" in capsys.readouterr().err

    second = ScriptedAnswerGenerator({})
    use(monkeypatch, second)

    assert generate(files) == 0
    assert second.calls == [QUESTION_2, "Who invented SVMs?"]


def test_score_requires_records(files: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    files["run"].parent.mkdir(parents=True)
    files["run"].write_text("", encoding="utf-8")

    code = cli.main(
        [
            "score-answers",
            str(files["run"]),
            "--dataset",
            str(files["dataset"]),
            "--gold",
            str(files["gold"]),
        ]
    )

    assert code == 2
    assert "aucun enregistrement" in capsys.readouterr().err


def test_context_neighbors_recorded_in_config_and_not_mixed_with_baseline(
    files: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use(monkeypatch, ScriptedAnswerGenerator({}))
    monkeypatch.setenv("DOCASSIST_CONTEXT_NEIGHBORS", "0")  # configuration du run de référence
    assert generate(files) == 0
    baseline = [json.loads(line) for line in files["run"].read_text(encoding="utf-8").splitlines()]
    assert all("context_neighbors" not in r["config"] for r in baseline)  # 0 : config inchangée

    monkeypatch.setenv("DOCASSIST_CONTEXT_NEIGHBORS", "1")
    assert generate(files) == 2  # même fichier, autre configuration : refus
    assert "autre configuration" in capsys.readouterr().err

    other = files["root"] / "out" / "run_neighbors.jsonl"
    code = cli.main(
        ["generate-answers", str(files["dataset"]), "--live", "--delay", "0", "-o", str(other)]
    )
    assert code == 0
    records = [json.loads(line) for line in other.read_text(encoding="utf-8").splitlines()]
    assert all(r["config"]["context_neighbors"] == 1 for r in records)
