"""Non-régression : E5 réel sur synthetic_v1 comparé au rapport de référence versionné.

Lancer avec : ``python -m pytest -m slow``. Échoue si un changement de code modifie les
rangs ou les métriques ; dans ce cas, analyser l'écart avant de régénérer la référence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc_assistant import cli
from doc_assistant.config import Settings

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation" / "datasets" / "synthetic_v1.json"
BASELINE = ROOT / "evaluation" / "results" / "synthetic_v1__e5-small__cs800-co100.json"


@pytest.fixture
def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    settings = Settings(models_dir=ROOT / "models")
    if not (settings.embedding_model_dir / "model.safetensors").is_file():
        pytest.skip("modèle E5 absent : python -m doc_assistant download-model")
    monkeypatch.setenv("DOCASSIST_MODELS_DIR", str(ROOT / "models"))
    output = tmp_path / "report.json"

    assert cli.main(["evaluate", str(DATASET), "--output", str(output)]) == 0
    result: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    return result


def test_matches_versioned_baseline(report: dict[str, Any]) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert report["dataset"]["sha256"] == baseline["dataset"]["sha256"], "jeu modifié"
    assert report["config"] == baseline["config"]
    assert report["retrieval"] == baseline["retrieval"]
    assert [q["reference_ranks"] for q in report["questions"]] == [
        q["reference_ranks"] for q in baseline["questions"]
    ]
    for got, expected in zip(report["questions"], baseline["questions"], strict=True):
        assert got["top1_score"] == pytest.approx(expected["top1_score"], abs=1e-4)
    assert report["absence_detection"]["auroc"] == pytest.approx(
        baseline["absence_detection"]["auroc"], abs=1e-3
    )
