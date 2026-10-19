"""CLI de bout en bout : vrai ChromaDB temporaire, embeddings factices."""

import json
import runpy
from collections.abc import Callable
from pathlib import Path

import pytest

from doc_assistant import bootstrap, cli
from doc_assistant.config import Settings
from fakes import FakeEmbedder

MakePdf = Callable[..., Path]
SYNTHETIC_V1 = Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "synthetic_v1.json"


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Données dans un dossier temporaire et FakeEmbedder à la place d'E5."""
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(bootstrap, "build_embedder", lambda _settings: FakeEmbedder())
    return tmp_path


def test_index_then_search(
    isolated: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    make_pdf([["Les chats dorment seize heures par jour."], []], name="chats.pdf")
    make_pdf([["La bourse de Paris a clôturé en hausse."]], name="bourse.pdf")
    pdf_dir = next(isolated.glob("*.pdf")).parent

    assert cli.main(["index", str(pdf_dir)]) == 0
    out = capsys.readouterr().out
    assert "INDEXÉ  chats.pdf : 2 pages, 1 chunks" in out
    assert "pages sans texte : [2]" in out

    assert cli.main(["search", "combien dorment les chats", "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "chats.pdf, page 1" in out
    assert "bourse.pdf" not in out


def test_reindexing_is_skipped_unless_forced(
    isolated: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    path = make_pdf([["Contenu."]], name="a.pdf")
    cli.main(["index", str(path)])
    capsys.readouterr()

    cli.main(["index", str(path)])
    assert "IGNORÉ  a.pdf" in capsys.readouterr().out

    cli.main(["index", "--force", str(path)])
    assert "INDEXÉ  a.pdf" in capsys.readouterr().out


def test_invalid_pdf_is_reported_and_others_are_indexed(
    isolated: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    good = make_pdf([["Texte."]], name="bon.pdf")
    bad = isolated / "mauvais.pdf"
    bad.write_bytes(b"pas un pdf")

    assert cli.main(["index", str(bad), str(good)]) == 1
    captured = capsys.readouterr()
    assert "ÉCHEC   mauvais.pdf" in captured.err
    assert "INDEXÉ  bon.pdf" in captured.out


def test_search_on_empty_index(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["search", "question"]) == 1
    assert "Aucun passage indexé" in capsys.readouterr().out


def test_empty_question_is_an_error(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["search", "   "]) == 2
    assert "vide" in capsys.readouterr().err


def test_no_pdf_found(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = isolated / "vide"
    empty.mkdir()

    assert cli.main(["index", str(empty)]) == 1
    assert "Aucun PDF" in capsys.readouterr().err


def test_long_excerpts_are_truncated(
    isolated: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    make_pdf([["mot " * 150]], name="long.pdf")
    cli.main(["index", str(isolated / "long.pdf")])
    capsys.readouterr()

    cli.main(["search", "mot", "-k", "1"])

    assert "…" in capsys.readouterr().out


def test_missing_model_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCASSIST_MODELS_DIR", str(tmp_path / "aucun-modele"))

    assert cli.main(["search", "question"]) == 2
    assert "download-model" in capsys.readouterr().err


def test_download_command_uses_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    received: list[Settings] = []

    def fake_download(settings: Settings) -> Path:
        received.append(settings)
        return settings.embedding_model_dir

    monkeypatch.setattr(cli, "download_embedding_model", fake_download)

    assert cli.main(["download-model"]) == 0
    assert received[0].embedding_model == "intfloat/multilingual-e5-small"
    assert "Modèle prêt" in capsys.readouterr().out


def test_evaluate_uses_a_temporary_index_and_writes_report(
    isolated: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    # Un index utilisateur existe déjà : l'évaluation ne doit ni le lire ni le modifier.
    cli.main(["index", str(make_pdf([["Document personnel."]], name="perso.pdf"))])
    user_index = isolated / "data" / "chroma"
    before = {p: p.stat().st_mtime_ns for p in user_index.rglob("*") if p.is_file()}
    capsys.readouterr()
    output = isolated / "results" / "report.json"

    code = cli.main(["evaluate", str(SYNTHETIC_V1), "--output", str(output), "-k", "3"])

    assert code == 0
    out = capsys.readouterr().out
    assert "SYNTHÉTIQUE" in out
    assert "Rapport écrit" in out
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset"]["name"] == "synthetic_v1"
    assert report["config"]["top_k"] == 3
    assert report["config"]["embedding_model"] == "fake-hash"
    after = {p: p.stat().st_mtime_ns for p in user_index.rglob("*") if p.is_file()}
    assert after == before


def test_evaluate_reports_invalid_dataset(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = isolated / "bad.json"
    bad.write_text('{"name": "x", "kind": "inconnu"}', encoding="utf-8")

    assert cli.main(["evaluate", str(bad)]) == 2
    assert "kind" in capsys.readouterr().err


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["doc_assistant", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("doc_assistant", run_name="__main__")

    assert exit_info.value.code == 0
    assert "index" in capsys.readouterr().out
