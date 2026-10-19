"""Commande ask : vrai ChromaDB temporaire, embeddings et LLM factices, aucun appel réseau."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest
from google import genai
from google.genai import errors

from doc_assistant import bootstrap, cli
from doc_assistant.config import Settings
from doc_assistant.domain.models import Answer
from fakes import FakeAnswerGenerator, FakeEmbedder

MakePdf = Callable[..., Path]
SECRET = "AIza-TEST-SECRET-never-printed"


@pytest.fixture
def courses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_pdf: MakePdf) -> Path:
    """Index de cours séparé (DOCASSIST_DATA_DIR) contenant deux PDF de test."""
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "courses"))
    monkeypatch.setattr(bootstrap, "build_embedder", lambda _settings: FakeEmbedder())
    make_pdf([["Les vecteurs de support sont sur la marge."]], name="svm.pdf")
    make_pdf([["La régression ridge ajoute une pénalité."], ["Page deux."]], name="ridge.pdf")
    assert cli.main(["index", str(tmp_path)]) == 0
    return tmp_path


def use_generator(monkeypatch: pytest.MonkeyPatch, generator: FakeAnswerGenerator) -> None:
    monkeypatch.setattr(bootstrap, "build_generator", lambda _settings: generator)


def test_ask_prints_answer_and_real_sources(
    courses: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Le message affiche le fournisseur configuré : on le fixe pour ne pas dépendre du shell
    # (DOCASSIST_LLM_PROVIDER=gemini peut y rester après une session ask/generate-answers).
    monkeypatch.delenv("DOCASSIST_LLM_PROVIDER", raising=False)
    generator = FakeAnswerGenerator()
    use_generator(monkeypatch, generator)
    capsys.readouterr()

    assert cli.main(["ask", "vecteurs de support marge", "-k", "2"]) == 0

    out = capsys.readouterr().out
    sent = generator.calls[0][1]
    first = sent[0].chunk
    assert f"Index : {courses / 'courses' / 'chroma'}" in out
    # Par défaut (context_neighbors=1), les voisins des 2 passages retrouvés sont aussi envoyés.
    assert len(sent) >= 2
    assert f"Envoi de {len(sent)} passages à none (fake-llm)" in out
    assert f"Réponse fondée sur {first.filename} [1]." in out
    assert "Sources :" in out
    assert f"[1] {first.filename}, page {first.page_number}" in out


def test_ask_abstention(
    courses: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_generator(monkeypatch, FakeAnswerGenerator(abstain=True))
    capsys.readouterr()

    assert cli.main(["ask", "Quelle est la capitale du Pérou ?"]) == 0

    out = capsys.readouterr().out
    assert "Pas de réponse trouvée dans les documents." in out
    assert "Sources :" not in out


def test_show_passages_lists_what_is_sent(
    courses: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_generator(monkeypatch, FakeAnswerGenerator())
    capsys.readouterr()

    cli.main(["ask", "ridge", "-k", "3", "--show-passages"])

    out = capsys.readouterr().out
    assert out.count("score=") == 3


def test_empty_index_does_not_call_the_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "vide"))
    monkeypatch.setattr(bootstrap, "build_embedder", lambda _settings: FakeEmbedder())
    generator = FakeAnswerGenerator()
    use_generator(monkeypatch, generator)

    assert cli.main(["ask", "question"]) == 1
    assert "Aucun passage indexé" in capsys.readouterr().out
    assert generator.calls == []


def test_provider_none_fails_clearly_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("DOCASSIST_LLM_PROVIDER", raising=False)

    def must_not_load(_settings: Settings) -> NoReturn:
        raise AssertionError("le modèle d'embeddings ne doit pas être chargé")

    monkeypatch.setattr(bootstrap, "build_embedder", must_not_load)

    assert cli.main(["ask", "question"]) == 2
    assert "DOCASSIST_LLM_PROVIDER=gemini" in capsys.readouterr().err


def test_missing_key_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCASSIST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCASSIST_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert cli.main(["ask", "question"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_quota_error_through_real_adapter_never_prints_the_key(
    courses: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class QuotaModels:
        def generate_content(self, **_kwargs: Any) -> NoReturn:
            raise errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})

    class QuotaClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.models = QuotaModels()

    monkeypatch.setenv("DOCASSIST_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setattr(genai, "Client", QuotaClient)
    capsys.readouterr()

    assert cli.main(["ask", "ridge"]) == 2

    captured = capsys.readouterr()
    assert "quota gratuit Gemini atteint" in captured.err
    assert SECRET not in captured.out + captured.err


def test_existing_commands_work_without_llm_configuration(
    courses: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DOCASSIST_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    capsys.readouterr()

    assert cli.main(["search", "ridge", "-k", "1"]) == 0
    assert "ridge.pdf" in capsys.readouterr().out


def test_abstention_shows_model_explanation(
    courses: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class ExplainingGenerator(FakeAnswerGenerator):
        def generate(self, question: str, passages: Any) -> Answer:
            return Answer("Les cours ne traitent pas ce sujet.", (), True, "fake", "raison test")

    use_generator(monkeypatch, ExplainingGenerator())
    capsys.readouterr()

    assert cli.main(["ask", "sujet absent"]) == 0

    out = capsys.readouterr().out
    assert "(raison : raison test)" in out
    assert "Les cours ne traitent pas ce sujet." in out
