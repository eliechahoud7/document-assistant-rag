"""Traduction FR -> EN des requêtes (recherche uniquement) : hors ligne, faux traducteur.

Aucun modèle de traduction n'est chargé : le traducteur Marian reçoit un faux tokenizer et un
faux modèle, et ``build_embedder`` est testé avec un dossier de modèles vide.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

from doc_assistant import bootstrap, cli
from doc_assistant.config import ConfigError, Settings
from doc_assistant.domain.errors import TranslationError, TranslationModelNotFoundError
from doc_assistant.domain.models import Chunk
from doc_assistant.indexing.embedders import E5Embedder
from doc_assistant.indexing.model_download import (
    REQUIRED_TRANSLATION_FILES,
    download_translation_model,
)
from doc_assistant.retrieval.marian_translator import REQUIRED_MODEL_FILES, MarianTranslator
from doc_assistant.retrieval.query_translation import TranslatingEmbedder, looks_french
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.qa_service import QAService
from fakes import FakeEmbedder, InMemoryVectorStore, ScriptedAnswerGenerator

ROOT = Path(__file__).resolve().parents[2]
FR_QUESTION = "Quelle pénalité annule certains coefficients ?"
EN_TRANSLATION = "Which penalty zeroes some weights?"


class FakeTranslator:
    """Traducteur scripté : renvoie une traduction fixe, ou lève l'exception fournie."""

    def __init__(self, result: str | Exception = EN_TRANSLATION) -> None:
        self.result = result
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-translator"

    def translate(self, text: str) -> str:
        self.calls.append(text)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def translating(result: str | Exception = EN_TRANSLATION) -> tuple[TranslatingEmbedder, Any, Any]:
    inner, translator = FakeEmbedder(), FakeTranslator(result)
    return TranslatingEmbedder(inner, translator), inner, translator


# ---------------------------------------------------------------- détection du français


def labelled_questions() -> list[tuple[str, str, str]]:
    rows = []
    for name in ("real_v1", "real_v1_en", "real_v2", "synthetic_v1"):
        data = json.loads((ROOT / "evaluation" / "datasets" / f"{name}.json").read_text("utf-8"))
        rows += [(f"{name}:{q['id']}", q["lang"], q["question"]) for q in data["questions"]]
    return rows


def test_language_detection_matches_every_labelled_dataset_question() -> None:
    rows = labelled_questions()

    wrong = [(qid, lang) for qid, lang, text in rows if looks_french(text) != (lang == "fr")]

    assert len(rows) > 100
    assert wrong == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Comment choisir k dans les k-moyennes ?", True),
        ("Qu'est-ce que le surapprentissage ?", True),
        ("How do you choose k in k-means?", False),
        ("What is the 'déjà vu' effect in the course?", False),  # accent isolé dans de l'anglais
        ("SVM ?", False),  # aucun indice : pas de traduction
        ("", False),
    ],
)
def test_language_detection_examples(text: str, expected: bool) -> None:
    assert looks_french(text) is expected


# ---------------------------------------------------------------- embedder traduisant


def test_french_question_is_translated_before_embedding() -> None:
    embedder, inner, translator = translating()

    embedder.embed_query(FR_QUESTION)

    assert translator.calls == [FR_QUESTION]
    assert inner.query_calls == [EN_TRANSLATION]


def test_english_question_bypasses_translation() -> None:
    embedder, inner, translator = translating()

    embedder.embed_query("Which penalty zeroes some coefficients?")

    assert translator.calls == []
    assert inner.query_calls == ["Which penalty zeroes some coefficients?"]


@pytest.mark.parametrize("result", [TranslationError("échec"), "   "])
def test_translation_failure_falls_back_to_original_question(
    result: str | Exception, caplog: pytest.LogCaptureFixture
) -> None:
    embedder, inner, _ = translating(result)

    with caplog.at_level(logging.WARNING, logger="doc_assistant.retrieval.query_translation"):
        embedder.embed_query(FR_QUESTION)

    assert inner.query_calls == [FR_QUESTION]
    assert "question d'origine utilisée" in caplog.text


def test_missing_model_at_translation_time_is_not_silenced() -> None:
    embedder, _, _ = translating(TranslationModelNotFoundError("modèle absent"))

    with pytest.raises(TranslationModelNotFoundError):
        embedder.embed_query(FR_QUESTION)


def test_documents_model_and_dimension_are_delegated() -> None:
    embedder, inner, translator = translating()

    vectors = embedder.embed_documents(["Une page en français."])

    assert vectors == inner.embed_documents(["Une page en français."])
    assert translator.calls == []  # les passages indexés ne sont jamais traduits
    assert (embedder.model_name, embedder.dimension) == (inner.model_name, inner.dimension)
    assert embedder.translator_name == "fake-translator"


def qa_service(embedder: Any) -> tuple[QAService, ScriptedAnswerGenerator]:
    store = InMemoryVectorStore()
    # Piège lexical : la question française partage « coefficients » avec le passage Ridge.
    texts = ["Lasso zeroes some weights.", "Ridge penalty shrinks coefficients smoothly."]
    chunks = [Chunk(f"d:{i}", "d", "notes.pdf", i + 1, i, 0, t) for i, t in enumerate(texts)]
    store.add(chunks, embedder.embed_documents(texts))
    generator = ScriptedAnswerGenerator({})
    return QAService(Retriever(embedder, store, default_top_k=1), generator), generator


def test_translation_changes_retrieval_but_generation_gets_the_original_question() -> None:
    embedder, _, _ = translating()
    service, generator = qa_service(embedder)

    result = service.ask(FR_QUESTION)

    assert result.passages[0].chunk.text == "Lasso zeroes some weights."  # grâce à la traduction
    assert generator.calls == [FR_QUESTION]  # question d'origine, intacte


def test_without_translation_the_same_french_question_retrieves_differently() -> None:
    service, generator = qa_service(FakeEmbedder())

    result = service.ask(FR_QUESTION)

    assert result.passages[0].chunk.text != "Lasso zeroes some weights."
    assert generator.calls == [FR_QUESTION]


# ---------------------------------------------------------------- assemblage et configuration


def test_translation_is_disabled_by_default() -> None:
    settings = Settings.from_env({})

    assert settings.query_translation == "none"
    assert isinstance(bootstrap.build_embedder(settings), E5Embedder)  # aucune enveloppe


def test_enabled_translation_without_model_fails_with_a_clear_error(tmp_path: Path) -> None:
    settings = Settings(models_dir=tmp_path, query_translation="fr-en")

    with pytest.raises(TranslationModelNotFoundError, match="download-translation-model"):
        bootstrap.build_embedder(settings)


def test_enabled_translation_with_model_files_wraps_the_embedder(tmp_path: Path) -> None:
    settings = Settings(models_dir=tmp_path, query_translation="fr-en")
    settings.translation_model_dir.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (settings.translation_model_dir / name).touch()

    embedder = bootstrap.build_embedder(settings)  # aucun chargement : le modèle est paresseux

    assert isinstance(embedder, TranslatingEmbedder)
    assert embedder.model_name == settings.embedding_model


def test_translation_settings_from_environment() -> None:
    settings = Settings.from_env({"DOCASSIST_QUERY_TRANSLATION": "FR-EN"})

    assert settings.query_translation == "fr-en"
    assert settings.translation_model_dir == Path("models") / "opus-mt-fr-en"


@pytest.mark.parametrize(
    "env", [{"DOCASSIST_QUERY_TRANSLATION": "en-fr"}, {"DOCASSIST_TRANSLATION_MODEL": " "}]
)
def test_invalid_translation_settings_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


# ---------------------------------------------------------------- adaptateur Marian


class FakeBatch(dict[str, Any]):
    def to(self, _device: str) -> FakeBatch:
        return self


class FakeTokenizer:
    def __call__(self, texts: list[str], **_kwargs: Any) -> FakeBatch:
        return FakeBatch(input_ids=texts)

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens
        return f"  {ids}  "


class FakeModel:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def generate(self, **batch: Any) -> list[str]:
        if self.error is not None:
            raise self.error
        return [EN_TRANSLATION for _ in batch["input_ids"]]


def test_marian_translator_decodes_the_generated_translation() -> None:
    translator = MarianTranslator(
        Path("absent"), "opus", tokenizer=FakeTokenizer(), model=FakeModel()
    )

    assert translator.translate(FR_QUESTION) == EN_TRANSLATION


def test_marian_translator_wraps_model_errors() -> None:
    translator = MarianTranslator(
        Path("absent"), "opus", tokenizer=FakeTokenizer(), model=FakeModel(RuntimeError("boom"))
    )

    with pytest.raises(TranslationError, match="RuntimeError"):
        translator.translate(FR_QUESTION)


def test_model_error_inside_the_pipeline_falls_back_to_original_question() -> None:
    marian = MarianTranslator(
        Path("absent"), "opus", tokenizer=FakeTokenizer(), model=FakeModel(RuntimeError("boom"))
    )
    inner = FakeEmbedder()

    TranslatingEmbedder(inner, marian).embed_query(FR_QUESTION)

    assert inner.query_calls == [FR_QUESTION]


def test_marian_translator_reports_missing_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").touch()
    translator = MarianTranslator(tmp_path, "opus")

    with pytest.raises(TranslationModelNotFoundError, match=r"model\.safetensors"):
        translator.translate(FR_QUESTION)  # chargement paresseux : même vérification


# ---------------------------------------------------------------- téléchargement et CLI


def test_translation_download_fetches_only_required_files_at_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    settings = Settings(models_dir=tmp_path)

    target = download_translation_model(settings)

    assert target == tmp_path / "opus-mt-fr-en"
    assert calls == [
        {
            "repo_id": "Helsinki-NLP/opus-mt-fr-en",
            "revision": "c4aed37b318c763fd177aa449b44e3b783cc6c02",
            "local_dir": target,
            "allow_patterns": REQUIRED_TRANSLATION_FILES,
        }
    ]
    assert set(REQUIRED_MODEL_FILES) <= set(REQUIRED_TRANSLATION_FILES)
    assert not any(n.endswith((".bin", ".h5", ".msgpack")) for n in REQUIRED_TRANSLATION_FILES)


def test_download_translation_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    targets: list[Path] = []

    def fake_download(settings: Settings) -> Path:
        targets.append(settings.translation_model_dir)
        return settings.translation_model_dir

    monkeypatch.setenv("DOCASSIST_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "download_translation_model", fake_download)

    assert cli.main(["download-translation-model"]) == 0
    assert targets == [tmp_path / "opus-mt-fr-en"]


def test_cli_reports_missing_translation_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCASSIST_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DOCASSIST_QUERY_TRANSLATION", "fr-en")

    code = cli.main(["search", FR_QUESTION])

    assert code == 2
    assert "modèle de traduction introuvable" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()  # erreur avant toute ouverture d'index


def test_translation_config_is_recorded_only_when_enabled() -> None:
    assert cli._translation_config(Settings()) == {}
    assert cli._translation_config(Settings(query_translation="fr-en")) == {
        "query_translation": "fr-en",
        "translation_model": "Helsinki-NLP/opus-mt-fr-en",
        "translation_model_revision": "c4aed37b318c763fd177aa449b44e3b783cc6c02",
    }
