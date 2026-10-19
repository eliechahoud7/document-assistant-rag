from pathlib import Path

import pytest

from doc_assistant.config import ConfigError, Settings


def test_defaults_are_offline_and_valid() -> None:
    settings = Settings.from_env({})

    assert settings.llm_provider == "none"
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.chunk_overlap < settings.chunk_size


def test_environment_overrides_defaults() -> None:
    settings = Settings.from_env(
        {
            "DOCASSIST_DATA_DIR": "/tmp/docs",
            "DOCASSIST_LLM_PROVIDER": "Ollama",
            "DOCASSIST_LOG_LEVEL": "debug",
            "DOCASSIST_CHUNK_SIZE": "500",
            "DOCASSIST_CHUNK_OVERLAP": "50",
            "DOCASSIST_RETRIEVAL_TOP_K": "10",
        }
    )

    assert settings.data_dir == Path("/tmp/docs")
    assert settings.llm_provider == "ollama"
    assert settings.log_level == "DEBUG"
    assert (settings.chunk_size, settings.chunk_overlap, settings.retrieval_top_k) == (500, 50, 10)


def test_variables_without_prefix_are_ignored() -> None:
    assert Settings.from_env({"CHUNK_SIZE": "1"}).chunk_size == 800


def test_derived_directories() -> None:
    settings = Settings(data_dir=Path("store"))

    assert settings.chroma_dir == Path("store") / "chroma"
    assert settings.uploads_dir == Path("store") / "uploads"


@pytest.mark.parametrize(
    "env",
    [
        {"DOCASSIST_CHUNK_SIZE": "abc"},
        {"DOCASSIST_CHUNK_SIZE": "0"},
        {"DOCASSIST_CHUNK_SIZE": "100", "DOCASSIST_CHUNK_OVERLAP": "100"},
        {"DOCASSIST_CHUNK_OVERLAP": "-1"},
        {"DOCASSIST_RETRIEVAL_TOP_K": "0"},
        {"DOCASSIST_LLM_PROVIDER": "openai"},
        {"DOCASSIST_LOG_LEVEL": "verbose"},
    ],
)
def test_invalid_values_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_embedding_and_store_settings() -> None:
    settings = Settings.from_env(
        {
            "DOCASSIST_MODELS_DIR": "D:/modeles",
            "DOCASSIST_EMBEDDING_MODEL_REVISION": "abc123",
            "DOCASSIST_EMBEDDING_DEVICE": "CPU",
            "DOCASSIST_EMBEDDING_BATCH_SIZE": "8",
            "DOCASSIST_CHROMA_COLLECTION": "essai_v2",
        }
    )

    assert settings.embedding_model_dir == Path("D:/modeles") / "multilingual-e5-small"
    assert settings.embedding_model_revision == "abc123"
    assert (settings.embedding_device, settings.embedding_batch_size) == ("cpu", 8)
    assert settings.chroma_collection == "essai_v2"


def test_default_embedding_settings_are_cpu_and_pinned() -> None:
    settings = Settings()

    assert settings.embedding_device == "cpu"
    assert len(settings.embedding_model_revision) == 40  # hash de commit complet
    assert settings.embedding_model_dir == Path("models") / "multilingual-e5-small"


@pytest.mark.parametrize(
    "env",
    [
        {"DOCASSIST_EMBEDDING_BATCH_SIZE": "0"},
        {"DOCASSIST_EMBEDDING_MODEL": " "},
        {"DOCASSIST_CHROMA_COLLECTION": "ab"},
        {"DOCASSIST_CHROMA_COLLECTION": "-chunks"},
        {"DOCASSIST_CHROMA_COLLECTION": "mes chunks"},
    ],
)
def test_invalid_embedding_settings_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_llm_defaults_make_no_api_call() -> None:
    settings = Settings.from_env({})

    assert settings.llm_provider == "none"
    assert settings.llm_model == "gemini-3.5-flash-lite"
    assert (settings.llm_timeout_s, settings.llm_max_retries) == (30, 1)


def test_llm_settings_from_environment() -> None:
    settings = Settings.from_env(
        {
            "DOCASSIST_LLM_PROVIDER": "Gemini",
            "DOCASSIST_LLM_MODEL": "gemini-x",
            "DOCASSIST_LLM_TIMEOUT_S": "10",
            "DOCASSIST_LLM_MAX_RETRIES": "0",
            "DOCASSIST_LLM_MAX_OUTPUT_TOKENS": "512",
        }
    )

    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-x"
    assert (settings.llm_timeout_s, settings.llm_max_retries) == (10, 0)
    assert settings.llm_max_output_tokens == 512


@pytest.mark.parametrize(
    "env",
    [
        {"DOCASSIST_LLM_MODEL": " "},
        {"DOCASSIST_LLM_TIMEOUT_S": "0"},
        {"DOCASSIST_LLM_TIMEOUT_S": "abc"},
        {"DOCASSIST_LLM_MAX_RETRIES": "-1"},
        {"DOCASSIST_LLM_MAX_OUTPUT_TOKENS": "0"},
    ],
)
def test_invalid_llm_settings_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_settings_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(AttributeError):
        settings.chunk_size = 1  # type: ignore[misc]


def test_context_neighbors_default_and_environment() -> None:
    assert Settings.from_env({}).context_neighbors == 1  # configuration retenue (E6a, ADR 0008)
    assert Settings.from_env({"DOCASSIST_CONTEXT_NEIGHBORS": "0"}).context_neighbors == 0


@pytest.mark.parametrize("value", ["-1", "4", "x"])
def test_invalid_context_neighbors(value: str) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"DOCASSIST_CONTEXT_NEIGHBORS": value})
