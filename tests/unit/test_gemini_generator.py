"""Adaptateur Gemini avec un faux client : aucun appel réseau, aucune vraie clé."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import errors, types

from doc_assistant import bootstrap
from doc_assistant.config import Settings
from doc_assistant.domain.errors import (
    LLMConfigurationError,
    LLMQuotaError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from doc_assistant.domain.models import Chunk, SearchResult
from doc_assistant.domain.ports import AnswerGenerator
from doc_assistant.generation.gemini_generator import (
    API_KEY_ENV,
    RETRYABLE_STATUS_CODES,
    GeminiGenerator,
)
from doc_assistant.generation.prompts import ANSWER_SCHEMA, SYSTEM_INSTRUCTION

SECRET = "AIza-TEST-SECRET-never-printed"
PASSAGES = [
    SearchResult(
        Chunk("a:0", "a", "lecture2.pdf", 3, 0, 0, "Support vectors lie on the margin."), 0.9
    ),
    SearchResult(Chunk("b:4", "b", "lecture1.pdf", 38, 4, 120, "Ridge adds a penalty."), 0.8),
]


def response(
    payload: dict[str, Any] | str,
    *,
    finish: types.FinishReason = types.FinishReason.STOP,
    block: types.BlockedReason | None = None,
) -> types.GenerateContentResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    if block is not None:
        return types.GenerateContentResponse(
            candidates=[],
            prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block),
        )
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=finish,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=120, candidates_token_count=30
        ),
    )


class FakeModels:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeClient:
    def __init__(self, outcome: Any) -> None:
        self.models = FakeModels(outcome)


def generator(outcome: Any) -> tuple[GeminiGenerator, FakeClient]:
    client = FakeClient(outcome)
    return GeminiGenerator(client, "gemini-3.5-flash-lite", max_output_tokens=512), client


# ---------------------------------------------------------------- réponses valides


def test_valid_answer_cites_real_metadata() -> None:
    gen, _ = generator(
        response(
            {"abstain": False, "answer": "They are support vectors [1].", "cited_passages": [1]}
        )
    )

    answer = gen.generate("What lies on the margin?", PASSAGES)

    assert not answer.abstained
    assert answer.text == "They are support vectors [1]."
    assert [(c.number, c.filename, c.page_number) for c in answer.citations] == [
        (1, "lecture2.pdf", 3)
    ]
    assert answer.citations[0].chunk is PASSAGES[0].chunk
    assert (answer.input_tokens, answer.output_tokens) == (120, 30)
    assert answer.model == "gemini-3.5-flash-lite"


def test_request_uses_structured_output_and_numbered_passages() -> None:
    gen, client = generator(response({"abstain": True, "answer": "", "cited_passages": []}))

    gen.generate("Quelle pénalité ?", PASSAGES)

    call = client.models.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"
    config = call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == ANSWER_SCHEMA
    assert config.system_instruction == SYSTEM_INSTRUCTION
    assert config.max_output_tokens == 512
    prompt = call["contents"]
    assert "Question: Quelle pénalité ?" in prompt
    assert "[1] (source: lecture2.pdf, page 3)" in prompt
    assert "[2] (source: lecture1.pdf, page 38)" in prompt


def test_invalid_citations_are_dropped() -> None:
    gen, _ = generator(
        response({"abstain": False, "answer": "Penalty [2] and [7].", "cited_passages": [2, 9, 0]})
    )

    answer = gen.generate("q", PASSAGES)

    assert [c.number for c in answer.citations] == [2]
    assert answer.citations[0].filename == "lecture1.pdf"
    assert "[7]" not in answer.text


def test_answer_without_any_valid_citation_becomes_abstention() -> None:
    gen, _ = generator(
        response({"abstain": False, "answer": "Trust me [5].", "cited_passages": [5]})
    )

    answer = gen.generate("q", PASSAGES)

    assert answer.abstained
    assert answer.citations == ()
    assert answer.abstain_reason == "la réponse ne cite aucun passage valide"


def test_model_abstention() -> None:
    gen, _ = generator(
        response(
            {"abstain": True, "answer": "Les documents ne le disent pas.", "cited_passages": [1]}
        )
    )

    answer = gen.generate("q", PASSAGES)

    assert answer.abstained
    assert answer.citations == ()  # une abstention ne cite rien, même si le modèle l'a fait
    assert answer.text == "Les documents ne le disent pas."


def test_no_passages_means_no_api_call() -> None:
    gen, client = generator(response({"abstain": False, "answer": "x [1]", "cited_passages": [1]}))

    answer = gen.generate("q", [])

    assert answer.abstained
    assert client.models.calls == []


# ---------------------------------------------------------------- réponses inexploitables


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (response("pas du json"), "non JSON"),
        (response({"answer": "x"}), "schéma"),
        (
            response(
                {"abstain": False, "answer": "x [1]", "cited_passages": [1]},
                finish=types.FinishReason.MAX_TOKENS,
            ),
            "tronquée",
        ),
        (
            response(
                {"abstain": False, "answer": "x", "cited_passages": []},
                finish=types.FinishReason.SAFETY,
            ),
            "SAFETY",
        ),
        (response({}, block=types.BlockedReason.SAFETY), "bloquée"),
        (types.GenerateContentResponse(candidates=[]), "aucun candidat"),
    ],
)
def test_unusable_responses_raise(outcome: Any, message: str) -> None:
    gen, _ = generator(outcome)

    with pytest.raises(LLMResponseError, match=message):
        gen.generate("q", PASSAGES)


# ---------------------------------------------------------------- erreurs de l'API


def api_error(cls: type[errors.APIError], code: int, status: str, message: str) -> errors.APIError:
    return cls(code, {"error": {"code": code, "status": status, "message": message}})


@pytest.mark.parametrize(
    ("exc", "expected", "message"),
    [
        (api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", "quota"), LLMQuotaError, "quota"),
        (
            api_error(errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid."),
            LLMConfigurationError,
            "clé API",
        ),
        (
            api_error(errors.ClientError, 403, "PERMISSION_DENIED", "denied"),
            LLMConfigurationError,
            "clé API",
        ),
        (
            api_error(errors.ClientError, 404, "NOT_FOUND", "no such model"),
            LLMConfigurationError,
            "introuvable",
        ),
        (
            api_error(errors.ClientError, 400, "INVALID_ARGUMENT", "bad schema"),
            LLMResponseError,
            "refusée",
        ),
        (
            api_error(errors.ServerError, 503, "UNAVAILABLE", "overloaded"),
            LLMUnavailableError,
            "503",
        ),
        (httpx.ReadTimeout("timed out"), LLMTimeoutError, "30 s"),
        (httpx.ConnectError("unreachable"), LLMUnavailableError, "réseau"),
    ],
)
def test_api_errors_are_mapped(exc: BaseException, expected: type[Exception], message: str) -> None:
    gen = GeminiGenerator(FakeClient(exc), "gemini-3.5-flash-lite", timeout_s=30)

    with pytest.raises(expected, match=message):
        gen.generate("q", PASSAGES)


def test_quota_error_is_also_an_unavailability() -> None:
    gen, _ = generator(api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", "quota"))

    with pytest.raises(LLMUnavailableError):
        gen.generate("q", PASSAGES)


# ---------------------------------------------------------------- clé et configuration


def test_missing_key_is_a_clear_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(LLMConfigurationError, match=API_KEY_ENV):
        GeminiGenerator.from_settings(Settings(llm_provider="gemini"))


def test_blank_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "   ")

    with pytest.raises(LLMConfigurationError, match=API_KEY_ENV):
        GeminiGenerator.from_settings(Settings(llm_provider="gemini"))


def test_client_is_built_with_key_timeout_and_safe_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class RecordingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.models = FakeModels(None)

    monkeypatch.setenv(API_KEY_ENV, SECRET)
    monkeypatch.setenv("GOOGLE_API_KEY", "other-key-must-not-be-used")
    monkeypatch.setattr(genai, "Client", RecordingClient)
    settings = Settings(llm_provider="gemini", llm_timeout_s=12, llm_max_retries=1)

    gen = GeminiGenerator.from_settings(settings)

    assert captured["api_key"] == SECRET  # transmise explicitement : GOOGLE_API_KEY ignorée
    options = captured["http_options"]
    assert options.timeout == 12_000
    assert options.retry_options.attempts == 2
    assert options.retry_options.http_status_codes == RETRYABLE_STATUS_CODES
    assert 429 not in options.retry_options.http_status_codes
    assert SECRET not in repr(vars(gen))


def test_real_client_construction_stays_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, SECRET)

    gen: AnswerGenerator = GeminiGenerator.from_settings(Settings(llm_provider="gemini"))

    assert gen.model_name == "gemini-3.5-flash-lite"  # aucun accès réseau (fixture de blocage)


def test_error_messages_never_contain_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    for exc in (
        api_error(errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid."),
        api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", "quota"),
        httpx.ReadTimeout("t"),
    ):
        gen, _ = generator(exc)
        with pytest.raises(Exception) as info:
            gen.generate("q", PASSAGES)
        assert SECRET not in str(info.value)


def test_bootstrap_requires_an_explicit_provider() -> None:
    with pytest.raises(LLMConfigurationError, match="DOCASSIST_LLM_PROVIDER=gemini"):
        bootstrap.build_generator(Settings())


@pytest.mark.parametrize("provider", ["ollama", "anthropic"])
def test_bootstrap_rejects_unimplemented_providers(provider: str) -> None:
    with pytest.raises(LLMConfigurationError, match="non implémenté"):
        bootstrap.build_generator(Settings(llm_provider=provider))  # type: ignore[arg-type]


def test_bootstrap_builds_gemini_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, SECRET)

    gen = bootstrap.build_generator(Settings(llm_provider="gemini"))

    assert isinstance(gen, GeminiGenerator)


def test_unclassified_api_error_is_an_unavailability() -> None:
    gen, _ = generator(api_error(errors.APIError, 418, "TEAPOT", "odd"))

    with pytest.raises(LLMUnavailableError, match="418"):
        gen.generate("q", PASSAGES)


def test_response_without_text_raises() -> None:
    empty = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    gen, _ = generator(empty)

    with pytest.raises(LLMResponseError, match="sans texte"):
        gen.generate("q", PASSAGES)
