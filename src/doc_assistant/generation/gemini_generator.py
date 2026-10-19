"""Adaptateur Gemini (SDK google-genai) pour la génération de réponses fondées sur les passages.

La clé est lue uniquement dans ``GEMINI_API_KEY`` et transmise explicitement au client :
elle n'est jamais conservée par l'adaptateur, ni journalisée, ni incluse dans un message.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

from doc_assistant.config import Settings
from doc_assistant.domain.errors import (
    LLMConfigurationError,
    LLMQuotaError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from doc_assistant.domain.models import Answer, SearchResult
from doc_assistant.generation.answers import no_passage_answer, parse_answer
from doc_assistant.generation.prompts import ANSWER_SCHEMA, SYSTEM_INSTRUCTION, build_user_prompt

API_KEY_ENV = "GEMINI_API_KEY"
# Nouvelles tentatives sur erreurs serveur uniquement : réessayer un 429 consommerait du quota.
RETRYABLE_STATUS_CODES = [500, 502, 503, 504]


class _Models(Protocol):
    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any: ...


class GeminiClient(Protocol):
    """Sous-ensemble de ``google.genai.Client`` utilisé (permet d'injecter un faux client)."""

    @property
    def models(self) -> _Models: ...


def _transport_errors() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    """Exceptions httpx de délai dépassé et de réseau (httpx2 aussi, s'il est installé)."""
    import httpx

    timeouts: list[type[BaseException]] = [httpx.TimeoutException]
    network: list[type[BaseException]] = [httpx.TransportError]
    try:
        import httpx2  # type: ignore[import-not-found,unused-ignore]
    except ImportError:  # pragma: no cover - dépend de l'environnement
        pass
    else:
        timeouts.append(httpx2.TimeoutException)
        network.append(httpx2.TransportError)
    return tuple(timeouts), tuple(network)


class GeminiGenerator:
    def __init__(
        self,
        client: GeminiClient,
        model: str,
        *,
        max_output_tokens: int = 2048,
        timeout_s: int = 30,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s

    @classmethod
    def from_settings(cls, settings: Settings) -> GeminiGenerator:
        api_key = os.environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            raise LLMConfigurationError(
                f"variable d'environnement {API_KEY_ENV} absente ou vide : créez une clé dans "
                'Google AI Studio puis, dans PowerShell, $env:GEMINI_API_KEY = "..."'
            )
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=settings.llm_timeout_s * 1000,  # millisecondes
                retry_options=types.HttpRetryOptions(
                    attempts=1 + settings.llm_max_retries,
                    http_status_codes=RETRYABLE_STATUS_CODES,
                ),
            ),
        )
        return cls(
            client,
            settings.llm_model,
            max_output_tokens=settings.llm_max_output_tokens,
            timeout_s=settings.llm_timeout_s,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer:
        if not passages:
            return no_passage_answer(self._model)
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_json_schema=ANSWER_SCHEMA,
            max_output_tokens=self._max_output_tokens,
        )
        timeouts, network = _transport_errors()
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=build_user_prompt(question, passages), config=config
            )
        except errors.ClientError as exc:
            raise self._client_error(exc) from exc
        except errors.ServerError as exc:
            raise LLMUnavailableError(f"erreur du service Gemini ({exc.code})") from exc
        except errors.APIError as exc:
            raise LLMUnavailableError(f"erreur de l'API Gemini ({exc.code})") from exc
        except timeouts as exc:
            raise LLMTimeoutError(f"pas de réponse de Gemini en {self._timeout_s} s") from exc
        except network as exc:
            raise LLMUnavailableError("impossible de joindre l'API Gemini (réseau)") from exc
        return self._to_answer(response, passages)

    def _client_error(self, exc: Any) -> Exception:
        code, status, message = exc.code, str(exc.status or ""), str(exc.message or "")
        if code == 429:
            return LLMQuotaError(
                "quota gratuit Gemini atteint (429 RESOURCE_EXHAUSTED) : réessayez plus tard ; "
                "les quotas journaliers se réinitialisent à minuit, heure du Pacifique"
            )
        if code in (401, 403) or "API key" in message or "API_KEY" in status:
            return LLMConfigurationError(f"clé API Gemini refusée ({code} {status})")
        if code == 404:
            return LLMConfigurationError(f"modèle Gemini introuvable : {self._model!r}")
        return LLMResponseError(f"requête refusée par Gemini ({code} {status})")

    def _to_answer(self, response: Any, passages: Sequence[SearchResult]) -> Answer:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMResponseError(f"requête bloquée par Gemini ({feedback.block_reason})")
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise LLMResponseError("réponse Gemini vide (aucun candidat)")
        reason = getattr(candidates[0].finish_reason, "name", str(candidates[0].finish_reason))
        if reason == "MAX_TOKENS":
            raise LLMResponseError(
                "réponse tronquée (max_output_tokens atteint) : augmentez "
                "DOCASSIST_LLM_MAX_OUTPUT_TOKENS"
            )
        if reason not in ("STOP", "FINISH_REASON_UNSPECIFIED", "None"):
            raise LLMResponseError(f"génération interrompue par Gemini ({reason})")
        text = response.text
        if not text:
            raise LLMResponseError("réponse Gemini sans texte")
        usage = getattr(response, "usage_metadata", None)
        return parse_answer(
            text,
            passages,
            model=self._model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
