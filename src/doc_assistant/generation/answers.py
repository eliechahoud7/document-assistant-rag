"""Validation de la sortie du LLM, indépendante du fournisseur.

Les citations sont reconstruites à partir des passages réellement fournis : un numéro de
passage inexistant est écarté, et une réponse sans citation valide devient une abstention.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from doc_assistant.domain.errors import LLMResponseError
from doc_assistant.domain.models import Answer, Citation, SearchResult

_MARKER_RE = re.compile(r"\[(\d+)\]")

ABSTAIN_MODEL = "le modèle juge que les passages ne contiennent pas la réponse"
ABSTAIN_NO_VALID_CITATION = "la réponse ne cite aucun passage valide"
ABSTAIN_NO_PASSAGES = "aucun passage retrouvé dans l'index"


def parse_answer(
    raw_json: str,
    passages: Sequence[SearchResult],
    *,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> Answer:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"réponse du modèle non JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("réponse du modèle : objet JSON attendu")
    abstain, text, cited = data.get("abstain"), data.get("answer"), data.get("cited_passages")
    if not isinstance(abstain, bool) or not isinstance(text, str) or not isinstance(cited, list):
        raise LLMResponseError("réponse du modèle non conforme au schéma attendu")

    if abstain:
        return Answer(
            text=text.strip(),
            citations=(),
            abstained=True,
            model=model,
            abstain_reason=ABSTAIN_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    valid = range(1, len(passages) + 1)
    declared = {n for n in cited if isinstance(n, int) and not isinstance(n, bool)}
    inline = {int(n) for n in _MARKER_RE.findall(text)}
    numbers = sorted((declared | inline) & set(valid))
    # Les marqueurs [n] qui ne désignent aucun passage fourni sont retirés du texte.
    cleaned = _MARKER_RE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if not numbers or not cleaned:
        return Answer(
            text="",
            citations=(),
            abstained=True,
            model=model,
            abstain_reason=ABSTAIN_NO_VALID_CITATION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    return Answer(
        text=cleaned,
        citations=tuple(Citation(n, passages[n - 1].chunk) for n in numbers),
        abstained=False,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def no_passage_answer(model: str) -> Answer:
    return Answer(
        text="", citations=(), abstained=True, model=model, abstain_reason=ABSTAIN_NO_PASSAGES
    )
