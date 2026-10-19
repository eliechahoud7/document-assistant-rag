"""Appel RÉEL à l'API Gemini : désélectionné par défaut (marqueurs llm et network).

Lancer uniquement de façon volontaire : ``python -m pytest -m llm`` avec GEMINI_API_KEY défini.
Consomme une requête du quota gratuit ; n'envoie qu'un passage factice, aucun document.
"""

import os

import pytest

from doc_assistant.config import Settings
from doc_assistant.domain.models import Chunk, SearchResult
from doc_assistant.generation.gemini_generator import API_KEY_ENV, GeminiGenerator

pytestmark = [pytest.mark.llm, pytest.mark.network]


@pytest.mark.skipif(not os.environ.get(API_KEY_ENV), reason=f"{API_KEY_ENV} non défini")
def test_live_answer_is_grounded() -> None:
    passage = Chunk(
        "t:0", "t", "test.pdf", 1, 0, 0, "The capital of the fictional land Zorbia is Quillon."
    )
    generator = GeminiGenerator.from_settings(Settings(llm_provider="gemini"))

    answer = generator.generate("What is the capital of Zorbia?", [SearchResult(passage, 0.9)])

    assert not answer.abstained
    assert "Quillon" in answer.text
    assert [(c.filename, c.page_number) for c in answer.citations] == [("test.pdf", 1)]
