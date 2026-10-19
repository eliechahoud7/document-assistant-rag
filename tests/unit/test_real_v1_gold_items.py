"""Non-régression des faits requis de l'annexe real_v1_answers.json (sans les PDF).

Chaque cas est une réponse plausible du modèle ; on vérifie le verdict automatique obtenu.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc_assistant.evaluation.generation import fact_found, normalize

ANNEX = Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "real_v1_answers.json"
ANSWERS: dict[str, Any] = json.loads(ANNEX.read_text(encoding="utf-8"))["answers"]


def verdict(qid: str, answer: str) -> str:
    text = normalize(answer)
    found = [fact_found(fact, text) for fact in ANSWERS[qid]["required_facts"]]
    return "correct" if all(found) else "partial" if any(found) else "incorrect"


@pytest.mark.parametrize(
    ("qid", "answer", "expected"),
    [
        # Q01 : la frontière loin des exemples ; « so far » ne doit plus compter.
        (
            "q01",
            "The boundary is placed as far from the training examples as possible, so noisy "
            "inputs do not cross it.",
            "correct",
        ),
        (
            "q01",
            "La frontière est éloignée des exemples, ce qui la rend robuste au bruit.",
            "correct",
        ),
        ("q01", "So far, this classifier handles noisy examples well.", "partial"),
        # Q07 : vers zéro, pas « non nul » ni « exactement zéro ».
        ("q07", "La pénalité pousse tous les paramètres vers zéro.", "correct"),
        ("q07", "It shrinks the parameters.", "correct"),
        ("q07", "Les paramètres deviennent non nuls (nonzero).", "incorrect"),
        ("q07", "Elle met les paramètres exactement à zéro.", "incorrect"),
        # Q08 : inverser une matrice n x n (ou la matrice de Gram).
        ("q08", "We must invert an n × n matrix.", "correct"),  # noqa: RUF001
        ("q08", "Il faut inverser la matrice de Gram.", "correct"),
        ("q08", "It requires the inverse of an n^2 sized matrix.", "correct"),
        ("q08", "It requires inverting a d × d matrix of features.", "partial"),  # noqa: RUF001
        # Q13 : 63,2 % et ses arrondis ; pas un autre nombre commençant par 63.
        ("q13", "Environ 63,2 % des exemples sont uniques.", "correct"),
        ("q13", "About 63% of the samples are unique.", "correct"),
        ("q13", "Environ 63 % sont distincts.", "correct"),
        ("q13", "About 63.25 percent are unique.", "incorrect"),
        ("q13", "Roughly two thirds are unique.", "incorrect"),
        # Q14 : 60 % / 40 % ; ni l'exemple 80/20 de la p. 39 ni « instances 21 à 40 ».
        (
            "q14",
            "The course trains on 60% of the training set and validates on the remaining 40%.",
            "correct",
        ),
        ("q14", "60 % pour l'entraînement, 40 % pour la validation.", "correct"),
        (
            "q14",
            "Train on instances 1 to 80 and test on 81 to 100, e.g. instances 21 to 40.",
            "incorrect",
        ),
        ("q14", "An 80% / 20% split with 5 folds.", "incorrect"),
    ],
)
def test_revised_items(qid: str, answer: str, expected: str) -> None:
    assert verdict(qid, answer) == expected


@pytest.mark.parametrize("qid", ["q03", "q10"])
def test_partial_verification_items_keep_their_human_review_flag(qid: str) -> None:
    item = ANSWERS[qid]

    assert item["support"] == "weak"
    assert any("NE vérifie PAS" in flag for flag in item["flags"])


def test_q03_keyword_scorer_accepts_answer_missing_the_bound() -> None:
    # Limite documentée : la borne « / n » n'est pas vérifiée automatiquement.
    assert verdict("q03", "It depends on the support vectors.") == "correct"


def test_q10_keyword_scorer_cannot_check_the_full_procedure() -> None:
    # Limite documentée : une réponse incomplète peut contenir les mots-clés génériques.
    assert verdict("q10", "Test on the remaining data.") == "correct"


def test_weak_items_are_explained_and_no_bare_far_or_zero_remains() -> None:
    assert all(item["flags"] for item in ANSWERS.values() if item["support"] == "weak")
    variants = {v for item in ANSWERS.values() for fact in item["required_facts"] for v in fact}
    assert "far" not in variants
    assert "zero" not in variants
