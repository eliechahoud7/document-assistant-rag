import pytest

from doc_assistant.retrieval.metrics import (
    auroc,
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
)

RANKS = [1, 2, None, 4]


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(RANKS, 1) == 0.25
    assert hit_rate_at_k(RANKS, 3) == 0.5
    assert hit_rate_at_k(RANKS, 10) == 0.75


def test_mean_reciprocal_rank() -> None:
    assert mean_reciprocal_rank(RANKS) == pytest.approx((1 + 1 / 2 + 0 + 1 / 4) / 4)
    assert mean_reciprocal_rank([1, 1]) == 1.0
    assert mean_reciprocal_rank([None]) == 0.0


@pytest.mark.parametrize("call", [lambda: hit_rate_at_k([], 1), lambda: mean_reciprocal_rank([])])
def test_empty_input_is_rejected(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        hit_rate_at_k(RANKS, 0)
    with pytest.raises(ValueError):
        recall_at_k([[1]], 0)


def test_recall_with_single_reference_equals_hit_rate() -> None:
    single = [[rank] for rank in RANKS]

    for k in (1, 3, 5):
        assert recall_at_k(single, k) == hit_rate_at_k(RANKS, k)


def test_recall_counts_each_reference() -> None:
    # Question 1 : 2 références (rangs 1 et 4) ; question 2 : 1 référence non trouvée.
    ranks: list[list[int | None]] = [[1, 4], [None]]

    assert recall_at_k(ranks, 1) == pytest.approx((1 / 2 + 0) / 2)
    assert recall_at_k(ranks, 5) == pytest.approx((2 / 2 + 0) / 2)


def test_recall_rejects_questions_without_reference() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], 1)
    with pytest.raises(ValueError):
        recall_at_k([[1], []], 1)


def test_auroc_direction_higher_score_means_answerable() -> None:
    answerable, unanswerable = [0.9, 0.8], [0.3, 0.2]

    assert auroc(answerable, unanswerable) == 1.0  # bon sens : séparation parfaite
    assert auroc(unanswerable, answerable) == 0.0  # sens inversé : détecté, pas masqué


def test_auroc_overlap_and_ties() -> None:
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([0.6, 0.4], [0.5]) == 0.5
    # 3 paires gagnées, 1 égalité, 2 perdues sur 6.
    assert auroc([0.9, 0.5, 0.1], [0.5, 0.3]) == pytest.approx((3 + 0.5) / 6)


def test_auroc_requires_both_groups() -> None:
    with pytest.raises(ValueError):
        auroc([], [0.1])
    with pytest.raises(ValueError):
        auroc([0.1], [])
