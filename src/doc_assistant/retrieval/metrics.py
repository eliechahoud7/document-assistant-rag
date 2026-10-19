"""Métriques de recherche à partir du rang du premier passage pertinent.

Un rang vaut 1 pour le premier résultat ; ``None`` signifie « non trouvé dans le top-k ».
"""

from __future__ import annotations

from collections.abc import Sequence


def hit_rate_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Proportion de questions dont un passage pertinent figure dans les ``k`` premiers."""
    if k <= 0:
        raise ValueError("k doit être strictement positif")
    if not ranks:
        raise ValueError("aucune question à évaluer")
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Moyenne de 1/rang (0 si non trouvé) : récompense les bons passages classés en tête."""
    if not ranks:
        raise ValueError("aucune question à évaluer")
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def recall_at_k(reference_ranks: Sequence[Sequence[int | None]], k: int) -> float:
    """Part des passages de référence retrouvés dans les ``k`` premiers, moyennée par question.

    ``reference_ranks[i][j]`` est le rang du premier résultat correspondant à la référence
    ``j`` de la question ``i``. Avec une seule référence par question, Recall@k = Hit@k ;
    avec deux références, trouver l'une des deux donne 0,5 pour cette question.
    """
    if k <= 0:
        raise ValueError("k doit être strictement positif")
    if not reference_ranks:
        raise ValueError("aucune question à évaluer")
    if any(not ranks for ranks in reference_ranks):
        raise ValueError("chaque question évaluée doit avoir au moins une référence")
    per_question = [
        sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)
        for ranks in reference_ranks
    ]
    return sum(per_question) / len(per_question)


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Probabilité qu'un score positif tiré au hasard dépasse un score négatif (ex aequo = 1/2).

    Sens : les positifs doivent avoir les scores les plus élevés. 1,0 = séparation parfaite ;
    0,5 = pas mieux que le hasard ; < 0,5 = le score varie en sens inverse de l'attendu.
    Calcul exact par comptage des paires (statistique de Mann-Whitney), sans seuil.
    """
    if not positive_scores or not negative_scores:
        raise ValueError("il faut au moins un score positif et un score négatif")
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))
