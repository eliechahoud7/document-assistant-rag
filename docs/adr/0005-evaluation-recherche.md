# ADR 0005 — Évaluation de la recherche par citations exactes

- **Statut** : accepté
- **Date** : 2026-10-19

## Contexte

Avant d'ajouter un LLM, il faut mesurer la recherche de façon reproductible, sans que les
références dépendent du découpage ni du moteur évalué, et distinguer « retrouver le bon
passage » de « reconnaître qu'il n'y a pas de réponse ».

## Décision

- Références = citations exactes (`filename`, `page`, `quote`), validées au chargement
  (présentes exactement une fois). Pertinence = couverture ≥ 50 % des caractères non blancs
  de la citation par un chunk de la même page.
- Métriques sur les questions répondables : Recall@1/3/5, Hit@1/3/5, MRR, ventilées par
  catégorie et par paire de langues. Questions sans réponse : AUROC du score du meilleur
  passage (sens vérifié par test : score élevé = répondable), sans choix de seuil.
- Commande `evaluate` : index ChromaDB temporaire, refus d'un index non vide, rapport JSON
  avec configuration, versions et SHA-256 du jeu ; référence versionnée et test `slow` de
  non-régression.
- `IndexingService.index_document()` permet d'indexer des documents synthétiques sans PDF.

## Conséquences

- ➕ Le même jeu reste valable si `chunk_size` / `chunk_overlap` changent : comparaison
  honnête de configurations.
- ➕ Deux exécutions donnent des rapports identiques (hors horodatage).
- ➖ L'annotation demande de copier le texte tel qu'extrait et nettoyé (ligatures, césures).
- ➖ Le jeu `synthetic_v1` est petit et auto-rédigé ; un jeu réel annoté reste nécessaire.
