# ADR 0007 — Évaluation de la génération : enregistrement en ligne, notation hors ligne

- **Statut** : accepté
- **Date** : 2026-10-19

## Contexte

Mesurer la justesse des réponses, l'exactitude des citations, l'abstention et la part des
échecs imputables à la recherche ou à la génération, avec un budget de 0 €, un modèle non
déterministe et un quota gratuit limité.

## Décision

- **Deux étapes** : `generate-answers --live` (seule étape qui appelle le LLM) enregistre une
  ligne JSON par question (passages envoyés avec leur texte, réponse, citations, jetons) ;
  `score-answers` note cet enregistrement hors ligne, de façon déterministe.
- **Reprise** : sauvegarde après chaque question ; une relance ignore les questions réussies,
  retente les erreurs ; arrêt propre sur quota (429, jamais retenté), code de sortie 3.
  Une relance refuse un fichier issu d'un autre jeu ou d'une autre configuration.
- **Même recherche que l'évaluation de la recherche** : index temporaire construit par
  `index_dataset()` (extraite de `run_evaluation`, rapports de recherche inchangés).
- **Référence** : annexe `real_v1_answers.json` (réponse attendue + faits requis avec variantes
  FR/EN), liée au SHA-256 du jeu ; chaque fait doit figurer près d'une citation de référence,
  sinon l'annexe est refusée. Les éléments fragiles sont marqués `weak` avec une explication.
- **Correction = indicateur approximatif** (présence des faits requis), corrigeable par des
  verdicts humains (`--overrides`) ; une fiche de relecture Markdown est produite.
- **Attribution des échecs** à partir du passage de référence présent ou non dans le top-k :
  `success`, `citation_failure`, `generation_failure`, `unjustified_abstention`,
  `retrieval_failure_abstained`, `retrieval_failure_answered` ; sans réponse :
  `correct_abstention`, `unsupported_answer`.
- **Exclu** : LLM juge (quota, non-déterminisme, auto-évaluation par le même modèle).

## Conséquences

- ➕ Notation rejouable sans réseau ; tests entièrement hors ligne (générateurs scriptés).
- ➕ Un quota épuisé n'oblige pas à tout refaire.
- ➖ Les mots-clés ratent des paraphrases et peuvent accepter une réponse fausse contenant le
  bon mot : la relecture humaine reste nécessaire.
- ➖ Un enregistrement = un échantillon d'un modèle non déterministe ; 15 + 5 questions.
- ➖ Les enregistrements bruts (`*.jsonl`, extraits des PDF et réponses) restent locaux
  (ignorés par Git) ; seuls les rapports notés et les fiches sont destinés à être versionnés.
