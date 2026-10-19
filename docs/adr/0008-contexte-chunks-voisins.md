# ADR 0008 — Contexte du LLM étendu aux chunks voisins (expérience E6a)

- **Statut** : accepté (défaut `DOCASSIST_CONTEXT_NEIGHBORS=1`)
- **Date** : 2026-10-19

## Constat (run de référence `real_v1__gen__gemini-3.5-flash-lite__run1`, top-5 seul)

- Q05 (abstention injustifiée) : le chunk envoyé s'arrête sur « …the simplest way to permit
  errors in the maximum margin linear classifier is to » ; la suite (« introduce “slack”
  variables ») est dans le chunk suivant, non envoyé.
- Q08 et Q14 (`retrieval_failure_answered`) : l'information demandée n'était pas dans le
  contexte ; elle termine le chunk précédent (Q08 : « requires inverting a n × n matrix ») ou la
  page précédente (Q14 : 60 % / 40 %, p. 38, alors que la p. 39 a été envoyée). Le modèle a
  répondu avec une information voisine (bénéfice des noyaux ; exemple 80/20).
- Q15 (`retrieval_failure_abstained`) : « elbow » et « silhouette » sont dans le chunk qui suit
  celui envoyé ; l'abstention était correcte au vu du contexte.
- Hors ligne, ajouter les voisins (±1, même document) au top-5 fait passer les questions
  répondables ayant une référence dans le contexte de 12/15 à 15/15.

## Décision

- `QAService` ajoute à chaque passage retrouvé ses `DOCASSIST_CONTEXT_NEIGHBORS` chunks voisins
  du même document, lus dans l'index (métadonnées et citations réelles). Défaut : **1**.
  `0` rétablit le contexte du run de référence.
- Inchangés : classement de la recherche (top-5 identique au run de référence pour les 20
  questions, vérifié hors ligne), embeddings, découpage, index, prompt et modèle.
- La valeur est enregistrée dans la configuration des enregistrements `generate-answers` quand
  elle est non nulle ; une reprise refuse de mélanger deux configurations.

## Résultats E6a (`real_v1__gen__gemini-3.5-flash-lite__e6a_neighbors1`, une seule variable)

Scores **automatiques par mots-clés, non relus par un humain** :

| | Référence (`run1`) | E6a |
|---|---|---|
| Répondables correctes | 10/15 | 14/15 |
| Abstentions sur répondables | 2 | 0 |
| Abstentions correctes (sans réponse) | 5/5 | 5/5 |
| Citations invalides | 0 / 15 | 0 / 18 |
| Précision des citations vs références | 0,80 | 1,00 |

Q05, Q08, Q14 et Q15 passent en succès ; aucune question ne régresse. Le critère fixé avant
l'expérience (aucune abstention perdue sur les questions sans réponse, plus de réponses
correctes, aucune régression par question) est rempli.

## Limites

- Une seule exécution par configuration d'un modèle non déterministe ; 20 questions, rédigées
  par l'assistant de développement ; le même jeu a servi au diagnostic et au test : l'amélioration
  reste à confirmer sur un jeu indépendant et après relecture humaine.
- Q10 reste incorrecte : les passages décrivant la procédure leave-one-out (lecture 2, p. 3-4) ne
  sont pas voisins d'un passage retrouvé ; l'extension ne peut pas les apporter.
- Q14 reste ambiguë dans le jeu (exemples 60/40 et 80/20 tous deux présents dans le contexte).
- Contexte plus long : 9 à 15 chunks par question au lieu de 5 (≈ 2,3 fois plus de texte),
  donc davantage de jetons par requête et un risque de dilution sur d'autres documents.
