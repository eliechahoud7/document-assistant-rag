# ADR 0002 — Configuration par la bibliothèque standard (phase 0)

- **Statut** : accepté, à réévaluer en phase 5
- **Date** : 2026-10-19

## Contexte

La phase 0 ne doit installer que des outils de développement. La configuration doit
néanmoins être centralisée, validée et testable dès maintenant.

## Décision

`doc_assistant.config.Settings` est une `dataclass` immuable, validée dans
`__post_init__`, construite depuis les variables d'environnement préfixées
`DOCASSIST_`. `Settings.from_env()` accepte un dictionnaire, ce qui permet de la
tester sans modifier l'environnement réel.

## Conséquences

- ➕ Zéro dépendance d'exécution ; validation explicite et lisible.
- ➖ Le fichier `.env` n'est pas chargé automatiquement.
- À la phase 5 (FastAPI, donc Pydantic déjà présent), migration vers
  `pydantic-settings`, en conservant les mêmes noms de variables.
