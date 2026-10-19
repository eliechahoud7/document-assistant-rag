# ADR 0001 — Architecture ports et adaptateurs

- **Statut** : accepté
- **Date** : 2026-10-19

## Contexte

Le projet doit pouvoir comparer plusieurs modèles d'embeddings, plusieurs LLM (local et
API) et éventuellement plusieurs bases vectorielles. Les tests doivent tourner sans
réseau, sans GPU et sans clé API. LangChain évolue rapidement et casse régulièrement son
API publique.

## Décision

Les services métier dépendent d'interfaces définies dans `domain/` (`typing.Protocol`).
Chaque technologie concrète (pypdf, ChromaDB, sentence-transformers, Ollama, API LLM,
LangChain) est encapsulée dans un adaptateur.

## Conséquences

- ➕ Changer de composant revient à changer d'adaptateur et de configuration.
- ➕ Les tests utilisent des implémentations factices déterministes.
- ➕ LangChain est confiné : une rupture d'API n'affecte qu'un adaptateur.
- ➖ Un peu plus de code qu'un usage direct des chaînes LangChain.
