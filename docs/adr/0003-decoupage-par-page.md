# ADR 0003 — Découpage page par page, tailles en caractères

- **Statut** : accepté
- **Date** : 2026-10-19

## Contexte

Chaque réponse doit citer précisément sa source (fichier et page). Compter la taille des
chunks en tokens exigerait le tokenizer du modèle d'embeddings, donc un téléchargement.

## Décision

- `RecursiveChunker` découpe chaque page séparément avec `RecursiveCharacterTextSplitter`
  (langchain-text-splitters) : un chunk appartient toujours à une seule page.
- Les tailles sont en caractères (800 par défaut, chevauchement de 100), soit environ
  150 à 200 tokens, bien en dessous de la limite de 512 tokens des modèles e5.
- Chaque chunk conserve `start_char`, sa position dans le texte de sa page.
- `doc_id` = 16 premiers caractères hexadécimaux du SHA-256 du fichier ;
  `chunk_id` = `"{doc_id}:{chunk_index}"`. Les identifiants sont déterministes : réimporter
  le même fichier avec les mêmes paramètres produit les mêmes identifiants.

## Conséquences

- ➕ Citation de page exacte, et extrait localisable grâce à `start_char`.
- ➕ Aucun modèle requis pour découper.
- ➖ Une phrase à cheval sur deux pages est coupée en deux chunks.
- ➖ Le nombre de tokens par chunk varie selon la langue et le vocabulaire.
- ➖ Changer `chunk_size` ou `chunk_overlap` change les `chunk_id` : il faudra réindexer.
