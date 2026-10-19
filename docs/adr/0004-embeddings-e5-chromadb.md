# ADR 0004 — Embeddings E5 locaux, ChromaDB, garde-fou sur le modèle

- **Statut** : accepté
- **Date** : 2026-10-19

## Contexte

Recherche sémantique en français et en anglais, sur un portable sans GPU NVIDIA
(i7-1165G7, 16 Go), sans envoyer de document à un service externe.

## Décision

- **Modèle** : `intfloat/multilingual-e5-small` (MIT, 384 dimensions, 512 tokens),
  exécuté avec sentence-transformers et torch CPU. Révision figée (commit `614241f6`),
  fichiers stockés dans `models/` (≈ 470 Mio : safetensors, tokenizer, configuration).
- **Préfixes E5** : `"query: "` pour les questions, `"passage: "` pour les chunks, gérés dans
  l'adaptateur. Le port `Embedder` distingue `embed_query` et `embed_documents`.
- **Hors ligne par construction** : téléchargement par une commande explicite
  (`download-model`) ; à l'exécution, chargement local uniquement (`local_files_only`,
  `HF_HUB_OFFLINE=1`). Vérifié avec les connexions réseau bloquées.
- **ChromaDB** persistant (`data/chroma/`), distance cosinus, télémétrie désactivée,
  vecteurs toujours fournis par l'application (`embedding_function=None`) : la fonction
  par défaut de Chroma téléchargerait un autre modèle.
- **Garde-fou** : le nom et la dimension du modèle sont enregistrés dans les métadonnées de
  la collection ; une réouverture avec un autre modèle lève `EmbeddingModelMismatchError`.
  Chaque chunk porte aussi `embedding_model`. Le nom vient de l'embedder qui produit
  réellement les vecteurs, pas de la configuration.
- **Idempotence** : `upsert` sur des `chunk_id` déterministes ; un PDF déjà indexé
  (même SHA-256) est ignoré, `--force` le remplace.

## Mesures (2026-10-19)

- Vitesse : ≈ 16 passages de 800 caractères par seconde sur CPU ; chargement ≈ 13 s.
- Test de fumée (12 questions, dont 7 interlingues) : E5 Hit@1 = 1,00, MRR = 1,00 ;
  base lexicale MRR = 0,22. Corpus trop facile pour comparer des modèles (phase 3).
- Les similarités cosinus d'E5 sont resserrées (≈ 0,73–0,75 pour un passage sans rapport,
  ≈ 0,85 pour le bon) : le classement est fiable, la valeur absolue ne doit pas servir de
  seuil sans calibration.

## Conséquences

- ➕ Aucune donnée ne quitte la machine ; coût nul.
- ➖ ≈ 1,4 Go dans le venv (torch, scipy, chromadb et ses dépendances serveur inutilisées).
- ➖ Changer de modèle impose une nouvelle collection et une réindexation complète.
