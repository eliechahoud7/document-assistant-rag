# Architecture

## Vue d'ensemble

```
                ┌───────────────────────────┐
                │      API (FastAPI)        │  HTTP, validation, schémas
                └────────────┬──────────────┘
                             │
                ┌────────────▼──────────────┐
                │   Services (cas d'usage)  │  IngestionService, QAService
                └────────────┬──────────────┘
                             │ dépend d'interfaces (typing.Protocol)
     ┌──────────────┬────────┴──────┬───────────────┬──────────────┐
     ▼              ▼               ▼               ▼              ▼
 DocumentLoader  Chunker        Embedder       VectorStore      LLMClient
     │              │               │               │              │
  [pypdf]   [text splitter]  [e5 / bge-m3]     [ChromaDB]   [Ollama | API]
```

## Règles de dépendance

- `domain/` ne dépend de rien d'autre que de la bibliothèque standard.
- `ingestion/`, `indexing/`, `retrieval/`, `generation/` implémentent les interfaces du
  domaine (adaptateurs) et peuvent dépendre de bibliothèques tierces.
- `services/` orchestre les adaptateurs **via les interfaces** uniquement.
- `bootstrap.py` est le seul module qui assemble les adaptateurs concrets ; la CLI
  (`cli.py`) et, plus tard, l'API l'utilisent.
- `api/` sera la seule couche qui connaît FastAPI.

## Modules (phase 2)

| Module | Rôle |
|---|---|
| `domain/ports.py` | `DocumentLoader`, `Chunker`, `Embedder`, `VectorStore` |
| `ingestion/` | `PypdfLoader`, `clean_text`, `RecursiveChunker` |
| `indexing/embedders.py` | `E5Embedder` (préfixes `query:` / `passage:`, CPU, local) |
| `indexing/vector_store.py` | `ChromaVectorStore` (cosinus, garde-fou sur le modèle) |
| `indexing/model_download.py` | téléchargement explicite du modèle (seul accès réseau) |
| `services/indexing_service.py` | PDF -> chunks -> vecteurs -> store, idempotent |
| `retrieval/retriever.py` | question -> top-k `SearchResult` |
| `retrieval/metrics.py` | Hit@k, MRR |

## Pipeline

**Indexation** : PDF → pages (avec numéro) → nettoyage → chunks avec métadonnées
(`doc_id`, `filename`, `page`, `chunk_index`, `sha256`, `embedding_model`) → vecteurs →
ChromaDB.

**Requête** : question → vecteur → top-k chunks → (reranking) → prompt avec contexte
numéroté → LLM → réponse → validation des citations `[n]`.
