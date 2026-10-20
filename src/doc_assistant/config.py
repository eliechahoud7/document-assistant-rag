"""Configuration centralisée, lue depuis les variables d'environnement.

Toutes les variables sont préfixées par ``DOCASSIST_``. La phase 0 n'utilise que la
bibliothèque standard ; voir docs/adr/0002-configuration-stdlib.md.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ENV_PREFIX = "DOCASSIST_"

LLMProvider = Literal["none", "gemini", "ollama", "anthropic"]
_LLM_PROVIDERS: tuple[str, ...] = ("none", "gemini", "ollama", "anthropic")
QueryTranslation = Literal["none", "fr-en"]
_QUERY_TRANSLATIONS: tuple[str, ...] = ("none", "fr-en")
_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# Contrainte de ChromaDB sur les noms de collection.
_COLLECTION_NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]")


class ConfigError(ValueError):
    """Valeur de configuration invalide."""


@dataclass(frozen=True, slots=True)
class Settings:
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    embedding_model: str = "intfloat/multilingual-e5-small"
    # Commit figé du dépôt Hugging Face, pour des embeddings reproductibles.
    embedding_model_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    chroma_collection: str = "chunks"
    # "none" par défaut : aucun appel à une API tant que l'utilisateur ne l'a pas choisi.
    llm_provider: LLMProvider = "none"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_timeout_s: int = 30
    # Nouvelles tentatives sur erreurs serveur ou réseau uniquement, jamais sur quota (429).
    llm_max_retries: int = 1
    llm_max_output_tokens: int = 2048
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_top_k: int = 5
    # Chunks voisins ajoutés au contexte du LLM (1 = configuration retenue après E6a, ADR 0008 ;
    # 0 = contexte réduit au top-k, comportement du run de référence).
    context_neighbors: int = 1
    # Traduction des questions françaises vers l'anglais, pour la recherche uniquement
    # ("none" par défaut : comportement inchangé). Modèle local, révision figée.
    query_translation: QueryTranslation = "none"
    translation_model: str = "Helsinki-NLP/opus-mt-fr-en"
    translation_model_revision: str = "c4aed37b318c763fd177aa449b44e3b783cc6c02"

    def __post_init__(self) -> None:
        if not 0 <= self.context_neighbors <= 3:
            raise ConfigError("context_neighbors doit être compris entre 0 et 3")
        if not self.embedding_model:
            raise ConfigError("embedding_model ne peut pas être vide")
        if self.embedding_batch_size <= 0:
            raise ConfigError("embedding_batch_size doit être strictement positif")
        if not _COLLECTION_NAME_RE.fullmatch(self.chroma_collection):
            raise ConfigError(
                "chroma_collection doit contenir 3 à 512 caractères parmi [a-zA-Z0-9._-], "
                f"commencer et finir par une lettre ou un chiffre ; reçu {self.chroma_collection!r}"
            )
        if self.log_level not in _LOG_LEVELS:
            raise ConfigError(f"log_level doit être parmi {_LOG_LEVELS}, reçu {self.log_level!r}")
        if self.llm_provider not in _LLM_PROVIDERS:
            raise ConfigError(
                f"llm_provider doit être parmi {_LLM_PROVIDERS}, reçu {self.llm_provider!r}"
            )
        if not self.llm_model:
            raise ConfigError("llm_model ne peut pas être vide")
        if self.llm_timeout_s <= 0:
            raise ConfigError("llm_timeout_s doit être strictement positif")
        if self.llm_max_retries < 0:
            raise ConfigError("llm_max_retries doit être positif ou nul")
        if self.llm_max_output_tokens <= 0:
            raise ConfigError("llm_max_output_tokens doit être strictement positif")
        if self.chunk_size <= 0:
            raise ConfigError("chunk_size doit être strictement positif")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("chunk_overlap doit vérifier 0 <= chunk_overlap < chunk_size")
        if self.retrieval_top_k <= 0:
            raise ConfigError("retrieval_top_k doit être strictement positif")
        if self.query_translation not in _QUERY_TRANSLATIONS:
            raise ConfigError(
                f"query_translation doit être parmi {_QUERY_TRANSLATIONS}, "
                f"reçu {self.query_translation!r}"
            )
        if not self.translation_model:
            raise ConfigError("translation_model ne peut pas être vide")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def embedding_model_dir(self) -> Path:
        """Dossier local du modèle : ``models/multilingual-e5-small`` par défaut."""
        return self.models_dir / self.embedding_model.rsplit("/", 1)[-1]

    @property
    def translation_model_dir(self) -> Path:
        """Dossier local du modèle de traduction : ``models/opus-mt-fr-en`` par défaut."""
        return self.models_dir / self.translation_model.rsplit("/", 1)[-1]

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Construit la configuration ; les variables absentes gardent leur valeur par défaut."""
        env = os.environ if environ is None else environ
        defaults = cls()

        def get(name: str, default: str) -> str:
            return env.get(ENV_PREFIX + name, default).strip()

        def get_int(name: str, default: int) -> int:
            raw = get(name, str(default))
            try:
                return int(raw)
            except ValueError as exc:
                raise ConfigError(f"{ENV_PREFIX}{name} doit être un entier, reçu {raw!r}") from exc

        return cls(
            log_level=get("LOG_LEVEL", defaults.log_level).upper(),
            data_dir=Path(get("DATA_DIR", str(defaults.data_dir))),
            models_dir=Path(get("MODELS_DIR", str(defaults.models_dir))),
            embedding_model=get("EMBEDDING_MODEL", defaults.embedding_model),
            embedding_model_revision=get(
                "EMBEDDING_MODEL_REVISION", defaults.embedding_model_revision
            ),
            embedding_device=get("EMBEDDING_DEVICE", defaults.embedding_device).lower(),
            embedding_batch_size=get_int("EMBEDDING_BATCH_SIZE", defaults.embedding_batch_size),
            chroma_collection=get("CHROMA_COLLECTION", defaults.chroma_collection),
            llm_provider=cast(LLMProvider, get("LLM_PROVIDER", defaults.llm_provider).lower()),
            llm_model=get("LLM_MODEL", defaults.llm_model),
            llm_timeout_s=get_int("LLM_TIMEOUT_S", defaults.llm_timeout_s),
            llm_max_retries=get_int("LLM_MAX_RETRIES", defaults.llm_max_retries),
            llm_max_output_tokens=get_int("LLM_MAX_OUTPUT_TOKENS", defaults.llm_max_output_tokens),
            chunk_size=get_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=get_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            retrieval_top_k=get_int("RETRIEVAL_TOP_K", defaults.retrieval_top_k),
            context_neighbors=get_int("CONTEXT_NEIGHBORS", defaults.context_neighbors),
            query_translation=cast(
                QueryTranslation, get("QUERY_TRANSLATION", defaults.query_translation).lower()
            ),
            translation_model=get("TRANSLATION_MODEL", defaults.translation_model),
            translation_model_revision=get(
                "TRANSLATION_MODEL_REVISION", defaults.translation_model_revision
            ),
        )
