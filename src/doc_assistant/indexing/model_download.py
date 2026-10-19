"""Téléchargement explicite du modèle d'embeddings : la seule étape qui utilise le réseau."""

from __future__ import annotations

from pathlib import Path

from doc_assistant.config import Settings

# Uniquement les fichiers nécessaires à sentence-transformers (~493 Mo au lieu de ~2,3 Go :
# le dépôt contient aussi les formats PyTorch .bin, ONNX et OpenVINO).
REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


def download_embedding_model(settings: Settings) -> Path:
    from huggingface_hub import snapshot_download

    target = settings.embedding_model_dir
    snapshot_download(
        repo_id=settings.embedding_model,
        revision=settings.embedding_model_revision,
        local_dir=target,
        allow_patterns=REQUIRED_FILES,
    )
    return target
