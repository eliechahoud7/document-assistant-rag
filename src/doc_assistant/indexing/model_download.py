"""Téléchargement explicite des modèles locaux (embeddings, traduction) : seule étape réseau."""

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


# Modèle de traduction Opus-MT : poids safetensors uniquement (~304 Mo ; le dépôt contient
# aussi les formats PyTorch .bin, TensorFlow et Flax, ~900 Mo de doublons).
REQUIRED_TRANSLATION_FILES = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "source.spm",
    "target.spm",
    "vocab.json",
]


def download_translation_model(settings: Settings) -> Path:
    from huggingface_hub import snapshot_download

    target = settings.translation_model_dir
    snapshot_download(
        repo_id=settings.translation_model,
        revision=settings.translation_model_revision,
        local_dir=target,
        allow_patterns=REQUIRED_TRANSLATION_FILES,
    )
    return target


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
