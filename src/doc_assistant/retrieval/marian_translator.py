"""Adaptateur de traduction MarianMT (Opus-MT, ``transformers``, CPU, fichiers locaux seulement)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from doc_assistant.config import Settings
from doc_assistant.domain.errors import TranslationError, TranslationModelNotFoundError

# Fichiers indispensables au chargement (voir indexing/model_download.py pour le téléchargement).
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
)
# Une question tient largement dans cette longueur ; au-delà, l'entrée est tronquée. La
# longueur de sortie est bornée par generation_config.json du modèle (max_length = 512) :
# la redéfinir ici déclencherait un avertissement de transformers à chaque traduction.
MAX_INPUT_TOKENS = 256


class MarianTranslator:
    def __init__(
        self,
        model_dir: Path,
        model_name: str,
        *,
        device: str = "cpu",
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        self.model_dir = model_dir
        self._model_name = model_name
        self.device = device
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> MarianTranslator:
        """Vérifie dès la construction que le modèle est présent (erreur explicite sinon)."""
        translator = cls(
            settings.translation_model_dir,
            settings.translation_model,
            device=settings.embedding_device,
        )
        translator.check_files()
        return translator

    @property
    def model_name(self) -> str:
        return self._model_name

    def check_files(self) -> None:
        missing = [name for name in REQUIRED_MODEL_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise TranslationModelNotFoundError(
                f"modèle de traduction introuvable dans {self.model_dir} (fichiers manquants : "
                f"{', '.join(missing)}). Téléchargez-le avec : python -m doc_assistant "
                "download-translation-model, ou désactivez la traduction "
                "(DOCASSIST_QUERY_TRANSLATION=none)"
            )

    def translate(self, text: str) -> str:
        tokenizer, model = self._load()
        try:
            batch = tokenizer(
                [text], return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
            ).to(self.device)
            output = model.generate(**batch)
            translated: str = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as exc:  # modèle ou tokenizer : erreurs variées, toutes récupérables
            raise TranslationError(f"échec de la traduction ({type(exc).__name__})") from exc
        return translated.strip()

    def _load(self) -> tuple[Any, Any]:
        """Charge le modèle au premier usage (l'import de ``transformers`` est lent)."""
        if self._tokenizer is None or self._model is None:
            self.check_files()
            # Garantie supplémentaire : la bibliothèque Hugging Face ne contacte pas le Hub.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import MarianMTModel, MarianTokenizer

            self._tokenizer = MarianTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            model: Any = MarianMTModel.from_pretrained(str(self.model_dir), local_files_only=True)
            self._model = model.to(self.device).eval()
        return self._tokenizer, self._model
