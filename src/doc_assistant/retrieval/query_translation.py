"""Traduction des questions françaises vers l'anglais, pour la recherche uniquement.

``TranslatingEmbedder`` enveloppe l'embedder : seul ``embed_query`` change, et seulement pour
une question détectée comme française. La question d'origine n'est jamais modifiée ailleurs :
le générateur, l'affichage et les enregistrements la reçoivent telle quelle. Si la traduction
échoue, la question d'origine est utilisée (recherche dégradée plutôt qu'interrompue) ; un
modèle absent reste en revanche une erreur explicite.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

from doc_assistant.domain.errors import TranslationError
from doc_assistant.domain.ports import Embedder, QueryTranslator

logger = logging.getLogger(__name__)

# Mots-outils fréquents dans les questions : une heuristique, pas un détecteur de langue général.
_FRENCH_WORDS = frozenset(
    {
        *("le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "et", "ou", "est"),
        *("sont", "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "comment"),
        *("pourquoi", "combien", "dans", "pour", "avec", "sur", "par", "entre", "sans", "ce"),
        *("cette", "ces", "son", "sa", "ses", "leur", "leurs", "il", "elle", "on", "ils"),
        *("elles", "se", "ne", "pas", "plus", "peut", "faut", "fait", "doit", "quand"),
        "lorsque",
    }
)
_ENGLISH_WORDS = frozenset(
    {
        *("the", "a", "an", "and", "or", "is", "are", "was", "were", "what", "which", "who"),
        *("whom", "how", "why", "when", "where", "does", "do", "did", "of", "in", "on", "for"),
        *("with", "by", "from", "into", "between", "without", "this", "that", "these"),
        *("those", "its", "their", "it", "they", "can", "should", "must", "has", "have"),
    }
)
_WORD_RE = re.compile(r"[^\W\d_]+")
# Élisions françaises (l', qu', n'...), apostrophe droite ou typographique (U+2019).
_ELISION_RE = re.compile(r"\b(?:l|d|qu|j|c|n|s|m)['" + chr(0x2019) + r"]\w", re.IGNORECASE)
_FRENCH_ACCENTS_RE = re.compile(r"[àâçéèêëîïôûùüÿœ]", re.IGNORECASE)


def looks_french(text: str) -> bool:
    """Vrai si la question semble française (mots-outils, élisions et accents)."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    french = sum(w in _FRENCH_WORDS for w in words) + len(_ELISION_RE.findall(text))
    french += min(2, len(_FRENCH_ACCENTS_RE.findall(text)))
    english = sum(w in _ENGLISH_WORDS for w in words)
    return french > english


class TranslatingEmbedder:
    """Embedder qui traduit les requêtes françaises avant de les encoder.

    Même modèle et même dimension que l'embedder enveloppé : l'index reste compatible
    (``ensure_same_embedding_model``) et ``embed_documents`` est inchangé.
    """

    def __init__(
        self,
        inner: Embedder,
        translator: QueryTranslator,
        *,
        is_source_language: Callable[[str], bool] = looks_french,
    ) -> None:
        self._inner = inner
        self._translator = translator
        self._is_source_language = is_source_language

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def translator_name(self) -> str:
        return self._translator.model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(self.retrieval_query(text))

    def retrieval_query(self, text: str) -> str:
        """Texte réellement encodé : la traduction, ou la question d'origine."""
        if not self._is_source_language(text):
            return text
        try:
            translated = self._translator.translate(text).strip()
        except TranslationError as exc:
            logger.warning("traduction impossible, question d'origine utilisée (%s)", exc)
            return text
        if not translated:
            logger.warning("traduction vide, question d'origine utilisée")
            return text
        return translated
