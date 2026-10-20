"""Exceptions métier."""


class IngestionError(Exception):
    """Erreur lors de l'import d'un document."""


class InvalidPDFError(IngestionError):
    """Le fichier n'est pas un PDF lisible (corrompu, chiffré, autre format)."""


class EmptyDocumentError(IngestionError):
    """Le document ne contient aucun texte extractible (PDF vide ou scanné)."""


class EmbeddingModelNotFoundError(Exception):
    """Le modèle d'embeddings n'est pas présent localement."""


class EmbeddingModelMismatchError(Exception):
    """L'index a été construit avec un autre modèle d'embeddings (vecteurs incompatibles)."""


class TranslationModelNotFoundError(Exception):
    """La traduction des requêtes est activée mais son modèle n'est pas présent localement."""


class TranslationError(Exception):
    """La traduction d'une requête a échoué (la requête d'origine est alors utilisée)."""


class LLMError(Exception):
    """Erreur liée à la génération de réponses par un LLM."""


class LLMConfigurationError(LLMError):
    """Fournisseur, modèle ou clé API absents ou refusés par le fournisseur."""


class LLMUnavailableError(LLMError):
    """Le service LLM n'a pas pu répondre (réseau, erreur serveur)."""


class LLMQuotaError(LLMUnavailableError):
    """Quota ou limite de débit atteint (HTTP 429)."""


class LLMTimeoutError(LLMUnavailableError):
    """Le service LLM n'a pas répondu dans le délai imparti."""


class LLMResponseError(LLMError):
    """Réponse inexploitable : bloquée, tronquée ou mal formée."""
