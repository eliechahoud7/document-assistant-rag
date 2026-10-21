"""Exceptions métier."""


class IngestionError(Exception):
    """Erreur lors de l'import d'un document."""


class InvalidPDFError(IngestionError):
    """Le fichier n'est pas un PDF lisible (corrompu, chiffré, autre format)."""


class EmptyDocumentError(IngestionError):
    """Le document ne contient aucun texte extractible (PDF vide ou scanné)."""


class UploadRejectedError(IngestionError):
    """Fichier envoyé refusé avant indexation (nom, taille ou format invalides)."""


class LibraryClosedError(UploadRejectedError):
    """La bibliothèque a été supprimée (suppression du compte) pendant ou avant l'envoi : rien
    n'y est ajouté."""


class DocumentRegistryError(Exception):
    """Registre des documents illisible ou impossible à écrire (il n'est jamais écrasé)."""


class DocumentNotFoundError(Exception):
    """Aucun document de la bibliothèque ne correspond à cet identifiant."""


class DocumentProtectedError(Exception):
    """Document protégé (document de démonstration) : il ne peut pas être supprimé."""


class AuthenticationError(Exception):
    """Identité absente ou incomplète : aucune bibliothèque personnelle n'est ouverte."""


class DocumentScopeError(Exception):
    """Un passage d'un autre document a été retrouvé alors qu'un seul document était demandé
    (garde-fou : rien n'est alors envoyé au LLM)."""


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
