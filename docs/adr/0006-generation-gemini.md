# ADR 0006 — Génération de réponses avec l'API Gemini (niveau gratuit)

- **Statut** : accepté
- **Date** : 2026-10-19 (conditions vérifiées ce jour sur les pages officielles)

## Contexte

Phase 4 : rédiger des réponses citées à partir des passages retrouvés, avec un budget de 0 €,
sans modifier la recherche (e5-small, 800/100, ChromaDB, top-5).

## Décision

- **Fournisseur** : API Gemini, modèle `gemini-3.5-flash-lite` (stable, sans date d'arrêt
  annoncée), SDK `google-genai`. Mistral écarté : sur son offre gratuite, les entrées et
  sorties peuvent servir à l'entraînement (désactivable).
- **Données (EEE)** : selon les conditions Gemini, pour un utilisateur de l'EEE, de la Suisse
  ou du Royaume-Uni, les règles des services payants s'appliquent aussi au quota gratuit :
  Google n'utilise pas les requêtes ni les réponses pour améliorer ses produits et ne les
  conserve que pour une durée limitée afin de détecter les abus. Seuls les passages retrouvés
  (top-5) sont envoyés, depuis un index de cours séparé (`DOCASSIST_DATA_DIR=data\courses`).
- **Limite contractuelle** : « You may use only Paid Services when making API Clients
  available to users in the European Economic Area » — usage local et personnel uniquement ;
  un déploiement public pour des utilisateurs de l'EEE exigerait l'offre payante.
- **Clé** : lue uniquement dans `GEMINI_API_KEY`, transmise explicitement au client (le SDK
  donnerait sinon priorité à `GOOGLE_API_KEY`), jamais stockée, journalisée ni affichée.
- **Ancrage** : sortie JSON imposée `{abstain, answer, cited_passages}` ; citations
  reconstruites à partir des chunks réellement fournis ; numéros invalides écartés ; réponse
  sans citation valide transformée en abstention ; aucun appel sans passage. Pas de seuil de
  similarité (phase 3 et E3a : scores non séparables et dépendants de la langue).
- **Erreurs** : 429 → `LLMQuotaError`, sans nouvelle tentative (préserve le quota) ; 5xx et
  réseau → au plus `DOCASSIST_LLM_MAX_RETRIES` nouvelles tentatives ; clé refusée, modèle
  introuvable → `LLMConfigurationError` ; réponse bloquée, tronquée ou mal formée →
  `LLMResponseError`. Délai : `DOCASSIST_LLM_TIMEOUT_S`.
- **Par défaut** `DOCASSIST_LLM_PROVIDER=none` : aucun appel tant que l'utilisateur ne
  l'active pas ; les autres commandes n'en dépendent pas.

## Conséquences

- ➕ Coût nul tant que le projet AI Studio n'a pas de facturation associée (dépassement →
  erreur 429, réinitialisation quotidienne à minuit, heure du Pacifique).
- ➖ Quotas gratuits non publiés : visibles uniquement dans AI Studio, par projet.
- ➖ Réponses non déterministes (température par défaut du modèle conservée).
- ➖ Qualité des réponses et de l'abstention non encore mesurée (évaluation de la génération à
  faire) ; elle est bornée par la recherche (Hit@5 = 0,80 sur `real_v1`).
- ➖ Les conditions d'utilisation gratuites peuvent changer : à revérifier avant usage.
