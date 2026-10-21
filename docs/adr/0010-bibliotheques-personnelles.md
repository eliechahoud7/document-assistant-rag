# ADR 0010 — Bibliothèques personnelles persistantes (connexion Google, Chroma Cloud)

- **Statut** : proposé — implémenté et testé hors ligne, **désactivé par défaut**
  (`DOCASSIST_USER_LIBRARIES=false`) ; non déployé
- **Date** : 2026-10-21 (tarifs, région et quotas de Chroma Cloud vérifiés ce jour sur ses pages
  officielles)
- **Remplace** pour l'usage public : les envois temporaires par session
  ([ADR 0009](0009-envois-prives-des-visiteurs.md)), qui ne survivent ni à 2 heures d'inactivité,
  ni à un redémarrage, ni au changement d'appareil. Les deux modes sont incompatibles.

## Contexte

Objectif : chaque utilisateur retrouve ses PDF après déconnexion, plus tard, et sur un autre
appareil, sans jamais voir, interroger ni supprimer ceux d'un autre. Le disque de Streamlit
Community Cloud est effacé à chaque redémarrage ou mise en veille : il faut une identité
vérifiée et un stockage hors du serveur de l'application.

## Décision

- **Connexion** : `st.login` / `st.user` / `st.logout` de Streamlit (OpenID Connect, Google),
  inscription ouverte. Le serveur Streamlit vérifie la connexion et conserve l'identité dans un
  cookie signé avec `[auth].cookie_secret` (itsdangerous, 30 jours) : le navigateur ne peut pas la
  forger. Exige Authlib (`streamlit[auth]`).
- **Identité** : `UserIdentity.from_claims` (`services/user_libraries.py`) ; clé = début du
  SHA-256 de l'émetteur (`iss`) et de l'identifiant stable du compte (`sub`). Jamais l'e-mail
  (qui peut changer), jamais une valeur de `st.session_state`, des paramètres d'adresse ou d'un
  widget. `web_identity.current_user()` est le seul point de lecture de `st.user`, relu à
  chaque exécution de la page ; une identité incomplète n'ouvre aucune bibliothèque.
- **Stockage** : Chroma Cloud, même API et même adaptateur `ChromaVectorStore` qu'en local
  (`chromadb.CloudClient`), base créée en région **GCP europe-west1** (données stockées et
  traitées dans cette région, d'après Chroma). Identifiants `CHROMA_API_KEY`, `CHROMA_TENANT`,
  `CHROMA_DATABASE` dans les Secrets uniquement. `DOCASSIST_USER_STORE=local` : même code sur des
  dossiers locaux (une base par utilisateur, `<DOCASSIST_DATA_DIR>/users/<clé>`), pour le
  développement et les tests.
- **Isolation par construction** : une collection par utilisateur, nommée d'après sa clé
  (`user-<clé>-chunks`) ; l'enregistrement de chaque document (nom, pages, date) est une
  métadonnée de son premier passage. Une `UserLibrary` n'ouvre que la sienne ; ajout, liste,
  recherche (`Retriever` sur son seul store), suppression et service de réponses passent par
  elle. Aucune fonction n'accepte un nom de collection ou une clé venant de l'interface ; un
  `doc_id` d'un autre utilisateur est introuvable. Les conversations sont effacées dès que
  l'identité change dans un même navigateur.
- **Défaut du moteur local de ChromaDB 1.5.9** (diagnostiqué le 2026-10-21, reproduit sans le
  code du projet) : écrire dans une collection puis dans une autre de la même base peut rendre la
  recherche de la première impossible (`InternalError : Error creating hnsw segment reader:
  Nothing found on disk`) jusqu'à sa prochaine écriture ; aucune donnée perdue, jamais de
  résultat d'une autre collection. D'où un registre sans collection à part (un ajout n'écrit que
  dans la collection de l'utilisateur) et, en local, une base par utilisateur. Pas de nouvelle
  tentative automatique : elle ne corrige rien (mesuré). Chroma Cloud utilise un autre moteur,
  non vérifiable hors ligne : test de charge obligatoire sur une base Chroma Cloud de test avant
  ouverture.
- **Aucun PDF conservé** : seuls les passages, leurs vecteurs et le registre (nom, pages, date)
  sont stockés ; le PDF reçu ne quitte pas un dossier temporaire.
- **Suppression du compte** : supprime la collection de l'utilisateur, et elle seule.
- **Visiteurs déconnectés** : démonstration en lecture seule, jamais l'index local ; aucune base
  ouverte pour une simple visite.
- **Création paresseuse** : la collection d'un utilisateur n'est créée qu'à son premier envoi ;
  se connecter sans rien envoyer, ou revenir après la suppression du compte, n'écrit rien dans
  la base.
- **Conditions avant tout usage** (exigence des conditions Gemini : utilisateurs de 18 ans ou
  plus, application non destinée aux mineurs) : 18 ans ou plus, aucune donnée sensible
  (article 9 du RGPD, interdite aussi par les conditions de Chroma), passages conservés par Chroma
  et envoyés à Gemini, lien vers la notice de l'exploitant (`DOCASSIST_PRIVACY_URL`,
  obligatoire). Version `TERMS_VERSION` enregistrée dans les métadonnées de la collection au
  premier envoi ; une nouvelle version est redemandée. Un simple engagement déclaratif : l'âge
  n'est pas vérifié. La suppression des documents et du compte reste toujours possible.
- **Limites (serveur)** : `DOCASSIST_UPLOAD_MAX_MB` (10, au plus `server.maxUploadSize` = 20),
  `DOCASSIST_UPLOAD_MAX_PAGES` (50), `DOCASSIST_ACCOUNT_MAX_DOCUMENTS` (20 documents par
  utilisateur) ; les fichiers en trop d'un envoi ne passent pas par le pipeline. Par utilisateur
  et par jour : `DOCASSIST_ACCOUNT_DAILY_QUESTIONS` (50, vérifié avant tout appel au LLM) et
  `DOCASSIST_ACCOUNT_DAILY_UPLOADS` (20) ; compteurs en mémoire, remis à zéro chaque jour (UTC)
  et à chaque redémarrage (garde-fou de coût, pas un quota garanti). `DOCASSIST_MAX_ACCOUNTS`
  (100) : nombre de collections de la base, compté et créé sous un même verrou ; la base Chroma
  Cloud doit donc être réservée à l'application.
- **Concurrence** : un verrou par utilisateur, partagé par tous les objets de sa bibliothèque
  (même après éviction du cache) ; la suppression du compte attend un envoi en cours, puis tout
  envoi tardif sur l'ancienne bibliothèque est refusé (`LibraryClosedError`), sans rien recréer.
- **Journaux** : jamais le nom d'un fichier envoyé (identifiant du document seulement) ni le
  message d'une erreur d'indexation (type seulement).
- **Quotas Chroma Cloud** respectés partout (local compris) : au plus 300 enregistrements par
  écriture et par lecture (lectures paginées, suppressions par lots).
- **Configuration refusée tant qu'elle est incomplète** : section `[auth]` (5 clés),
  `cookie_secret` d'au moins 32 caractères, Authlib, identifiants Chroma Cloud ; les messages ne
  contiennent jamais de valeur secrète.

## Coût (tarifs officiels du 2026-10-21)

5 $ de crédits offerts aux nouveaux comptes, puis facturation à l'usage : écritures 2,50 $/Gio,
stockage 0,33 $/Gio par mois, requêtes 0,0075 $/Tio, données renvoyées 0,09 $/Gio. Un passage
(800 caractères, vecteur de 384 dimensions, métadonnées) pèse environ 3 Ko : un PDF de 50 pages
(environ 150 passages) coûte environ 0,001 $ à écrire et moins de 0,001 $ par mois à stocker. Montants
indicatifs ; une carte peut être demandée après les crédits (à vérifier dans le tableau de bord).

## Confidentialité

- **Données personnelles stockées** : texte des documents envoyés (passages), nom de fichier,
  date ; l'identité n'est stockée que sous forme de clé dérivée (aucun e-mail dans la base).
  Responsable du traitement : l'exploitant de l'application. Sous-traitants : Chroma
  (stockage, région UE), Google (connexion ; réponses via Gemini).
- **Envoyé au LLM** : question et passages retrouvés de l'utilisateur. Une application publique
  pour des utilisateurs de l'EEE doit utiliser l'offre **payante** de Gemini (ADR 0006).
- **Conservation** : jusqu'à suppression du document ou du compte par l'utilisateur.
- **À faire avant l'ouverture** : notice de confidentialité rédigée et publiée par l'exploitant
  (finalité, base légale, données, sous-traitants, transferts, durée, droits, contact),
  signature du DPA de Chroma, conditions de traitement de Google (offre payante) et de
  Streamlit Community Cloud à vérifier.
- **Hors périmètre** : chiffrement de bout en bout, vérification de l'âge, quotas persistants
  (les compteurs quotidiens sont en mémoire), modération des contenus, sauvegardes, export des
  données de l'utilisateur.

## Conséquences

- ➕ Bibliothèque retrouvée après déconnexion, redémarrage ou sur un autre appareil.
- ➕ Isolation vérifiée par 58 tests hors ligne (identité, liste, recherche, suppression, compte,
  persistance, régression « Nothing found on disk », conditions, limites, concurrence), sur une
  vraie base ChromaDB locale, plus 4 tests unitaires des limites quotidiennes.
- ➕ Pipeline, découpage, embeddings et interface inchangés ; mode local et démonstration
  inchangés.
- ➖ Dépendance à deux services externes (Google, Chroma Cloud) et coût faible mais non nul.
- ➖ Le PDF d'origine n'est pas conservé : pas de téléchargement du fichier depuis l'interface.
- ➖ La vraie connexion Google et la vraie base Chroma Cloud ne sont pas testées hors ligne :
  vérification manuelle obligatoire avant ouverture.
