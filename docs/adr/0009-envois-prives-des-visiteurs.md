# ADR 0009 — Envois privés des visiteurs : isolation, stockage temporaire, conservation

- **Statut** : proposé — implémenté, **désactivé par défaut et non activé sur le déploiement**
  (bloqué par la décision sur Gemini, voir [ADR 0006](0006-generation-gemini.md))
- **Date** : 2026-10-20
- **Remplacé pour l'usage public par** [ADR 0010](0010-bibliotheques-personnelles.md)
  (bibliothèques personnelles persistantes) ; conservé, désactivé par défaut

## Contexte

Le déploiement public (mode démonstration, `DOCASSIST_DEMO_DOCS_DIR`) ne propose que deux cours
MIT OCW, en lecture seule. Permettre à un visiteur d'interroger son propre PDF pose quatre
problèmes : un visiteur ne doit jamais voir ni modifier les documents d'un autre ; les documents
de démonstration doivent rester intacts ; le disque et le CPU du serveur sont limités (CPU
seulement, embeddings e5-small) ; les passages des PDF envoyés partent chez le fournisseur LLM.

## Décision

- **Activation explicite** : `DOCASSIST_SESSION_UPLOADS=true`, refusé (erreur de configuration
  affichée, rien n'est créé) hors mode démonstration. Par défaut : désactivé. Les secrets du
  déploiement actuel ne le définissent pas.
- **Création à la demande** : un visiteur qui ne fait que consulter interroge la bibliothèque de
  démonstration partagée, en lecture seule (tous ses documents sont protégés). Sa bibliothèque
  privée n'est créée qu'à son **premier envoi** : une simple visite (onglet, robot, sonde de
  santé) ne copie aucun index et n'ouvre aucun client ChromaDB.
- **Isolation par session** : la bibliothèque privée est créée dans
  `<DOCASSIST_DATA_DIR>/sessions/<32 caractères hexadécimaux>/` (nom tiré par
  `secrets.token_hex(16)`, jamais affiché) et commence par une copie de l'index et du registre
  de démonstration. Ajouts, suppressions et questions ne portent alors que sur ce dossier ;
  l'index de démonstration partagé n'est plus jamais modifié après son amorçage. La bibliothèque
  est rattachée à la session Streamlit (côté serveur), pas à un identifiant transmis par le
  navigateur.
- **Plafond de sessions** : au plus `DOCASSIST_MAX_SESSIONS` (20) bibliothèques privées à la
  fois, comptées sur le disque après nettoyage des inactives ; la place est réservée (dossier
  créé) avant la copie, sous le même verrou que le comptage. Plafond atteint : l'envoi est
  refusé avec un message (« réessayez dans quelques minutes ») et le visiteur garde la
  démonstration ; rien n'est créé.
- **Documents de démonstration protégés** : marqués « Démo », sans bouton de suppression, et
  `DocumentLibrary.delete` les refuse (`DocumentProtectedError`) quelle que soit l'interface.
- **Limites vérifiées côté serveur** (`DocumentLibrary`, avant extraction et embeddings) :
  `DOCASSIST_UPLOAD_MAX_MB` (10), `DOCASSIST_UPLOAD_MAX_PAGES` (50), `DOCASSIST_UPLOAD_MAX_FILES`
  (3 documents ajoutés par session ; les documents de démonstration ne comptent pas). Un envoi
  de plus de `DOCASSIST_UPLOAD_MAX_FILES` fichiers n'en traite que les premiers ; les autres
  sont refusés, nommés dans un message, sans passer par le pipeline. Valeurs : entiers
  strictement positifs en chiffres ASCII, sinon erreur de configuration. Une seule indexation
  à la fois dans le processus.
- **Réception par le serveur** : la limite du composant d'envoi n'est vérifiée que par le
  navigateur ; Streamlit n'applique, pendant la réception, que `server.maxUploadSize` (200 Mo
  par défaut). `.streamlit/config.toml` la fixe à **20 Mo**, la plus grande limite de
  l'application (envois locaux) ; `DOCASSIST_UPLOAD_MAX_MB` au-delà de cette valeur est refusé
  au démarrage (erreur de configuration). Un fichier plus gros est rejeté par Streamlit avant
  d'atteindre l'application ; entre 10 et 20 Mo, il est reçu puis refusé par `DocumentLibrary`.
- **Unité de taille** : 1 Mo = 1 024 × 1 024 octets partout. Le composant d'envoi de Streamlit
  compte ainsi dans un navigateur sous Windows et compte 1 000 000 octets ailleurs ; la limite du
  serveur n'est donc jamais plus stricte que celle du navigateur (pas de fichier accepté par le
  navigateur puis refusé).
- **Conservation** : une bibliothèque inactive depuis plus de **2 heures** est supprimée (dossier
  entier : index, PDF, registre). Chaque interaction de la session repousse l'échéance. Le
  nettoyage a lieu à chaque création de bibliothèque privée et au plus toutes les 10 minutes
  sinon. Une session dont la bibliothèque a disparu revient à la démonstration partagée, avec
  une conversation vide et un message « Votre session a expiré » ; une nouvelle bibliothèque ne
  sera créée qu'à son prochain envoi. Sur Streamlit Community Cloud, le disque est en outre
  effacé à chaque redémarrage ou mise en veille de l'application.
- **Index ouverts** : le nettoyage ferme d'abord l'index ChromaDB d'une session encore ouverte
  dans le processus (`ChromaVectorStore.close`), sans quoi Windows refuse de supprimer ses
  fichiers. La bibliothèque et le service de réponses d'une session partagent un seul accès à
  l'index.
- **Information du visiteur** : sous le bouton d'envoi, l'interface indique que les documents ne
  sont visibles que dans la session, qu'ils sont supprimés après 2 heures d'inactivité, et que les
  passages utiles aux questions sont envoyés au service de réponses : « n'ajoutez pas de document
  confidentiel ou personnel ».

## Confidentialité

- **Envoyé au LLM** : la question et les passages retrouvés (top-5 et leurs voisins), jamais le
  PDF entier. Avec le niveau gratuit de Gemini, l'ouverture à des utilisateurs de l'EEE, de la
  Suisse ou du Royaume-Uni exige l'offre payante (ADR 0006) : **à trancher avant toute
  activation publique**.
- **Sur le serveur** : PDF, passages et vecteurs restent en clair dans le dossier de la session
  jusqu'à sa suppression. Les journaux ne contiennent pas le contenu des documents, mais un
  avertissement d'extraction peut citer le nom du fichier envoyé (journaux visibles du seul
  propriétaire de l'application).
- **Index de départ** : chaque visiteur reçoit une copie de **tout** l'index de
  `DOCASSIST_DATA_DIR`, pas seulement des PDF de démonstration. Ce dossier doit donc être vide
  au démarrage (disque neuf du cloud) ; en local, ne jamais utiliser `data` (il contient le CV)
  mais un dossier dédié, par exemple `DOCASSIST_DATA_DIR=data\demo-test`.
- **Hors périmètre** : comptes, authentification, chiffrement au repos, antivirus. Une session
  n'est protégée que par l'isolation de Streamlit ; pas de limitation de débit par adresse
  (supprimer puis renvoyer un document reste possible sans limite de fréquence). Le disque est
  borné par le plafond de sessions, les limites par session et la conservation de 2 heures.

## Conséquences

- ➕ Aucun document d'un visiteur n'est visible d'un autre ; la démonstration reste intacte.
- ➕ Rien n'est conservé au-delà de 2 heures d'inactivité.
- ➕ Une simple visite ne coûte ni disque ni client ChromaDB supplémentaire.
- ➖ Chaque visiteur qui envoie un document copie l'index de démonstration et ouvre son propre
  client ChromaDB : coût disque et mémoire proportionnel aux bibliothèques privées actives, au
  plus `DOCASSIST_MAX_SESSIONS`.
- ➖ Plafond atteint : les visiteurs suivants ne peuvent pas envoyer de document avant qu'une
  place se libère (2 heures d'inactivité d'une autre session).
- ➖ Les documents ajoutés sont perdus à l'expiration, au redémarrage de l'application, ou si le
  visiteur ouvre un nouvel onglet (nouvelle session).
- ➖ Indexation séquentielle : plusieurs visiteurs qui envoient en même temps attendent leur tour.
