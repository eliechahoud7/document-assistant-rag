# Intelligent Document Assistant — RAG

**English summary.** A local-first retrieval-augmented question-answering assistant for PDF
documents in French and English. PDFs are parsed page by page (pypdf), split into chunks, embedded
on CPU with `intfloat/multilingual-e5-small`, and stored in ChromaDB. For a question, the top-5
chunks — plus their immediate neighbouring chunks — are sent to Google Gemini
(`gemini-3.5-flash-lite`, free tier), which must answer only from those passages, cite them, or
abstain. Citations always come from the retrieved chunks' real file and page metadata, never from
the model's text. The project includes a reproducible evaluation harness: retrieval metrics
(Recall@k, Hit@k, MRR) on annotated quotes, and an offline, deterministic scorer for recorded
answers. On a 20-question set built from three real course PDFs, automatic keyword-based scoring
(not human-reviewed) rose from 10/15 to 14/15 correct answers after adding neighbouring chunks to
the context, with 0 invalid citations and 5/5 correct abstentions on unanswerable questions. The
sample is small and self-annotated: these are indicative results, not a production benchmark.

---

Assistant de questions-réponses sur des PDF (français et anglais) : il retrouve les passages
pertinents, rédige une réponse **fondée uniquement sur ces passages**, cite le **fichier et la
page réels**, et **s'abstient** quand les documents ne contiennent pas la réponse.

> **Statut : version finalisée (v0.1.0).** Ingestion, recherche sémantique, génération citée
> (`ask`) et évaluation de la recherche et de la génération sont implémentées et testées hors
> ligne. Voir [Résultats](#résultats), [Limites](#limites) et [Hors périmètre](#hors-périmètre-de-cette-version).

## Fonctionnement

```
Indexation : PDF → pages (pypdf) → nettoyage → chunks de 800 caractères (chevauchement 100,
             jamais à cheval sur deux pages) → embeddings E5 « passage: » (CPU) → ChromaDB

Question   : embedding E5 « query: » → top-5 chunks (cosinus) → + chunks voisins (±1, même
             document) → Gemini (sortie JSON imposée : répondre ou s'abstenir, passages cités)
             → citations reconstruites à partir des chunks réellement envoyés
```

- **Hors ligne** : indexation et recherche ne contactent aucun service ; seuls `download-model`
  (téléchargement unique du modèle) et la génération (`ask`, `generate-answers`) utilisent le
  réseau.
- **Ancrage** : un numéro de passage inexistant est écarté ; une réponse sans citation valide
  devient une abstention ; sans passage retrouvé, aucun appel au LLM.
- **Architecture** ports et adaptateurs : les services dépendent d'interfaces (`DocumentLoader`,
  `Chunker`, `Embedder`, `VectorStore`, `AnswerGenerator`), jamais d'une bibliothèque concrète.
  Détails : [docs/architecture.md](docs/architecture.md) et les décisions dans [docs/adr/](docs/adr/).

## Installation (Windows, PowerShell)

Prérequis : Python ≥ 3.11 (testé avec 3.13), environ 2 Go de disque (environnement + modèle).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m doc_assistant download-model    # une fois : ~470 Mio dans .\models\
```

> Les commandes ci-dessous appellent `.\.venv\Scripts\python.exe` directement, ce qui évite
> d'activer l'environnement (`Activate.ps1` peut être bloqué par la politique d'exécution).

## Démarrage rapide

**1. Recherche seule (hors ligne, sans clé).**

```powershell
$env:DOCASSIST_DATA_DIR = "data\courses"     # index dédié aux documents que vous acceptez d'envoyer
.\.venv\Scripts\python.exe -m doc_assistant index chemin\vers\vos\pdfs
.\.venv\Scripts\python.exe -m doc_assistant search "Comment choisir k dans les k-moyennes ?" -k 3
```

**2. Réponses citées (`ask`, API Gemini, niveau gratuit).** Seuls les passages retrouvés sont
envoyés au fournisseur. Créez une clé dans Google AI Studio, dans un projet **sans facturation**.

```powershell
$env:DOCASSIST_LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "..."                  # à saisir vous-même ; jamais dans un fichier versionné
.\.venv\Scripts\python.exe -m doc_assistant ask "Comment choisir k dans les k-moyennes ?" --show-passages
```

Sans clé ou avec `DOCASSIST_LLM_PROVIDER=none` (défaut), `ask` s'arrête avec un message clair ;
`index`, `search` et `evaluate` fonctionnent sans configuration LLM. Quota dépassé (429), délai ou
erreur de l'API : message explicite, aucune nouvelle tentative sur quota.

## Configuration

Variables d'environnement (voir [.env.example](.env.example) ; le fichier `.env` n'est pas chargé
automatiquement : définissez les variables dans la session PowerShell).

| Variable | Défaut | Rôle |
|---|---|---|
| `DOCASSIST_DATA_DIR` | `data` | dossier de l'index ChromaDB (`<dir>\chroma`) |
| `DOCASSIST_MODELS_DIR` | `models` | dossier des modèles locaux (embeddings, traduction facultative) |
| `DOCASSIST_RETRIEVAL_TOP_K` | `5` | passages retrouvés par question |
| `DOCASSIST_CONTEXT_NEIGHBORS` | `1` | chunks voisins ajoutés à chaque passage envoyé au LLM (0 = top-k seul) |
| `DOCASSIST_CHUNK_SIZE` / `_OVERLAP` | `800` / `100` | découpage (caractères) ; changer impose de réindexer |
| `DOCASSIST_LLM_PROVIDER` | `none` | `gemini` pour activer `ask` |
| `DOCASSIST_LLM_MODEL` | `gemini-3.5-flash-lite` | modèle Gemini |
| `DOCASSIST_LLM_TIMEOUT_S` / `_MAX_RETRIES` | `30` / `1` | délai ; nouvelles tentatives sur erreurs serveur uniquement |
| `GEMINI_API_KEY` | — | clé API, lue uniquement dans l'environnement, jamais affichée ni enregistrée |
| `DOCASSIST_QUERY_TRANSLATION` | `none` | `fr-en` : traduit les questions françaises pour la recherche uniquement (expérimental, voir [Résultats](#expérience--traduction-fr--en-des-requêtes-et-fusion-rrf-real_v2)) |
| `DOCASSIST_APP_UPLOADS` | `true` | ajout et suppression de documents depuis l'interface web ; `false` pour une bibliothèque en lecture seule (déploiement public) |
| `DOCASSIST_MODEL_AUTO_DOWNLOAD` | `false` | `true` : télécharge le modèle d'embeddings au premier démarrage s'il est absent (déploiement) |
| `DOCASSIST_DEMO_DOCS_DIR` | — | mode démonstration : PDF publics indexés au démarrage ; ajout et suppression toujours désactivés dans la bibliothèque partagée (voir [Déploiement](#déploiement-streamlit-community-cloud)) |
| `DOCASSIST_SESSION_UPLOADS` | `false` | mode démonstration uniquement : `true` donne à chaque visiteur une bibliothèque privée et temporaire ([ADR 0009](docs/adr/0009-envois-prives-des-visiteurs.md)) |
| `DOCASSIST_UPLOAD_MAX_MB` / `_MAX_PAGES` / `_MAX_FILES` | `10` / `50` / `3` | limites des envois d'un visiteur : taille (1 Mo = 1 024 × 1 024 octets, au plus `server.maxUploadSize` = 20) et pages par PDF, documents ajoutés par session |
| `DOCASSIST_MAX_SESSIONS` | `20` | bibliothèques privées de visiteurs simultanées ; au-delà, l'envoi est refusé et la démonstration reste disponible |
| `DOCASSIST_USER_LIBRARIES` | `false` | `true` : connexion Google et bibliothèque personnelle persistante par utilisateur ([ADR 0010](docs/adr/0010-bibliotheques-personnelles.md)) ; incompatible avec `DOCASSIST_SESSION_UPLOADS` |
| `DOCASSIST_USER_STORE` | `chroma-cloud` | base des bibliothèques personnelles : `chroma-cloud` (exige `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`) ou `local` (`<DOCASSIST_DATA_DIR>\users`, développement uniquement) |
| `DOCASSIST_ACCOUNT_MAX_DOCUMENTS` | `20` | documents par utilisateur connecté |
| `DOCASSIST_ACCOUNT_DAILY_QUESTIONS` / `_UPLOADS` | `50` / `20` | questions et envois par utilisateur et par jour (compteurs en mémoire, remis à zéro chaque jour et au redémarrage) |
| `DOCASSIST_MAX_ACCOUNTS` | `100` | bibliothèques personnelles au plus dans la base ; au-delà, aucun nouveau compte |
| `DOCASSIST_PRIVACY_URL` | — | obligatoire avec `DOCASSIST_USER_LIBRARIES` : adresse `https://` de votre notice de confidentialité, affichée avant tout envoi |

## Exemple (tiré d'un enregistrement existant)

Réponses réelles extraites de
`evaluation/results/real_v1__gen__gemini-3.5-flash-lite__e6a_neighbors1.jsonl` (enregistrement
local de l'expérience E6a, configuration par défaut actuelle), mises en forme comme la sortie de
`ask`. Aucun nouvel appel n'a été fait pour cette documentation.

```
Question : Comment choisir le nombre de groupes dans l'algorithme des k-moyennes ?

Pour choisir le nombre de groupes (k) dans l'algorithme des k-moyennes, on peut augmenter k
progressivement et rechercher une baisse soudaine de l'erreur, ce qui apparaît comme un « coude »
(elbow) sur un graphique représentant k en fonction de l'erreur. Une autre technique couramment
utilisée est l'analyse de silhouette (silhouette analysis), qui attribue un score basé sur le
calcul de la distance intra-groupe par rapport à la distance inter-groupe.

Sources :
  [15] lecture1.pdf, page 81
```

```
Question : Comment la normalisation par lots (batch normalization) accélère-t-elle
           l'entraînement des réseaux profonds ?

Pas de réponse trouvée dans les documents.
(raison : le modèle juge que les passages ne contiennent pas la réponse)
Les documents fournis ne contiennent pas d'information sur la normalisation par lots (batch
normalization) ni sur la manière dont elle accélère l'entraînement des réseaux profonds.
```

La question est en français, le cours en anglais : la recherche et la réponse sont interlingues.

## Résultats

Petits échantillons, questions et réponses de référence rédigées par l'assistant de
développement : résultats **indicatifs**. Détails et protocole : [evaluation/README.md](evaluation/README.md).

### Recherche (sans LLM, configuration par défaut, top-5)

| Jeu | Questions répondables | Recall@1 | Recall@5 | Hit@5 | MRR |
|---|---|---|---|---|---|
| `synthetic_v1` (synthétique, rédigé pour le projet) | 30 | 0,900 | 0,983 | 1,000 | 0,967 |
| `real_v1` (3 PDF de cours réels, 117 + 6 + 8 pages) | 15 | 0,367 | 0,694 | 0,800 | 0,552 |

Le score de similarité ne permet pas de reconnaître une question sans réponse (AUROC 0,73 sur
`real_v1`, distributions qui se chevauchent) : l'abstention est confiée au LLM.

### Génération (`real_v1`, 15 questions répondables + 5 sans réponse)

> ⚠️ **Scores automatiques par mots-clés, non relus par un humain.** Une réponse est comptée
> « correcte » si elle contient les faits requis de l'annexe
> [`real_v1_answers.json`](evaluation/datasets/real_v1_answers.json) ; c'est un indicateur
> approximatif (paraphrases manquées, négations non détectées). Les fiches de relecture existent
> mais n'ont pas été remplies.

| Configuration | Correctes (auto) | Incorrectes | Abstentions (répondables) | Citations invalides | Abstentions correctes (sans réponse) |
|---|---|---|---|---|---|
| Référence : top-5 seul (`run1`) | **10/15** | 3 | 2 | 0 / 15 | 5/5 |
| E6a : top-5 + voisins ±1 (défaut actuel) | **14/15** | 1 (Q10) | 0 | 0 / 18 | 5/5 |

E6a corrige Q05, Q08, Q14 et Q15, dont l'information se trouvait dans le chunk voisin d'un
passage retrouvé ; aucune question ne régresse. Décision : [ADR 0008](docs/adr/0008-contexte-chunks-voisins.md).

### Expérience : traduction FR → EN des requêtes et fusion RRF (`real_v2`)

Jeu `real_v2` corrigé (47 questions : 38 répondables, dont 18 en français, et 9 sans réponse ;
SHA-256 `75abb1ee…`). Réglages identiques pour les trois runs : `multilingual-e5-small`
(révision `614241f6`), chunks 800/100, top-5, chevauchement minimal 50 %, 370 chunks, index
temporaire, mêmes versions des bibliothèques. Recherche seule, sans LLM.

- **Traduction des requêtes (facultative, désactivée par défaut).** Avec
  `DOCASSIST_QUERY_TRANSLATION=fr-en`, une question détectée comme française est traduite en
  anglais par un modèle local (`Helsinki-NLP/opus-mt-fr-en`, révision figée) **pour la recherche
  uniquement** : la question d'origine reste celle envoyée au LLM et affichée. Les questions
  anglaises ne sont pas traduites ; si la traduction échoue, la question d'origine est utilisée.
- **Fusion RRF (expérience isolée, non intégrée à l'application).**
  [`evaluation/experiments/rrf_fusion.py`](evaluation/experiments/rrf_fusion.py) fusionne le
  classement de la question d'origine et celui de sa traduction par *Reciprocal Rank Fusion*
  (Cormack et al., 2009) : score d'un passage = Σ 1 / (60 + rang), k = 60 (valeur standard, non
  réglée sur ce jeu). Pour l'AUROC, le score du passage en tête est son meilleur cosinus.

| Run (38 questions répondables) | Hit@1 | Hit@3 | Hit@5 | MRR | AUROC (absence) |
|---|---|---|---|---|---|
| Référence ([`…__corrected.json`](evaluation/results/real_v2__e5-small__cs800-co100__corrected.json)) | 0,474 | 0,684 | 0,895 | 0,609 | 0,918 |
| Traduction seule ([`…__fr2en-opusmt.json`](evaluation/results/real_v2__e5-small__cs800-co100__fr2en-opusmt.json)) | 0,500 | 0,789 | 0,895 | 0,648 | 0,942 |
| Origine + traduction, RRF ([`…__fr2en-rrf60.json`](evaluation/results/real_v2__e5-small__cs800-co100__fr2en-rrf60.json)) | 0,474 | 0,763 | 0,895 | 0,641 | 0,942 |

Questions françaises (18), référence / traduction / RRF : Hit@1 0,333 / 0,389 / 0,333 ;
Hit@3 0,556 / 0,778 / 0,722 ; Hit@5 0,833 dans les trois cas ; MRR 0,478 / 0,560 / 0,546.
Questions anglaises (20) : résultats identiques dans les trois runs (elles ne sont pas traduites).

- **Traduction seule** : 8 questions mieux classées (dont Q18, rang 17 → 3, qui entre dans le
  top-5, et Q16, 93 → 31), 3 moins bien (Q12, 3 → 6, qui sort du top-5 ; Q22, 1 → 3 ; Q06,
  100 → 102), 27 inchangées.
- **RRF** : récupère Q12 (rang 3) mais perd Q18 (rang 7). La fusion favorise les passages bien
  classés par les deux requêtes et pénalise un passage que seule la traduction trouve ; elle
  laisse hors du top-5 les mêmes questions que la référence.
- **Échecs communs** : Q06 et Q16 restent au-delà du rang 30 dans les trois runs ; le modèle
  traduit littéralement des termes du domaine (« norme » → *standard*, « noyaux » → *nuclei*,
  « ACP » non traduit en *PCA*). Q24 (anglaise) reste au rang 8 : la liste attendue est coupée
  entre deux chunks.
- **AUROC** : la hausse est à lire avec prudence, car les questions françaises sans réponse
  obtiennent elles aussi des scores plus élevés après traduction.

**Conclusion.** Ni la traduction ni la RRF n'améliorent le Hit@5 (34/38 dans les trois runs).
La traduction améliore le classement des questions françaises (MRR +0,08) mais en fait sortir
une du top-5 ; la RRF ne fait pas mieux que la traduction seule. La traduction reste donc
**désactivée par défaut** et la RRF **n'est pas intégrée** à l'application. Piste suivante, non
mesurée : dans les trois runs, 36 des 38 questions ont un passage pertinent dans le top-20 ; un
reranking de ces 20 candidats pourrait au mieux gagner 2 questions (il faudrait un modèle local
supplémentaire). Q06 et Q16 exigeraient plutôt une meilleure représentation de la requête.

**Limites.** 18 questions françaises seulement (une question = 5,6 points de Hit@5 en français) ;
un seul modèle de traduction ; documents uniquement en anglais ; jeu rédigé par l'assistant de
développement et utilisé aussi pour le diagnostic (références de Q04, Q09, Q20, Q21 et Q33
complétées après lecture des résultats), sans jeu de test indépendant ; détection du français
heuristique, vérifiée seulement sur les 151 questions des jeux du dépôt.

Reproduire (index temporaires ; la RRF refuse d'écrire si elle ne reproduit pas exactement les
deux rapports de comparaison, et n'écrase jamais un rapport existant) :

```powershell
# Une fois : dépendances facultatives et modèle de traduction local (~304 Mo)
.\.venv\Scripts\python.exe -m pip install -e ".[translation]"
.\.venv\Scripts\python.exe -m doc_assistant download-translation-model

# Référence, puis traduction seule
.\.venv\Scripts\python.exe -m doc_assistant evaluate evaluation\datasets\real_v2.json -k 5 -o reference.json
$env:DOCASSIST_QUERY_TRANSLATION = "fr-en"
.\.venv\Scripts\python.exe -m doc_assistant evaluate evaluation\datasets\real_v2.json -k 5 -o traduction.json
Remove-Item Env:DOCASSIST_QUERY_TRANSLATION

# Fusion RRF, comparée aux deux rapports versionnés
.\.venv\Scripts\python.exe evaluation\experiments\rrf_fusion.py evaluation\datasets\real_v2.json `
    --baseline evaluation\results\real_v2__e5-small__cs800-co100__corrected.json `
    --translation evaluation\results\real_v2__e5-small__cs800-co100__fr2en-opusmt.json -o rrf.json
```

## Limites

- **Q10 (procédure leave-one-out) reste incorrecte** : la réponse nomme la méthode sans décrire la
  procédure (retirer un exemple, entraîner sur le reste, tester sur l'exemple retiré) ; les
  passages qui la décrivent (lecture 2, p. 3-4) ne sont pas retrouvés pour cette question.
- **Évaluation limitée** : 20 questions, 3 documents, une seule exécution d'un modèle non
  déterministe ; questions et réponses de référence rédigées par l'assistant ; le même jeu a servi
  au diagnostic et au test (un jeu `real_v2` indépendant serait nécessaire pour confirmer).
- **Correction automatique approximative** (voir ci-dessus) ; Q14 est ambiguë dans le jeu : le
  cours donne un exemple 60/40 et un exemple 80/20.
- **Extraction** : PDF à texte sélectionnable uniquement (pas d'OCR) ; formules et mots parfois
  collés à l'extraction ; un chunk ne franchit jamais une page.
- **Niveau gratuit Gemini** : quotas non publiés (visibles dans AI Studio), conditions
  susceptibles de changer ; usage local et personnel. Selon les conditions Gemini, rendre l'outil
  accessible à des utilisateurs de l'EEE exigerait l'offre payante ([ADR 0006](docs/adr/0006-generation-gemini.md)).
- **Contexte plus long** avec les voisins (9 à 15 chunks par question au lieu de 5).

## Hors périmètre de cette version

Inclus : une interface web Streamlit ([`app.py`](app.py),
`.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost` ; l'option
limite l'accès à votre machine) qui travaille sur une seule bibliothèque :
l'index de `DOCASSIST_DATA_DIR` (`data` par défaut), le même que celui des commandes `index`,
`search` et `ask`. La section « Mes documents » liste les PDF indexés (nom, pages, passages, date
d'ajout, état), y compris ceux indexés avec la CLI, et persiste entre les sessions grâce au
registre `<DOCASSIST_DATA_DIR>\documents.json`. On peut y ajouter plusieurs PDF à la fois (même
pipeline que `index`, même déduplication par SHA-256, sans appel au LLM ; PDF à texte
sélectionnable, 20 Mo au plus, fichier conservé dans `<DOCASSIST_DATA_DIR>\uploads\`) et
supprimer un document : seuls ses passages (même `doc_id`), son PDF conservé et son entrée du
registre sont retirés. Les questions portent sur toute la bibliothèque, avec la même
configuration LLM que `ask`. Pour un déploiement public, choisissez un dossier sans le CV
(par exemple `DOCASSIST_DATA_DIR=data\courses`) et `DOCASSIST_APP_UPLOADS=false`.

Hors périmètre : API FastAPI (le paquet `api/` est vide), OCR des PDF scannés, Docker, reranking
ou recherche hybride, et toute
nouvelle expérience de recherche (modèle d'embeddings, découpage, top-k) ; seules exceptions, la
traduction facultative des requêtes et l'expérience RRF décrites dans
[Résultats](#expérience--traduction-fr--en-des-requêtes-et-fusion-rrf-real_v2).

## Évaluation

```powershell
# Recherche : index temporaire, n'utilise jamais vos index
.\.venv\Scripts\python.exe -m doc_assistant evaluate evaluation\datasets\real_v1.json -o rapport.json

# Génération : notation hors ligne et déterministe d'un enregistrement existant
.\.venv\Scripts\python.exe -m doc_assistant score-answers evaluation\results\real_v1__gen__gemini-3.5-flash-lite__e6a_neighbors1.jsonl `
    --dataset evaluation\datasets\real_v1.json --gold evaluation\datasets\real_v1_answers.json --review relecture.md
```

L'enregistrement de nouvelles réponses (`generate-answers --live`) appelle l'API et consomme du
quota : il n'est jamais lancé par les tests. Les enregistrements bruts (`*.jsonl`) et les PDF
d'évaluation (`evaluation/documents/`) sont exclus de Git.

## Structure

```
src/doc_assistant/
├── config.py      # configuration (variables DOCASSIST_*)
├── domain/        # modèles (Document, Chunk, Answer, Citation…), interfaces, erreurs
├── ingestion/     # extraction pypdf, nettoyage, découpage
├── indexing/      # embeddings E5, ChromaDB, téléchargement du modèle
├── retrieval/     # recherche, voisins de chunks, métriques
├── generation/    # prompt, validation des réponses, adaptateur Gemini
├── services/      # indexation, questions-réponses
├── evaluation/    # jeux annotés, évaluation de la recherche et de la génération
├── bootstrap.py   # assemblage des adaptateurs
└── cli.py         # download-model, index, search, ask, evaluate, generate-answers, score-answers
tests/             # unit/, integration/ (aucun accès réseau)
evaluation/        # jeux d'évaluation, annexe de réponses, rapports
docs/              # architecture et ADR 0001 à 0010
```

## Qualité et tests

```powershell
.\.venv\Scripts\python.exe -m pytest --cov            # suite par défaut, réseau bloqué
.\.venv\Scripts\python.exe -m pytest -m slow          # vrai modèle E5 (rapports de référence)
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy                    # mode strict
```

Une fixture bloque toute connexion réseau pendant les tests ; la génération y est simulée par des
générateurs factices. Le test réel Gemini (`-m llm`) n'est jamais sélectionné par défaut.

## Déploiement (Streamlit Community Cloud)

> **Avant toute mise en ligne publique** : selon les conditions de l'API Gemini, rendre l'outil
> accessible à des utilisateurs de l'EEE, de la Suisse ou du Royaume-Uni exige l'offre payante
> ([ADR 0006](docs/adr/0006-generation-gemini.md)). Vérifiez les conditions en vigueur.

Fichiers utilisés : [`requirements.txt`](requirements.txt) (installe le projet et PyTorch
**CPU** `torch==2.14.1+cpu`, sans paquet NVIDIA), [`.streamlit/config.toml`](.streamlit/config.toml)
(sans adresse d'écoute : en local, gardez `--server.address localhost`) et
[`demo/documents/`](demo/documents/SOURCES.md) (seuls PDF publiés : cours MIT OCW sous
CC BY-NC-SA 4.0). Ni `data/` (index, CV, registre), ni `models/`, ni `.env`, ni
`.streamlit/secrets.toml` ne sont versionnés.

1. Poussez la branche sur GitHub, puis sur share.streamlit.io : *Create app*, dépôt, branche,
   fichier `app.py` ; dans *Advanced settings*, choisissez **Python 3.13**.
2. Dans *Secrets*, définissez (clés au premier niveau : elles deviennent des variables
   d'environnement) :

   ```toml
   GEMINI_API_KEY = "..."
   DOCASSIST_LLM_PROVIDER = "gemini"
   DOCASSIST_MODEL_AUTO_DOWNLOAD = "true"
   DOCASSIST_DEMO_DOCS_DIR = "demo/documents"
   DOCASSIST_APP_UPLOADS = "false"
   ```

3. Premier démarrage : installation des dépendances, téléchargement du modèle (~493 Mo), puis
   indexation des PDF de démonstration. Le disque du cloud est temporaire : après une mise en
   veille, le modèle est retéléchargé et les documents réindexés automatiquement.

En mode démonstration, l'ajout et la suppression de documents sont toujours désactivés, même si
`DOCASSIST_APP_UPLOADS` vaut `true`.

### Envois privés des visiteurs (désactivés)

Implémentés mais **désactivés sur le déploiement** tant que la question Gemini/EEE n'est pas
tranchée ([ADR 0006](docs/adr/0006-generation-gemini.md)) : les passages des PDF envoyés par les
visiteurs partiraient chez Gemini. Décision et détails : [ADR 0009](docs/adr/0009-envois-prives-des-visiteurs.md).

- Avec `DOCASSIST_SESSION_UPLOADS = "true"` (mode démonstration exigé), un visiteur consulte la
  démonstration partagée en lecture seule ; à son **premier envoi**, sa session reçoit sa propre
  copie de la bibliothèque de démonstration dans `data/sessions/<identifiant aléatoire>/`. Un
  visiteur ne voit jamais les documents d'un autre et ne peut pas supprimer les documents de
  démonstration.
- Au plus `DOCASSIST_MAX_SESSIONS` (20) bibliothèques privées à la fois : au-delà, l'envoi est
  refusé (« réessayez dans quelques minutes ») et la démonstration reste utilisable.
- Limites vérifiées par le serveur : `DOCASSIST_UPLOAD_MAX_MB` (10), `DOCASSIST_UPLOAD_MAX_PAGES`
  (50), `DOCASSIST_UPLOAD_MAX_FILES` (3 documents ajoutés par session ; les fichiers en trop
  d'un envoi sont refusés sans être traités) ; une valeur invalide affiche une erreur de
  configuration. Streamlit refuse lui-même, dès la réception, tout fichier de plus de 20 Mo
  (`server.maxUploadSize` dans `.streamlit/config.toml`) ; `DOCASSIST_UPLOAD_MAX_MB` ne peut pas
  dépasser cette valeur.
- Conservation : bibliothèque supprimée après 2 heures d'inactivité (ou au redémarrage de
  l'application sur le cloud) ; le visiteur en est averti sous le bouton d'envoi, avec la
  consigne de ne pas envoyer de document confidentiel ou personnel.
- Chaque visiteur reçoit une copie de **tout** l'index de `DOCASSIST_DATA_DIR` : pour un essai
  local, utilisez un dossier vide dédié (`$env:DOCASSIST_DATA_DIR = "data\demo-test"`), jamais
  `data` (qui contient le CV).

Pour les activer plus tard, ajoutez aux *Secrets* (en gardant `DOCASSIST_APP_UPLOADS = "false"`) :

```toml
DOCASSIST_SESSION_UPLOADS = "true"
# facultatif, valeurs par défaut :
DOCASSIST_UPLOAD_MAX_MB = "10"
DOCASSIST_UPLOAD_MAX_PAGES = "50"
DOCASSIST_UPLOAD_MAX_FILES = "3"
DOCASSIST_MAX_SESSIONS = "20"
```

### Bibliothèques personnelles (connexion Google, désactivées)

Implémentées et testées hors ligne, **désactivées par défaut** ; décision, coûts et
confidentialité : [ADR 0010](docs/adr/0010-bibliotheques-personnelles.md). Chaque utilisateur
connecté avec Google retrouve sa propre bibliothèque après déconnexion, redémarrage de
l'application ou sur un autre appareil ; les visiteurs déconnectés gardent la démonstration en
lecture seule. Les passages sont conservés dans Chroma Cloud (région UE), une collection par
utilisateur ; aucun PDF n'est gardé sur le serveur. « Supprimer mon compte » efface toutes les
données de l'utilisateur, et seulement les siennes.

Avant son premier envoi et sa première question, chaque utilisateur doit confirmer avoir 18 ans
ou plus et accepter de courtes conditions (pas de données sensibles ; passages conservés par
Chroma et envoyés à Gemini), avec un lien vers **votre** notice de confidentialité
(`DOCASSIST_PRIVACY_URL`, obligatoire). L'application ne rédige aucune notice et n'affirme
aucune conformité. L'acceptation est enregistrée avec la bibliothèque dès le premier envoi ;
rien n'est écrit dans la base pour un utilisateur qui se connecte sans rien envoyer. Limites
par utilisateur (questions et envois par jour) et nombre maximal de comptes : voir
[Configuration](#configuration).

**Avant toute activation publique** : offre payante de Gemini pour les utilisateurs de l'EEE
([ADR 0006](docs/adr/0006-generation-gemini.md)), notice de confidentialité publiée, essai
manuel de la connexion Google et de Chroma Cloud.

Préparation (une seule fois, hors du dépôt) :

1. La dépendance de connexion est déclarée dans `pyproject.toml` (`streamlit[auth]`, qui ajoute
   Authlib) ; installez-la en local avec `.\.venv\Scripts\python.exe -m pip install -e .`.
2. Google Cloud Console : créez un client OAuth « Application Web » avec l'URI de redirection
   `https://<votre-app>.streamlit.app/oauth2callback` (et `http://localhost:8501/oauth2callback`
   pour un essai local).
3. Chroma Cloud : créez une base en région **europe-west1** et une clé API limitée à cette base.
4. Générez un `cookie_secret` aléatoire (au moins 32 caractères), par exemple
   `.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"`.
5. Dans les *Secrets* de Streamlit Community Cloud (jamais dans un fichier versionné) :

   ```toml
   DOCASSIST_USER_LIBRARIES = "true"
   DOCASSIST_PRIVACY_URL = "https://.../confidentialite"
   CHROMA_API_KEY = "..."
   CHROMA_TENANT = "..."
   CHROMA_DATABASE = "..."

   [auth]
   redirect_uri = "https://<votre-app>.streamlit.app/oauth2callback"
   cookie_secret = "..."
   client_id = "....apps.googleusercontent.com"
   client_secret = "..."
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```

   en gardant `DOCASSIST_APP_UPLOADS = "false"` et sans `DOCASSIST_SESSION_UPLOADS`.

Essai local sans Chroma Cloud : `DOCASSIST_USER_STORE=local` (une base par utilisateur dans
`<DOCASSIST_DATA_DIR>\users`, jamais `data` qui contient le CV) et un
`.streamlit\secrets.toml` local (ignoré par Git) avec la section `[auth]`.

## Sources des documents d'évaluation et licences

Les PDF d'évaluation ne sont **pas** inclus dans le dépôt (`evaluation/documents/` est ignoré),
à l'exception de copies non modifiées de `lecture2.pdf` et `lecture3.pdf` dans
`demo/documents/` pour la démonstration (attribution : [`SOURCES.md`](demo/documents/SOURCES.md)).
Les jeux d'évaluation, l'annexe de réponses, les rapports et les fiches de relecture contiennent
de **courts extraits** de ces documents :

- `lecture1.pdf` : Jesse Read, *Machine Learning — CSC_2S004_EP (previously CSE204)*, notes de
  cours, École polytechnique, version du 11 mai 2025 — sous licence
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (mention en p. 1).
- `lecture2.pdf`, `lecture3.pdf` : Tommi Jaakkola, *6.867 Machine Learning*, cours 3 et 7, Fall
  2006, MIT OpenCourseWare (https://ocw.mit.edu/) — sous licence
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Ces extraits (fichiers de `evaluation/datasets/` et `evaluation/results/` qui les contiennent)
restent soumis à CC BY-NC-SA 4.0 : attribution, usage non commercial, partage dans les mêmes
conditions. Aucune approbation des auteurs n'est sous-entendue.

## Licence

Code source : [MIT](LICENSE). Les extraits de documents tiers suivent leur propre licence (voir
ci-dessus).
