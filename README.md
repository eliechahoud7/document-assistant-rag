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
| `DOCASSIST_MODELS_DIR` | `models` | dossier du modèle d'embeddings |
| `DOCASSIST_RETRIEVAL_TOP_K` | `5` | passages retrouvés par question |
| `DOCASSIST_CONTEXT_NEIGHBORS` | `1` | chunks voisins ajoutés à chaque passage envoyé au LLM (0 = top-k seul) |
| `DOCASSIST_CHUNK_SIZE` / `_OVERLAP` | `800` / `100` | découpage (caractères) ; changer impose de réindexer |
| `DOCASSIST_LLM_PROVIDER` | `none` | `gemini` pour activer `ask` |
| `DOCASSIST_LLM_MODEL` | `gemini-3.5-flash-lite` | modèle Gemini |
| `DOCASSIST_LLM_TIMEOUT_S` / `_MAX_RETRIES` | `30` / `1` | délai ; nouvelles tentatives sur erreurs serveur uniquement |
| `GEMINI_API_KEY` | — | clé API, lue uniquement dans l'environnement, jamais affichée ni enregistrée |

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

API FastAPI (le paquet `api/` est vide), interface utilisateur, Docker, reranking ou recherche
hybride, et toute nouvelle expérience de recherche (modèle d'embeddings, découpage, top-k).

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
docs/              # architecture et ADR 0001 à 0008
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

## Sources des documents d'évaluation et licences

Les PDF d'évaluation ne sont **pas** inclus dans le dépôt (`evaluation/documents/` est ignoré).
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
