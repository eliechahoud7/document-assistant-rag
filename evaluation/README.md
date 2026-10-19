# Évaluation

Deux évaluations **distinctes** : la **recherche** (sans LLM, sections suivantes) et la
**génération** ([plus bas](#évaluation-de-la-génération)). Leurs chiffres ne se comparent pas
entre eux.

| Fichier de résultats | Contenu |
|---|---|
| `results/synthetic_v1__e5-small__cs800-co100.json` | recherche, jeu synthétique (référence) |
| `results/real_v1__e5-small__cs800-co100.json` | recherche, jeu réel (référence) |
| `results/real_v1__e5-small__cs800-co100__top50.json` | recherche, rangs réels jusqu'à 50 (E0) |
| `results/*__cs800-co200.json` | recherche, chevauchement 200 (E1, non retenu) |
| `results/real_v1_en__e5-small__cs800-co100.json` | recherche, questions traduites en anglais (E3a) |
| `results/real_v1__gen__run1__scores.json` / `__review.md` | génération, référence (top-5 seul) |
| `results/real_v1__gen__e6a__scores.json` / `__review.md` | génération, E6a (top-5 + voisins ±1, défaut actuel) |

Les enregistrements bruts de génération (`*.jsonl`) et les PDF (`documents/`) restent locaux
(ignorés par Git). Fichiers contenant des extraits de cours : voir la section « Sources des
documents d'évaluation et licences » du [README principal](../README.md).

# Évaluation de la recherche

> ⚠️ **Résultats synthétiques ≠ qualité en production.** Les chiffres de `synthetic_v1`
> proviennent d'un petit corpus rédigé pour le projet. Ils servent à comparer des
> configurations et à détecter des régressions, **pas** à prouver la qualité du système sur
> de vrais documents. Résultats sur de vrais documents : [section dédiée](#résultats--real_v1-réel).

## Lancer une évaluation (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m doc_assistant evaluate evaluation\datasets\synthetic_v1.json `
    --output evaluation\results\synthetic_v1__e5-small__cs800-co100.json
```

L'évaluation construit un **index ChromaDB temporaire** (supprimé à la fin) : elle ne lit ni
ne modifie l'index de l'utilisateur (`data/chroma/`). Elle refuse d'ailleurs tout index non
vide. Elle utilise la configuration courante (`DOCASSIST_CHUNK_SIZE`, etc.), enregistrée
dans le rapport avec les versions des bibliothèques et l'empreinte SHA-256 du jeu.

## Protocole

**Références indépendantes du découpage.** Une référence est une citation exacte d'une page
(`filename`, `page`, `quote`), pas un identifiant de chunk. Un chunk retrouvé est pertinent
s'il est sur la bonne page et couvre **au moins 50 % des caractères non blancs** de la
citation. Les blancs sont exclus parce que le découpeur supprime l'espace entre deux
chunks : sans cela, une citation de longueur impaire coupée pile à une frontière
n'atteignait 50 % dans aucun chunk (cas trouvé et testé, `tests/unit/test_eval_relevance.py`).
Une citation plus longue que `chunk_size` est refusée.

**Annotation sans le retriever.** Les citations sont choisies en lisant les documents, jamais
à partir des résultats du moteur évalué. Le chargement vérifie que chaque citation apparaît
exactement une fois sur sa page.

**Métriques (questions répondables uniquement)**

| Métrique | Définition |
|---|---|
| Recall@k | part des citations de référence retrouvées dans le top k, moyennée par question |
| Hit@k | part des questions avec au moins une référence dans le top k |
| MRR | moyenne de 1/rang de la première référence trouvée (0 si absente du top k) |

Avec deux références, Recall@1 vaut au plus 0,5 pour cette question : il faut lire Recall@1
avec Hit@1 pour les questions `multi_passage`.

**Détection d'absence (questions sans réponse, évaluées à part).** Le moteur renvoie toujours
k passages : on mesure si le **score cosinus du meilleur passage** distingue les questions
répondables (scores attendus plus élevés) des questions sans réponse, par l'**AUROC**
(1 = séparation parfaite, 0,5 = hasard, < 0,5 = sens inverse). Aucun seuil n'est choisi sur
ces données (ce serait du surapprentissage) : c'est l'objet de la phase 4.

## Résultats — `synthetic_v1` (SYNTHÉTIQUE)

Configuration : `intfloat/multilingual-e5-small` (commit `614241f6`), `chunk_size=800`,
`chunk_overlap=100`, top-5, ChromaDB HNSW cosinus. 6 documents, 27 pages, 29 chunks ;
30 questions répondables, 10 sans réponse. Mesuré le 2026-10-19, **deux exécutions
identiques** (hors horodatage).

| | n | Recall@1 | Recall@3 | Recall@5 | Hit@1 | MRR |
|---|---|---|---|---|---|---|
| **Global** | 30 | 0,900 | 0,983 | 0,983 | 0,933 | 0,967 |
| cross_lingual | 13 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| paraphrase | 6 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| fine_grained | 5 | 0,800 | 1,000 | 1,000 | 0,800 | 0,900 |
| long_page | 3 | 0,667 | 1,000 | 1,000 | 0,667 | 0,833 |
| multi_passage | 2 | 0,500 | 0,750 | 0,750 | 1,000 | 1,000 |
| numeric | 1 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |

Détection d'absence : **AUROC = 0,760**. Score du meilleur passage : répondables
0,858 en moyenne [0,787 ; 0,927], sans réponse 0,819 [0,756 ; 0,866].
**Les distributions se chevauchent** : 17 des 30 questions répondables ont un score inférieur
ou égal au maximum des questions sans réponse. Aucun seuil sur ce score ne sépare les deux
groupes ; il mesure la proximité thématique, pas la présence de la réponse.

### Échecs observés

- **q03 (négation)** : « sans jamais les mettre à zéro » → la page L1 (qui *met* les
  coefficients à zéro) passe devant la page L2.
- **q07 (page longue)** : la page sur l'arrêt précoce traite aussi de validation croisée et
  d'augmentation de données ; le chunk est « dilué » et classé 2ᵉ.
- **q06 (biais de langue)** : question anglaise sur le dropout ; la page anglaise est 1ʳᵉ mais
  la page française sur le même sujet est absente du top 5, devancée par d'autres pages
  anglaises.
- **u01, u05, u03** : questions sans réponse mais sur le sujet (prix d'une pompe à chaleur,
  nombre de signataires) obtenant des scores supérieurs à la plupart des questions répondables.

## Limites

- **Synthétique et auto-rédigé** : documents et questions écrits par le même auteur, d'où un
  risque de biais (vocabulaire, difficulté). Les faits sont illustratifs.
- **Petit échantillon** : 30 questions répondables, une question ≈ 3,3 points ; les catégories
  comptent 1 à 13 questions (une question de `long_page` = 33 points). Pas d'intervalle de
  confiance calculé : les différences de quelques points ne sont pas significatives.
- **Documents courts et propres** : pages d'un seul paragraphe, sans en-têtes, tableaux, ni
  erreurs d'extraction ; 29 chunks seulement, contre des milliers sur un vrai corpus, où la
  concurrence entre passages est bien plus forte et l'approximation HNSW peut jouer.
- **Une seule configuration mesurée** : aucun réglage n'a été modifié pour améliorer les scores.
- **Reproductibilité** vérifiée sur cette machine (CPU, versions dans le rapport) ; d'autres
  processeurs ou versions peuvent produire de légers écarts de score.

## Résultats — `real_v1` (RÉEL)

3 PDF de cours (anglais), 15 questions répondables (7 FR, 8 EN) et 5 sans réponse, annotées par
citations exactes. Configuration : e5-small, 800/100, top-5.

| | n | Recall@1 | Recall@3 | Recall@5 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| **Global** | 15 | 0,367 | 0,606 | 0,694 | 0,400 | 0,800 | 0,552 |

Détection d'absence : AUROC = 0,733 (distributions qui se chevauchent). Expériences de recherche
menées sur ce jeu, à titre de diagnostic : E0 (rangs jusqu'à 50 : les références manquées sont
aux rangs 6 à 27, ou au-delà de 50), E1 (chevauchement 200 : non retenu, 4 questions améliorées et
5 dégradées), E3a (questions françaises traduites en anglais : meilleur classement, Hit@5
inchangé, effet non significatif sur 7 questions appariées). Aucune n'a modifié la configuration.

# Évaluation de la génération

> Résultats **distincts** de ceux de la recherche ci-dessus.

## Résultats — `real_v1` (RÉEL), modèle `gemini-3.5-flash-lite`

> ⚠️ **Scores automatiques par mots-clés, non relus par un humain.** Les fiches de relecture
> (`results/*__review.md`) n'ont pas été remplies et aucun verdict humain (`--overrides`) n'a
> été appliqué. Une réponse est comptée correcte si elle contient tous les faits requis de
> l'annexe : c'est un indicateur approximatif.

| | Référence `run1` (top-5 seul) | E6a (top-5 + voisins ±1) |
|---|---|---|
| Répondables correctes (auto) | **10/15** | **14/15** |
| Incorrectes | 3 (Q08, Q10, Q14) | 1 (Q10) |
| Abstentions sur répondables | 2 (Q05 injustifiée, Q15 justifiée) | 0 |
| Abstentions correctes (sans réponse) | 5/5 | 5/5 |
| Citations (invalides) | 15 (0) | 18 (0) |
| Précision des citations vs références | 0,80 | 1,00 |

Par question, E6a corrige Q05, Q08, Q14 et Q15 et ne dégrade aucune question. Échec restant :
Q10 (procédure leave-one-out), dont les passages qui décrivent la procédure (lecture 2,
p. 3-4) ne sont pas dans le contexte ; la réponse nomme la méthode sans la décrire. Une seule
exécution par configuration : la variabilité du modèle n'est pas mesurée. Décision :
[ADR 0008](../docs/adr/0008-contexte-chunks-voisins.md).

## Protocole

1. **Relire l'annexe** `datasets/real_v1_answers.json` (réponses attendues, faits requis,
   éléments `weak` et leurs `flags`). Elle est liée au SHA-256 de `real_v1.json`.
2. **Enregistrer** (appels à l'API, ~20 requêtes du quota gratuit, reprise possible) :
   `generate-answers ... --live`. Code de sortie 3 = arrêt sur quota : relancer la même
   commande plus tard.
3. **Noter hors ligne**, autant de fois que nécessaire : `score-answers ...` produit un
   rapport JSON et une fiche de relecture Markdown.
4. **Relire** la fiche et, si besoin, corriger des verdicts dans un fichier passé à
   `--overrides` (`{"q03": {"correctness": "correct", "citation_support": true}}`).

La correction automatique (présence des faits requis) **n'est qu'un indicateur approximatif** :
elle peut manquer une paraphrase correcte ou accepter une réponse fausse contenant le bon mot.

| Issue | Signification |
|---|---|
| `success` | référence retrouvée, réponse correcte, cite un passage de référence |
| `citation_failure` | réponse correcte mais ne cite aucun passage de référence (ou appui refusé en relecture) |
| `generation_failure` | référence retrouvée mais réponse partielle ou incorrecte |
| `unjustified_abstention` | référence retrouvée mais abstention |
| `retrieval_failure_abstained` | référence non retrouvée, abstention (comportement correct) |
| `retrieval_failure_answered` | référence non retrouvée mais réponse donnée (à vérifier) |
| `correct_abstention` / `unsupported_answer` | question sans réponse : abstention / réponse |

## Évaluer sur de vrais documents

1. Copier `datasets/real_template.json` en `datasets/real_v1.json`.
2. Placer les PDF dans `evaluation/documents/` (ignoré par Git : ne pas versionner de
   documents personnels ou confidentiels).
3. Pour chaque question, lire le PDF et copier une citation exacte. Le texte de référence est
   celui **extrait et nettoyé**, qui peut différer de l'affichage du PDF (ligatures, césures,
   espaces multiples, retours à la ligne) : en cas d'erreur « citation trouvée 0 fois »,
   raccourcir la citation à un fragment tenant sur une seule ligne du PDF.
4. Viser au moins 30 questions répondables et 10 sans réponse, réparties FR/EN.
5. Lancer `evaluate` et publier les résultats dans une section **séparée** « réels ».

Les jeux actuels (`real_v1`, son annexe et sa variante `real_v1_en`) contiennent de courts
extraits de documents sous licence CC BY-NC-SA 4.0 ; un nouveau jeu construit sur d'autres
documents doit en vérifier la licence avant toute publication.
