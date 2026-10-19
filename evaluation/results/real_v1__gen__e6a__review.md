# Relecture — real_v1

> La correction est estimée par la présence de faits requis (mots-clés avec variantes FR/EN) : c'est un indicateur approximatif, à confirmer par la relecture humaine (paraphrases non reconnues, erreurs factuelles contenant le bon mot, appui des citations).

Pour corriger un verdict, créez un fichier JSON passé à `score-answers --overrides` :

```json
{"q03": {"correctness": "correct", "citation_support": true, "note": "paraphrase"}}
```

## q01 — paraphrase (en)

**Question** : Why is a maximum-margin boundary less sensitive to noise in the inputs?

**Réponse attendue** : Because the maximum-margin boundary is drawn as far from the training examples as possible, small perturbations (noise) of the inputs do not push examples across it; it is robust to noisy examples, though not to noisy labels.
**Limites de la référence (weak)** : L'idée « frontière éloignée des exemples » reste formulable de nombreuses façons : une paraphrase non listée sera comptée comme absente (relecture). La variante isolée « far » a été retirée : elle correspondait aussi à « so far ».
**Faits requis** : ✔ as far from | far from the training | far from the examples | far from the data | far away from | loin des exemples | loin des donnees | loin des points | le plus loin | eloign* | large margin | wide margin | grande marge | marge large | distance maximale | maximum distance | maximal distance ; ✔ noisy | noise | bruit* | perturbation*

**Recherche** : passage de référence retrouvé = True (rangs [1])
**Génération (réponse)** : A maximum-margin boundary is less sensitive to noisy examples because it draws the separating boundary as far from the training examples as possible.
- [1] lecture2.pdf, page 3 : 3 6.867 Machine learning, lecture 3 (Jaakkola) decision boundary θˆT x + θˆ0 = 0 decision boundary θˆT x = 0 x x θˆ x x θˆ x x x x γgeom x γgeom x x x a) b) Figure 2: a) Maximum margin linear classifier through origin; b) Maximum margin linear classifier with an offset parameter Properties of the ma…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q02 — cross_lingual (fr)

**Question** : Comment appelle-t-on les exemples d'entraînement qui se trouvent exactement sur la marge ?

**Réponse attendue** : They are called support vectors.
**Faits requis** : ✔ support vector* | vecteurs de support | vecteur de support | vecteurs supports | vecteur support

**Recherche** : passage de référence retrouvé = True (rangs [5])
**Génération (réponse)** : Les exemples d'entraînement qui se trouvent exactement sur la marge sont appelés des vecteurs de support (support vectors).
- [5] lecture2.pdf, page 3 : from the training examples as possible makes it robust to noisy examples (though not noisy labels). The maximum margin linear boundary also has the curious property that the solution depends only on a subset of the examples, those that appear exactly on the margin (the dashed lines parallel to the b…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q03 — fine_grained (en)

**Question** : When each training point is set aside in turn and the max-margin classifier is retrained without it, what limits the share of set-aside points that end up misclassified?

**Réponse attendue** : The leave-one-out error is bounded above by the number of support vectors divided by the number of training examples n: only support vectors can be misclassified when they are held out.
**Limites de la référence (weak)** : Le correcteur automatique vérifie seulement que la réponse mentionne les vecteurs de support ; il NE vérifie PAS la borne complète (nombre de vecteurs de support / n), dont la formule est en outre mal extraite du PDF (« # of support vectors leave-one-out CV error ≤ (7) n ») : à juger en relecture.
**Faits requis** : ✔ support vector* | vecteurs de support | vecteur de support

**Recherche** : passage de référence retrouvé = True (rangs [2])
**Génération (réponse)** : The share of set-aside points (leave-one-out cross-validation error) that end up misclassified is limited by the number of support vectors divided by the total number of training examples n.
- [2] lecture2.pdf, page 4 : classifier would generalize to each training example if it had not been part of the training set. A classifier that has a low leave-one-out cross-validation error is likely to generalize well though it is not guaranteed to do so. Now, what is the leave-one-out CV error of the maximum margin linear c…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q04 — cross_lingual (fr)

**Question** : Que se passe-t-il si l'on augmente beaucoup le coût C des violations de marge ?

**Réponse attendue** : If C is increased enough, all slack variables become zero and we recover the maximum-margin (hard-margin) linear separator, if the data are separable.
**Faits requis** : ✔ maximum margin | max-margin | marge maximale | marge maximum | hard margin | hard-margin | marge dure | marge rigide

**Recherche** : passage de référence retrouvé = True (rangs [3])
**Génération (réponse)** : Si l'on augmente la pénalité C pour les violations de marge, alors à un moment donné toutes les variables d'écart s'annulent ($\xi_t = 0$) et l'on retrouve le séparateur linéaire à marge maximale (si c'est possible).
- [3] lecture2.pdf, page 5 : ξt > 0 for some example. The penalty for this violation is Cξt and it is traded-off with the possible gain in minimizing the squared norm of the parameter vector or �θ�2 . If we increase the penalty C for margin violations then at some point all ξt = 0 and we get back the maximum margin linear separ…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q05 — paraphrase (en)

**Question** : How can the max-margin classifier be modified to tolerate mislabeled or non-separable training examples?

**Réponse attendue** : By introducing slack variables that let margin constraints be violated, each violation being penalized (cost C times the slack) in the objective: a relaxed (soft-margin) formulation.
**Faits requis** : ✔ slack | variable d'ecart | variables d'ecart | variable de relachement | variables de relachement | relax* | soft margin | soft-margin | marge souple

**Recherche** : passage de référence retrouvé = True (rangs [5])
**Génération (réponse)** : The maximum margin classifier can be modified to tolerate errors by introducing slack variables for the classification and margin constraints in the optimization problem. This measures the degree to which each margin constraint is violated, associates a cost for the violation, and minimizes these costs together with the norm of the parameter vector.
- [5] lecture2.pdf, page 5 : 5 6.867 Machine learning, lecture 3 (Jaakkola) what would happen if we switched the label of the top right support vector in Figure 3. Allowing misclassified examples, relaxation Labeling errors are common in many practical problems and we should try to mitigate their effect. We typically do not kno…
- [6] lecture2.pdf, page 5 : Perhaps the simplest way to permit errors in the maximum margin linear classifier is to introduce “slack” variables for the classification/margin constraints in the optimization problem. In other words, we measure the degree to which each margin constraint is vio­ lated and associate a cost for the …

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q06 — paraphrase (en)

**Question** : What type of optimization problem has to be solved to train a linear SVM?

**Réponse attendue** : A quadratic programming problem: a quadratic objective in the parameters with linear constraints.
**Faits requis** : ✔ quadratic program* | programmation quadratique | programme quadratique | quadratic optimization | optimisation quadratique

**Recherche** : passage de référence retrouvé = True (rangs [2])
**Génération (réponse)** : To train a linear SVM, a quadratic programming problem has to be solved, where the objective is quadratic in the parameters with linear constraints.
- [2] lecture2.pdf, page 2 : 2 6.867 Machine learning, lecture 3 (Jaakkola) In other words, our classification problem (the data, setup) only tells us something about the ratio θ/γ, not θ or γ separately. For example, the geometric margin is defined only on the basis of this ratio. Scaling θ by a constant also doesn’t change th…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q07 — cross_lingual (fr)

**Question** : Quel effet la pénalité de régularisation a-t-elle sur les paramètres dans la régression à noyau ?

**Réponse attendue** : The regularization penalty pulls all the parameters towards zero (it shrinks them).
**Limites de la référence (strong)** : Le mot isolé « zero » n'est plus accepté : il correspondait aussi à « nonzero » ou à « mis exactement à zéro » (comportement du lasso, que le passage n'attribue qu'aux dimensions non informatives). Une négation (« ne pousse pas vers zéro ») n'est pas détectée.
**Faits requis** : ✔ towards zero | toward zero | towards 0 | toward 0 | close to zero | near zero | vers zero | vers 0 | vers le zero | proche de zero | proches de zero | pres de zero | shrink* | retreci*

**Recherche** : passage de référence retrouvé = True (rangs [2])
**Génération (réponse)** : L'effet de la pénalité de régularisation est de tirer tous les paramètres vers zéro et de fixer explicitement à zéro les dimensions linéaires des paramètres auxquelles les vecteurs de caractéristiques d'entraînement ne se rapportent pas [2].
- [2] lecture3.pdf, page 1 : be minimized, with parameter λ, is given by n J(θ) = �� yt − θT φ(xt) �2 + λ�θ�2 (1) t=1 This form can be derived from penalized log-likelihood estimation (see previous lecture notes). The effect of the regularization penalty is to pull all the parameters towards zero. So any linear dimensions in th…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q08 — fine_grained (en)

**Question** : What computational price do we pay when regression is expressed with inner products between examples?

**Réponse attendue** : Estimating the coefficients requires inverting an n × n matrix (λI + K, where K is the Gram matrix of inner products between the n training examples); this is the cost of working with inner products instead of the feature vectors.
**Limites de la référence (strong)** : La matrice inversée est λI + K ; « inverser la matrice de Gram (n × n) » est accepté comme équivalent. Une taille décrite autrement (ex. « matrice carrée de la taille du jeu d'entraînement ») ne sera pas reconnue.
**Faits requis** : ✔ invert* | invers* ; ✔ n × n | n×n | n x n | nxn | n by n | n-by-n | n*n | n^2 | n² | n par n | gram | kernel matrix | matrice noyau | matrice du noyau | matrice des noyaux

**Recherche** : passage de référence retrouvé = True (rangs [1])
**Génération (réponse)** : When regression is expressed with inner products between examples, finding the estimates requires inverting an n x n matrix, which is the computational cost of dealing with inner products as opposed to handling feature vectors directly.
- [1] lecture3.pdf, page 2 : � � � � 6.867 Machine learning, lecture 7 (Jaakkola) 2 penalty we added. But how do we set αt? The values for αt can be found by insisting that they indeed can be interpreted as prediction differences: n 1 � αt = yt − θT φ(xt) = yt − αt� φ(xt� )T φ(xt) (4) λ t�=1 Thus αt depends only on the actual r…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q09 — multi_passage (fr)

**Question** : Quels types de noyaux peut-on utiliser à la place du simple produit scalaire ?

**Réponse attendue** : Standard examples are polynomial kernels and radial basis (Gaussian, RBF) kernels; lecture 1 also mentions sequence/string kernels and graph kernels.
**Faits requis** : ✔ polynom* ; ✔ gauss* | radial | rbf

**Recherche** : passage de référence retrouvé = True (rangs [None, 14, None])
**Génération (réponse)** : À la place du simple produit scalaire (ou noyau linéaire), on peut utiliser des noyaux polynomiaux, des noyaux gaussiens, des noyaux de séquences et de chaînes (string kernels), ainsi que des noyaux de graphes, entre autres.
- [14] lecture1.pdf, page 113 : There are numerous obvious uses of the inner product here:⟨ ̃x, x⟩ > 0 tells us that ̃x is pointing in the same direction asx (and therefore might perhaps take the same class label). And we can compare tow in the same way to find out which side of the decision boundary the test point is on, leading …
- [15] lecture1.pdf, page 113 : We can also use many functionsK to get many kinds of kernels: polynomial kernels, Gaussian kernels, sequence and string kernels, graph kernels, etc. In many cases, the nature of the kernel depends on the nature ofX and, like explicit feature maps, we can choose random kernels or they can be designed…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q10 — multi_passage (en)

**Question** : How does the validation scheme that holds out a single sample at a time actually work?

**Réponse attendue** : Each training example is held out in turn: the model is trained on all the remaining examples, tested on the held-out one, and the errors are counted (averaged) over all examples. It is k-fold cross-validation with k = n.
**Limites de la référence (weak)** : Le correcteur automatique NE vérifie PAS la procédure complète : ses mots-clés génériques (« remaining », « test ») peuvent figurer dans une réponse incomplète. La relecture doit vérifier les trois étapes : retirer un exemple, entraîner sur le reste, tester sur l'exemple retiré (et répéter pour chacun).
**Faits requis** : ✘ remaining | rest of | the rest | all other* | the others | other examples | restant* | autres exemples | tous les autres | all but | n-1 | n − 1 | n - 1 ; ✘ test* | evaluat* | evalu* | error* | erreur*

**Recherche** : passage de référence retrouvé = True (rangs [None, None, 5])
**Génération (réponse)** : The validation scheme that holds out a single sample at a time is called leave-one-out cross validation, which is considered on small datasets where k equals the number of data points n (k = n).
- [5] lecture1.pdf, page 39 : 4.10. UNDERFITTING 39 60%? More, less? How much of our data can we spare for a validation set (knowing thatwe cannot use this same data for training, thus we reduce the size of our training set)? The technique ofk-fold5 cross validationallows us to use all of it. Suppose a dataset ofn = 100 points, …

**Verdict automatique** : incorrect → issue `generation_failure`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q11 — multi_passage (fr)

**Question** : Quelle pénalité la régression ridge ajoute-t-elle à la fonction de perte ?

**Réponse attendue** : Ridge regression adds a penalty λw² proportional to the squared Euclidean (L2) norm of the weights.
**Faits requis** : ✔ l2 | euclid* | w2 | squared | carre* | norme 2 | w^2 | w⊤w

**Recherche** : passage de référence retrouvé = True (rangs [2, 2, 8, 6])
**Génération (réponse)** : La régression ridge ajoute une pénalité de type λw^2 (ou liée à la norme euclidienne / norme L2) à la fonction de perte.
- [2] lecture1.pdf, page 38 : that is the difference between the two methods. Consider a simple example of fitting a line to a set of data points{(xi,yi)}ni=1; each xi is a single dimension (e.g., as in Figure 1.1), thus we only regularize (restrict) the slope w of the line (a single parameter). Ridge regression adds a penalty λ…
- [6] lecture1.pdf, page 115 : B.4. EXEMPLIFIED: KERNEL RIDGE REGRESSION 115 B.4 Exemplified: Kernel Ridge Regression We exemplify on ridge regression. The estimator for this method is derived (you may recall from earlier topics on regression) by minimizing the error function as follows: E(w) = (y− Xw)⊤(y− Xw) +λw⊤w =... ▷ (take …

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q12 — fine_grained (en)

**Question** : Which of ridge or lasso can drive some weights exactly to zero, and what is the consequence?

**Réponse attendue** : Lasso (L1 penalty) can set some coefficients exactly to zero, so it performs feature selection: the corresponding features can be discarded.
**Faits requis** : ✔ lasso | l1 ; ✔ select* | discard* | elimin* | supprim* | ecart*

**Recherche** : passage de référence retrouvé = True (rangs [2])
**Génération (réponse)** : Lasso regression can drive coefficients (weights) to zero. The consequence is that it performs feature selection; if a coefficient is zero, the corresponding feature is worthless and can be discarded from the data.
- [2] lecture1.pdf, page 38 : With ridge regression we have the advantage thatw2 has a straightforward derivative (thus a closed form solution is possible). Lasso regression has the advantage that it may set coefficients to0 and thus performs feature selection (ifwj = 0, then thej-th feature xj is worthless as the model only eve…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q13 — numeric (fr)

**Question** : Quelle proportion d'exemples distincts trouve-t-on dans un échantillon bootstrap utilisé pour le bagging ?

**Réponse attendue** : About 63.2% of the samples in a bootstrap sample are unique (the others are duplicates).
**Limites de la référence (strong)** : Les arrondis « environ 63 % » sont acceptés (le passage donne ≈ 63,2 %). « Deux tiers » ou « 1 − 1/e » ne sont pas acceptés : le premier est inexact, le second ne figure pas dans le passage (relecture).
**Faits requis** : ✔ 63.2 | 63,2 | 63 % | 63% | environ 63 | about 63 | approximately 63 | roughly 63 | around 63 | pres de 63 | ≈ 63 | ~63 | ~ 63

**Recherche** : passage de référence retrouvé = True (rangs [3])
**Génération (réponse)** : Environ 63,2 % des échantillons d'un échantillon bootstrap (utilisé pour le bagging) sont uniques, les autres étant des doublons.
- [3] lecture1.pdf, page 72 : • samples are independent of each other (an important assumption; recall Section 2.7); • Dm can contain the same number of samples asD (or fewer, or more); and • ≈ 63.2% of samples inDm will be unique, with others being duplicates. 8.7.2 Random Forest There are other ways to increase diversity. For …

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q14 — numeric (en)

**Question** : What train/validation split percentages does the course give as an example?

**Réponse attendue** : The course's example fits the model on 60% of the training set and uses the remaining 40% as a validation set.
**Limites de la référence (strong)** : Référence vérifiée : l'exemple 60 % / 40 % figure entièrement en page 38. La page 39 ne le prolonge pas : elle s'ouvre sur une question rhétorique (« 60%? More, less? ») puis présente un autre exemple (validation croisée à 5 plis, 80 / 20). Une citation de la page 39 compte donc à juste titre comme hors référence. Les faits exigent un pourcentage pour ne pas accepter « instances 21 à 40 ».
**Faits requis** : ✔ 60% | 60 % | 60 percent | 60 per cent | 60 pour cent ; ✔ 40% | 40 % | 40 percent | 40 per cent | 40 pour cent

**Recherche** : passage de référence retrouvé = True (rangs [4])
**Génération (réponse)** : The course gives an example of fitting a model on 60% of the training set and using the remaining 40% as a validation set to check model performance.
- [4] lecture1.pdf, page 38 : test data, as we never have access to our ‘real’ test data (which our deployed model will see). But we can simulate test data, by cutting off a bit of the training set. We call this a validation set. For example, we fit our model on 60% of the training set, and use the remaining 40% to check model p…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## q15 — cross_lingual (fr)

**Question** : Comment choisir le nombre de groupes dans l'algorithme des k-moyennes ?

**Réponse attendue** : Increase k gradually and look for a sudden drop in the error (an 'elbow' in the plot of k versus error); silhouette analysis, which scores clusterings between -1 and +1, is another common technique.
**Faits requis** : ✔ elbow | coude ; ✔ silhouette*

**Recherche** : passage de référence retrouvé = True (rangs [15, 15])
**Génération (réponse)** : Pour choisir le nombre de groupes (k) dans l'algorithme des k-moyennes, on peut augmenter k progressivement et rechercher une baisse soudaine de l'erreur, ce qui apparaît comme un « coude » (elbow) sur un graphique représentant k en fonction de l'erreur. Une autre technique couramment utilisée est l'analyse de silhouette (silhouette analysis), qui attribue un score basé sur le calcul de la distance intra-groupe par rapport à la distance inter-groupe.
- [15] lecture1.pdf, page 81 : look at Eq. (10.1) and note that the errorE should decrease inversely tok (because clusters are smaller), with k = n leading to a minimum (zero distance from each point to its cluster centre). Such a solution would be of no use to us. We can instead increasek gradually and look for asudden drop in e…

**Verdict automatique** : correct → issue `success`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## u01 — unanswerable_on_topic (en)

**Question** : What is the time complexity of training an SVM with the SMO algorithm?

**Recherche** : —
**Génération (abstention)** : The provided documents do not contain information about the time complexity of training an SVM with the SMO algorithm.
**Raison** : le modèle juge que les passages ne contiennent pas la réponse

**Verdict automatique** : — → issue `correct_abstention`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## u02 — unanswerable_on_topic (fr)

**Question** : Comment la normalisation par lots (batch normalization) accélère-t-elle l'entraînement des réseaux profonds ?

**Recherche** : —
**Génération (abstention)** : Les documents fournis ne contiennent pas d'information sur la normalisation par lots (batch normalization) ni sur la manière dont elle accélère l'entraînement des réseaux profonds.
**Raison** : le modèle juge que les passages ne contiennent pas la réponse

**Verdict automatique** : — → issue `correct_abstention`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## u03 — unanswerable_on_topic (en)

**Question** : How does AdaBoost reweight the training examples that were misclassified?

**Recherche** : —
**Génération (abstention)** : The provided documents do not contain information about how AdaBoost reweights misclassified training examples.
**Raison** : le modèle juge que les passages ne contiennent pas la réponse

**Verdict automatique** : — → issue `correct_abstention`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## u04 — unanswerable_on_topic (fr)

**Question** : Quelle valeur du paramètre C est recommandée par défaut pour un SVM ?

**Recherche** : —
**Génération (abstention)** : Les documents fournis ne contiennent pas d'information sur une valeur recommandée par défaut pour le paramètre C d'un SVM.
**Raison** : le modèle juge que les passages ne contiennent pas la réponse

**Verdict automatique** : — → issue `correct_abstention`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____

## u05 — unanswerable_on_topic (fr)

**Question** : Combien d'arbres faut-il utiliser dans une forêt aléatoire ?

**Recherche** : —
**Génération (abstention)** : Les documents fournis ne contiennent pas d'information indiquant le nombre d'arbres qu'il faut utiliser dans une forêt aléatoire.
**Raison** : le modèle juge que les passages ne contiennent pas la réponse

**Verdict automatique** : — → issue `correct_abstention`

**Relecture** : correction = ____ ; citations appuient la réponse = oui / non ; note = ____
