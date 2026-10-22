# Protocole - Prédiction du vieillissement humide d'une pièce polymère (E/E0)

Projet de recherche SeaTech MTX3A - axe « pure data-driven ».

Ce document décrit la démarche complète, de la recherche bibliographique jusqu'à la
prédiction dans l'application. Chaque étape indique ce qui est **implémenté** dans le dépôt
et ce qui reste **à produire** (section 7).

---

## 0. Question posée et périmètre

**Question.** Pour une pièce polymère de matériau et d'épaisseur donnés, exposée à une
température T et une humidité relative HR constantes pendant une durée t, quel est son
module d'Young résiduel E/E0, et donc E = (E/E0) × E0 ?

**Ce que l'approche peut faire.** Un modèle appris sur des courbes publiées **interpole**
entre les conditions et les durées couvertes par ces courbes.

**Ce qu'elle ne peut pas faire.** Un modèle à arbres (Random Forest, XGBoost) **n'extrapole
pas** : hors du domaine d'entraînement, il renvoie la valeur de sa dernière feuille. Les
conditions d'usage courantes (local à 20-25 °C, HR modérée, plusieurs années) sont en général
hors du domaine des essais accélérés publiés (40-90 °C, forte HR, 1 000 à 5 000 h). Ce point
est une limite de fond de l'axe data-driven et doit être discuté dans le rapport.

**Statut des informations** (à respecter dans tout le rapport) :

| Nature | Exemples | Où elle se trouve |
|---|---|---|
| Donnée réelle | points digitalisés, valeurs de tableaux publiés | `data/dataset_reel.csv`, colonnes de mesure |
| Valeur calculée | E/E0 normalisé, E en MPa affiché, métriques | scripts, `results/` |
| Hypothèse | HR = 100 % en immersion, tolérances du domaine, transférabilité de E/E0 | ce document, section 8 |

---

## 1. Installation

Python 3.11 ou plus récent (testé en 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_chaine.py      # doit afficher : TOUS LES TESTS PASSENT
```

Le test de bout en bout travaille sur un jeu **synthétique** écrit dans un dossier
temporaire. Il ne touche jamais `models/`.

---

## 2. Constitution du dataset réel

### 2.1 Sélection des sources

Une source est retenue si elle donne **explicitement** :
- la température et l'humidité de vieillissement (ou le milieu d'immersion) ;
- l'épaisseur des éprouvettes ;
- une valeur du module à t = 0 **mesurée dans la même étude** ;
- le type d'essai (traction ou flexion) ;
- l'état des éprouvettes au moment de l'essai (testées humides ou reséchées), si possible.

Chaque article consulté est noté dans `data/journal_sources.csv`, **y compris ceux rejetés**,
avec la raison du rejet. Sans ce journal, un biais de sélection est invisible.

Le DOI est **recopié depuis l'article lui-même**, jamais depuis une liste générée ou un
moteur de réponse. Chaque DOI est ouvert au moins une fois pour vérifier qu'il mène au bon
article. `verifier_dataset.py` vérifie le format d'un DOI, pas son existence.

### 2.2 Digitalisation (WebPlotDigitizer)

1. Calibrer chaque axe sur deux graduations **éloignées** ; déclarer les axes logarithmiques
   comme tels.
2. Si l'abscisse est en √t (h^0,5 ou j^0,5), élever au carré **avant** de convertir en heures.
3. Un dataset WebPlotDigitizer = une courbe = un `curve_id`.
4. Digitaliser le **centre des marqueurs expérimentaux**, jamais la courbe d'ajustement des
   auteurs : la digitaliser reviendrait à injecter leur modèle comme donnée réelle. Une
   figure sans marqueurs n'est pas utilisable.
5. Barres d'erreur : digitaliser l'extrémité ; incertitude = |extrémité − centre|, exprimée
   en unités de E/E0. Type (`ecart_type`, `min_max`, `IC95`) selon la légende, `inconnu` si
   la légende ne le précise pas.
6. **Erreur de digitalisation mesurée, pas supposée** : un second membre du groupe
   re-digitalise environ 10 % des courbes, tirées au hasard. L'écart moyen entre les deux
   lectures donne l'incertitude de digitalisation à citer dans le rapport.

### 2.3 Remplissage du CSV

**Le dataset actuel est généré par script, pas saisi à la main :**

```bash
python data/construire_dataset.py --pdf-dir sources_pdf --extraction sources_pdf/Extraction_vieillissement_hygrothermique.csv
```

Le script refait à chaque exécution :
- la lecture des valeurs de **tableaux et de texte**, avec la page citée dans `source_fig` ;
- la **digitalisation des figures**. Pour une figure vectorielle, il lit les coordonnées exactes
  des marqueurs dans le PDF. Pour une figure raster, il détecte le centre des marqueurs par
  analyse d'image. Dans les deux cas, la calibration se fait sur les graduations ;
- des **contrôles de calibration** contre les valeurs tabulées. Par exemple, la Fig. 4.4a de
  Chilali doit redonner le Tab. 4.2 aux bornes. Si un contrôle échoue, le script s'arrête et
  n'écrit aucun fichier ;
- la reprise de l'extraction tierce, marquée `verification = non_controle` car les PDF de ces
  articles n'ont pas été fournis.

Il écrit aussi `data/sources_sha256.csv`, l'empreinte de chaque PDF utilisé. Les PDF eux-mêmes
restent hors du dépôt, dans `sources_pdf/` qui est ignoré par Git, pour des raisons de droits
d'auteur.

Pour ajouter une source à la main, partir de `data/template_dataset_reel.csv` (dictionnaire
des colonnes en commentaire dans le fichier). Règles :

- **Une ligne = un point mesuré.** Les métadonnées sont constantes dans une courbe, sauf
  `source_fig` et `extraction`, qui décrivent la provenance de chaque point : une même courbe
  peut mêler un tableau et une figure.
- **Donnée réelle et valeur calculée sont séparées.** `E_t_MPa` contient le module publié
  (lu ou digitalisé). `Propriete_residuelle = E_t_MPa / E0_MPa` est calculé, et le vérificateur
  bloque toute incohérence entre les deux.
- **`verification`** vaut `controle_pdf` si la valeur a été relue sur le PDF, `non_controle`
  sinon.
- **Temps en heures** (1 j = 24 h, 1 sem = 168 h) ; point initial à `Temps_h = 0` exactement.
- **Normalisation courbe par courbe** : E/E0 = E(t) / E(t=0) de la **même** courbe ; la ligne
  t = 0 vaut donc 1.
- **Polyamides** : vérifier que l'état de référence est sec (sortie de moulage) et non
  conditionné ; sinon E0 ne correspond pas au même état. Le noter dans `commentaire`.
- **Immersion** : `Humidite_HR = 100` par convention, la réalité est portée par `milieu`
  et `fluide`.
- **`etat_essai`** : `humide` si l'éprouvette est testée à la sortie de l'enceinte,
  `reseche` si elle est séchée avant l'essai, `reconditionne` si elle est rééquilibrée
  (typiquement 23 °C / 50 %HR) avant l'essai, `inconnu` si l'article ne le dit pas. C'est la
  seule façon de séparer la plastification (réversible) de la dégradation (irréversible).
- **Case manquante = case vide.** Ne jamais compléter, ne jamais imputer.

### 2.4 Contrôle

```bash
python src/verifier_dataset.py data/dataset_reel.csv --sortie results
```

- **BLOQUANT** : courbe rejetée → corriger la saisie ou retirer la courbe.
- **EXCLU** : moins de 2 points, c'est-à-dire pas de point vieilli en plus de la référence
  t = 0 → conservée dans le CSV, écartée de l'entraînement. Une courbe à 2 points apporte une
  mesure valide pour l'apprentissage. Elle ne suffit pas pour une analyse cinétique
  (`--min-points 3` dans ce cas).
- **AVERT** : courbe conservée, information manquante ou suspecte → à lire une par une.

Répéter jusqu'à ce que chaque courbe rejetée soit corrigée ou volontairement écartée, avec
la raison dans `journal_sources.csv`. Committer le CSV **et** `results/rapport_courbes.csv`.

Lire aussi la couverture affichée :
- le tableau T × HR, compté en courbes ;
- la corrélation de Spearman entre T et HR, qui annonce si SHAP pourra séparer leurs effets ;
- le nombre de séries à HR fixe couvrant au moins 3 températures. S'il vaut 0, le test
  d'Arrhenius de la section 7 est impossible.

---

## 3. Choix du périmètre d'entraînement

Deux options, à trancher **avant** de regarder un score :

| Option | Commande | Avantage | Coût |
|---|---|---|---|
| Tous les matériaux | `python src/train.py --data ...` | plus de données | erreur irréductible entre polymères aux mêmes conditions |
| Un seul matériau | `python src/train.py --data ... --materiau <material_id>` | pas de mélange de matériaux | peu de courbes ; protocoles C et D impossibles |

Règle de décision proposée : entraîner sur un seul matériau seulement si
`verifier_dataset.py` montre pour lui au moins une dizaine de courbes et au moins 3
températures. Ce seuil est un choix d'ingénieur, à justifier.

Le jeu de variables du modèle déployé (`--features-final imposees | contexte | materiau`)
est lui aussi choisi a priori et consigné dans le rapport. Le choisir d'après le score du
split officiel biaiserait ce score.

---

## 4. Entraînement et validation

```bash
python src/train.py --data data/dataset_reel.csv [--materiau <id>] [--features-final contexte]
```

### 4.1 Protocoles (mêmes données, mêmes modèles, mêmes hyperparamètres)

| Protocole | Découpage | Rôle | À rapporter |
|---|---|---|---|
| A | 80/20 aléatoire sur les **lignes** | témoin de fuite : les points d'une même courbe tombent des deux côtés | à montrer, **jamais** comme score |
| B | 80/20 sur les **courbes** (GroupShuffleSplit), 20 tirages | score de référence | moyenne ± écart-type |
| C | LeaveOneGroupOut par **matériau** | matériau jamais vu | R², RMSE sur les prédictions poolées ; RMSE médiane par matériau |
| D | LeaveOneGroupOut par **source** | labo et lot jamais vus (fuite résiduelle de B) | idem C |

L'écart A − B mesure la fuite de données ; l'écart B − C mesure la difficulté de
généralisation à un nouveau matériau. Ces deux écarts sont des résultats à part entière.

### 4.2 Split officiel et modèle déployé

- `GroupShuffleSplit(test_size=0.2, random_state=42)` sur `curve_id`. Aucune courbe ne se
  retrouve des deux côtés, ce qui est vérifié par une assertion.
- Le modèle XGBoost déployé est entraîné sur les 80 % de courbes et évalué sur les 20 %.
  L'option `--refit-complet` le réentraîne sur 100 % après évaluation ; le score test devient
  alors une estimation, légèrement pessimiste.
- Le score d'un unique split sur quelques dizaines de courbes est fragile : c'est la
  distribution du protocole B qui fait foi.

### 4.3 Métriques

R², RMSE et MAE **hors points t = 0**. Ces points valent 1 par construction et gonflent
artificiellement le R². Les valeurs t = 0 inclus sont données à côté, pour montrer l'écart.

**Référence obligatoire : `Constante_moyenne`**, qui prédit la moyenne d'entraînement sans
regarder les variables. Un modèle qui ne fait pas mieux qu'elle en RMSE n'a rien appris
d'utilisable dans ce protocole.

**Le R² est instable quand les courbes test varient peu.** Avec des courbes test comme Wang
(E/E0 entre 0,88 et 1), la variance de référence est minuscule et le R² explose en négatif.
Pour comparer les modèles, la RMSE et la MAE font foi. Le R² est donné en médiane, à titre
indicatif.

### 4.4 Interdits (fuites de données)

- Régler les hyperparamètres ou faire de l'early stopping sur le jeu test.
- Choisir le modèle ou les variables d'après le score test.
- Normaliser avec des statistiques calculées sur tout le jeu. Ici, la normalisation est
  dans le pipeline, donc apprise sur l'entraînement seul.
- Évaluer sur des courbes ou des sources ayant servi à l'entraînement. L'application le
  détecte.

### 4.5 Sorties

| Fichier | Contenu |
|---|---|
| `models/modele.json` | modèle XGBoost, format JSON natif (pas de pickle/joblib) |
| `models/features.json` | variables, split, domaine d'entraînement, métriques, SHA-256 du CSV, versions |
| `results/protocoles.csv` | tableau A/B/C/D × jeux de variables × modèles |
| `results/split_officiel.csv` | courbes train / test |
| `results/split_officiel_scores.csv` | Ridge / RF / XGBoost sur le split officiel |
| `results/predictions_test.csv` | prédictions et résidus sur les courbes test |
| `figures/parite_residus_test.png` | parité et résidus en fonction du temps |

---

## 5. Prédiction du cas fixé

Renseigner le bloc `CAS` en tête de `src/predire_piece.py` (T, HR, durée, épaisseur, E0),
puis :

```bash
python src/predire_piece.py [--seuil 0.8]
```

Le script refuse de tourner tant qu'un champ obligatoire est vide. Il renvoie E/E0 au temps
fixé, E en MPa si E0 est fourni, et un **verdict** :

| Verdict | Signification | Ce qu'on peut écrire |
|---|---|---|
| `DOMAINE` | au moins une courbe d'entraînement à ±10 °C et ±10 %HR, même milieu, même état d'essai, couvrant la durée | valeur interpolée, avec la RMSE test comme ordre de grandeur de l'erreur |
| `HORS_CONDITIONS` | aucune courbe voisine | **pas de prédiction** : relève de la baseline physique |
| `HORS_TEMPS` | courbes voisines trop courtes | **pas de prédiction** au-delà de leur durée maximale |
| `HORS_EPAISSEUR` | épaisseur hors plage | **pas de prédiction** |

Les tolérances ±10 °C et ±10 %HR sont un choix d'ingénieur (`TOL_T_C`, `TOL_HR`), à
justifier. La RMSE test n'est **pas** un intervalle de confiance pour le cas étudié.

---

## 6. Application Streamlit

```bash
streamlit run app.py
```

- **Prédiction du cas** : saisie du cas (sans valeur par défaut), E/E0 et E, verdict, courbe
  (rouge hors domaine), carte des conditions d'entraînement, export CSV.
- **Tester avec des données expérimentales** : import d'un CSV au format du template.
  Il est vérifié, puis les prédictions sont comparées aux mesures. L'application signale si
  des courbes ou des sources ont servi à l'entraînement ; dans ce cas, le score n'est pas une
  validation.
- **Modèle et traçabilité** : dataset, SHA-256, split, métriques, protocoles, versions.

L'application ne contient aucune logique de prédiction propre : elle appelle les mêmes
fonctions que la ligne de commande (`evaluer_cas`, `construire_features`). Après un
réentraînement, le bouton « Recharger le modèle » ou un rechargement de la page suffit.

---

## 6 bis. Modèle hybride (prédiction de l'application)

**Principe.** Chaque matériau des sources fournies reçoit le niveau de modèle que **ses** données permettent
d'identifier ; aucun paramètre n'est inventé. Les cas hors du domaine identifiable sont **refusés** avec la raison
(exception `NonPredictible`), au lieu d'être extrapolés en silence. Données : `data/donnees_physiques.csv`,
produit par `python data/digitaliser_physique.py` (tableaux, texte, figures vectorielles et raster avec contrôles).

| Matériau | Source | Données | Modèle | T | HR | Épaisseur |
|---|---|---|---|---|---|---|
| PA6 Goodfellow | Le Gac 2017 | relues | chaîne : isotherme, Fick, E(M), Tg(M), GP | équilibre seulement (D à 25 °C) | oui | Fick |
| Époxy RIMR 135 | Gibhardt 2022 | relues | chaîne + GP | Arrhenius 8-50 °C | oui | Fick |
| PA6 Akulon F223-D | Zarbali 2026 | relues | chaîne (D par humidité), sans mesure E(t) | **non publiée** | oui | Fick |
| Lin / époxy, lin / Elium | Chilali 2017 | relues | apparent : D publié, amplitude ajustée | **non publiée** | immersion seule | Fick |
| PA6 Takmid | Guerira 2016 | relues | apparent : D ajusté (incohérent avec le D publié) | **non publiée** | immersion seule | douteux |
| Époxy Wang (flexion) | Wang 2016 | **non relues** | apparent : D et Δ∞(a_w) ajustés | 60 °C seulement | 65-100 % | Fick |
| PLA 2002D | Niaounakis 2011 | **non relues** | cinétique : Arrhenius, 20-50 °C | oui | 80 % seulement | sans effet |
| PLA BQ (FFF) | Bergaliyeva 2022 | **non relues** | cinétique, une condition | 50 °C seulement | 70 % seulement | non publiée |
| PLA L130, PLA L130 + amidon | Goetjes 2024 | **non relues** | **aucun** : pas de E0, une durée | - | - | - |

Non exploitables : PA6 Alyamid (Ksouri 2018, aucune valeur chiffrée) ; matériaux du fichier
« dataset_reel_..._complet » (valeurs introuvables dans les sources citées).

**Incertitude** : 200 tirages des paramètres (covariances des ajustements) x écart de modèle (GP pour les
chaînes avec mesures E(t), résidu de l'ajustement sinon), pondéré par l'avancement ; IC 95 %.
**Contraintes** : E(t=0) = E0 ; 0 < E ≤ E0 ; E(t) non croissant ; équilibre indépendant de l'épaisseur et temps
de saturation en h² (vérifiés pour tous les matériaux par `tests/test_hybride.py`).

**Validation** (`python src/hybride.py --rapport`, planche `figures/hybride_tous_materiaux.png`) : une seule
validation indépendante (PA6 Goodfellow, Le Gac Fig. 7a, jamais ajustée : RMSE 0,13 en E/E0). Toutes les autres
RMSE sont des qualités d'ajustement, pas des capacités de prédiction ; PA6 Akulon n'a aucune mesure E(t).

**Résultat à discuter** : pour le module du PA6, l'effet de l'eau et celui de la température ne se superposent
pas en fonction de Tg − T ; le modèle ne combine donc pas eau et température de service, et le signale.

**Ce que le modèle ne fait pas** : hydrolyse au-delà des durées mesurées, vieillissement physique, séchage et
cycles, effet de la température de service sur le module, diffusion non fickienne explicite, diffusion 3D des
composites, état initial autre que sec.

---

## 7. Analyses du rapport - À PRODUIRE (non incluses dans cette version)

1. **Baseline physique Fick + Arrhenius**, ajustée par `scipy.optimize.curve_fit`
   (réutiliser `src/physics.py`). C'est la référence à battre et le seul outil capable
   d'extrapoler. Ses paramètres (D, ΔE∞) sont propres à chaque matériau : décider entre
   ajustement par matériau, par famille ou global.
2. **Courbes d'apprentissage** : score en fonction du nombre de **courbes** d'entraînement,
   avec découpage par courbe.
3. **Résidus** en fonction de t, T, HR, épaisseur, matériau, source.
4. **Extrapolation** : entraîner sur les temps courts, projeter à 10 000 h, superposer la
   baseline physique. Figure obligatoire.
5. **SHAP** : importance ≠ causalité. Avec T et HR corrélées, la répartition des
   contributions entre elles est en partie arbitraire.
6. **Test d'Arrhenius**, séparé de SHAP : ln(t pour atteindre un seuil de E/E0) en fonction
   de 1/T en **kelvin**, à HR fixe, pour chaque série à au moins 3 températures.
7. **Cohérence physique** : E/E0(t=0) ≈ 1 ; E/E0 non croissant en t, sauf post-réticulation
   documentée ; effet de HR de même signe que la prise en eau ; écart humide / reséché
   cohérent avec une plastification réversible.

---

## 8. Hypothèses et limites (à reprendre dans le rapport)

- T et HR constantes : les cycles réels (jour/nuit, saisons) et l'endommagement par
  gonflement/séchage alterné ne sont pas représentés.
- HR = 100 % en immersion : convention. L'activité de l'eau salée est légèrement inférieure
  à 1, et la prise en eau en immersion peut différer de celle à 100 %HR vapeur.
- La perte relative E/E0 mesurée sur éprouvettes est supposée transférable à la pièce
  réelle ; procédé, cristallinité et orientation peuvent différer.
- La propagation de l'incertitude sur E0 dans E/E0 est négligée.
- Une cible E/E0 unique mélange plastification (réversible) et dégradation (irréversible)
  lorsque `etat_essai` est `humide` ou `inconnu`. Le sujet affirme à tort que le
  vieillissement humide est irréversible.
- Aux mêmes conditions, deux polymères différents donnent des E/E0 différents : c'est une
  erreur irréductible si le matériau n'est pas décrit.
- Les points d'une même courbe ne sont pas indépendants, et les courbes d'une même source
  non plus : d'où les protocoles B, C et D.
- Pas d'extrapolation hors du domaine d'entraînement (section 0).

---

## 9. Traçabilité et reproductibilité

- À committer : `data/dataset_reel.csv`, `data/journal_sources.csv`,
  `results/rapport_courbes.csv`, `models/`, `results/`, `figures/`.
- `models/features.json` contient le SHA-256 du CSV d'entraînement. Un CSV modifié, même
  d'un caractère, produit une autre empreinte.
- Graine unique `--seed 42`, versions figées dans `requirements.txt`, versions réellement
  utilisées enregistrées dans `features.json`.
- Modèle au format JSON natif XGBoost : un pickle/joblib devient illisible dès que la version
  de XGBoost change.
- Les données synthétiques (`tests/`) sont refusées par défaut ; un modèle entraîné avec
  `--autoriser-synthetique` est marqué comme tel et l'application l'affiche en rouge.

---

## 10. Check-list avant rendu

- [ ] `python tests/test_chaine.py` passe sur une installation propre
- [ ] Aucun DOI non vérifié ; `journal_sources.csv` à jour, rejets compris
- [ ] Erreur de digitalisation mesurée par double lecture (section 2.2)
- [ ] `verifier_dataset.py` : chaque courbe rejetée corrigée ou justifiée
- [ ] Périmètre (tous matériaux / un matériau) et jeu de variables choisis avant le score test
- [ ] Rapport : protocole B en moyenne ± écart-type, écarts A − B et B − C discutés
- [ ] Métriques hors t = 0
- [ ] Cas d'usage : verdict affiché et assumé (DOMAINE ou non)
- [ ] Figures de la section 7 produites, baseline physique comprise
- [ ] Hypothèses de la section 8 reprises dans le rapport
