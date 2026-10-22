# Prédiction du vieillissement humide d'une pièce polymère (E/E0)

Projet de recherche SeaTech MTX3A, axe « pure data-driven ».
La démarche complète, les conventions et les limites sont dans **[PROTOCOLE.md](PROTOCOLE.md)**.

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_chaine.py                              # pipeline data-driven (données synthétiques)
python tests/test_hybride.py                             # cohérence physique du modèle hybride

python src/hybride.py --rapport                          # lois ajustées, validation, figures du rapport
streamlit run app.py                                     # application (modèle hybride + page XGBoost)
```

Reconstruire les données depuis les PDF (dans `sources_pdf/`, non versionné) :

```bash
python data/digitaliser_physique.py --pdf-dir sources_pdf --extraction sources_pdf/Extraction_vieillissement_hygrothermique.csv
python data/construire_dataset.py --pdf-dir sources_pdf --extraction sources_pdf/Extraction_vieillissement_hygrothermique.csv
python src/verifier_dataset.py data/dataset_reel.csv --sortie results
python src/train.py --data data/dataset_reel.csv --features-final materiau   # référence XGBoost
```

## Deux modèles

| Modèle | Rôle | Fichiers |
|---|---|---|
| **Hybride** (application principale) | 11 matériaux des sources ; pour chacun le modèle que ses données permettent (chaîne physique, cinétique apparente, cinétique de dégradation ou données seules) ; E0 propre au matériau ; incertitudes, points de vigilance et refus motivés hors domaine | `src/hybride.py`, `data/donnees_physiques.csv` |
| **Data-driven** (référence exigée par le sujet) | Ridge, Random Forest, XGBoost ; protocoles A/B/C/D ; split 80/20 par courbe | `src/train.py`, `data/dataset_reel.csv`, `pages/2_Comparaison_XGBoost.py` |

## Organisation

| Chemin | Rôle |
|---|---|
| `data/construire_dataset.py` | construit `dataset_reel.csv` depuis les PDF (tableaux, texte, digitalisation contrôlée) |
| `data/dataset_reel.csv` | dataset réel généré (ne pas éditer à la main : relancer le script) |
| `data/sources_sha256.csv` | empreintes des PDF et de l'extraction utilisés |
| `data/template_dataset_reel.csv` | en-tête et dictionnaire des colonnes du dataset réel |
| `data/journal_sources.csv` | journal de TOUS les articles consultés, rejets compris |
| `src/verifier_dataset.py` | contrôle du schéma, des unités, du point t=0, de la couverture |
| `src/train.py` | protocoles A/B/C/D, split 80/20 par courbe, modèle JSON XGBoost |
| `src/predire_piece.py` | prédiction d'un cas fixé et verdict de domaine |
| `app.py` | application Streamlit : modèle hybride |
| `pages/2_Comparaison_XGBoost.py` | page de l'application : modèle XGBoost |
| `src/hybride.py` | modèle hybride : ajustements, incertitudes, vigilance, validation, figures |
| `data/digitaliser_physique.py` | extrait les lois physiques des PDF (figures vectorielles, tableaux, texte) |
| `data/donnees_physiques.csv` | données physiques sourcées (absorption, isotherme, D, Tg, E(M), E(t)) |
| `tests/` | jeu synthétique (tests du code uniquement) et test de bout en bout |

Aucune donnée du dossier `tests/` n'est une donnée expérimentale.
