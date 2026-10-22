#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement et validation - découpage 80/20 PAR COURBE.

Usage (depuis la racine du dépôt) :
    python src/train.py --data data/dataset_reel.csv
    python src/train.py --data data/dataset_reel.csv --repetitions 5 --jeux contexte
    python src/train.py --data data/dataset_reel.csv --features-final materiau --ponderer-courbes

Protocoles (même données, mêmes modèles, mêmes hyperparamètres) :
    A  80/20 aléatoire sur les LIGNES      témoin de fuite, ne pas utiliser comme score
    B  80/20 par COURBE (GroupShuffleSplit) score de référence, répété sur --repetitions graines
    C  LeaveOneGroupOut par MATÉRIAU       usage réel : matériau jamais vu
    D  LeaveOneGroupOut par SOURCE         même labo / même lot jamais vus (fuite résiduelle de B)

Split officiel : GroupShuffleSplit(test_size=0.2, random_state=--seed) sur curve_id.
Le modèle XGBoost déployé (models/modele.json) est entraîné sur ses 80 % de courbes,
et son score est mesuré sur les 20 % restants. Hyperparamètres FIXÉS A PRIORI : aucun
réglage, aucun early stopping sur le jeu test (ce serait une fuite).

Métriques principales calculées HORS points t=0 : ces points valent 1 par construction
et gonfleraient R² artificiellement. Les métriques t=0 inclus sont données à côté.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, ShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    from src.verifier_dataset import imprimer_rapport, verifier
except ImportError:  # exécution par "python src/train.py"
    from verifier_dataset import imprimer_rapport, verifier

CIBLE = "Propriete_residuelle"
KELVIN = 273.15
JEUX = ("imposees", "contexte", "materiau")
RACINE = Path(__file__).resolve().parents[1]
VARIABLES_BASE = ["Temperature_C", "Humidite_HR", "Temps_h", "Epaisseur_mm"]


# =========================================================================== features
def categories_depuis(df: pd.DataFrame) -> dict:
    """Vocabulaire des variables catégorielles matériau. Ce n'est pas une fuite de cible :
    seule la liste des modalités est lue, jamais la propriété mesurée."""
    cats = {}
    for col in ("famille", "renfort"):
        v = df[col] if col in df else pd.Series("", index=df.index)
        v = v.fillna("").astype(str).str.strip().replace("", "inconnu")
        cats[col] = sorted(v.unique().tolist())
    return cats


def construire_features(df: pd.DataFrame, jeu: str, pour: str = "arbre",
                        categories: dict | None = None) -> pd.DataFrame:
    """Fonction UNIQUE de construction des variables, partagée par train.py et app.py
    (évite tout écart entraînement / service).

    jeu  : imposees  -> T, HR, t, h (colonnes imposées par le sujet)
           contexte  -> + immersion, état d'essai, type de module
           materiau  -> + famille et renfort en indicatrices (descripteurs transférables)
    pour : arbre     -> variables brutes (un arbre est invariant aux transformations
                        monotones d'UNE variable)
           lineaire  -> variables physiques : 1/T en KELVIN, activité de l'eau,
                        sqrt(t)/h (Fick temps courts), ln(1+t)
    """
    if jeu not in JEUX:
        raise ValueError(f"jeu '{jeu}' inconnu, choisir parmi {JEUX}")
    T = df["Temperature_C"].to_numpy(float)
    HR = df["Humidite_HR"].to_numpy(float)
    t = df["Temps_h"].to_numpy(float)
    h = df["Epaisseur_mm"].to_numpy(float)
    X: dict[str, np.ndarray] = {}
    if pour == "arbre":
        X.update({"Temperature_C": T, "Humidite_HR": HR, "Temps_h": t, "Epaisseur_mm": h})
    elif pour == "lineaire":
        X["inv_T_K"] = 1.0 / (T + KELVIN)            # 1/K ; jamais 1/(T en degC)
        X["a_w"] = HR / 100.0                        # hypothèse : a_w = 1 en immersion
        X["sqrt_t_sur_h"] = np.sqrt(t) / h           # h^0.5 / mm
        X["log1p_t"] = np.log1p(t)                   # t en h
    else:
        raise ValueError("pour = 'arbre' ou 'lineaire'")
    if jeu in ("contexte", "materiau"):
        X["immersion"] = (df["milieu"].astype(str).to_numpy() == "immersion").astype(float)
        etat = df["etat_essai"].astype(str).to_numpy()
        X["essai_reseche"] = (etat == "reseche").astype(float)
        X["essai_inconnu"] = (etat == "inconnu").astype(float)
        X["essai_reconditionne"] = (etat == "reconditionne").astype(float)
        X["flexion"] = (df["propriete"].astype(str).to_numpy() == "E_flexion").astype(float)
    if jeu == "materiau":
        if categories is None:
            raise ValueError("le jeu 'materiau' exige le dictionnaire categories")
        for col in ("famille", "renfort"):
            v = df[col] if col in df else pd.Series("", index=df.index)
            v = v.fillna("").astype(str).str.strip().replace("", "inconnu").to_numpy()
            for c in categories[col]:
                X[f"{col}={c}"] = (v == c).astype(float)
    return pd.DataFrame(X, index=df.index)


# =========================================================================== modèles
def modeles(seed: int) -> dict:
    """Hyperparamètres fixés a priori (pas de réglage sur le test)."""
    return {
        # référence « zéro compétence » : prédit la moyenne d'entraînement, ignore les variables.
        # Un modèle qui ne bat pas cette ligne n'a rien appris d'utilisable.
        "Constante_moyenne": ("arbre", DummyRegressor(strategy="mean")),
        "Ridge_var_physiques": ("lineaire",
                                make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "RandomForest": ("arbre",
                         RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               random_state=seed, n_jobs=1)),
        "XGBoost": ("arbre",
                    xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     min_child_weight=1.0, reg_lambda=1.0,
                                     tree_method="hist", random_state=seed, n_jobs=4)),
    }


def ajuster(modele, X, y, w=None):
    m = clone(modele)
    if w is None:
        m.fit(X, y)
    elif hasattr(m, "steps"):          # Pipeline : poids transmis à la dernière étape
        m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    else:
        m.fit(X, y, sample_weight=w)
    return m


def poids_par_courbe(courbes: np.ndarray) -> np.ndarray:
    """Chaque courbe pèse autant, quel que soit son nombre de points digitalisés."""
    _, inv, n = np.unique(courbes, return_inverse=True, return_counts=True)
    w = 1.0 / n[inv]
    return w * len(w) / w.sum()


def metriques(y, p, t) -> dict:
    out = {}
    for suffixe, m in (("", t > 0), ("_avec_t0", np.ones(len(t), bool))):
        if m.sum() >= 2:
            out["R2" + suffixe] = r2_score(y[m], p[m])
            out["RMSE" + suffixe] = float(np.sqrt(mean_squared_error(y[m], p[m])))
            out["MAE" + suffixe] = mean_absolute_error(y[m], p[m])
        else:
            out.update({"R2" + suffixe: np.nan, "RMSE" + suffixe: np.nan,
                        "MAE" + suffixe: np.nan})
        out["n_points" + suffixe] = int(m.sum())
    return out


# =========================================================================== protocoles
def lancer_protocoles(df, cats, jeux, repetitions, seed, ponderer):
    y = df[CIBLE].to_numpy(float)
    t = df["Temps_h"].to_numpy(float)
    courbes = df["curve_id"].to_numpy()
    idx = np.arange(len(df))
    w_tout = poids_par_courbe(courbes) if ponderer else None

    protos = {
        "A_aleatoire_lignes": ("repete", list(ShuffleSplit(
            n_splits=repetitions, test_size=0.2, random_state=seed).split(idx)), None),
        "B_80_20_par_courbe": ("repete", list(GroupShuffleSplit(
            n_splits=repetitions, test_size=0.2, random_state=seed).split(idx, groups=courbes)),
            None),
    }
    for nom, col in (("C_materiau_inconnu", "material_id"), ("D_source_inconnue", "source_doi")):
        g = df[col].to_numpy()
        if len(np.unique(g)) >= 2:
            protos[nom] = ("groupe", list(LeaveOneGroupOut().split(idx, groups=g)), g)
        else:
            print(f"  [info] protocole {nom} impossible : un seul groupe '{col}'")

    lignes, detail_groupes = [], []
    for jeu in jeux:
        for nom_m, (pour, modele) in modeles(seed).items():
            X = construire_features(df, jeu, pour, cats)
            for nom_p, (genre, splits, groupes) in protos.items():
                if genre == "repete":
                    scores = []
                    for tr, te in splits:
                        m = ajuster(modele, X.iloc[tr], y[tr],
                                    None if w_tout is None else w_tout[tr])
                        scores.append(metriques(y[te], m.predict(X.iloc[te]), t[te]))
                    s = pd.DataFrame(scores)
                    ligne = {"protocole": nom_p, "jeu": jeu, "modele": nom_m,
                             "n_evaluations": len(splits)}
                    for k in ("R2", "RMSE", "MAE", "R2_avec_t0"):
                        ligne[f"{k}_moy"] = s[k].mean()
                        ligne[f"{k}_ecart_type"] = s[k].std(ddof=1) if len(s) > 1 else np.nan
                    # sur peu de courbes test, un tirage à variance quasi nulle fait exploser R2 :
                    # la médiane et la RMSE sont plus lisibles que la moyenne de R2
                    ligne["R2_mediane"] = s["R2"].median()
                else:
                    oof = np.full(len(df), np.nan)
                    for tr, te in splits:
                        m = ajuster(modele, X.iloc[tr], y[tr],
                                    None if w_tout is None else w_tout[tr])
                        oof[te] = m.predict(X.iloc[te])
                        mt = metriques(y[te], oof[te], t[te])
                        detail_groupes.append({"protocole": nom_p, "jeu": jeu,
                                               "modele": nom_m, "groupe": groupes[te][0],
                                               "RMSE": mt["RMSE"], "n_points": mt["n_points"]})
                    mg = metriques(y, oof, t)
                    rmse_g = [d["RMSE"] for d in detail_groupes
                              if d["protocole"] == nom_p and d["jeu"] == jeu
                              and d["modele"] == nom_m]
                    ligne = {"protocole": nom_p, "jeu": jeu, "modele": nom_m,
                             "n_evaluations": len(splits),
                             "R2_moy": mg["R2"], "RMSE_moy": mg["RMSE"], "MAE_moy": mg["MAE"],
                             "R2_avec_t0_moy": mg["R2_avec_t0"],
                             "RMSE_mediane_par_groupe": float(np.nanmedian(rmse_g))}
                lignes.append(ligne)
                med = ligne.get("R2_mediane")
                print(f"  {nom_p:20s} {jeu:9s} {nom_m:20s} R2={ligne['R2_moy']:+.3f} "
                      + (f"(médiane {med:+.3f}) " if med is not None else "")
                      + f"RMSE={ligne['RMSE_moy']:.4f}")
    return pd.DataFrame(lignes), pd.DataFrame(detail_groupes)


# =========================================================================== sérialisation
def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


CLES_FEATURES_JSON = ("cible", "jeu_features", "features", "categories", "split",
                      "domaine_entrainement", "dataset", "metriques_test_hors_t0",
                      "metriques_test_avec_t0", "versions")


def charger_modele(dossier: str | Path = RACINE / "models"):
    """Pour app.py et predire_piece.py : modèle XGBoost (JSON natif) + métadonnées.
    Lève FileNotFoundError ou ValueError avec un message lisible."""
    dossier = Path(dossier)
    manquants = [f for f in ("modele.json", "features.json") if not (dossier / f).exists()]
    if manquants:
        raise FileNotFoundError(f"{', '.join(manquants)} absent(s) de {dossier} : "
                                f"aucun modèle entraîné avec src/train.py")
    meta = json.loads((dossier / "features.json").read_text(encoding="utf-8"))
    absentes = [c for c in CLES_FEATURES_JSON if c not in meta]
    if absentes:
        raise ValueError(f"{dossier / 'features.json'} est d'un ancien format (clés absentes : "
                         f"{absentes}) : réentraîner avec src/train.py")
    m = xgb.XGBRegressor()
    m.load_model(str(dossier / "modele.json"))
    return m, meta


def predire(modele, meta: dict, conditions: pd.DataFrame) -> np.ndarray:
    """E/E0 prédit. `conditions` contient les colonnes de base (+ milieu, etat_essai,
    propriete, famille, renfort selon le jeu). Le module absolu E = (E/E0) x E0_MPa se
    calcule dans l'app avec un E0 FOURNI, jamais prédit."""
    X = construire_features(conditions, meta["jeu_features"], "arbre", meta["categories"])
    return modele.predict(X[meta["features"]])


# =========================================================================== figures
def figure_test(y, p, t, titre, chemin):
    m = t > 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y[m], p[m], s=14, alpha=0.7, label="t > 0")
    ax[0].scatter(y[~m], p[~m], s=14, c="0.6", marker="x", label="t = 0 (triviaux)")
    lo = min(y.min(), p.min()) - 0.02
    hi = max(y.max(), p.max()) + 0.02
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="E/E0 source (digitalisé)",
              ylabel="E/E0 prédit", title="Parité - 20 % de courbes test")
    ax[0].legend()
    r = p[m] - y[m]
    ax[1].scatter(t[m], r, s=14, alpha=0.7)
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set_xscale("log")
    ax[1].set(xlabel="Temps_h (h, échelle log)", ylabel="résidu prédit - source",
              title="Résidus vs temps (t > 0)")
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


# =========================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(RACINE / "data" / "dataset_reel.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repetitions", type=int, default=20,
                    help="nombre de tirages pour les protocoles A et B (défaut 20)")
    ap.add_argument("--jeux", default="imposees,contexte,materiau",
                    help="jeux de variables comparés dans les protocoles")
    ap.add_argument("--features-final", default="contexte", choices=JEUX,
                    help="jeu de variables du modèle déployé (choisi AVANT de voir le test)")
    ap.add_argument("--ponderer-courbes", action="store_true",
                    help="poids 1/n_points par courbe à l'entraînement")
    ap.add_argument("--refit-complet", action="store_true",
                    help="après évaluation, réentraîne le modèle déployé sur 100 %% des courbes")
    ap.add_argument("--materiau", default=None,
                    help="entraîne UNIQUEMENT sur ce material_id (cas d'usage à matériau fixé)")
    ap.add_argument("--sortie", default=None,
                    help="racine où écrire models/, results/, figures/ (défaut : racine du dépôt)")
    ap.add_argument("--verifie-seulement", action="store_true",
                    help="n'entraîne que sur les courbes verification = controle_pdf")
    ap.add_argument("--min-points", type=int, default=2)
    ap.add_argument("--strict", action="store_true",
                    help="refuse de tourner si une courbe est rejetée par la vérification")
    ap.add_argument("--autoriser-synthetique", action="store_true",
                    help="tests du code uniquement")
    a = ap.parse_args(argv)
    jeux = [j.strip() for j in a.jeux.split(",") if j.strip()]

    # ---------------------------------------------------------------- données
    res = verifier(a.data, a.min_points, a.autoriser_synthetique)
    imprimer_rapport(res, detail=False)
    n_rej = int((res.rapport_courbes["statut"] != "EXPLOITABLE").sum()) if len(
        res.rapport_courbes) else 0
    if n_rej and a.strict:
        print(f"\n--strict : {n_rej} courbe(s) non exploitable(s). Arrêt.")
        return 1
    df = res.df_exploitable.copy()
    filtre = {}
    if a.materiau:
        dispo = sorted(df["material_id"].unique().tolist())
        df = df[df["material_id"] == a.materiau].reset_index(drop=True)
        filtre["material_id"] = a.materiau
        print(f"\nFiltre matériau '{a.materiau}' : {df['curve_id'].nunique()} courbes, "
              f"{len(df)} points, {df['source_doi'].nunique()} source(s)")
        if df.empty:
            print(f"Aucune courbe exploitable pour ce matériau. Disponibles : {dispo}")
            return 1
        if "materiau" in jeux or a.features_final == "materiau":
            print("Un seul matériau : le jeu 'materiau' est constant, donc inutile. Retiré.")
            jeux = [j for j in jeux if j != "materiau"] or ["contexte"]
            if a.features_final == "materiau":
                a.features_final = "contexte"
    if a.verifie_seulement:
        avant = df["curve_id"].nunique()
        df = df[df["verification"].astype(str) == "controle_pdf"].reset_index(drop=True)
        filtre["verification"] = "controle_pdf"
        print(f"\n--verifie-seulement : {df['curve_id'].nunique()} courbes conservées sur {avant}")
    n_courbes = df["curve_id"].nunique()
    if n_courbes < 5:
        print(f"\n{n_courbes} courbe(s) exploitable(s) : insuffisant pour un 80/20 par courbe.")
        return 1
    if n_rej:
        print(f"\n{n_rej} courbe(s) écartée(s) par la vérification (voir verifier_dataset.py).")
    besoin_mat = "materiau" in jeux or a.features_final == "materiau"
    if besoin_mat and (df["famille"].astype(str) == "").all():
        print("Jeu 'materiau' demandé mais colonne famille vide partout : retiré.")
        jeux = [j for j in jeux if j != "materiau"]
        if a.features_final == "materiau":
            print("--features-final materiau impossible. Arrêt.")
            return 1
    cats = categories_depuis(df)
    y = df[CIBLE].to_numpy(float)
    t = df["Temps_h"].to_numpy(float)
    courbes = df["curve_id"].to_numpy()

    base = Path(a.sortie) if a.sortie else RACINE
    dossier_res = base / "results"
    dossier_fig = base / "figures"
    dossier_mod = base / "models"
    for d in (dossier_res, dossier_fig, dossier_mod):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- protocoles
    print(f"\n=== PROTOCOLES ({a.repetitions} tirages pour A et B) ===")
    table, detail = lancer_protocoles(df, cats, jeux, a.repetitions, a.seed,
                                      a.ponderer_courbes)
    table.to_csv(dossier_res / "protocoles.csv", index=False)
    detail.to_csv(dossier_res / "protocoles_detail_par_groupe.csv", index=False)

    # ---------------------------------------------------------------- split officiel
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed)
    tr, te = next(gss.split(np.arange(len(df)), groups=courbes))
    c_tr, c_te = np.unique(courbes[tr]), np.unique(courbes[te])
    assert not set(c_tr) & set(c_te), "fuite : une courbe des deux côtés du split"
    print(f"\n=== SPLIT OFFICIEL (seed={a.seed}) ===")
    print(f"  train : {len(c_tr)} courbes / {len(tr)} points   "
          f"test : {len(c_te)} courbes / {len(te)} points "
          f"({100 * len(te) / len(df):.1f} % des points)")
    if len(c_te) < 5:
        print(f"  ATTENTION : score test calculé sur {len(c_te)} courbes seulement, "
              f"statistiquement fragile ; se fier à la distribution du protocole B.")
    print(f"  courbes test : {', '.join(map(str, c_te))}")
    mat_seuls_test = sorted(set(df["material_id"].iloc[te]) - set(df["material_id"].iloc[tr]))
    if mat_seuls_test:
        print(f"  matériaux présents UNIQUEMENT dans le test : {mat_seuls_test}")
    pd.DataFrame({"curve_id": np.concatenate([c_tr, c_te]),
                  "jeu": ["train"] * len(c_tr) + ["test"] * len(c_te)}).to_csv(
        dossier_res / "split_officiel.csv", index=False)

    w = poids_par_courbe(courbes) if a.ponderer_courbes else None
    officiel = []
    for nom_m, (pour, modele) in modeles(a.seed).items():
        X = construire_features(df, a.features_final, pour, cats)
        m = ajuster(modele, X.iloc[tr], y[tr], None if w is None else w[tr])
        officiel.append({"modele": nom_m, "jeu": a.features_final,
                         **metriques(y[te], m.predict(X.iloc[te]), t[te])})
    officiel = pd.DataFrame(officiel)
    officiel.to_csv(dossier_res / "split_officiel_scores.csv", index=False)
    print(officiel[["modele", "R2", "RMSE", "MAE", "R2_avec_t0", "n_points"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------------- modèle déployé
    pour, modele = modeles(a.seed)["XGBoost"]
    X = construire_features(df, a.features_final, pour, cats)
    final = ajuster(modele, X.iloc[tr], y[tr], None if w is None else w[tr])
    p_te = final.predict(X.iloc[te])
    m_te = metriques(y[te], p_te, t[te])
    figure_test(y[te], p_te, t[te],
                f"XGBoost, jeu '{a.features_final}' - R2 (t>0) = {m_te['R2']:.3f}, "
                f"RMSE = {m_te['RMSE']:.4f}", dossier_fig / "parite_residus_test.png")
    pred = df.iloc[te][["curve_id", "material_id", "source_doi", *VARIABLES_BASE,
                        "milieu", "etat_essai", CIBLE]].copy()
    pred["prediction"] = p_te
    pred["residu"] = p_te - y[te]
    pred.to_csv(dossier_res / "predictions_test.csv", index=False)

    lignes_dom = tr
    if a.refit_complet:
        final = ajuster(modele, X, y, w)
        lignes_dom = np.arange(len(df))
        print("\n--refit-complet : modèle déployé réentraîné sur 100 % des courbes ; "
              "le score test ci-dessus est celui du modèle 80 %.")
    final.save_model(str(dossier_mod / "modele.json"))

    d_dom = df.iloc[lignes_dom]
    conditions = (d_dom.groupby("curve_id").agg(
        Temperature_C=("Temperature_C", "first"), Humidite_HR=("Humidite_HR", "first"),
        Epaisseur_mm=("Epaisseur_mm", "first"), milieu=("milieu", "first"),
        etat_essai=("etat_essai", "first"), famille=("famille", "first"),
        material_id=("material_id", "first"), t_max_h=("Temps_h", "max"))
        .reset_index(drop=True).to_dict(orient="records"))
    meta = {
        "cible": CIBLE,
        "jeu_features": a.features_final,
        "features": list(X.columns),
        "categories": cats,
        "modele": "XGBRegressor",
        "hyperparametres": {k: v for k, v in final.get_params().items()
                            if v is not None and isinstance(v, (int, float, str, bool))
                            and not (isinstance(v, float) and np.isnan(v))},
        "pondere_par_courbe": bool(a.ponderer_courbes),
        "filtre_entrainement": filtre,
        "split": {"methode": "GroupShuffleSplit(groups=curve_id)", "test_size": 0.2,
                  "random_state": a.seed, "curves_train": c_tr.tolist(),
                  "curves_test": c_te.tolist()},
        "refit_complet": bool(a.refit_complet),
        "metriques_test_hors_t0": {k: m_te[k] for k in ("R2", "RMSE", "MAE", "n_points")},
        "metriques_test_avec_t0": {k: m_te[k + "_avec_t0"] for k in ("R2", "RMSE", "MAE")},
        "domaine_entrainement": {
            "bornes": {c: [float(d_dom[c].min()), float(d_dom[c].max())]
                       for c in VARIABLES_BASE},
            "conditions_courbes": conditions,
            "materiaux": sorted(d_dom["material_id"].unique().tolist()),
            "sources": sorted(d_dom["source_doi"].unique().tolist()),
        },
        "dataset": {"fichier": str(Path(a.data).name), "sha256": res.empreinte_sha256,
                    "n_courbes": int(n_courbes), "n_points": int(len(df)),
                    "synthetique_autorise": bool(a.autoriser_synthetique)},
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
        "avertissement": "E0_MPa est une donnée matériau fournie par l'utilisateur, "
                         "jamais une prédiction. Hors du domaine d'entraînement, un modèle "
                         "à arbres renvoie une constante (pas d'extrapolation).",
    }
    (dossier_mod / "features.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")

    # contrôle de relecture : le JSON rechargé doit redonner exactement les mêmes prédictions
    m2, meta2 = charger_modele(dossier_mod)
    p2 = predire(m2, meta2, df.iloc[te])
    ecart = float(np.max(np.abs(p2 - final.predict(X.iloc[te]))))
    print(f"\nModèle écrit : {dossier_mod / 'modele.json'} (écart relecture = {ecart:.2e})")
    print(f"Résultats : {dossier_res}/   Figure : {dossier_fig / 'parite_residus_test.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
