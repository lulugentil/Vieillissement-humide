#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vieillissement d'une pièce polymère : matériau, T, HR, épaisseur et temps FIXÉS.

Usage (depuis la racine du dépôt, après python src/train.py [--materiau ...]) :
    python src/predire_piece.py                      # lit le bloc CAS ci-dessous
    python src/predire_piece.py --T 28 --HR 60       # variante autour du cas nominal

Sortie : E/E0 au temps fixé (et E en MPa si E0 est fourni), plus la courbe jusqu'à ce
temps, en ROUGE là où aucune courbe d'entraînement ne couvre la condition. Hors de ce
domaine, un modèle à arbres recopie sa dernière feuille : la valeur n'est pas une prédiction.

Le calcul est dans evaluer_cas() / figure_cas(), appelées AUSSI par app.py (Streamlit) :
la ligne de commande et l'application ne peuvent pas diverger.

Hypothèses à déclarer : T et HR constantes (pas de cycles), rigidité en service = état
humide, perte relative E/E0 transférable des éprouvettes publiées à la pièce réelle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from src.train import RACINE, charger_modele, predire
except ImportError:
    from train import RACINE, charger_modele, predire

# =====================================================================================
# CAS FIXÉ - à remplir avec TES valeurs. None = non renseigné (le script refuse de tourner).
# Aucune valeur par défaut n'est une donnée.
# =====================================================================================
CAS = {
    "Temperature_C": None,    # degC, température du local
    "Humidite_HR": None,      # %, humidité relative du local (0-100)
    "Temps_h": None,          # h, durée d'exposition visée (1 an = 8760 h)
    "Epaisseur_mm": None,     # mm, épaisseur de la pièce
    "E0_MPa": None,           # MPa, module initial de la pièce (optionnel, pour E en MPa)
    "famille": "",            # utile seulement si le modèle déployé a le jeu 'materiau'
    "renfort": "aucun",
    "milieu": "vapeur",       # pièce dans un local = atmosphère humide
    "etat_essai": "humide",   # rigidité en service, plastification comprise
    "propriete": "E_traction",
}
# Tolérances du domaine : choix d'ingénieur, à justifier dans le rapport.
TOL_T_C = 10.0
TOL_HR = 10.0
JALONS = ((168, "1 semaine"), (720, "1 mois"), (8760, "1 an"), (43800, "5 ans"),
          (87600, "10 ans"))


def valider_cas(cas: dict) -> list[str]:
    """Liste des problèmes bloquants du cas (vide = cas utilisable)."""
    erreurs = [f"{k} non renseigné" for k in ("Temperature_C", "Humidite_HR", "Temps_h",
                                              "Epaisseur_mm") if cas.get(k) is None]
    if erreurs:
        return erreurs
    if not 0 <= cas["Humidite_HR"] <= 100:
        erreurs.append("HR doit être en % (0-100)")
    if cas["Temps_h"] <= 0:
        erreurs.append("le temps doit être > 0 h")
    if cas["Epaisseur_mm"] <= 0:
        erreurs.append("l'épaisseur doit être > 0 mm")
    if cas.get("E0_MPa") is not None and cas["E0_MPa"] <= 0:
        erreurs.append("E0 doit être > 0 MPa")
    if cas.get("milieu") == "immersion" and cas["Humidite_HR"] != 100:
        erreurs.append("convention du dataset : HR = 100 en immersion")
    return erreurs


def courbes_voisines(meta: dict, cas: dict) -> pd.DataFrame:
    """Courbes d'entraînement qui 'couvrent' la condition (mêmes milieu et état d'essai,
    |ΔT| <= TOL_T_C, |ΔHR| <= TOL_HR, même famille si le modèle utilise le matériau)."""
    cond = pd.DataFrame(meta["domaine_entrainement"]["conditions_courbes"])
    m = ((cond["Temperature_C"] - cas["Temperature_C"]).abs() <= TOL_T_C) & \
        ((cond["Humidite_HR"] - cas["Humidite_HR"]).abs() <= TOL_HR) & \
        (cond["milieu"] == cas["milieu"]) & (cond["etat_essai"] == cas["etat_essai"])
    if meta["jeu_features"] == "materiau" and "famille" in cond:
        m &= cond["famille"] == cas["famille"]
    return cond[m]


def avertissements_modele(meta: dict, cas: dict) -> list[tuple[str, str]]:
    """(niveau, message) ; niveau 'erreur' = le cas ne peut pas être calculé."""
    out = []
    filtre = meta.get("filtre_entrainement", {})
    if meta["dataset"].get("synthetique_autorise"):
        out.append(("alerte", "MODÈLE ENTRAÎNÉ AVEC DES DONNÉES SYNTHÉTIQUES : "
                              "résultat sans valeur physique."))
    if "material_id" in filtre:
        out.append(("info", f"Modèle spécifique au matériau '{filtre['material_id']}' "
                            f"({meta['dataset']['n_courbes']} courbes)."))
    elif meta["jeu_features"] != "materiau":
        out.append(("alerte", "Modèle entraîné sur plusieurs matériaux sans descripteur "
                              "matériau : la valeur est une moyenne sur les polymères du dataset."))
    elif cas.get("famille") not in meta["categories"]["famille"]:
        out.append(("erreur", f"Famille '{cas.get('famille')}' inconnue du modèle : "
                              f"{meta['categories']['famille']}"))
    return out


def evaluer_cas(modele, meta: dict, cas: dict, seuil: float | None = None,
                n_points: int = 300) -> dict:
    """Prédiction E/E0(t) jusqu'au temps fixé + diagnostic de domaine."""
    erreurs = valider_cas(cas)
    if erreurs:
        raise ValueError("; ".join(erreurs))
    avert = avertissements_modele(meta, cas)
    if any(n == "erreur" for n, _ in avert):
        raise ValueError(" ; ".join(m for n, m in avert if n == "erreur"))

    tf = float(cas["Temps_h"])
    t = np.unique(np.concatenate([[0.0], np.geomspace(1.0, max(tf, 1.0), n_points), [tf]]))
    cond = pd.DataFrame({"Temperature_C": cas["Temperature_C"], "Humidite_HR": cas["Humidite_HR"],
                         "Temps_h": t, "Epaisseur_mm": cas["Epaisseur_mm"],
                         "milieu": cas["milieu"], "etat_essai": cas["etat_essai"],
                         "propriete": cas["propriete"], "famille": cas.get("famille", ""),
                         "renfort": cas.get("renfort", "")})
    ratio = predire(modele, meta, cond)

    bornes = meta["domaine_entrainement"]["bornes"]
    h_ok = bornes["Epaisseur_mm"][0] <= cas["Epaisseur_mm"] <= bornes["Epaisseur_mm"][1]
    vois = courbes_voisines(meta, cas)
    t_valide = float(vois["t_max_h"].max()) if (len(vois) and h_ok) else 0.0
    dans = t <= t_valide
    i_f = int(np.flatnonzero(t == tf)[0])

    if not h_ok:
        verdict = ("HORS_EPAISSEUR", f"Épaisseur hors du domaine "
                   f"[{bornes['Epaisseur_mm'][0]:g} ; {bornes['Epaisseur_mm'][1]:g}] mm : "
                   f"la valeur n'est pas une prédiction.")
    elif len(vois) == 0:
        verdict = ("HORS_CONDITIONS", f"Aucune courbe d'entraînement à ±{TOL_T_C:g} degC et "
                   f"±{TOL_HR:g} %HR (mêmes milieu et état d'essai). Domaine : T "
                   f"{bornes['Temperature_C']} degC, HR {bornes['Humidite_HR']} %. "
                   f"La valeur n'est pas une prédiction.")
    elif tf > t_valide:
        verdict = ("HORS_TEMPS", f"Les courbes voisines s'arrêtent à {t_valide:g} h : au-delà, "
                   f"la valeur est la constante de la dernière feuille, pas une prédiction.")
    else:
        verdict = ("DOMAINE", f"Dans le domaine : {len(vois)} courbe(s) voisine(s), "
                   f"couvertes jusqu'à {t_valide:g} h.")

    t_seuil = None
    if seuil is not None:
        sous = np.flatnonzero(ratio <= seuil)
        t_seuil = float(t[sous[0]]) if len(sous) else None

    E0 = cas.get("E0_MPa")
    jalons = []
    for th, lib in JALONS + ((tf, "temps fixé"),):
        if th <= tf:
            i = int(np.argmin(np.abs(t - th)))
            jalons.append({"jalon": lib, "Temps_h": float(t[i]), "E_E0": float(ratio[i]),
                           "E_MPa_calcule": float(ratio[i] * E0) if E0 else None,
                           "dans_domaine": bool(dans[i])})
    return {"t": t, "ratio": ratio, "dans": dans, "t_valide": t_valide, "voisines": vois,
            "i_f": i_f, "Temps_h": tf, "E_E0": float(ratio[i_f]),
            "E_MPa": float(ratio[i_f] * E0) if E0 else None,
            "verdict": verdict[0], "message": verdict[1], "t_seuil": t_seuil,
            "seuil": seuil, "avertissements": avert,
            "rmse_test": meta["metriques_test_hors_t0"]["RMSE"],
            "jalons": pd.DataFrame(jalons)}


def figure_cas(r: dict, cas: dict):
    t, ratio, dans = r["t"], r["ratio"], r["dans"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = t > 0
    ax.plot(t[pos & dans], ratio[pos & dans], c="C0", lw=2, label="couvert par les données")
    k = np.flatnonzero(pos & ~dans)
    if len(k):
        if k[0] > 0 and dans[k[0] - 1] and pos[k[0] - 1]:
            k = np.concatenate([[k[0] - 1], k])
        ax.plot(t[k], ratio[k], c="red", lw=2, label="hors domaine : pas une prédiction")
    ax.plot([r["Temps_h"]], [r["E_E0"]], "ko", label=f"temps fixé : E/E0 = {r['E_E0']:.3f}")
    if r["seuil"] is not None:
        ax.axhline(r["seuil"], c="k", ls=":", lw=1, label=f"seuil E/E0 = {r['seuil']}")
    ax.set_xscale("log")
    ax.set(xlabel="Temps (h, échelle log)", ylabel="E/E0 prédit",
           title=f"T = {cas['Temperature_C']:g} degC, HR = {cas['Humidite_HR']:g} %, "
                 f"h = {cas['Epaisseur_mm']:g} mm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=float, help="remplace CAS['Temperature_C'] (degC)")
    ap.add_argument("--HR", type=float, help="remplace CAS['Humidite_HR'] (%%)")
    ap.add_argument("--temps", type=float, help="remplace CAS['Temps_h'] (h)")
    ap.add_argument("--epaisseur", type=float, help="remplace CAS['Epaisseur_mm'] (mm)")
    ap.add_argument("--E0", type=float, help="remplace CAS['E0_MPa'] (MPa)")
    ap.add_argument("--famille", help="remplace CAS['famille']")
    ap.add_argument("--milieu", choices=["vapeur", "immersion"], help="remplace CAS['milieu']")
    ap.add_argument("--etat", choices=["humide", "reseche", "reconditionne", "inconnu"],
                    help="remplace CAS['etat_essai']")
    ap.add_argument("--seuil", type=float, default=None,
                    help="critère de fin de vie en E/E0 (ex. 0.8)")
    ap.add_argument("--modeles", default=str(RACINE / "models"),
                    help="dossier contenant modele.json et features.json")
    a = ap.parse_args(argv)

    cas = dict(CAS)
    for cle, val in (("Temperature_C", a.T), ("Humidite_HR", a.HR), ("Temps_h", a.temps),
                     ("Epaisseur_mm", a.epaisseur), ("E0_MPa", a.E0), ("famille", a.famille),
                     ("milieu", a.milieu), ("etat_essai", a.etat)):
        if val is not None:
            cas[cle] = val
    erreurs = valider_cas(cas)
    if erreurs:
        print("Cas incomplet, à renseigner dans CAS ou par option : " + " ; ".join(erreurs))
        return 1
    try:
        modele, meta = charger_modele(a.modeles)
    except (FileNotFoundError, ValueError) as e:
        print(f"Modèle inutilisable : {e}")
        return 1
    try:
        r = evaluer_cas(modele, meta, cas, a.seuil)
    except ValueError as e:
        print(f"Cas non calculable : {e}")
        return 1

    for niveau, msg in r["avertissements"]:
        print(("!!! " if niveau == "alerte" else "") + msg)
    tf = r["Temps_h"]
    print(f"\nCAS : T = {cas['Temperature_C']:g} degC, HR = {cas['Humidite_HR']:g} %, "
          f"h = {cas['Epaisseur_mm']:g} mm, t = {tf:g} h ({tf / 8760:.2f} an), "
          f"{cas['milieu']}, état {cas['etat_essai']}")
    print(f"\n  E/E0 prédit à t = {tf:g} h : {r['E_E0']:.3f}")
    if r["E_MPa"] is not None:
        print(f"  E calculé = (E/E0) x E0 = {r['E_E0']:.3f} x {cas['E0_MPa']:g} = "
              f"{r['E_MPa']:.0f} MPa (E0 est une donnée de la pièce, pas une prédiction)")
    print(f"  Erreur typique sur les courbes test : RMSE = {r['rmse_test']:.3f} en E/E0 "
          f"(indicatif, PAS un intervalle de confiance)")
    print(f"\n  VERDICT [{r['verdict']}] : {r['message']}")
    if a.seuil is not None:
        if r["t_seuil"] is None:
            print(f"  E/E0 reste > {a.seuil} jusqu'à {tf:g} h.")
        else:
            ou = "dans le" if r["t_seuil"] <= r["t_valide"] else "HORS du"
            print(f"  E/E0 <= {a.seuil} atteint vers {r['t_seuil']:g} h - {ou} domaine.")
    print("\n" + r["jalons"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    nom = (f"cas_T{cas['Temperature_C']:g}_HR{cas['Humidite_HR']:g}"
           f"_h{cas['Epaisseur_mm']:g}_t{tf:g}")
    (RACINE / "results").mkdir(exist_ok=True)
    (RACINE / "figures").mkdir(exist_ok=True)
    E0 = cas.get("E0_MPa")
    pd.DataFrame({"Temps_h": r["t"], "E_E0_predit": r["ratio"],
                  "E_MPa_calcule": r["ratio"] * E0 if E0 else np.nan,
                  "dans_domaine": r["dans"]}).to_csv(
        RACINE / "results" / f"prediction_{nom}.csv", index=False)
    fig = figure_cas(r, cas)
    fig.savefig(RACINE / "figures" / f"prediction_{nom}.png", dpi=150)
    plt.close(fig)
    print(f"\nÉcrit : results/prediction_{nom}.csv et figures/prediction_{nom}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
