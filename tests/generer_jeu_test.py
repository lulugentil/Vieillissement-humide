#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jeu SYNTHÉTIQUE servant UNIQUEMENT à tester le code (vérification, entraînement, app).

AUCUNE valeur produite ici n'est une donnée expérimentale. La loi utilisée (saturation
exponentielle + terme lent) est arbitraire et n'a été validée sur aucun matériau.
source_doi = SYNTHETIQUE : verifier_dataset.py rejette ces courbes sauf avec
--autoriser-synthetique, et train.py marque alors le modèle comme synthétique.

Usage :
    python tests/generer_jeu_test.py              # écrit tests/donnees/
    python tests/generer_jeu_test.py --sortie /tmp/x
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MATERIAUX = [("M1", "epoxy", "aucun", 2.0), ("M2", "epoxy", "verre", 3.0),
             ("M3", "polyamide", "aucun", 2.0), ("M4", "vinylester", "verre", 4.0),
             ("M5", "polyamide", "verre", 3.0), ("M6", "epoxy", "aucun", 1.0)]
TEMPS_H = np.array([0, 24, 72, 168, 336, 720, 1440, 2000], dtype=float)


def jeu_valide(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lignes, k = [], 0
    for mat, fam, ren, h in MATERIAUX:
        d0, amp = rng.uniform(0.5, 2.0), rng.uniform(0.1, 0.4)
        for T in (40, 60, 80):
            for HR, milieu in ((50, "vapeur"), (85, "vapeur"), (100, "immersion")):
                if rng.random() < 0.35:
                    continue
                k += 1
                inv = 1 / (T + 273.15) - 1 / 333.15
                frac = 1 - np.exp(-d0 * np.exp(-4000 * inv) * TEMPS_H / h ** 2 / 50)
                y = 1 - amp * (HR / 100) * frac - 1e-5 * TEMPS_H * np.exp(-6000 * inv)
                y[1:] += rng.normal(0, 0.015, len(TEMPS_H) - 1)
                for ti, yi in zip(TEMPS_H, y):
                    lignes.append(dict(
                        curve_id=f"S{k:02d}", material_id=mat, famille=fam, renfort=ren,
                        source_doi=f"SYNTHETIQUE_{mat}", source_fig="Fig. 1", extraction="figure",
                        milieu=milieu, fluide="eau_distillee" if milieu == "immersion" else "",
                        Temperature_C=T, Humidite_HR=HR, Epaisseur_mm=h, etat_essai="humide",
                        propriete="E_traction", Temps_h=ti,
                        Propriete_residuelle=round(float(yi), 4), incertitude=0.02,
                        type_incertitude="ecart_type", E0_MPa=3000, Tg0_C="", commentaire=""))
    return pd.DataFrame(lignes)


def jeu_avec_erreurs(d: pd.DataFrame) -> pd.DataFrame:
    """Une erreur typique de saisie injectée par courbe S01 à S11."""
    d = d.astype(str).replace("nan", "")
    d.loc[(d.curve_id == "S01") & (d.Temps_h == "0.0"), "Temps_h"] = "0.4"   # t0 non nul
    d.loc[d.curve_id == "S02", "Humidite_HR"] = "0.85"                       # fraction
    d.loc[d.curve_id == "S03", "Temperature_C"] = "333.15"                   # kelvin
    d.loc[(d.curve_id == "S04") & (d.Temps_h == "168.0"), "Temperature_C"] = "61"
    d.loc[(d.curve_id == "S05") & (d.Temps_h == "0.0"), "Propriete_residuelle"] = "0.97"
    d.loc[d.curve_id == "S06", "source_doi"] = ""                            # sans source
    d.loc[d.curve_id == "S07", "etat_essai"] = "inconnu"                     # avertissement
    d.loc[d.curve_id == "S08", "milieu"] = "Immersion"                       # casse
    d.loc[d.curve_id == "S09", "E0_MPa"] = "3,2"                             # virgule
    d = d[~((d.curve_id == "S10") & (d.Temps_h != "0.0"))]                    # t=0 seul
    d.loc[(d.curve_id == "S11") & (d.Temps_h == "720.0"), "Propriete_residuelle"] = "95"
    return d


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", default=str(Path(__file__).resolve().parent / "donnees"))
    a = ap.parse_args(argv)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    d = jeu_valide()
    d.to_csv(out / "synthetique_valide.csv", index=False)
    jeu_avec_erreurs(d).to_csv(out / "synthetique_erreurs.csv", index=False)
    print(f"{d['curve_id'].nunique()} courbes synthétiques écrites dans {out}/")


if __name__ == "__main__":
    main()
