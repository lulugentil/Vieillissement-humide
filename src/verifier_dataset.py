#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vérification du dataset réel de vieillissement hygrothermique, courbe par courbe.

Usage (depuis la racine du dépôt) :
    python src/verifier_dataset.py data/dataset_reel.csv
    python src/verifier_dataset.py data/dataset_reel.csv --sortie results

Niveaux de diagnostic :
    BLOQUANT  la courbe est rejetée (schéma, unités, point t=0, incohérence interne, source).
    EXCLU     la courbe est valide mais trop pauvre (< --min-points points, t=0 inclus).
    AVERT     la courbe est conservée, mais une information manque ou est suspecte.

Ce script ne corrige rien et n'impute rien : il signale. Il vérifie le FORMAT d'un DOI,
pas son existence : chaque DOI doit être ouvert à la main au moins une fois.

Code retour : 0 = aucune courbe bloquée, 1 = au moins une courbe bloquée,
              2 = fichier illisible ou colonnes obligatoires absentes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- schéma
COLONNES_IMPOSEES = ["Temperature_C", "Humidite_HR", "Temps_h", "Epaisseur_mm",
                     "Propriete_residuelle"]
COLONNES_OBLIGATOIRES = COLONNES_IMPOSEES + [
    "curve_id", "material_id", "source_doi", "source_fig", "extraction",
    "milieu", "etat_essai", "propriete",
]
COLONNES_RECOMMANDEES = ["famille", "renfort", "fluide", "E_t_MPa", "E0_MPa", "incertitude",
                         "type_incertitude", "Tg0_C", "verification", "commentaire"]
ORDRE_TEMPLATE = [
    "curve_id", "material_id", "famille", "renfort",
    "source_doi", "source_fig", "extraction",
    "milieu", "fluide", "Temperature_C", "Humidite_HR", "Epaisseur_mm",
    "etat_essai", "propriete",
    "Temps_h", "Propriete_residuelle", "incertitude", "type_incertitude",
    "E_t_MPa", "E0_MPa", "Tg0_C", "verification", "commentaire",
]
COLONNES_NUMERIQUES = ["Temperature_C", "Humidite_HR", "Temps_h", "Epaisseur_mm",
                       "Propriete_residuelle", "incertitude", "E_t_MPa", "E0_MPa", "Tg0_C"]
VOCABULAIRE = {
    "milieu": {"vapeur", "immersion"},
    "etat_essai": {"humide", "reseche", "reconditionne", "inconnu"},
    "verification": {"controle_pdf", "non_controle"},
    "propriete": {"E_traction", "E_flexion"},
    "extraction": {"figure", "tableau", "texte"},
    "type_incertitude": {"ecart_type", "min_max", "IC95", "digitalisation", "inconnu"},
    "renfort": {"aucun", "verre", "carbone", "naturel", "autre"},
    "fluide": {"eau_distillee", "eau_deionisee", "eau_robinet", "eau_de_mer",
               "solution_saline", "inconnu"},
}
# source_fig et extraction décrivent la provenance de CHAQUE point (une même courbe peut
# mêler tableau et figure) : elles ne sont pas imposées constantes dans la courbe.
CONSTANTES_PAR_COURBE = [
    "material_id", "famille", "renfort", "source_doi",
    "milieu", "fluide", "Temperature_C", "Humidite_HR", "Epaisseur_mm",
    "etat_essai", "propriete", "E0_MPa", "Tg0_C", "verification",
]

# --------------------------------------------------------------------------- seuils
# Seuils d'alerte : choix d'ingénieur, PAS des valeurs issues de la littérature.
TOL_T0 = 1e-3            # tolérance sur E/E0 = 1 au point t=0
TOL_RATIO = 2e-3         # tolérance sur E/E0 = E_t_MPa / E0_MPa quand les deux sont renseignés
SEUIL_REMONTEE = 0.05    # hausse de E/E0 entre deux points successifs jugée suspecte
ECART_TG_C = 20.0        # alerte si T_vieillissement >= Tg0 - 20 degC
RE_DOI = re.compile(r"^10\.\d{4,9}/\S+$")
RE_COMMENTAIRE = re.compile(r'^\s*"?\s*#')
BLOQUANT, EXCLU, AVERT = "BLOQUANT", "EXCLU", "AVERT"
SANS_ID = "(curve_id vide)"


@dataclass
class Resultat:
    df: pd.DataFrame                      # toutes les lignes lues (colonne _ligne = n° dans le fichier)
    df_exploitable: pd.DataFrame          # lignes des courbes EXPLOITABLE uniquement
    rapport_courbes: pd.DataFrame         # une ligne par courbe
    problemes: pd.DataFrame               # un diagnostic par ligne
    empreinte_sha256: str
    separateur: str
    couverture: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- lecture
def lire_csv(chemin: str | Path) -> tuple[pd.DataFrame, str, str]:
    """Lit le CSV en texte brut (aucune conversion implicite), ignore les commentaires #."""
    brut = Path(chemin).read_bytes()
    empreinte = hashlib.sha256(brut).hexdigest()
    texte = brut.decode("utf-8-sig")
    gardees, numeros = [], []
    for n, ligne in enumerate(texte.splitlines(), start=1):
        if ligne.strip() and not RE_COMMENTAIRE.match(ligne):
            gardees.append(ligne)
            numeros.append(n)
    if not gardees:
        raise ValueError("fichier vide une fois les commentaires retirés")
    entete = gardees[0]
    sep = ";" if entete.count(";") > entete.count(",") else ","
    df = pd.read_csv(io.StringIO("\n".join(gardees)), sep=sep, dtype=str,
                     keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    # colonnes fantômes ajoutées par Excel (Unnamed: 21, ...) si entièrement vides
    fantomes = [c for c in df.columns if c.startswith("Unnamed") and
                df[c].astype(str).str.strip().eq("").all()]
    df = df.drop(columns=fantomes)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df["_ligne"] = numeros[1:] if len(numeros) - 1 == len(df) else np.arange(len(df)) + 2
    vides = df.drop(columns="_ligne").eq("").all(axis=1)
    df = df.loc[~vides].reset_index(drop=True)
    return df, sep, empreinte


# --------------------------------------------------------------------------- vérification
def verifier(chemin: str | Path, min_points: int = 2,
             autoriser_synthetique: bool = False) -> Resultat:
    problemes: list[dict] = []

    def signaler(cid, niveau, colonne, message, ligne=None):
        problemes.append({"curve_id": cid, "niveau": niveau, "colonne": colonne,
                          "ligne_fichier": ligne, "message": message})

    df, sep, empreinte = lire_csv(chemin)

    # ---- schéma
    manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in df.columns]
    if manquantes:
        raise KeyError(f"colonnes obligatoires absentes : {manquantes}. "
                       f"En-tête attendu : {','.join(ORDRE_TEMPLATE)}")
    for c in COLONNES_RECOMMANDEES:
        if c not in df.columns:
            signaler("(global)", AVERT, c, "colonne recommandée absente : créée vide")
            df[c] = ""
    inconnues = [c for c in df.columns if c not in ORDRE_TEMPLATE and c != "_ligne"]
    if inconnues:
        signaler("(global)", AVERT, ",".join(inconnues), "colonnes non prévues : ignorées")

    df["curve_id"] = df["curve_id"].replace("", SANS_ID)
    cid_ligne = df["curve_id"].to_numpy()
    lignes = df["_ligne"].to_numpy()

    # ---- conversion numérique (décimale virgule tolérée si séparateur ;)
    for col in COLONNES_NUMERIQUES:
        s = df[col].astype(str)
        if sep == ";":
            s = s.str.replace(",", ".", regex=False)
        num = pd.to_numeric(s.where(s != "", np.nan), errors="coerce")
        invalides = (s != "") & num.isna()
        for i in np.flatnonzero(invalides.to_numpy()):
            signaler(cid_ligne[i], BLOQUANT, col,
                     f"valeur non numérique '{df[col].iat[i]}'", lignes[i])
        df[col] = num.astype(float)

    # ---- vocabulaire contrôlé
    for col, autorises in VOCABULAIRE.items():
        v = df[col].astype(str).to_numpy()
        for i in np.flatnonzero((v != "") & ~np.isin(v, list(autorises))):
            signaler(cid_ligne[i], BLOQUANT, col,
                     f"valeur '{v[i]}' hors vocabulaire {sorted(autorises)}", lignes[i])
        if col in COLONNES_OBLIGATOIRES:
            for i in np.flatnonzero(v == ""):
                signaler(cid_ligne[i], BLOQUANT, col, "valeur obligatoire vide", lignes[i])
    for col in ("material_id", "source_fig"):
        for i in np.flatnonzero(df[col].astype(str).to_numpy() == ""):
            signaler(cid_ligne[i], BLOQUANT, col, "valeur obligatoire vide", lignes[i])
    for i in np.flatnonzero(cid_ligne == SANS_ID):
        signaler(SANS_ID, BLOQUANT, "curve_id", "curve_id vide", lignes[i])

    # ---- plages physiques, ligne par ligne
    def verifier_plage(col, condition, niveau, message):
        vals = df[col].to_numpy()
        for i in np.flatnonzero(condition(vals)):
            signaler(cid_ligne[i], niveau, col, message.format(v=vals[i]), lignes[i])

    for col in COLONNES_IMPOSEES:
        verifier_plage(col, np.isnan, BLOQUANT, "colonne imposée vide")
    with np.errstate(invalid="ignore"):
        verifier_plage("Temperature_C", lambda x: (x > 250) | (x < -60), BLOQUANT,
                       "T = {v} : hors plage ; saisie en kelvin ?")
        verifier_plage("Temperature_C", lambda x: (x > 150) & (x <= 250), AVERT,
                       "T = {v} degC : inhabituel pour du vieillissement humide")
        verifier_plage("Humidite_HR", lambda x: (x < 0) | (x > 100), BLOQUANT,
                       "HR = {v} : hors [0 ; 100] %")
        verifier_plage("Humidite_HR", lambda x: (x > 0) & (x <= 1), BLOQUANT,
                       "HR = {v} : fraction 0-1 au lieu de % ?")
        verifier_plage("Temps_h", lambda x: x < 0, BLOQUANT, "temps négatif")
        verifier_plage("Temps_h", lambda x: x > 1e5, AVERT,
                       "t = {v} h (> 11 ans) : unité ?")
        verifier_plage("Epaisseur_mm", lambda x: x <= 0, BLOQUANT, "épaisseur <= 0")
        verifier_plage("Epaisseur_mm", lambda x: x > 50, BLOQUANT,
                       "épaisseur = {v} mm : saisie en µm ?")
        verifier_plage("Epaisseur_mm", lambda x: (x > 0) & (x < 0.05), AVERT,
                       "épaisseur = {v} mm : film très mince ou saisie en m ?")
        verifier_plage("Propriete_residuelle", lambda x: (x <= 0) | (x > 1.5), BLOQUANT,
                       "E/E0 = {v} hors ]0 ; 1.5] : saisie en % ou non normalisée ?")
        verifier_plage("Propriete_residuelle", lambda x: (x > 1.15) & (x <= 1.5), AVERT,
                       "E/E0 = {v} > 1.15 : post-réticulation / antiplastification ou erreur ?")
        verifier_plage("incertitude", lambda x: x < 0, BLOQUANT, "incertitude négative")
        verifier_plage("incertitude", lambda x: x > 0.5, AVERT,
                       "incertitude = {v} : exprimée en MPa ou en % au lieu de E/E0 ?")
        verifier_plage("E0_MPa", lambda x: x <= 0, BLOQUANT, "E0 <= 0")
        verifier_plage("E0_MPa", lambda x: (x > 0) & (x < 50), AVERT,
                       "E0 = {v} MPa : saisi en GPa ? (ou élastomère)")
        verifier_plage("E0_MPa", lambda x: x > 3e5, AVERT, "E0 = {v} MPa : unité ?")
        verifier_plage("Tg0_C", lambda x: (x < -150) | (x > 450), BLOQUANT,
                       "Tg0 = {v} degC : hors plage")
    # cohérence donnée réelle (E_t_MPa) / valeur calculée (Propriete_residuelle = E_t / E0)
    et = df["E_t_MPa"].to_numpy()
    e0 = df["E0_MPa"].to_numpy()
    yr = df["Propriete_residuelle"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ecart = np.abs(et / e0 - yr)
    for i in np.flatnonzero(np.isfinite(ecart) & (ecart > TOL_RATIO)):
        signaler(cid_ligne[i], BLOQUANT, "Propriete_residuelle",
                 f"E/E0 = {yr[i]:g} incohérent avec E_t/E0 = {et[i]:g}/{e0[i]:g} = "
                 f"{et[i] / e0[i]:.4f}", lignes[i])
    for i in np.flatnonzero(np.isnan(yr) & np.isfinite(et) & np.isnan(e0)):
        signaler(cid_ligne[i], BLOQUANT, "E0_MPa",
                 "E_t publié mais E0 non publié : E/E0 non calculable "
                 "(ne pas choisir une référence arbitraire)", lignes[i])
    inc = df["incertitude"].to_numpy()
    typ = df["type_incertitude"].astype(str).to_numpy()
    for i in np.flatnonzero(~np.isnan(inc) & (typ == "")):
        signaler(cid_ligne[i], AVERT, "type_incertitude",
                 "incertitude renseignée sans son type", lignes[i])

    # ---- contrôles par courbe
    for cid, g in df.groupby("curve_id", sort=False):
        if cid == SANS_ID:
            continue
        # constance des métadonnées
        for col in CONSTANTES_PAR_COURBE:
            valeurs = g[col].astype(str).replace("nan", "").unique()
            if len(valeurs) > 1:
                signaler(cid, BLOQUANT, col,
                         f"valeurs différentes dans la même courbe : {sorted(valeurs)[:5]}")
        # source
        src = str(g["source_doi"].iat[0])
        if src == "":
            signaler(cid, BLOQUANT, "source_doi", "source absente : donnée non traçable")
        elif src.upper().startswith("EXEMPLE") or str(cid).upper().startswith("EX_"):
            signaler(cid, BLOQUANT, "source_doi", "ligne d'exemple du template : à supprimer")
        elif src.upper().startswith("SYNTHETIQUE"):
            signaler(cid, AVERT if autoriser_synthetique else BLOQUANT, "source_doi",
                     "donnée SYNTHÉTIQUE : interdite dans le dataset réel")
        elif not RE_DOI.match(src) and not re.match(r"^(HAL|ISBN|URL|THESE|ACTES):", src):
            signaler(cid, AVERT, "source_doi",
                     f"'{src}' n'a pas le format d'un DOI (10.xxxx/...) ni HAL:/ISBN:/URL:/THESE:/ACTES:")
        # point t = 0
        t = g["Temps_h"].to_numpy()
        y = g["Propriete_residuelle"].to_numpy()
        n0 = int(np.sum(t == 0))
        if n0 == 0:
            tmin = np.nanmin(t) if np.isfinite(t).any() else np.nan
            signaler(cid, BLOQUANT, "Temps_h",
                     f"pas de point t=0 (t min = {tmin:g} h) ; si c'est le point initial "
                     f"digitalisé, écrire 0 exactement")
        elif n0 > 1:
            signaler(cid, BLOQUANT, "Temps_h", f"{n0} points à t=0 : un seul attendu")
        else:
            y0 = float(y[t == 0][0])
            if np.isfinite(y0) and abs(y0 - 1.0) > TOL_T0:
                signaler(cid, BLOQUANT, "Propriete_residuelle",
                         f"E/E0 à t=0 = {y0:g} ≠ 1 : normaliser par la valeur t=0 de la courbe")
        # nombre de points, doublons, remontées
        n = len(g)
        if n < min_points:
            signaler(cid, EXCLU, "Temps_h", f"{n} point(s) < min_points = {min_points}")
        tpos = t[t > 0]
        if len(tpos) != len(np.unique(tpos)):
            signaler(cid, AVERT, "Temps_h", "plusieurs points au même temps (doublon ?)")
        ordre = np.argsort(t, kind="stable")
        dy = np.diff(y[ordre])
        if len(dy) and np.isfinite(dy).any() and np.nanmax(dy) > SEUIL_REMONTEE:
            k = int(np.nanargmax(dy))
            signaler(cid, AVERT, "Propriete_residuelle",
                     f"remontée de +{dy[k]:.3f} après t = {t[ordre][k]:g} h "
                     f"(post-réticulation, réséchage, dispersion ou erreur de digitalisation ?)")
        # cohérence des conditions
        milieu, hr = g["milieu"].iat[0], g["Humidite_HR"].iat[0]
        T, fluide = g["Temperature_C"].iat[0], str(g["fluide"].iat[0])
        if milieu == "immersion" and np.isfinite(hr) and hr != 100:
            signaler(cid, BLOQUANT, "Humidite_HR",
                     f"immersion avec HR = {hr:g} : convention HR = 100 en immersion")
        if milieu == "immersion" and fluide == "":
            signaler(cid, AVERT, "fluide", "immersion sans nature du fluide")
        if milieu == "immersion" and np.isfinite(T) and T > 100:
            signaler(cid, AVERT, "Temperature_C",
                     f"eau liquide à {T:g} degC : autoclave ? à vérifier dans la source")
        if milieu == "vapeur" and hr == 100:
            signaler(cid, AVERT, "Humidite_HR",
                     "100 %HR en vapeur : condensation probable, vérifier que ce n'est pas une immersion")
        if milieu == "vapeur" and fluide not in ("", "nan"):
            signaler(cid, AVERT, "fluide", "fluide renseigné en vapeur : ne s'applique qu'à l'immersion")
        if str(g["verification"].iat[0]) == "non_controle":
            signaler(cid, AVERT, "verification",
                     "valeurs non recontrôlées sur le PDF source")
        if g["etat_essai"].iat[0] == "inconnu":
            signaler(cid, AVERT, "etat_essai",
                     "humide ou reséché inconnu : plastification réversible et dégradation "
                     "irréversible indiscernables")
        # métadonnées recommandées
        if g["E0_MPa"].isna().all():
            signaler(cid, AVERT, "E0_MPa", "E0 non renseigné : E en MPa non affichable")
        if g["incertitude"].isna().all():
            signaler(cid, AVERT, "incertitude", "aucune incertitude")
        for col in ("famille", "renfort"):
            if (g[col].astype(str) == "").all():
                signaler(cid, AVERT, col, f"{col} vide : pas de descripteur matériau transférable")
        tg = g["Tg0_C"].iat[0]
        if np.isfinite(tg) and np.isfinite(T) and T >= tg - ECART_TG_C:
            signaler(cid, AVERT, "Tg0_C",
                     f"T = {T:g} degC >= Tg0 - {ECART_TG_C:g} (Tg0 = {tg:g}) ; la Tg humide "
                     f"est plus basse : régime proche/au-dessus de Tg")

    # ---- doublons probables de courbes (même figure, mêmes conditions)
    cles = ["source_doi", "source_fig", "material_id", "Temperature_C", "Humidite_HR",
            "Epaisseur_mm", "milieu", "etat_essai", "propriete"]
    resume = df[df["curve_id"] != SANS_ID].groupby("curve_id")[cles].first()
    for _, bloc in resume.groupby(cles, dropna=False):
        if len(bloc) > 1:
            for cid in bloc.index:
                signaler(cid, AVERT, "curve_id",
                         f"mêmes source/figure/conditions que {sorted(set(bloc.index) - {cid})} : "
                         f"courbe digitalisée deux fois ?")

    # ---- synthèse par courbe
    probs = pd.DataFrame(problemes, columns=["curve_id", "niveau", "colonne",
                                             "ligne_fichier", "message"])
    lignes_rapport = []
    for cid, g in df.groupby("curve_id", sort=False):
        p = probs[probs["curve_id"] == cid]
        if (p["niveau"] == BLOQUANT).any():
            statut = "REJETEE"
        elif (p["niveau"] == EXCLU).any():
            statut = "EXCLUE"
        else:
            statut = "EXPLOITABLE"
        lignes_rapport.append({
            "curve_id": cid, "statut": statut,
            "material_id": g["material_id"].iat[0], "source_doi": g["source_doi"].iat[0],
            "n_points": len(g), "t_max_h": g["Temps_h"].max(),
            "n_bloquant": int((p["niveau"] == BLOQUANT).sum()),
            "n_avert": int((p["niveau"] == AVERT).sum()),
            "diagnostic": " | ".join(f"[{n}] {c}: {m}{o}" for n, c, m, o in regrouper(p)),
        })
    rapport = pd.DataFrame(lignes_rapport)
    ok = rapport.loc[rapport["statut"] == "EXPLOITABLE", "curve_id"] if len(rapport) else []
    df_ok = df[df["curve_id"].isin(ok)].reset_index(drop=True)
    res = Resultat(df=df, df_exploitable=df_ok, rapport_courbes=rapport, problemes=probs,
                   empreinte_sha256=empreinte, separateur=sep)
    res.couverture = couverture(df_ok)
    return res


def regrouper(p: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """Fusionne les messages identiques d'une courbe (même niveau, colonne, message)
    et liste les lignes du fichier concernées. BLOQUANT d'abord."""
    rang = {BLOQUANT: 0, EXCLU: 1, AVERT: 2}
    sortie = []
    for (niv, col, msg), b in p.groupby(["niveau", "colonne", "message"], sort=False):
        nums = sorted(int(x) for x in b["ligne_fichier"].dropna())
        if not nums:
            ou = ""
        elif len(nums) == 1:
            ou = f" (ligne {nums[0]})"
        elif nums == list(range(nums[0], nums[-1] + 1)):
            ou = f" (lignes {nums[0]}-{nums[-1]})"
        else:
            ou = f" (lignes {', '.join(map(str, nums[:8]))}{' ...' if len(nums) > 8 else ''})"
        sortie.append((niv, col, msg, ou))
    return sorted(sortie, key=lambda x: rang.get(x[0], 9))


# --------------------------------------------------------------------------- couverture
def couverture(df_ok: pd.DataFrame) -> dict:
    """Couverture expérimentale des courbes exploitables (au niveau COURBE, pas point)."""
    if df_ok.empty:
        return {}
    c = df_ok.groupby("curve_id").agg(
        material_id=("material_id", "first"), famille=("famille", "first"),
        T=("Temperature_C", "first"), HR=("Humidite_HR", "first"),
        milieu=("milieu", "first"), etat=("etat_essai", "first"),
        propriete=("propriete", "first"), h=("Epaisseur_mm", "first"),
        t_max=("Temps_h", "max"), n=("Temps_h", "size"))
    classes_T = pd.cut(c["T"], [-np.inf, 30, 50, 70, 90, np.inf], right=False,
                       labels=["<30", "30-50", "50-70", "70-90", ">=90"])
    classes_HR = pd.cut(c["HR"], [-np.inf, 50, 75, 90, 100, np.inf], right=False,
                        labels=["<50", "50-75", "75-90", "90-<100", "100"]).astype(str)
    classes_HR = classes_HR.where(c["milieu"] != "immersion", "immersion")
    table_T_HR = pd.crosstab(classes_T, classes_HR).rename_axis(index="T (degC)",
                                                                 columns="HR (%)")
    vap = c[c["milieu"] == "vapeur"]
    rho = np.nan
    if len(vap) >= 3 and vap["T"].nunique() > 1 and vap["HR"].nunique() > 1:
        rho = float(vap["T"].rank().corr(vap["HR"].rank()))
    par_mat = c.groupby("material_id").agg(
        n_courbes=("T", "size"), famille=("famille", "first"),
        T_niveaux=("T", lambda s: sorted(s.unique().tolist())),
        HR_niveaux=("HR", lambda s: sorted(s.unique().tolist())),
        t_max_h=("t_max", "max"))
    iso = c.groupby(["material_id", "HR", "milieu", "etat", "propriete", "h"])["T"].nunique()
    return {"courbes": c, "table_T_HR": table_T_HR, "spearman_T_HR_vapeur": rho,
            "par_materiau": par_mat,
            "series_arrhenius_3T": int((iso >= 3).sum()),
            "series_arrhenius_detail": iso[iso >= 3]}


# --------------------------------------------------------------------------- affichage
def imprimer_rapport(res: Resultat, detail: bool = True) -> None:
    r = res.rapport_courbes
    print(f"\nFichier SHA-256 : {res.empreinte_sha256}   séparateur : '{res.separateur}'")
    print(f"Lignes lues : {len(res.df)}   courbes : {len(r)}")
    glob = res.problemes[res.problemes["curve_id"] == "(global)"]
    for p in glob.itertuples():
        print(f"  [{p.niveau}] {p.colonne} : {p.message}")
    if detail and len(r):
        print("\n=== DIAGNOSTIC COURBE PAR COURBE ===")
        for ligne in r.itertuples():
            print(f"\n{ligne.curve_id}  [{ligne.statut}]  {ligne.material_id}  "
                  f"n={ligne.n_points}  t_max={ligne.t_max_h:g} h  source={ligne.source_doi}")
            p = res.problemes[res.problemes["curve_id"] == ligne.curve_id]
            for niv, col, msg, ou in regrouper(p):
                print(f"   {niv:8s} {col:22s} {msg}{ou}")
    if len(r):
        compte = r["statut"].value_counts()
        print("\n=== BILAN ===")
        for s in ("EXPLOITABLE", "EXCLUE", "REJETEE"):
            print(f"  {s:12s} {int(compte.get(s, 0)):4d} courbe(s)")
    cov = res.couverture
    if not cov:
        print("\nAucune courbe exploitable : pas de couverture à afficher.")
        return
    d = res.df_exploitable
    print(f"\nExploitable : {d['curve_id'].nunique()} courbes, {len(d)} points, "
          f"{d['material_id'].nunique()} matériaux, {d['source_doi'].nunique()} sources")
    print("\nCourbes par classe T x HR :")
    print(cov["table_T_HR"].to_string())
    rho = cov["spearman_T_HR_vapeur"]
    print(f"\nCorrélation de Spearman T/HR entre courbes (vapeur) : "
          f"{'non calculable' if np.isnan(rho) else f'{rho:+.2f}'}")
    if np.isfinite(rho) and abs(rho) > 0.5:
        print("  -> T et HR fortement corrélées : leurs contributions SHAP seront "
              "difficiles à séparer.")
    print(f"\nSéries à HR fixe couvrant >= 3 températures (même matériau, HR, milieu, "
          f"état d'essai, propriété, épaisseur) : {cov['series_arrhenius_3T']}")
    if cov["series_arrhenius_3T"] == 0:
        print("  -> test d'Arrhenius (ln t_seuil vs 1/T_K) IMPOSSIBLE en l'état.")
    print("\nPar matériau :")
    print(cov["par_materiau"].to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--min-points", type=int, default=2,
                    help="nombre minimal de points par courbe, t=0 inclus (défaut 2 : la "
                         "référence et au moins un point vieilli)")
    ap.add_argument("--sortie", default=None,
                    help="dossier où écrire rapport_courbes.csv et problemes.csv")
    ap.add_argument("--autoriser-synthetique", action="store_true",
                    help="tests du code uniquement : accepte source_doi=SYNTHETIQUE")
    ap.add_argument("--resume", action="store_true", help="n'affiche pas le détail par courbe")
    a = ap.parse_args(argv)
    try:
        res = verifier(a.csv, a.min_points, a.autoriser_synthetique)
    except (OSError, ValueError, KeyError) as e:
        print(f"ERREUR : {e}", file=sys.stderr)
        return 2
    imprimer_rapport(res, detail=not a.resume)
    if a.sortie:
        out = Path(a.sortie)
        out.mkdir(parents=True, exist_ok=True)
        res.rapport_courbes.to_csv(out / "rapport_courbes.csv", index=False)
        res.problemes.to_csv(out / "problemes.csv", index=False)
        print(f"\nRapports écrits dans {out}/")
    return 1 if (res.problemes["niveau"] == BLOQUANT).any() else 0


if __name__ == "__main__":
    sys.exit(main())
