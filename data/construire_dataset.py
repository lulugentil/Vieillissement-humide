#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construction TRAÇABLE de data/dataset_reel.csv à partir des sources.

Usage (racine du dépôt) :
    python data/construire_dataset.py --pdf-dir chemin/vers/pdfs \
        --extraction chemin/vers/Extraction_vieillissement_hygrothermique.csv

Chaque valeur a une origine explicite :
  - valeurs lues dans un TABLEAU ou le TEXTE d'un PDF fourni      -> codées ci-dessous, page citée
  - valeurs DIGITALISÉES sur une figure d'un PDF fourni            -> recalculées à chaque exécution :
        figures vectorielles : coordonnées exactes des marqueurs dans le PDF ;
        figures raster       : centre des marqueurs détecté par analyse d'image ;
    calibration sur les graduations, contrôlée contre les valeurs tabulées (arrêt si écart) ;
  - valeurs d'une extraction tierce dont le PDF n'a pas été fourni -> verification = non_controle.
Rien n'est imputé : une valeur absente reste vide et le vérificateur rejette la courbe.

Les PDF ne sont pas redistribués (droits d'auteur) : leur empreinte SHA-256 est enregistrée.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
from PIL import Image
from scipy import ndimage as ndi

COLONNES = ["curve_id", "material_id", "famille", "renfort", "source_doi", "source_fig",
            "extraction", "milieu", "fluide", "Temperature_C", "Humidite_HR", "Epaisseur_mm",
            "etat_essai", "propriete", "Temps_h", "Propriete_residuelle", "incertitude",
            "type_incertitude", "E_t_MPa", "E0_MPa", "Tg0_C", "verification", "commentaire"]
PDF = {"legac": "48321.pdf", "zarbali": "acsomega_6c01154.pdf",
       "chilali": "65911_CHILALI_2017_archivage.pdf", "guerira": "74-GUERIRA.pdf",
       "gibhardt": "fmats-09-826076.pdf"}
LIGNES: list[dict] = []
HORS_CIBLE: list[dict] = []     # données réelles conservées mais hors cible E/E0 (résistances...)


class ErreurControle(RuntimeError):
    pass


def controle(condition: bool, message: str) -> None:
    if not condition:
        raise ErreurControle(message)


def ajouter(cid: str, meta: dict, t_h: float, E_t: float | None, E0: float | None,
            inc_abs: float | None = None, type_inc: str = "", fig: str = "", extraction: str = "",
            commentaire: str = "") -> None:
    """E/E0 est CALCULÉ ici à partir des valeurs absolues ; il n'est jamais saisi à la main."""
    y = None if (E_t is None or E0 is None) else round(E_t / E0, 4)
    inc = None if (inc_abs is None or E0 is None) else round(inc_abs / E0, 4)
    LIGNES.append({**meta, "curve_id": cid, "Temps_h": t_h,
                   "Propriete_residuelle": y, "E_t_MPa": None if E_t is None else round(E_t, 2),
                   "E0_MPa": None if E0 is None else round(E0, 2),
                   "incertitude": inc, "type_incertitude": type_inc if inc is not None else "",
                   "source_fig": fig or meta.get("source_fig", ""),
                   "extraction": extraction or meta.get("extraction", ""),
                   "commentaire": commentaire.replace(",", " ;")})


# ============================================================ outils de digitalisation
def objets_vectoriels(pdf_path: Path, page: int, boite: tuple):
    pg = pdfplumber.open(str(pdf_path)).pages[page - 1]
    x0, t0, x1, t1 = boite
    dedans = lambda o: x0 <= o["x0"] and o["x1"] <= x1 and t0 <= o["top"] and o["bottom"] <= t1
    return [l for l in pg.lines if dedans(l)], [r for r in pg.rects if dedans(r)]


def marqueurs_carres(rects) -> list[tuple[float, float]]:
    """Centres des marqueurs carrés (Origin les dessine deux fois : fond puis contour)."""
    c = {(round((r["x0"] + r["x1"]) / 2, 2), round((r["top"] + r["bottom"]) / 2, 2))
         for r in rects if 2.5 < r["width"] < 6 and 2.5 < r["height"] < 6}
    return sorted(c)


def barre_erreur(cx: float, cy: float, lignes) -> float | None:
    """Demi-hauteur de la barre d'erreur (pt) d'après ses deux traits de capuchon."""
    caps = sorted({round(l["top"], 2) for l in lignes if abs(l["top"] - l["bottom"]) < 0.2
                   and abs((l["x0"] + l["x1"]) / 2 - cx) < 1.5 and 1.5 < abs(l["x1"] - l["x0"]) < 3.5})
    haut = [c for c in caps if c < cy]
    bas = [c for c in caps if c > cy]
    if not haut or not bas:
        return None
    h, b = max(haut), min(bas)
    return (b - h) / 2 if abs((h + b) / 2 - cy) < 0.3 else None


def image_pdf(pdf_path: Path, page: int, index: int = 0) -> np.ndarray:
    im = pdfplumber.open(str(pdf_path)).pages[page - 1].images[index]
    flux = im["stream"]
    donnees = flux.get_data()
    if any("DCT" in str(f[0]) for f in flux.get_filters()):
        return np.asarray(Image.open(io.BytesIO(donnees)).convert("RGB")).astype(int)
    w, h = im["srcsize"]
    return np.frombuffer(donnees, np.uint8)[: w * h * 3].reshape(h, w, 3).astype(int)


# ============================================================ 1. Le Gac et al. 2017 (PDF fourni)
def le_gac(pdf_path: Path) -> None:
    doi = "10.1016/j.polymdegradstab.2017.02.003"
    base = dict(material_id="PA6 Goodfellow AM303020 (plaque 2 mm compression)", famille="polyamide",
                renfort="aucun", source_doi=doi, Epaisseur_mm=2, propriete="E_traction", Tg0_C=66,
                verification="controle_pdf", extraction="figure")
    # ---- Fig. 7a (p. 11) : E en fonction du temps d'immersion, eau de mer naturelle à 25 °C
    lignes, rects = objets_vectoriels(pdf_path, 11, (95, 125, 275, 275))
    gx0, gx30, gy0, gy3 = 110.88, 255.72, 261.12, 139.44      # graduations majeures (pt)
    px_j, px_gpa = (gx30 - gx0) / 30, (gy0 - gy3) / 3
    pts = marqueurs_carres(rects)
    controle(len(pts) == 10, f"Le Gac Fig. 7a : {len(pts)} marqueurs au lieu de 10")
    E0 = (gy0 - pts[0][1]) / px_gpa * 1000
    controle(abs(E0 - 2300) < 60, f"Le Gac Fig. 7a : E(0) = {E0:.0f} MPa, Tab. 1 annonce 2.3 GPa")
    meta = {**base, "milieu": "immersion", "fluide": "eau_de_mer", "Temperature_C": 25,
            "Humidite_HR": 100, "etat_essai": "humide", "source_fig": "Fig. 7a p. 11"}
    for cx, cy in pts:
        t_j = (cx - gx0) / px_j
        t_nom = round(t_j * 2) / 2
        controle(abs(t_j - t_nom) < 0.05, f"Le Gac Fig. 7a : t = {t_j:.3f} j loin d'une durée nominale")
        b = barre_erreur(cx, cy, lignes)
        ajouter("LEGAC17_EDM25", meta, t_nom * 24, (gy0 - cy) / px_gpa * 1000, E0,
                None if b is None else b / px_gpa * 1000, "inconnu",
                commentaire=f"digitalisation vectorielle (coordonnées exactes du marqueur) ; t = {t_j:.3f} j "
                            f"arrondi à {t_nom:g} j ; essai à la sortie du bac (§2) ; 3 éprouvettes par "
                            f"condition ; nature de la barre d'erreur non précisée")
    # ---- Fig. 3a (p. 9) : E en fonction de la teneur en eau, conditionnement 48 h à 80 °C (Tab. 2)
    lignes, rects = objets_vectoriels(pdf_path, 9, (95, 190, 270, 340))
    gx0, gx10, gy0, gy3 = 110.28, 262.08, 326.52, 202.2
    px_w, px_gpa = (gx10 - gx0) / 10, (gy0 - gy3) / 3
    pts = marqueurs_carres(rects)
    controle(len(pts) == 6, f"Le Gac Fig. 3a : {len(pts)} marqueurs au lieu de 6")
    conditions = [("sec", None, None, 0.0), ("HR20", 20, "vapeur", 0.7), ("HR30", 30, "vapeur", 1.1),
                  ("HR40", 40, "vapeur", 1.7), ("HR50", 50, "vapeur", 2.4), ("EDM", 100, "immersion", 9.3)]
    E0 = (gy0 - pts[0][1]) / px_gpa * 1000
    b0 = barre_erreur(*pts[0], lignes)
    for (cx, cy), (nom, hr, milieu, w_tab) in zip(pts, conditions):
        w = (cx - gx0) / px_w
        controle(abs(w - w_tab) < 0.25, f"Le Gac Fig. 3a : teneur {w:.2f} % loin de {w_tab} % (Tab. 2)")
        if nom == "sec":
            continue
        meta = {**base, "milieu": milieu, "fluide": "eau_de_mer" if milieu == "immersion" else "",
                "Temperature_C": 80, "Humidite_HR": hr, "etat_essai": "humide",
                "source_fig": "Fig. 3a p. 9 + Tab. 2 p. 7"}
        cid = f"LEGAC17_T80_{nom}"
        ajouter(cid, meta, 0, E0, E0, None if b0 is None else b0 / px_gpa * 1000, "inconnu",
                commentaire="référence = éprouvette séchée (point 0 % de la Fig. 3a ; partagée par les 5 "
                            "conditions à 80 °C)")
        b = barre_erreur(cx, cy, lignes)
        note = " ; teneur 2.6 % sur la Fig. 3a contre 2.4 % au Tab. 2" if nom == "HR50" else ""
        ajouter(cid, meta, 48, (gy0 - cy) / px_gpa * 1000, E0,
                None if b is None else b / px_gpa * 1000, "inconnu",
                commentaire=f"48 h à 80 °C (Tab. 2) ; teneur en eau lue {w:.2f} %{note} ; état humide déduit : "
                            f"module tracé en fonction de la teneur en eau au moment de l'essai ; "
                            f"80 °C > Tg sèche (66 °C)")


# ============================================================ 2. Zarbali et al. 2026 (PDF fourni)
def zarbali(pdf_path: Path) -> None:
    im = image_pdf(pdf_path, 4)                                    # Fig. 3 p. 31295
    rouge = (im[..., 0] > 180) & (im[..., 1] < 110) & (im[..., 2] < 110)
    # calibration relevée sur les graduations : x(1 %) = 265 px, 143 px/% ; y(0.5 GPa) = 704 px, 230 px/GPa
    X = lambda x: (x - 122.0) / 143.0
    Y = lambda y: 0.5 + (704.0 - y) / 230.0
    lab, n = ndi.label(ndi.binary_erosion(rouge, iterations=5))
    pts = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.mean() > 100 and len(xs) > 40:                       # exclut la légende et les fragments
            pts.append((xs.mean(), ys.mean()))
    fen = rouge[105:155, 100:145]                                  # point 0 % coupé par l'axe
    lig = np.where(fen.sum(1) >= 8)[0] + 105
    pts.append((122.0, (lig.min() + lig.max()) / 2))
    pts.sort()
    table3 = [(4, 0.0), (36, 0.51), (57, 0.85), (73, 1.78), (99.9, 5.29)]   # Tab. 2 et 3
    controle(len(pts) == 5, f"Zarbali Fig. 3 : {len(pts)} points au lieu de 5")
    for (cx, _), (_, w_tab) in zip(pts, table3):
        controle(abs(X(cx) - w_tab) < 0.05, f"Zarbali Fig. 3 : teneur {X(cx):.3f} % contre {w_tab} % (Tab. 3)")
    E0 = Y(pts[0][1]) * 1000
    base = dict(material_id="PA6 Akulon F223-D (injection ISO 527-2)", famille="polyamide", renfort="aucun",
                source_doi="10.1021/acsomega.6c01154", source_fig="Fig. 3 p. 31295 + Tab. 2-3",
                extraction="figure", milieu="vapeur", fluide="", Temperature_C=None, Epaisseur_mm=4,
                etat_essai="humide", propriete="E_traction", Tg0_C=None, verification="controle_pdf")
    for (cx, cy), (hr, w_tab) in zip(pts[1:], table3[1:]):
        cid = f"ZARB26_HR{str(hr).replace('.', '')}"
        meta = {**base, "Humidite_HR": hr}
        ajouter(cid, meta, 0, E0, E0, commentaire="référence = éprouvette conditionnée à 4 %HR (teneur en eau 0) : "
                "HYPOTHÈSE d'état initial ; TEMPÉRATURE DE CONDITIONNEMENT NON PUBLIÉE")
        inc = 0.0857 * 1000 if hr == 57 else None                 # seule barre dépassant le marqueur
        ajouter(cid, meta, 650, Y(cy) * 1000, E0, inc, "inconnu" if inc else "",
                commentaire=f"digitalisation raster (centre du disque rouge ; 1 px = 0.004 GPa) ; HR réelle {hr} % "
                            f"(Tab. 2) ; 650 h selon §Experimental mais 25 j dans la conclusion ; "
                            f"teneur en eau {w_tab} % (Tab. 3)")


# ============================================================ 3. Chilali 2017 (PDF fourni)
def chilali(pdf_path: Path) -> None:
    src = "THESE:Chilali 2017 Université de Reims Champagne-Ardenne (identifiant HAL/theses.fr à ajouter)"
    note = ("stratifié 4 plis 3 mm nominal (p. 75) ; module longitudinal 0° ; eau du robinet à 'température "
            "ambiante' SANS valeur (p. 88) ; séchage à la température de vieillissement (p. 120)")
    base = dict(renfort="naturel", source_doi=src, extraction="tableau", milieu="immersion",
                fluide="eau_robinet", Temperature_C=None, Humidite_HR=100, Epaisseur_mm=3,
                propriete="E_traction", Tg0_C=None, verification="controle_pdf")
    tab = {"LINEPOX": ("Lin / époxyde (Vf 32 %)", "epoxy", 14.05, 0.32, 5.42, 0.25, 11.70, 0.56),
           "LINELIUM": ("Lin / Elium 150 acrylique Arkema (Vf 37 %)", "acrylique", 13.18, 0.92, 5.65, 0.52,
                        11.18, 0.97)}
    # ---- Fig. 4.4a (p. 117, image raster 501x379) : points intermédiaires du lin/époxyde
    im = image_pdf(pdf_path, 117)
    bleu = (im[..., 2] > 150) & (im[..., 0] < 110) & (im[..., 1] < 130)
    a, b = np.polyfit(np.arange(5, 13), [297, 269, 240, 212, 183, 155, 126, 97.5], 1)  # graduations Y
    Y = lambda y: (y - b) / a
    def triangle(t_j: float) -> float:
        xc = 57 + t_j * 13.5                                       # graduations X : 2 j -> 84 px, 13.5 px/j
        bande = bleu[16:323, int(round(xc - 6)): int(round(xc + 6)) + 1].sum(1)
        idx = np.where(bande >= 5)[0]
        serie = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
        return Y((serie[0] + serie[-1]) / 2 + 16)
    controle(abs(triangle(0) - 14.05) < 0.05 and abs(triangle(30) - 5.42) < 0.05,
             "Chilali Fig. 4.4a : la calibration ne retrouve pas le Tab. 4.2 aux bornes")
    for cle, (mat, fam, e0, s0, eh, sh, es, ss) in tab.items():
        for etat, suff, ev, sv, lib in (("humide", "HUM", eh, sh, "vieilli à saturation (30 j) non séché"),
                                        ("reseche", "SEC", es, ss, "vieilli à saturation (30 j) puis séché")):
            cid = f"CHIL17_{cle}_{suff}"
            meta = {**base, "material_id": mat, "famille": fam, "etat_essai": etat}
            ajouter(cid, meta, 0, e0 * 1000, e0 * 1000, s0 * 1000, "inconnu", fig="Tab. 4.2 p. 121",
                    commentaire=f"non vieilli EL = {e0} ± {s0} GPa ; {note}")
            if cle == "LINEPOX" and etat == "humide":
                for t_j in (2, 7, 15):
                    ajouter(cid, meta, t_j * 24, triangle(t_j) * 1000, e0 * 1000, 0.035 * 1000, "digitalisation",
                            fig="Fig. 4.4a p. 117", extraction="figure",
                            commentaire="digitalisation raster (centre du triangle bleu) ; calibration contrôlée "
                                        "sur le Tab. 4.2 à 0 et 30 j (écart < 0.02 GPa) ; incertitude = 1 px")
            ajouter(cid, meta, 720, ev * 1000, e0 * 1000, sv * 1000, "inconnu", fig="Tab. 4.2 p. 121",
                    commentaire=f"{lib} EL = {ev} ± {sv} GPa ; Temps_h = durée d'immersion (séchage non compté)")


# ============================================================ 4. Gibhardt et al. 2022 (PDF fourni, TEXTE)
def gibhardt() -> None:
    base = dict(material_id="Hexion EPIKOTE RIMR 135 / EPIKURE RIMH 137 (100:30)", famille="epoxy",
                renfort="aucun", source_doi="10.3389/fmats.2022.826076", extraction="texte",
                milieu="immersion", fluide="inconnu", Humidite_HR=100, Epaisseur_mm=0.5,
                etat_essai="humide", propriete="E_traction", Tg0_C=89.4, verification="controle_pdf")
    E0 = 2750.0
    ref = ("E0 = 2.75 GPa état sec de référence (§3.1) ; essai < 30 min après sortie du bain (§2.3) ; "
           "épaisseur 0.5 mm (Fig. 1) ; Tg0 = Tg onset de E' en DMTA (§3.1) ; nature de l'eau non précisée")
    for T, pts in ((8, [(1028, 2700, "§3.4 : 2.7 GPa à saturation (valeur arrondie) ; 1028 h au Tab. 1 et "
                                     "environ 1000 h sur les Fig. 9 et 11"),
                        (2600, 2500, "§3.9 : 2.50 GPa après vieillissement long")]),
                   (50, [(76, 2500, "§3.4 : 2.5 GPa à saturation (valeur arrondie) ; 76 h (Tab. 1 et Fig. 9)"),
                         (2600, 2550, "§3.9 : 2.55 GPa après vieillissement long (hausse attribuée au "
                                      "vieillissement physique par les auteurs)")]),
                   (70, [(250, 2150, "§3.9 : 2.15 GPa ; DURÉE CONTRADICTOIRE : 250 h sur les Fig. 9 / 11 / 13 "
                                     "mais 400 h au Tab. 1 ; vieillissement au-dessus de la Tg humide")])):
        cid = f"GIBH22_WB{T:02d}"
        meta = {**base, "Temperature_C": T, "source_fig": "§3.1 + §3.4 + §3.9 + Tab. 1"}
        ajouter(cid, meta, 0, E0, E0, commentaire="point t=0 = référence sèche ; " + ref)
        for t, E, com in pts:
            ajouter(cid, meta, t, E, E0, commentaire=com)


# ============================================================ 5. Guerira et al. 2016 (PDF fourni, TABLEAU)
def guerira() -> None:
    meta = dict(material_id="PA6 Takmid (plaque 7 mm)", famille="polyamide", renfort="aucun",
                source_doi="ACTES:ICEMAEP2016 Constantine pp. 1021-1029 (sans DOI)", source_fig="Tab. 3 p. 1026",
                extraction="tableau", milieu="immersion", fluide="eau_distillee", Temperature_C=None,
                Humidite_HR=100, Epaisseur_mm=7, etat_essai="inconnu", propriete="E_traction", Tg0_C=None,
                verification="controle_pdf")
    for j, E in ((0, 1268), (4, 763.35), (8, 779.8), (12, 404.9), (16, 674.1), (24, 433.15), (28, 527.95)):
        com = f"{j} j"
        if j == 0:
            com += (" ; 'température ambiante' SANS valeur ; E0 mesuré 1268 MPa contre 3.3 GPa au Tab. 1 "
                    "(donnée fournisseur) ; mode de mesure de la déformation non précisé ; 7 mm (texte) / 7.2 (Fig. 1)")
        if j == 12:
            com += " ; la Fig. 10 indique 404.5"
        ajouter("GUER16_PA6_EAU", meta, j * 24, E, 1268.0, commentaire=com)


# ============================================================ 6. extraction tierce (PDF NON fournis)
MOTIF = re.compile(r",(-?[\d.]+|NaN),(-?[\d.]+|NaN),(-?[\d.]+|NaN),(-?[\d.]+|NaN),([^,]+),"
                   r"(-?[\d.]+|NaN),(-?[\d.]+|NaN),(-?[\d.]+|NaN),([^,]*),")


def extraction_tierce(chemin: Path) -> list[str]:
    brut = chemin.read_bytes()
    try:
        texte = brut.decode("utf-8")
    except UnicodeDecodeError:
        texte = brut.decode("cp1252")
    ecartees: list[str] = []
    nombre = lambda v: None if v == "NaN" else float(v)
    commun = dict(verification="non_controle", extraction="tableau", Tg0_C=None)
    t0_faits: set[str] = set()
    for ligne in [l for l in texte.splitlines() if l.strip()][1:]:
        src = ligne.split(",")[0].strip('"')
        m = MOTIF.search(ligne)
        if not m:
            ecartees.append(f"{src} : ligne non lisible")
            continue
        T, HR, t, h, prop, P0, Pt, ratio, _ = m.groups()
        T, HR, t, h, P0, Pt = map(nombre, (T, HR, t, h, P0, Pt))
        fig = (re.search(r"(Tableau [^\t\",]+)", ligne) or re.search(r"(Figure [^\t\",]+)", ligne))
        fig = fig.group(1).strip() if fig else ""
        if src.startswith("Wang"):
            if not prop.startswith("Module"):
                ecartees.append(f"{src} t={t:g} h : {prop.strip()} -> data/donnees_hors_cible.csv")
                HORS_CIBLE.append(dict(source_doi="10.1016/j.compositesb.2016.09.067", source_fig=fig,
                                       materiau="Epoxy 618/F-44/AFG-90/CTBN (9:6:4:1) DDS/PPh3",
                                       cas=src.split("_")[-1], Temperature_C=T, Temps_h=t,
                                       propriete=prop.strip(), valeur_MPa=Pt, P0_MPa=P0,
                                       verification="non_controle",
                                       raison="résistance et non module : hors cible E/E0"))
                continue
            cas = src.split("_")[-1]
            milieu, hr, fluide = {"I": ("immersion", 100, "eau_deionisee"), "II": ("vapeur", 98, ""),
                                  "III": ("vapeur", 65, "")}[cas]
            meta = dict(commun, material_id="Epoxy 618/F-44/AFG-90/CTBN (9:6:4:1) DDS/PPh3", famille="epoxy",
                        renfort="aucun", source_doi="10.1016/j.compositesb.2016.09.067", milieu=milieu,
                        fluide=fluide, Temperature_C=T, Humidite_HR=hr, Epaisseur_mm=h, etat_essai="inconnu",
                        propriete="E_flexion", source_fig=fig)
            cid = f"WANG16_CAS{cas}"
            com = "temps reconvertis en heures par l'extraction (le Tab. 3 imprime la racine du temps en s^0.5)"
        elif src.startswith("Niaounakis"):
            meta = dict(commun, material_id="PLA NatureWorks 2002D (4.25 % D ; moulage par compression)",
                        famille="pla", renfort="aucun", source_doi="10.1002/app.32644", milieu="vapeur",
                        fluide="", Temperature_C=T, Humidite_HR=HR, Epaisseur_mm=h, etat_essai="inconnu",
                        propriete="E_traction", source_fig=fig)
            cid = f"NIAO11_T{T:g}"
            com = ("valeurs imprimées en GPa (1 décimale) ; l'extraction signale que le texte p. 478 annonce 8 % de "
                   "perte à 40 °C / 30 j contre 18.75 % dans le tableau")
        elif src.startswith("Bergaliyeva"):
            meta = dict(commun, material_id="PLA filament BQ (FFF 100 % ; 0/90)", famille="pla", renfort="aucun",
                        source_doi="10.3390/polym14235256", milieu="vapeur", fluide="", Temperature_C=T,
                        Humidite_HR=HR, Epaisseur_mm=h, etat_essai="reconditionne", propriete="E_traction",
                        source_fig=fig)
            cid = "BERG22_T50_HR70"
            com = "épaisseur non publiée (0.2 mm = hauteur de couche) ; reconditionné 88 h à 23 °C / 50 %HR"
        elif src.startswith("Goetjes"):
            amidon = "amidon" in ligne[: m.start()]                   # champ matériau seulement
            meta = dict(commun, material_id="PLA Luminy L130" + (" + 50 % amidon de pomme de terre" if amidon
                                                                   else " (injection 1A)"),
                        famille="pla", renfort="autre" if amidon else "aucun", source_doi="10.3390/ma17153683",
                        milieu="vapeur", fluide="", Temperature_C=T, Humidite_HR=HR, Epaisseur_mm=h,
                        etat_essai="reconditionne", propriete="E_traction", source_fig=fig)
            cid = f"GOET24_{'PLAAMI' if amidon else 'PLA'}_T{T:g}_HR{HR:g}"
            com = "E0 non publié (le témoin 23 °C / 10 %HR a déjà 168 h) ; épaisseur non publiée ; reconditionné 168 h"
        else:
            ecartees.append(f"{src} : aucune valeur de module chiffrée")
            continue
        if Pt is None:
            ecartees.append(f"{cid} t={t:g} h : module non chiffré")
            continue
        if P0 is not None and cid not in t0_faits:
            ajouter(cid, meta, 0, P0, P0, commentaire="point t=0 construit à partir de P0 publié ; " + com)
            t0_faits.add(cid)
        ajouter(cid, meta, t, Pt, P0, commentaire=com)
        if P0 is not None and ratio != "NaN" and abs(float(ratio) - Pt / P0) > 1e-4:
            print(f"  [!] {cid} t={t:g} : rapport de l'extraction {ratio} ≠ P_t/P0 = {Pt / P0:.6f}")
    return ecartees


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--extraction", default=None, help="Extraction_vieillissement_hygrothermique.csv")
    ap.add_argument("--sortie", default=str(Path(__file__).resolve().parent / "dataset_reel.csv"))
    a = ap.parse_args(argv)
    d = Path(a.pdf_dir)
    manquants = [f for f in PDF.values() if not (d / f).exists()]
    if manquants:
        print(f"PDF absents de {d} : {manquants}")
        return 1
    try:
        le_gac(d / PDF["legac"])
        zarbali(d / PDF["zarbali"])
        chilali(d / PDF["chilali"])
        gibhardt()
        guerira()
    except ErreurControle as e:
        print(f"CONTRÔLE DE DIGITALISATION ÉCHOUÉ : {e}\nAucun fichier écrit.")
        return 1
    ecartees = extraction_tierce(Path(a.extraction)) if a.extraction else []
    df = pd.DataFrame(LIGNES).reindex(columns=COLONNES)
    df.to_csv(a.sortie, index=False, encoding="utf-8")
    empreintes = [{"fichier": f, "sha256": hashlib.sha256((d / f).read_bytes()).hexdigest()}
                  for f in PDF.values()]
    if a.extraction:
        empreintes.append({"fichier": Path(a.extraction).name,
                           "sha256": hashlib.sha256(Path(a.extraction).read_bytes()).hexdigest()})
    pd.DataFrame(empreintes).to_csv(Path(a.sortie).with_name("sources_sha256.csv"), index=False)
    if HORS_CIBLE:
        pd.DataFrame(HORS_CIBLE).to_csv(Path(a.sortie).with_name("donnees_hors_cible.csv"), index=False)
        print(f"{len(HORS_CIBLE)} valeurs hors cible (résistances) conservées dans donnees_hors_cible.csv")
    print(f"{len(df)} lignes, {df['curve_id'].nunique()} courbes écrites dans {a.sortie}")
    print(df.groupby("verification")["curve_id"].nunique().to_string())
    if ecartees:
        print(f"\n{len(ecartees)} ligne(s) de l'extraction tierce NON intégrées :")
        motifs = pd.Series([re.sub(r" t=[\d.]+ h", "", e) for e in ecartees]).value_counts()
        for motif, n in motifs.items():
            print(f"  - {n:3d} x {motif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
