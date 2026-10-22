# Ligne « shebang » : lancement direct du fichier sous Linux/macOS (ignorée sous Windows).
#!/usr/bin/env python3
# Encodage UTF-8 du fichier.
# -*- coding: utf-8 -*-
# Docstring : rôle du script, colonnes produites et méthodes de lecture des figures.
"""
Extraction TRAÇABLE de toutes les données physiques -> data/donnees_physiques.csv

Usage (racine du dépôt) :
    python data/digitaliser_physique.py --pdf-dir sources_pdf \
        --extraction sources_pdf/Extraction_vieillissement_hygrothermique.csv

Une ligne = une mesure (x, y) d'une grandeur, pour un matériau, avec ses conditions et sa source.
  grandeur        x                              y
  absorption      temps (h)                      teneur en eau (% masse)
  isotherme       activité de l'eau a_w (-)      teneur en eau à l'équilibre (% masse)
  diffusivite     température (°C) ou vide       D (m²/s)
  tg_eau          teneur en eau (%)              Tg (°C)
  module_eau      teneur en eau (%)              E (MPa), éprouvette à teneur homogène
  module_temp     température d'essai (°C)       E (MPa), éprouvette sèche
  module_temps    temps d'exposition (h)         E (MPa)
  module_168h     (aucun)                        E (MPa) après 168 h, sans E0 publié
Colonnes de conditions : T_C (vide = non publiée), a_w, h_mm (vide = non publiée), milieu.
verification = controle_pdf (relu sur le PDF fourni) | non_controle (extraction tierce, PDF non fourni).

Figures vectorielles : coordonnées exactes des marqueurs (pdfplumber), calibration sur les graduations.
Figures raster : centre des marqueurs par couleur (analyse d'image), calibration sur les graduations,
contrôle contre les valeurs tabulées quand elles existent (arrêt si écart).
"""
# Annotations de type modernes sur toutes les versions de Python.
from __future__ import annotations

# argparse : options de la ligne de commande (--pdf-dir, --extraction, --sortie).
import argparse
# io : lit une image JPEG directement depuis des octets en mémoire.
import io
# re : expressions régulières, pour découper le CSV d'extraction mal formé.
import re
# sys : code de sortie du programme.
import sys
# Path : chemins de fichiers.
from pathlib import Path

# numpy : calculs sur tableaux et images.
import numpy as np
# pandas : construction et écriture du tableau final.
import pandas as pd
# pdfplumber : lecture des PDF (mots avec leurs coordonnées, objets vectoriels, images).
import pdfplumber
# PIL (Pillow) : décodage des images raster contenues dans les PDF.
from PIL import Image
# ndimage : analyse d'image (érosion, étiquetage des taches de couleur = marqueurs).
from scipy import ndimage as ndi

# Noms des PDF attendus dans le dossier des sources (ne pas les renommer).
PDF = {"legac": "48321.pdf", "zarbali": "acsomega_6c01154.pdf", "chilali": "65911_CHILALI_2017_archivage.pdf",
       "guerira": "74-GUERIRA.pdf", "gibhardt": "fmats-09-826076.pdf"}
# Ordre et noms des colonnes du fichier produit data/donnees_physiques.csv.
COLONNES = ["materiau", "grandeur", "x", "y", "T_C", "a_w", "h_mm", "milieu", "condition", "source_doi",
            "source_fig", "extraction", "verification", "commentaire"]
# Liste où s'accumulent toutes les mesures extraites (une entrée = une ligne du CSV).
LIGNES: list[dict] = []


# Erreur propre à ce script : levée quand un contrôle de lecture échoue ; aucun fichier n'est alors écrit.
class ErreurControle(RuntimeError):
    # Aucun comportement supplémentaire : la classe sert seulement à distinguer cette erreur des autres.
    pass


# Vérifie une condition de cohérence ; si elle est fausse, arrête l'extraction avec le message donné.
def controle(ok: bool, msg: str) -> None:
    # Condition fausse...
    if not ok:
        # ... arrêt.
        raise ErreurControle(msg)


# Ajoute une mesure à la liste. Arguments : matériau, source, grandeur, abscisse x, ordonnée y, puis conditions (T en
# °C, a_w, h en mm, milieu) et traçabilité (figure, mode d'extraction, vérification, commentaire). L'astérisque oblige
# à nommer les arguments optionnels, ce qui évite les inversions.
def ajouter(materiau, source, grandeur, x, y, *, T=None, a_w=None, h=None, milieu="", condition="",
            fig="", extraction="figure", verification="controle_pdf", commentaire=""):
    # Crée la ligne ; x est arrondi à 6 décimales et y à 6 chiffres significatifs (au-delà, c'est du bruit de
    # lecture).
    LIGNES.append(dict(materiau=materiau, grandeur=grandeur, x=None if x is None else round(float(x), 6),
                       y=float(f"{y:.6g}"), T_C=T, a_w=a_w, h_mm=h, milieu=milieu, condition=condition,
                       source_doi=source, source_fig=fig, extraction=extraction, verification=verification,
                       commentaire=commentaire))


# ========== OUTILS DE LECTURE
# ============================================================ outils
# Calibration d'un graphique vectoriel : relie les positions des graduations (en points PDF) à leurs valeurs.
def _calibration(pg, boite, xlab, ylab):
    # Cadre du graphique : gauche, haut, droite, bas (en points PDF).
    x0, t0, x1, t1 = boite
    # Centre horizontal d'un mot.
    cx = lambda w: (w["x0"] + w["x1"]) / 2
    # Centre vertical d'un mot.
    cy = lambda w: (w["top"] + w["bottom"]) / 2
    # Tous les mots de la page avec leurs coordonnées.
    W = pg.extract_words()
    # Graduations de l'axe X : textes attendus situés juste sous le bas du cadre.
    xs = [(float(w["text"]), cx(w)) for w in W
          if w["text"] in xlab and t1 - 2 < cy(w) < t1 + 14 and x0 - 15 < cx(w) < x1 + 5]
    # Graduations de l'axe Y : textes attendus situés juste à gauche du cadre.
    ys = [(float(w["text"]), cy(w)) for w in W
          if w["text"] in ylab and x0 - 25 < w["x0"] < x0 + 5 and t0 - 6 < cy(w) < t1 + 4]
    # Il faut au moins 3 graduations par axe pour une calibration fiable.
    controle(len(xs) >= 3 and len(ys) >= 3, f"calibration impossible ({len(xs)} x, {len(ys)} y)")
    # Droite valeur = a·position + b pour l'axe X (régression linéaire).
    ax = np.polyfit([p for _, p in xs], [v for v, _ in xs], 1)
    # Même chose pour l'axe Y.
    ay = np.polyfit([p for _, p in ys], [v for v, _ in ys], 1)
    # Renvoie les deux droites de calibration.
    return ax, ay


# Positions des marqueurs d'une figure vectorielle (losanges ou carrés), en points PDF.
def _marqueurs(pg, boite, mode):
    # Cadre de recherche.
    x0, t0, x1, t1 = boite
    # Teste si un objet est dans le cadre.
    dedans = lambda o: x0 < o["x0"] < x1 and t0 < o["top"] < t1
    # Losanges : dessinés comme des courbes noires d'environ 7 points de large.
    if mode == "losange":
        objs = [c for c in pg.curves if dedans(c) and str(c.get("stroking_color")) == "0.0"
                and 6.5 < c["x1"] - c["x0"] < 8]
    # Carrés...
    else:
        # ... dessinés comme des rectangles de 2,5 à 6 points.
        objs = [r for r in pg.rects if dedans(r) and 2.5 < r["width"] < 6 and 2.5 < r["height"] < 6]
    # Centres des marqueurs ; l'ensemble supprime les doublons (le logiciel dessine chaque marqueur deux fois : fond
    # et contour).
    return sorted({(round((o["x0"] + o["x1"]) / 2, 2), round((o["top"] + o["bottom"]) / 2, 2)) for o in objs})


# Extrait une image raster d'une page de PDF sous forme de tableau de pixels (hauteur x largeur x 3 couleurs).
def _image(pdf_path: Path, page: int, index: int = 0) -> np.ndarray:
    # Image n° index de la page (pages numérotées à partir de 1).
    im = pdfplumber.open(str(pdf_path)).pages[page - 1].images[index]
    # Flux de données de l'image.
    flux = im["stream"]
    # Octets décodés.
    donnees = flux.get_data()
    # Si l'image est un JPEG (filtre DCT)...
    if any("DCT" in str(f[0]) for f in flux.get_filters()):
        # ... décodage par Pillow puis conversion en tableau d'entiers.
        return np.asarray(Image.open(io.BytesIO(donnees)).convert("RGB")).astype(int)
    # Sinon, image brute : dimensions...
    w, h = im["srcsize"]
    # ... et lecture directe des octets en tableau de pixels RGB.
    return np.frombuffer(donnees, np.uint8)[: w * h * 3].reshape(h, w, 3).astype(int)


# Trouve les taches d'un masque de couleur (= les marqueurs) et renvoie leurs centres.
def _taches(masque, min_aire, erosion):
    # Érosion : retire les traits fins (courbes, barres d'erreur) pour ne garder que le cœur des marqueurs.
    m = ndi.binary_erosion(masque, iterations=erosion) if erosion else masque
    # Étiquetage : chaque tache connexe reçoit un numéro.
    lab, n = ndi.label(m)
    # Liste des centres.
    out = []
    # Pour chaque tache...
    for i in range(1, n + 1):
        # ... coordonnées de ses pixels ;
        ys, xs = np.where(lab == i)
        # ... si elle est assez grande (pas un résidu de trait)...
        if len(xs) >= min_aire:
            # ... on garde son centre (moyenne des pixels).
            out.append((xs.mean(), ys.mean()))
    # Centres triés de gauche à droite.
    return sorted(out)


# ========== 1. LE GAC 2017 : PA6 Goodfellow (figures vectorielles : coordonnées exactes)
# ============================================================ 1. Le Gac et al. 2017 — PA6 Goodfellow
# Extrait toutes les données de l'article de Le Gac.
def le_gac(p: Path) -> None:
    # Clé du matériau et DOI de la source.
    M, S = "PA6_GOODFELLOW", "10.1016/j.polymdegradstab.2017.02.003"
    # Ouverture du PDF.
    pdf = pdfplumber.open(str(p))
    # Fig. 1 gauche (p. 7) : absorption eau de mer 25 °C, h = 2 mm ; abscisse = racine(t)/h (s^0.5/m)
    # Page 7 (indice 6 : numérotation à partir de 0).
    pg = pdf.pages[6]
    # Cadre du graphique de gauche de la Fig. 1.
    boite = (108, 205, 275, 362)
    # Calibration sur les graduations des deux axes.
    ax, ay = _calibration(pg, boite, {"0.0", "0.5", "1.0", "1.5"}, {"0", "2", "4", "6", "8", "10", "12"})
    # Marqueurs convertis en valeurs (x = racine(t)/h, y = teneur en eau), triés.
    pts = sorted((np.polyval(ax, x), np.polyval(ay, y)) for x, y in _marqueurs(pg, boite, "losange"))
    # Retire le marqueur de la légende : seul point nettement plus bas que ses deux voisins.
    garde = [q for i, q in enumerate(pts)
             if not (0 < i < len(pts) - 1 and q[1] < pts[i - 1][1] - 1.0 and q[1] < pts[i + 1][1] - 1.0)]
    # Contrôle : exactement un point retiré.
    controle(len(pts) - len(garde) == 1, "Le Gac Fig. 1 : marqueur de légende non isolé")
    # Pour chaque point de la cinétique :
    for x, y in garde:
        # conversion de racine(t)/h (en 10⁶ s^0,5/m) en temps (h) : t = (x·h)², avec h = 2 mm ;
        t_h = (x * 1e6 * 2e-3) ** 2 / 3600
        # enregistrement (25 °C, eau de mer, a_w = 0,98, 2 mm).
        ajouter(M, S, "absorption", t_h, y, T=25, a_w=0.98, h=2.0, milieu="immersion_eau_de_mer",
                fig="Fig. 1 gauche p. 7", commentaire="vectoriel ; abscisse publiée racine(t)/h convertie en heures")
    # Fig. 10 (p. 13) : Tg (pic tan delta) en fonction de la teneur en eau
    # Page 13 : Fig. 10.
    pg = pdf.pages[12]
    # Cadre du graphique.
    boite = (215, 455, 385, 622)
    # Calibration.
    ax, ay = _calibration(pg, boite, {"0.0", "2.5", "5.0", "7.5", "10.0"}, {"80", "60", "40", "20", "0", "-20"})
    # Chaque marqueur...
    for x, y in _marqueurs(pg, boite, "losange"):
        # ... devient un point Tg(teneur en eau).
        ajouter(M, S, "tg_eau", np.polyval(ax, x), np.polyval(ay, y), fig="Fig. 10 p. 13", commentaire="vectoriel")
    # Fig. 3a (p. 9) : E en fonction de la teneur en eau homogène
    # Page 9 : Fig. 3a.
    pg = pdf.pages[8]
    # Positions (en points PDF) des graduations extrêmes relevées : 0 et 10 % en X, 0 et 3 GPa en Y.
    gx0, gx10, gy0, gy3 = 110.28, 262.08, 326.52, 202.2
    # Chaque marqueur carré...
    for cx, cy in _marqueurs(pg, (95, 190, 270, 340), "carre"):
        # ... converti en teneur en eau (%) et en module (MPa) par proportionnalité entre graduations.
        ajouter(M, S, "module_eau", max((cx - gx0) / ((gx10 - gx0) / 10), 0.0),
                (gy0 - cy) / ((gy0 - gy3) / 3) * 1000, h=2.0, fig="Fig. 3a p. 9",
                commentaire="vectoriel ; essai à la sortie du conditionnement ; T d'essai ambiante (21 °C supposé)")
    # Fig. 5a (p. 10) : E du PA6 sec en fonction de la température d'essai
    # Page 10 : Fig. 5a.
    pg = pdf.pages[9]
    # Mots de la page.
    W = pg.extract_words()
    # Centre horizontal d'un mot.
    cx = lambda w: (w["x0"] + w["x1"]) / 2
    # Centre vertical d'un mot.
    cy = lambda w: (w["top"] + w["bottom"]) / 2
    # Graduations de l'axe X (températures).
    xs = [(float(w["text"]), cx(w)) for w in W if w["text"] in {"0", "20", "40", "60", "80"} and 345 < cy(w) < 356 and cx(w) < 270]
    # Graduations de l'axe Y (GPa).
    ys = [(float(w["text"]), cy(w)) for w in W if w["text"] in {"0", "1", "2", "3"} and cx(w) < 92 and 200 < cy(w) < 346]
    # Calibration X.
    ax = np.polyfit([q for _, q in xs], [v for v, _ in xs], 1)
    # Calibration Y.
    ay = np.polyfit([q for _, q in ys], [v for v, _ in ys], 1)
    # Marqueurs carrés du graphique de gauche.
    R = [r for r in pg.rects if 2.5 < r["width"] < 6 and 200 < r["top"] < 345 and r["x0"] < 270]
    # Pour chaque marqueur (doublons supprimés, triés)...
    for x, y in sorted({(round((r["x0"] + r["x1"]) / 2, 2), round((r["top"] + r["bottom"]) / 2, 2)) for r in R}):
        # ... point module/température d'essai, éprouvette sèche (a_w = 0).
        ajouter(M, S, "module_temp", np.polyval(ax, x), np.polyval(ay, y) * 1000, a_w=0.0,
                fig="Fig. 5a p. 10", commentaire="vectoriel ; éprouvette sèche")
    # Tab. 2 (p. 7) : teneur en eau homogène après 48 h à 80 °C
    # Tab. 2 : teneur en eau après 48 h à 80 °C pour 4 humidités (recopiée du tableau)...
    for hr, w in ((20, 0.7), (30, 1.1), (40, 1.7), (50, 2.4)):
        # ... enregistrée comme point d'isotherme.
        ajouter(M, S, "isotherme", hr / 100, w, T=80, a_w=hr / 100, h=2.0, milieu="vapeur", fig="Tab. 2 p. 7",
                extraction="tableau", commentaire="équilibre supposé atteint en 48 h (teneur dite homogène)")
    # Point eau de mer du même tableau (a_w = 0,98).
    ajouter(M, S, "isotherme", 0.98, 9.3, T=80, a_w=0.98, h=2.0, milieu="immersion_eau_de_mer",
            fig="Tab. 2 p. 7 + §Modeling p. 6", extraction="tableau", commentaire="a_w eau de mer = 0.98 (auteurs)")
    # Fig. 7a (p. 11) : E(t) eau de mer 25 °C, 2 mm -> VALIDATION (jamais ajustée)
    # Page 11 : Fig. 7a.
    pg = pdf.pages[10]
    # Graduations extrêmes relevées : 0 et 30 jours, 0 et 3 GPa.
    gx0, gx30, gy0, gy3 = 110.88, 255.72, 261.12, 139.44
    # Chaque marqueur...
    for cx_, cy_ in _marqueurs(pg, (95, 125, 275, 275), "carre"):
        # ... temps en jours, arrondi à la demi-journée (durées nominales de l'essai) ;
        t_j = round((cx_ - gx0) / ((gx30 - gx0) / 30) * 2) / 2
        # ... point de VALIDATION E(t) (jamais utilisé pour ajuster le modèle).
        ajouter(M, S, "module_temps", t_j * 24, (gy0 - cy_) / ((gy0 - gy3) / 3) * 1000, T=25, a_w=0.98, h=2.0,
                milieu="immersion_eau_de_mer", fig="Fig. 7a p. 11",
                commentaire="vectoriel ; VALIDATION : non utilisé pour ajuster la chaîne physique")


# ========== 2. GIBHARDT 2022 : époxy (valeurs écrites dans le texte et les légendes)
# ============================================================ 2. Gibhardt et al. 2022 — époxy RIMR 135
# Recopie les valeurs publiées par Gibhardt (aucune lecture de figure).
def gibhardt() -> None:
    # Clé du matériau et DOI.
    M, S = "EPOXY_HEXION_RIMR135", "10.3389/fmats.2022.826076"
    # Référence commune.
    f5 = "Fig. 5 (légende) p. 6"
    # Coefficients de diffusion publiés à 8, 40 et 50 °C...
    for T, D in ((8, 3.2e-14), (40, 1.6e-13), (50, 5.5e-13)):
        # ... enregistrés (immersion, 0,5 mm).
        ajouter(M, S, "diffusivite", T, D, T=T, a_w=1.0, h=0.5, milieu="immersion_eau_douce", fig=f5,
                extraction="texte", commentaire="D fickien ajusté par les auteurs ; Q = 47.2 kJ/mol (§3.2)")
    # M∞ en immersion aux mêmes températures...
    for T, Mi in ((8, 2.57), (40, 2.80), (50, 3.00)):
        # ... enregistrés.
        ajouter(M, S, "isotherme", 1.0, Mi, T=T, a_w=1.0, h=0.5, milieu="immersion_eau_douce", fig=f5,
                extraction="texte", commentaire="M∞ ajusté par les auteurs ; pas d'augmentation au-delà de 50 °C (§3.2)")
    # M∞ en vapeur à 50 °C (50 et 80 %HR)...
    for hr, Mi in ((50, 0.64), (80, 1.90)):
        # ... enregistrés.
        ajouter(M, S, "isotherme", hr / 100, Mi, T=50, a_w=hr / 100, h=0.5, milieu="vapeur",
                fig="Fig. 5B (légende) p. 6", extraction="texte", commentaire="M∞ ajusté par les auteurs")
    # Module sec de référence (2,75 GPa).
    ajouter(M, S, "module_eau", 0.0, 2750, h=0.5, fig="§3.1 p. 5", extraction="texte", commentaire="sec (référence)")
    # Module saturé à 8 °C (teneur 2,57 %).
    ajouter(M, S, "module_eau", 2.57, 2700, h=0.5, fig="§3.4 p. 8 + Fig. 5A", extraction="texte",
            commentaire="saturé à 8 °C ; 2.7 GPa arrondi à 0.1 GPa")
    # Module saturé à 50 °C (teneur 3,00 %).
    ajouter(M, S, "module_eau", 3.00, 2500, h=0.5, fig="§3.4 p. 8 + Fig. 5A", extraction="texte",
            commentaire="saturé à 50 °C ; 2.5 GPa arrondi à 0.1 GPa")
    # Tg(M) recalculée en 4 points depuis la régression publiée Tg = 89,2 − 9,7·M (ce ne sont pas des mesures).
    for Mi in (0.0, 1.0, 2.0, 3.0):
        # Enregistrement.
        ajouter(M, S, "tg_eau", Mi, 89.2 - 9.7 * Mi, fig="Fig. 10 (régression des auteurs) p. 9", extraction="texte",
                commentaire="recalculé depuis la régression publiée Tg = 89.2 - 9.7 M (vieillissement <= 40 °C)")
    # Modules publiés dans le temps (8, 50 et 70 °C)...
    for T, t, E in ((8, 1028, 2700), (8, 2600, 2500), (50, 76, 2500), (50, 2600, 2550), (70, 250, 2150)):
        # ... enregistrés comme mesures E(t).
        ajouter(M, S, "module_temps", t, E, T=T, a_w=1.0, h=0.5, milieu="immersion_eau_douce", fig="§3.4 + §3.9",
                extraction="texte", commentaire="70 °C au-dessus de la Tg humide ; 2600 h : vieillissement physique")


# ========== 3. ZARBALI 2026 : PA6 Akulon (figures raster : analyse d'image)
# ============================================================ 3. Zarbali et al. 2026 — PA6 Akulon F223-D
# Extrait les données de Zarbali.
def zarbali(p: Path) -> None:
    # Clé du matériau et DOI.
    M, S = "PA6_AKULON_F223D", "10.1021/acsomega.6c01154"
    # Humidités réelles des dessiccateurs (Tab. 2) ; dictionnaire conservé pour mémoire, non utilisé dans les calculs.
    tab2 = {4: 4, 36: 36, 57: 57, 73: 73, 99.9: 99.9}        # HR réelles (Tab. 2)
    # Tab. 3 : (HR, teneur en eau mesurée à 25 j, M∞ prédit par les auteurs).
    tab3 = [(4, 0.0, None), (36, 0.51, 0.52), (57, 0.85, 0.94), (73, 1.78, 1.85), (99.9, 5.29, 5.34)]
    # Texte d'avertissement repris dans chaque ligne.
    note_T = "TEMPÉRATURE DE CONDITIONNEMENT NON PUBLIÉE (dessiccateurs, ambiante)"
    # Pour chaque humidité...
    for hr, w25, winf in tab3:
        # ... point d'isotherme (M∞ des auteurs, ou valeur mesurée à 4 %HR où il n'y a pas de M∞).
        ajouter(M, S, "isotherme", hr / 100, winf if winf is not None else w25, a_w=hr / 100, h=4.0, milieu="vapeur",
                fig="Tab. 3 p. 31294", extraction="tableau",
                commentaire=("M∞ prédit par les auteurs (Fick)" if winf is not None else "mesuré à 25 j") + " ; " + note_T)
    # Fig. 1 (p. 31294, raster 998x744) : prise en eau en fonction du temps, h = 4 mm
    # Page 3, première image : Fig. 1.
    im = _image(p, 3, 0)
    # Canaux rouge, vert, bleu de chaque pixel.
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    # Conversion pixel -> temps (h) : 95,2 px = 0 s et 956 px = 4·10⁶ s (graduations relevées).
    X = lambda x: (x - 95.2) / (956 - 95.2) * 4.0e6 / 3600        # h ; graduations 0 -> 4e6 s
    # Conversion pixel -> teneur en eau (%) : 649 px = 0 %, 106,5 px par %.
    Y = lambda y: (649 - y) / 106.5                               # % ; 106.5 px par %
    # Masque de couleur de chaque série : vert (99,9 %HR), magenta (73), bleu (57), rouge (36).
    masques = {99.9: (G > 110) & (R < 90) & (B < 90), 73: (R > 200) & (B > 200) & (G < 90),
               57: (B > 180) & (R < 90) & (G < 120), 36: (R > 200) & (G < 90) & (B < 90)}
    # Pour chaque série :
    for hr, m in masques.items():
        # copie du masque ;
        m = m.copy()
        # efface la zone de la légende ;
        m[20:215, 95:240] = False                                  # légende
        # centres des marqueurs ;
        pts = _taches(m, 25, 3)
        # contrôle : au moins 8 points trouvés ;
        controle(len(pts) >= 8, f"Zarbali Fig. 1 : {len(pts)} points pour {hr} %HR")
        # pour chaque point...
        for x, y in pts:
            # ... hors du point initial...
            if X(x) > 1:
                # ... enregistrement de la prise en eau.
                ajouter(M, S, "absorption", X(x), max(Y(y), 0.0), a_w=hr / 100, h=4.0, milieu="vapeur",
                        fig="Fig. 1 p. 31294", commentaire=f"raster ; {note_T} ; points masqués par superposition omis")
    # Fig. 2 (raster 997x830) : Tg en fonction de la teneur en eau
    # Page 3, deuxième image : Fig. 2.
    im = _image(p, 3, 1)
    # Canaux de couleur.
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    # Masque des disques rouges.
    rouge = (R > 200) & (G < 90) & (B < 90)
    # Copie...
    m = rouge.copy()
    # ... sans la zone de légende.
    m[30:110, 520:640] = False
    # Pixel -> teneur en eau (0 à 6 %).
    X = lambda x: (x - 123) / (988.5 - 123) * 6
    # Pixel -> Tg (80 °C en haut, −40 °C en bas).
    Y = lambda y: 80 - (y - 14) / (710 - 14) * 120
    # Centres des disques.
    pts = _taches(m, 40, 4)
    # Contrôle : exactement 5 points.
    controle(len(pts) == 5, f"Zarbali Fig. 2 : {len(pts)} points au lieu de 5")
    # Pour chaque point, associé à la ligne du Tab. 3 correspondante...
    for (x, y), (_, _, winf) in zip(pts, tab3):
        # ... si un M∞ est publié...
        if winf is not None:
            # ... contrôle que la teneur lue correspond au tableau (à 0,06 % près) ;
            controle(abs(X(x) - winf) < 0.06, f"Zarbali Fig. 2 : teneur {X(x):.2f} contre {winf} (Tab. 3)")
        # ... enregistrement Tg(teneur en eau).
        ajouter(M, S, "tg_eau", max(X(x), 0.0), Y(y), fig="Fig. 2 p. 31294", commentaire="raster ; Tg = pic tan delta")
    # Fig. 3 (p. 31295, raster 989x814) : E en fonction de la teneur en eau (25 j)
    # Page 4 : Fig. 3.
    im = _image(p, 4, 0)
    # Masque des disques rouges.
    rouge = (im[..., 0] > 180) & (im[..., 1] < 110) & (im[..., 2] < 110)
    # Pixel -> teneur en eau.
    X = lambda x: (x - 122.0) / 143.0
    # Pixel -> module (GPa).
    Y = lambda y: 0.5 + (704.0 - y) / 230.0
    # Érosion forte (supprime barres d'erreur et courbe) puis étiquetage des taches.
    lab, n = ndi.label(ndi.binary_erosion(rouge, iterations=5))
    # Centres retenus.
    pts = []
    # Pour chaque tache...
    for i in range(1, n + 1):
        # ... ses pixels ;
        ys, xs = np.where(lab == i)
        # ... hors légende (en haut) et assez grande...
        if ys.mean() > 100 and len(xs) > 40:
            # ... centre retenu.
            pts.append((xs.mean(), ys.mean()))
    # Le disque à 0 % est coupé par l'axe : on lit sa hauteur dans une petite fenêtre...
    fen = rouge[105:155, 100:145]
    # ... lignes où le rouge est assez large...
    lig = np.where(fen.sum(1) >= 8)[0] + 105
    # ... et on prend le milieu de ces lignes comme centre.
    pts.append((122.0, (lig.min() + lig.max()) / 2))
    # Tri de gauche à droite.
    pts.sort()
    # Contrôle : 5 points.
    controle(len(pts) == 5, f"Zarbali Fig. 3 : {len(pts)} points au lieu de 5")
    # Pour chaque point, associé au Tab. 3...
    for (x, y), (hr, w25, _) in zip(pts, tab3):
        # ... contrôle que la teneur lue correspond au tableau (à 0,05 % près) ;
        controle(abs(X(x) - w25) < 0.05, f"Zarbali Fig. 3 : teneur {X(x):.3f} contre {w25} (Tab. 3)")
        # ... enregistrement module/teneur en eau.
        ajouter(M, S, "module_eau", max(X(x), 0.0), Y(y) * 1000, h=4.0, fig="Fig. 3 p. 31295",
                commentaire=f"raster (1 px = 0.004 GPa) ; {hr} %HR ; 0 % = conditionné à 4 %HR")


# ========== 4. CHILALI 2017 : composites lin (tableaux + figure raster)
# ============================================================ 4. Chilali 2017 — composites lin
# Extrait les données de la thèse de Chilali.
def chilali(p: Path) -> None:
    # Référence de la thèse (pas de DOI).
    S = "THESE:Chilali 2017 Université de Reims Champagne-Ardenne"
    # Avertissement repris dans chaque ligne.
    note = "eau du robinet à 'température ambiante' NON CHIFFRÉE (p. 88) ; stratifié 0°, 4 plis, 3 mm (p. 75)"
    # Tab. 4.1 (p. 114) : diffusion (Fick 3D) des éprouvettes à 0°
    # Pour chaque composite : Dzz (mm²/s) et M∞ (%) du Tab. 4.1...
    for M, Dzz, Minf in (("LIN_EPOXY", 2.76e-6, 7.31), ("LIN_ELIUM", 3.12e-6, 6.63)):
        # ... D converti de mm²/s en m²/s (x 10⁻⁶) et enregistré ;
        ajouter(M, S, "diffusivite", None, Dzz * 1e-6, a_w=1.0, h=3.0, milieu="immersion_eau_douce",
                fig="Tab. 4.1 p. 114", extraction="tableau",
                commentaire="Dzz (épaisseur) du modèle de Fick 3D ; " + note)
        # ... M∞ enregistré.
        ajouter(M, S, "isotherme", 1.0, Minf, a_w=1.0, h=3.0, milieu="immersion_eau_douce", fig="Tab. 4.1 p. 114",
                extraction="tableau", commentaire="M∞ ; " + note)
    # Tab. 4.2 (p. 121) et Fig. 4.4a (p. 117, raster 501x379)
    # Image de la Fig. 4.4a (page 117).
    im = _image(p, 117)
    # Masque des triangles bleus (lin/époxy).
    bleu = (im[..., 2] > 150) & (im[..., 0] < 110) & (im[..., 1] < 130)
    # Masque des carrés noirs (lin/Elium).
    noir = (im[..., 0] < 90) & (im[..., 1] < 90) & (im[..., 2] < 90)
    # Calibration de l'axe Y : positions des graduations de 5 à 12 GPa (régression linéaire).
    a, b = np.polyfit(np.arange(5, 13), [297, 269, 240, 212, 183, 155, 126, 97.5], 1)
    # Pixel -> module (GPa).
    Y = lambda y: (y - b) / a

    # Lit le module d'une série à un temps donné.
    def lire(masque, t_j, seuil):
        # Position horizontale du temps : 57 px = 0 jour, 13,5 px par jour.
        xc = 57 + t_j * 13.5
        # Nombre de pixels de la couleur sur chaque ligne d'une bande verticale de 13 px autour de ce temps.
        bande = masque[16:323, int(round(xc - 6)): int(round(xc + 6)) + 1].sum(1)
        # Lignes où la couleur est assez présente.
        idx = np.where(bande >= seuil)[0]
        # Plus longue suite de lignes consécutives = le marqueur.
        serie = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
        # Milieu du marqueur converti en module.
        return Y((serie[0] + serie[-1]) / 2 + 16)

    # Contrôle : la figure doit redonner le Tab. 4.2 à 0 et 30 jours (à 0,05 GPa près), sinon arrêt.
    controle(abs(lire(bleu, 0, 5) - 14.05) < 0.05 and abs(lire(bleu, 30, 5) - 5.42) < 0.05,
             "Chilali Fig. 4.4a : calibration incohérente avec le Tab. 4.2")
    # Pour chaque composite : masque, seuil, puis valeurs du Tab. 4.2 (E0, E à 30 j, E séché, avec écarts-types) et
    # incertitude de lecture de la figure.
    for M, masque, seuil, e0, s0, e30, s30, es, ss, inc in (
            ("LIN_EPOXY", bleu, 5, 14.05, 0.32, 5.42, 0.25, 11.70, 0.56, 0.035),
            ("LIN_ELIUM", noir, 6, 13.18, 0.92, 5.65, 0.52, 11.18, 0.97, 0.15)):
        # Conditions communes : température non publiée, immersion, 3 mm.
        kw = dict(T=None, a_w=1.0, h=3.0, milieu="immersion_eau_douce")
        # Point t = 0 (Tab. 4.2).
        ajouter(M, S, "module_temps", 0, e0 * 1000, fig="Tab. 4.2 p. 121", extraction="tableau",
                commentaire=f"non vieilli {e0} ± {s0} GPa ; " + note, **kw)
        # Points à 2, 7 et 15 jours lus sur la figure...
        for t_j in (2, 7, 15):
            # ... enregistrés en MPa.
            ajouter(M, S, "module_temps", t_j * 24, lire(masque, t_j, seuil) * 1000, fig="Fig. 4.4a p. 117",
                    commentaire=f"raster ; incertitude de lecture ±{inc} GPa"
                    + (" (marqueurs carrés en partie masqués par les triangles)" if M == "LIN_ELIUM" else ""), **kw)
        # Point à 30 jours (Tab. 4.2).
        ajouter(M, S, "module_temps", 720, e30 * 1000, fig="Tab. 4.2 p. 121", extraction="tableau",
                commentaire=f"vieilli à saturation (30 j) {e30} ± {s30} GPa", **kw)
        # Module après séchage (Tab. 4.2) : grandeur à part, part irréversible de la perte.
        ajouter(M, S, "module_reseche", 720, es * 1000, fig="Tab. 4.2 p. 121", extraction="tableau",
                commentaire=f"30 j puis séché jusqu'à la masse initiale : {es} ± {ss} GPa (part irréversible)", **kw)


# ========== 5. GUERIRA 2016 : PA6 Takmid (tableau + figure vectorielle)
# ============================================================ 5. Guerira et al. 2016 — PA6 Takmid
# Extrait les données de Guerira.
def guerira(p: Path) -> None:
    # Clé et référence (actes de congrès, sans DOI).
    M, S = "PA6_TAKMID", "ACTES:ICEMAEP2016 Constantine pp. 1021-1029 (sans DOI)"
    # Avertissement repris dans chaque ligne.
    note = "eau distillée à 'température ambiante' NON CHIFFRÉE"
    # Modules du Tab. 3 en fonction du temps (jours)...
    for j, E in ((0, 1268), (4, 763.35), (8, 779.8), (12, 404.9), (16, 674.1), (24, 433.15), (28, 527.95)):
        # ... enregistrés (temps converti en heures).
        ajouter(M, S, "module_temps", j * 24, E, a_w=1.0, h=7.0, milieu="immersion_eau_douce", fig="Tab. 3 p. 1026",
                extraction="tableau", commentaire=f"{j} j ; {note} ; état d'essai (humide/séché) non précisé"
                + (" ; E0 mesuré 1268 MPa contre 3.3 GPa fournisseur (Tab. 1)" if j == 0 else ""))
    # D publié dans le texte (géométrie de l'échantillon inconnue : non utilisé par le modèle).
    ajouter(M, S, "diffusivite", None, 0.259839e-12, a_w=1.0, h=None, milieu="immersion_eau_douce",
            fig="§2.1.3 p. 1024", extraction="texte",
            commentaire="D gravimétrique des auteurs (unité imprimée m/s, lue m²/s) ; géométrie de l'échantillon "
                        "d'absorption (11.86 mg) non donnée")
    # Fig. 3 (p. 1024, vectorielle) : prise de masse en fonction de racine(t) en racine(min)
    # Page 4.
    pg = pdfplumber.open(str(p)).pages[3]
    # Mots de la page.
    W = pg.extract_words()
    # Centre horizontal d'un mot.
    cx = lambda w: (w["x0"] + w["x1"]) / 2
    # Centre vertical d'un mot.
    cy = lambda w: (w["top"] + w["bottom"]) / 2
    # Graduations de l'axe X (racine des minutes).
    xs = [(float(w["text"]), cx(w)) for w in W if w["text"] in {"0", "100", "200", "300", "400"} and 230 < w["top"] < 240]
    # Graduations de l'axe Y (%).
    ys = [(float(w["text"]), cy(w)) for w in W if w["text"] in {"0", "2", "4", "5", "6"} and w["x0"] < 150 and 70 < w["top"] < 230]
    # Calibration X.
    ax = np.polyfit([q for _, q in xs], [v for v, _ in xs], 1)
    # Calibration Y.
    ay = np.polyfit([q for _, q in ys], [v for v, _ in ys], 1)
    # Marqueurs carrés rouges de la série expérimentale.
    R = [r for r in pg.rects if 5.5 < r["x1"] - r["x0"] < 7.5 and str(r.get("non_stroking_color")) == "(0.753, 0.314, 0.302)"
         and r["top"] < 232]
    # Centres sans doublons, triés.
    pts = sorted({(round((r["x0"] + r["x1"]) / 2, 2), round((r["top"] + r["bottom"]) / 2, 2)) for r in R})
    # Pour chaque marqueur...
    for x, y in pts:
        # ... abscisse en racine des minutes ;
        racine_min = np.polyval(ax, x)
        # ... au-delà de 400, c'est le marqueur de la légende...
        if racine_min > 400:                                     # marqueur de légende
            # ... ignoré ;
            continue
        # ... temps converti en heures (racine(min)² / 60) et enregistrement.
        ajouter(M, S, "absorption", racine_min ** 2 / 60, max(np.polyval(ay, y), 0), a_w=1.0, h=None,
                milieu="immersion_eau_douce", fig="Fig. 3 p. 1024",
                commentaire=f"vectoriel ; {note} ; épaisseur de l'échantillon d'absorption non donnée")


# ========== 6. EXTRACTION TIERCE (articles dont le PDF n'a pas été fourni)
# ============================================================ 6. extraction tierce (PDF non fournis)
# Motif qui retrouve, dans chaque ligne du CSV d'extraction, les 9 champs numériques et texte (T, HR, t, h, propriété,
# P0, P_t, rapport, origine), malgré les guillemets et tabulations mal placés du fichier.
MOTIF = re.compile(r",(-?[\d.]+|NaN),(-?[\d.]+|NaN),(-?[\d.]+|NaN),(-?[\d.]+|NaN),([^,]+),"
                   r"(-?[\d.]+|NaN),(-?[\d.]+|NaN),(-?[\d.]+|NaN),([^,]*),")


# Lit le CSV d'extraction tierce ; toutes ses lignes sont marquées « non_controle ».
def extraction_tierce(chemin: Path) -> None:
    # Lecture brute des octets.
    brut = chemin.read_bytes()
    # Essaie l'encodage UTF-8...
    try:
        # Décodage en UTF-8 (lève UnicodeDecodeError si le fichier n'est pas en UTF-8).
        texte = brut.decode("utf-8")
    # ... sinon...
    except UnicodeDecodeError:
        # ... l'encodage Windows (cp1252), celui du fichier fourni.
        texte = brut.decode("cp1252")
    # Convertit un champ en nombre, ou None pour « NaN ».
    nb = lambda v: None if v == "NaN" else float(v)
    # Options communes à toutes ces lignes.
    kw = dict(extraction="tableau", verification="non_controle")
    # Conditions déjà dotées de leur point t = 0.
    deja = set()
    # Pour chaque ligne non vide, sauf l'en-tête :
    for ligne in [l for l in texte.splitlines() if l.strip()][1:]:
        # identifiant de la source (premier champ) ;
        src = ligne.split(",")[0].strip('"')
        # recherche des champs ;
        m = MOTIF.search(ligne)
        # ligne illisible...
        if not m:
            # ... ignorée ;
            continue
        # champs extraits ;
        T, HR, t, h, prop, P0, Pt, _, _ = m.groups()
        # conversion des champs numériques ;
        T, HR, t, h, P0, Pt = map(nb, (T, HR, t, h, P0, Pt))
        # référence du tableau source ;
        fig = re.search(r"(Tableau [^\t\",]+)", ligne)
        # texte de la référence, ou vide ;
        fig = fig.group(1).strip() if fig else ""
        # module absent...
        if Pt is None:
            # ... ligne ignorée ;
            continue
        # Wang :
        if src.startswith("Wang"):
            # les résistances ne sont pas des modules...
            if not prop.startswith("Module"):
                # ... ignorées ;
                continue
            # numéro du cas (I, II, III) ;
            cas = src.split("_")[-1]
            # milieu et activité de l'eau du cas ;
            milieu, a_w = {"I": ("immersion_eau_douce", 1.0), "II": ("vapeur", 0.98), "III": ("vapeur", 0.65)}[cas]
            # clé et DOI ;
            mat, doi = "EPOXY_WANG_618", "10.1016/j.compositesb.2016.09.067"
            # commentaire ;
            com = f"cas {cas} ; module de FLEXION 3 points ; temps reconvertis en heures par l'extraction"
        # Niaounakis : clé, DOI, milieu et activité de l'eau ;
        elif src.startswith("Niaounakis"):
            # PLA 2002D (Niaounakis) : atmosphère humide, a_w = HR/100.
            mat, doi, milieu, a_w = "PLA_2002D", "10.1002/app.32644", "vapeur", HR / 100
            # commentaire ;
            com = "valeurs imprimées en GPa (1 décimale)"
        # Bergaliyeva : idem ;
        elif src.startswith("Bergaliyeva"):
            # PLA BQ imprimé (Bergaliyeva) : atmosphère humide, a_w = HR/100.
            mat, doi, milieu, a_w = "PLA_BQ_FFF", "10.3390/polym14235256", "vapeur", HR / 100
            # commentaire ;
            com = "épaisseur non publiée ; reconditionné 88 h à 23 °C / 50 %HR avant essai"
        # Goetjes :
        elif src.startswith("Goetjes"):
            # matériau avec ou sans amidon (cherché dans le champ matériau seulement) ;
            amidon = "amidon" in ligne[: m.start()]
            # clé ;
            mat = "PLA_L130_AMIDON" if amidon else "PLA_L130"
            # module à 168 h enregistré comme grandeur à part (pas d'E0 publié) ;
            ajouter(mat, "10.3390/ma17153683", "module_168h", None, Pt, T=T, a_w=HR / 100, h=None, milieu="vapeur",
                    condition=f"{T:g} °C / {HR:g} %HR / 168 h", fig=fig,
                    commentaire="E0 non publié ; reconditionné 168 h à 23 °C / 50 %HR avant essai", **kw)
            # ligne suivante ;
            continue
        # autre source (sans valeur chiffrée)...
        else:
            # ... ignorée.
            continue
        # Identifiant de la condition (matériau, T, a_w).
        cle = (mat, T, a_w)
        # Si E0 (P0) est publié et que cette condition n'a pas encore son point t = 0...
        if P0 is not None and cle not in deja:
            # ... ajout du point t = 0...
            ajouter(mat, doi, "module_temps", 0, P0, T=T, a_w=a_w, h=h, milieu=milieu, fig=fig,
                    commentaire="t = 0 construit à partir de P0 publié ; " + com, **kw)
            # ... condition marquée comme traitée.
            deja.add(cle)
        # Ajout du point vieilli.
        ajouter(mat, doi, "module_temps", t, Pt, T=T, a_w=a_w, h=h, milieu=milieu, fig=fig, commentaire=com, **kw)


# Programme principal.
def main(argv=None) -> int:
    # Analyseur d'options ; la description est la docstring.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --pdf-dir : dossier des PDF (obligatoire).
    ap.add_argument("--pdf-dir", required=True)
    # --extraction : CSV d'extraction tierce (facultatif).
    ap.add_argument("--extraction", default=None)
    # --sortie : fichier produit (par défaut data/donnees_physiques.csv).
    ap.add_argument("--sortie", default=str(Path(__file__).resolve().parent / "donnees_physiques.csv"))
    # Lecture des options.
    a = ap.parse_args(argv)
    # Dossier des PDF.
    d = Path(a.pdf_dir)
    # PDF attendus absents du dossier.
    manquants = [f for f in PDF.values() if not (d / f).exists()]
    # S'il en manque...
    if manquants:
        # ... message...
        print(f"PDF absents de {d} : {manquants}")
        # ... et arrêt avec code d'erreur.
        return 1
    # Extraction de chaque source...
    try:
        # Le Gac (PDF).
        le_gac(d / PDF["legac"])
        # Gibhardt (valeurs du texte, aucun PDF lu).
        gibhardt()
        # Zarbali (PDF).
        zarbali(d / PDF["zarbali"])
        # Chilali (PDF).
        chilali(d / PDF["chilali"])
        # Guerira (PDF).
        guerira(d / PDF["guerira"])
    # ... si un contrôle de lecture échoue...
    except ErreurControle as e:
        # ... message...
        print(f"CONTRÔLE DE DIGITALISATION ÉCHOUÉ : {e}\nAucun fichier écrit.")
        # ... et arrêt SANS écrire de fichier.
        return 1
    # Si le CSV d'extraction est fourni...
    if a.extraction:
        # ... ses lignes sont ajoutées.
        extraction_tierce(Path(a.extraction))
    # Tableau final, colonnes dans l'ordre défini.
    df = pd.DataFrame(LIGNES).reindex(columns=COLONNES)
    # Écriture du CSV.
    df.to_csv(a.sortie, index=False, encoding="utf-8")
    # Résumé : nombre de lignes par matériau et par grandeur.
    print(df.groupby(["materiau", "grandeur"]).size().unstack(fill_value=0).to_string())
    # Nombre total de lignes.
    print(f"\n{len(df)} lignes écrites dans {a.sortie}")
    # Fin sans erreur.
    return 0


# Exécuté seulement quand le fichier est lancé directement.
if __name__ == "__main__":
    # Lance main() et transmet son code de sortie.
    sys.exit(main())
