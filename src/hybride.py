# Ligne « shebang » : permet de lancer le fichier directement comme un programme sous Linux/macOS (ignorée sous
# Windows).
#!/usr/bin/env python3
# Déclare l'encodage UTF-8 du fichier (accents, symboles ∞, ², Δ dans les textes).
# -*- coding: utf-8 -*-
# Docstring du module : description affichée par --help et lisible par help(hybride). Elle résume les 4 niveaux de
# modèle.
"""
Modèle HYBRIDE de vieillissement humide, pour TOUS les matériaux des sources fournies.

Chaque matériau reçoit le niveau de modèle que ses données permettent (aucune loi inventée) :

  chaine     teneur en eau puis module :  M∞ = isotherme(T, a_w) ; D(T, a_w) ; profil de Fick dans la
             plaque (Crank) ; E local = E(M) mesuré ; moyenne dans l'épaisseur ; + processus gaussien (GP)
             sur les écarts aux mesures E(t) quand il y en a.
  apparent   E/E0 = 1 - Δ∞(a_w) · g_b(F(D t / h²))   avec F = prise en eau moyenne de Fick (Crank) et
             g_b(F) = (1 - exp(-b F)) / (1 - exp(-b)) (g = F si b = 0 : module proportionnel à l'eau absorbée ;
             b > 0 : le module chute plus vite que l'eau n'entre, pente initiale finie) ;
             D publié (gravimétrie) ou ajusté sur E(t) ; Δ∞ (et b) ajustés sur E(t).
  cinetique  E/E0 = r∞ + (1 - r∞) exp(-k(T) t)   (dégradation lente, non limitée par la diffusion),
             k Arrhenius si plusieurs températures.
  donnees    aucune cinétique identifiable (pas de E0 ou une seule durée) : données affichées, pas de courbe.

Incertitude : 200 tirages des paramètres (covariances des ajustements) x écart de modèle (GP ou résidu).
Les cas hors du domaine identifiable sont REFUSÉS (exception NonPredictible) avec la raison.

Usage :
    python src/hybride.py --rapport
    python src/hybride.py --materiau LIN_EPOXY --T 23 --milieu immersion_eau_douce --epaisseur 3 --temps 1000
"""
# Permet d'écrire les annotations de type modernes (ex. « object | None ») même sur des versions de Python plus
# anciennes.
from __future__ import annotations

# argparse : lecture des options de la ligne de commande (--materiau, --T, --rapport...).
import argparse
# sys : accès à sys.exit pour renvoyer un code de sortie au terminal.
import sys
# warnings : permet de faire taire les avertissements numériques de scipy pendant les ajustements.
import warnings
# dataclass : crée une classe « conteneur » (Modele) sans écrire de constructeur ; field fixe les valeurs par défaut
# mutables.
from dataclasses import dataclass, field
# lru_cache : mémorise le résultat d'une fonction ; chaque matériau n'est ajusté qu'une seule fois.
from functools import lru_cache
# Path : manipulation des chemins de fichiers indépendante du système (Windows / Linux).
from pathlib import Path

# numpy : calcul numérique sur tableaux (séries de Fick, tirages aléatoires, statistiques).
import numpy as np
# pandas : lecture et filtrage du fichier de données physiques (tableau CSV).
import pandas as pd
# PchipInterpolator : interpolation par morceaux qui respecte la monotonie des points (pas d'oscillation entre deux
# mesures).
from scipy.interpolate import PchipInterpolator
# curve_fit : ajustement de paramètres par moindres carrés non linéaires ; renvoie aussi leur matrice de covariance.
from scipy.optimize import curve_fit
# erfc : fonction d'erreur complémentaire, solution exacte de Fick aux temps courts (milieu semi-infini).
from scipy.special import erfc
# Processus gaussien de scikit-learn : sert à corriger les écarts physique/mesures et à estimer l'incertitude.
from sklearn.gaussian_process import GaussianProcessRegressor
# Noyaux du processus gaussien : RBF (similarité qui décroît avec la distance), constante (amplitude), blanc (bruit de
# mesure).
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
# Régression isotone : force une suite de points à être monotone (ex. le module ne peut que baisser quand l'eau
# augmente).
from sklearn.isotonic import IsotonicRegression

# Racine du dépôt : dossier parent de src/ (resolve() donne le chemin absolu, parents[1] remonte de deux niveaux).
RACINE = Path(__file__).resolve().parents[1]
# Chemin du fichier de données physiques produit par data/digitaliser_physique.py.
DONNEES = RACINE / "data" / "donnees_physiques.csv"
# Constante des gaz parfaits en J/(mol·K), utilisée dans la loi d'Arrhenius.
R_GAZ = 8.314462618
# Décalage pour convertir les °C en kelvins (Arrhenius exige des températures absolues).
KELVIN = 273.15
# Constante de Crank : quand D·t/h² = 0,2818, la plaque a absorbé 95 % de son eau d'équilibre. Sert à calculer le
# temps de saturation.
TAU_95 = 0.2818               # D t / h² à 95 % de la prise en eau moyenne (plaque, Crank)
# Incertitude de mesure SUPPOSÉE (3 %) : hypothèse explicite, utilisée pour les points E(M) et le bruit du processus
# gaussien.
BRUIT = 0.03                  # écart-type de mesure SUPPOSÉ (3 %) des points E(M) et du GP : hypothèse
# Nombre de jeux de paramètres tirés au hasard pour propager l'incertitude (méthode de Monte-Carlo).
N_TIRAGES = 200
# Activité de l'eau a_w de chaque milieu : None = elle vaut HR/100 (vapeur) ; 1 pour l'eau douce ; 0,98 pour l'eau de
# mer.
MILIEUX = {"vapeur": None, "immersion_eau_douce": 1.0, "immersion_eau_de_mer": 0.98}
# a_w eau de mer = 0.98 : valeur donnée par Le Gac et al. 2017 (§Modeling). Eau douce : 1.

# Libellés lisibles des 4 niveaux de modèle, affichés dans l'application et la ligne de commande.
TYPES = {"chaine": "chaîne physique (sorption + Fick + E(M))",
         "apparent": "cinétique apparente (Fick, amplitude ajustée sur E(t))",
         "cinetique": "cinétique de dégradation (premier ordre)",
         "donnees": "données seulement (pas de cinétique identifiable)"}


# Exception propre au projet : levée quand un cas sort de ce que les données permettent de prédire. Hérite de
# ValueError.
class NonPredictible(ValueError):
    """Cas hors du domaine identifiable avec les données : la prédiction est refusée."""


# ---------- FICHES MATÉRIAUX : une entrée par matériau. Clés communes : nom, famille, type (niveau de modèle),
# source, verification (controle_pdf = relu sur le PDF, non_controle = extraction tierce), propriete
# (traction/flexion), E0_MPa (None = lu dans les données), Tg0_C, h_ref_mm (épaisseur des éprouvettes), T_essais,
# milieux autorisés, duree_module_max_h (dernière mesure de module), notes (affichées comme points de vigilance).
# =========================================================================== fiches matériaux
# Uniquement des valeurs publiées (source citée) ou des choix explicitement notés comme tels.
# Texte réutilisable pour les sources qui n'indiquent pas la température (« ambiante »).
_T_INCONNUE = "température non publiée (« ambiante ») : la courbe vaut pour la température du laboratoire"
# Dictionnaire principal : clé courte du matériau -> fiche descriptive.
MATERIAUX = {
    # PA6 de Le Gac 2017 : le mieux documenté (absorption, isotherme, E(M), Tg(M), E(t) de validation). Modèle «
    # chaine ».
    "PA6_GOODFELLOW": dict(
        nom="PA6 Goodfellow AM303020 (plaque compressée)", famille="polyamide", type="chaine",
        source="Le Gac et al. 2017, Polym. Degrad. Stab. 137:272-280, doi 10.1016/j.polymdegradstab.2017.02.003",
        verification="controle_pdf", propriete="traction",
        E0_MPa=2300, E0_source="Tab. 1 (module sec)", Tg0_C=66, Tg0_source="Tab. 1 (DMA)", h_ref_mm=2.0,
        # T_essais = 25 °C : température de la seule cinétique d'absorption mesurée.
        T_essais=25.0, T_essai_module="ambiante, 21 °C supposé", milieux={"vapeur", "immersion_eau_douce",
                                                                          "immersion_eau_de_mer"},
        # duree_sans_hydrolyse_h = 4380 h (6 mois) : durée jusqu'à laquelle les auteurs montrent l'absence d'hydrolyse
        # ; regime_caoutchoutique_mesure : les mesures couvrent l'état au-dessus de la Tg humide.
        duree_module_max_h=720, duree_sans_hydrolyse_h=4380, regime_caoutchoutique_mesure=True,
        # validation_independante = True : les mesures E(t) de ce matériau n'ont servi à aucun ajustement (vraie
        # validation).
        validation_independante=True,
        notes=["Diffusion NON fickienne (Le Gac Fig. 1) : Fick est une approximation ; la chute réelle du module "
               "est plus lente que Fick aux temps courts (validation Fig. 7a).",
               "Désorption beaucoup plus lente que l'absorption (Le Gac Fig. 9) : récupération au séchage non symétrique.",
               "Plastification réversible : après séchage le module revient près de sa valeur initiale (Le Gac Fig. 9)."]),
    # Époxy de Gibhardt 2022 : D à trois températures (donc Arrhenius), isotherme, E(M) partielle. Modèle « chaine ».
    "EPOXY_HEXION_RIMR135": dict(
        nom="Époxy Hexion EPIKOTE RIMR 135 / EPIKURE RIMH 137", famille="epoxy", type="chaine",
        source="Gibhardt et al. 2022, Front. Mater. 9:826076, doi 10.3389/fmats.2022.826076",
        verification="controle_pdf", propriete="traction",
        E0_MPa=2750, E0_source="§3.1 (référence sèche)", Tg0_C=89.4, Tg0_source="§3.1 (Tg onset E', DMTA)",
        # T_essais = None mais les températures de vieillissement sont connues ; c'est la température d'essai
        # mécanique qui ne l'est pas.
        h_ref_mm=0.5, T_essais=None, T_essai_module="ambiante (non chiffrée, §3.5)",
        milieux={"vapeur", "immersion_eau_douce", "immersion_eau_de_mer"},
        duree_module_max_h=2600, duree_sans_hydrolyse_h=None, regime_caoutchoutique_mesure=False,
        validation_independante=False,
        notes=["Le module dépend de l'HISTOIRE de vieillissement, pas seulement de la teneur en eau (Gibhardt §3.9).",
               "Au-delà d'environ 2600 h, le vieillissement physique fait remonter une partie des propriétés : "
               "non modélisé.",
               "Aucun module mesuré en atmosphère humide : la relation E(M) vient des immersions."]),
    # PA6 de Zarbali 2026 : isotherme, cinétiques d'absorption, E(M), Tg(M), mais aucune mesure E(t) et température
    # non publiée.
    "PA6_AKULON_F223D": dict(
        # T_inconnue=True : déclenche l'alerte critique « température non publiée » dans les points de vigilance.
        nom="PA6 Akulon F223-D (injecté)", famille="polyamide", T_inconnue=True, type="chaine",
        source="Zarbali et al. 2026, ACS Omega 11:31292-31299, doi 10.1021/acsomega.6c01154",
        verification="controle_pdf", propriete="traction",
        # E0_MPa=None : E0 sera pris sur la figure 3 (module à teneur en eau nulle) lors de l'ajustement.
        E0_MPa=None, E0_source="Fig. 3 à teneur en eau nulle (conditionné à 4 %HR)", Tg0_C=63,
        Tg0_source="Fig. 2 (DMA, pic de tan delta)", h_ref_mm=4.0, T_essais=None,
        T_essai_module="non publiée", milieux={"vapeur", "immersion_eau_douce", "immersion_eau_de_mer"},
        duree_module_max_h=650, duree_sans_hydrolyse_h=None, regime_caoutchoutique_mesure=True,
        validation_independante=False,
        notes=["Température de conditionnement NON publiée (dessiccateurs) : aucun effet de la température.",
               "Absorption non fickienne aux fortes humidités (auteurs) : D apparent ajusté par humidité.",
               "Aucune mesure E(t) : la courbe vient entièrement de la chaîne physique, sans correction GP ni "
               "validation sur une cinétique de module.",
               "Tg humide sous l'ambiante dès environ 1 % d'eau (Fig. 2) : phase amorphe caoutchoutique."]),
    # Composite lin/époxy (Chilali 2017) : D et M∞ publiés, module mesuré dans le temps. Modèle « apparent ».
    "LIN_EPOXY": dict(
        nom="Composite lin / époxy (stratifié 0°, Vf 32 %)", famille="composite_lin", T_inconnue=True, type="apparent",
        source="Chilali 2017, thèse, Université de Reims Champagne-Ardenne (Tab. 4.1, 4.2, Fig. 4.4a)",
        verification="controle_pdf", propriete="traction (module longitudinal)",
        E0_MPa=None, E0_source="Tab. 4.2 (non vieilli)", Tg0_C=None, Tg0_source="non publiée", h_ref_mm=3.0,
        T_essais=None, T_essai_module="ambiante", milieux={"immersion_eau_douce"},
        # D_mode='publie' : la vitesse vient du D gravimétrique publié, pas d'un ajustement sur le module.
        duree_module_max_h=720, duree_sans_hydrolyse_h=None, D_mode="publie", p_libre=True,
        notes=["Immersion en eau du robinet uniquement : l'humidité relative n'est pas prédictible.",
               "Diffusion 3D (bords et plan) dans les éprouvettes : le modèle 1D utilise Dzz (épaisseur) du Tab. 4.1.",
               "Aucune mesure avant 2 jours : la forme de la courbe aux temps courts n'est pas contrainte.",
               "Après séchage, le module revient à 0.83 E0 (Tab. 4.2) : environ 17 % de perte irréversible "
               "(endommagement des fibres et des interfaces).",
               "L'eau salée réduit la vitesse (5 à 10 %) et la saturation (20 %) d'absorption (p. 92) : non modélisé."]),
    # Composite lin/Elium (Chilali 2017) : même traitement que lin/époxy.
    "LIN_ELIUM": dict(
        nom="Composite lin / Elium 150 acrylique (stratifié 0°, Vf 37 %)", famille="composite_lin", T_inconnue=True, type="apparent",
        source="Chilali 2017, thèse, Université de Reims Champagne-Ardenne (Tab. 4.1, 4.2, Fig. 4.4a)",
        verification="controle_pdf", propriete="traction (module longitudinal)",
        E0_MPa=None, E0_source="Tab. 4.2 (non vieilli)", Tg0_C=None, Tg0_source="non publiée", h_ref_mm=3.0,
        T_essais=None, T_essai_module="ambiante", milieux={"immersion_eau_douce"},
        duree_module_max_h=720, duree_sans_hydrolyse_h=None, D_mode="publie", p_libre=True,
        notes=["Immersion en eau du robinet uniquement : l'humidité relative n'est pas prédictible.",
               "Points à 2, 7 et 15 j lus sur des marqueurs en partie masqués (incertitude ±0.15 GPa).",
               "Aucune mesure avant 2 jours : la forme de la courbe aux temps courts n'est pas contrainte.",
               "Après séchage, le module revient à 0.85 E0 (Tab. 4.2) : environ 15 % de perte irréversible.",
               "Diffusion 3D (bords et plan) dans les éprouvettes : le modèle 1D utilise Dzz du Tab. 4.1."]),
    # PA6 Takmid (Guerira 2016) : module mesuré dans le temps, D publié inutilisable (géométrie inconnue). Modèle «
    # apparent ».
    "PA6_TAKMID": dict(
        nom="PA6 Takmid (plaque 7 mm)", famille="polyamide", T_inconnue=True, type="apparent",
        source="Guerira et al. 2016, actes ICEMAEP2016 Constantine, pp. 1021-1029 (sans DOI)",
        verification="controle_pdf", propriete="traction",
        E0_MPa=None, E0_source="Tab. 3 (mesuré, 0 j)", Tg0_C=None, Tg0_source="non publiée", h_ref_mm=7.0,
        T_essais=None, T_essai_module="ambiante", milieux={"immersion_eau_douce"},
        # D_mode='ajuste' : D est ajusté directement sur les mesures de module.
        duree_module_max_h=672, duree_sans_hydrolyse_h=None, D_mode="ajuste", p_libre=False,
        notes=["E0 mesuré (1268 MPa) très inférieur à la valeur fournisseur du même article (3.3 GPa) : mesure "
               "de déformation probablement par la traverse. Seul E/E0 est fiable, pas E en MPa.",
               "Module non monotone dans le temps (405 MPa à 12 j, 674 à 16 j) : forte dispersion.",
               "D apparent ajusté sur E(t) très supérieur au D gravimétrique publié (0.26e-12 m²/s) : "
               "l'épaisseur réelle des éprouvettes de traction est incertaine, l'effet d'épaisseur n'est pas fiable.",
               "Immersion en eau distillée uniquement : l'humidité relative n'est pas prédictible."]),
    # Époxy de Wang 2016 (extraction tierce) : module de FLEXION à 60 °C dans trois milieux. Modèle « apparent ».
    "EPOXY_WANG_618": dict(
        nom="Époxy 618/F-44/AFG-90/CTBN durci DDS (flexion)", famille="epoxy", type="apparent",
        source="Wang et al. 2016, Compos. B, doi 10.1016/j.compositesb.2016.09.067 (PDF non fourni)",
        verification="non_controle", propriete="FLEXION 3 points",
        E0_MPa=None, E0_source="Tab. 3 (extraction tierce)", Tg0_C=None, Tg0_source="non extraite", h_ref_mm=3.0,
        T_essais=60.0, T_essai_module="non extraite", milieux={"vapeur", "immersion_eau_douce", "immersion_eau_de_mer"},
        duree_module_max_h=672, duree_sans_hydrolyse_h=None, D_mode="ajuste", p_libre=False,
        # T_tolerance=5 : seules les températures à ±5 °C de 60 °C sont acceptées ; aw_min=0,6 : humidités plus basses
        # refusées.
        T_tolerance=5.0, aw_min=0.6,
        notes=["Valeurs NON recontrôlées sur le PDF (extraction tierce).",
               "Module de FLEXION : ne pas comparer directement aux modules de traction.",
               "Une seule température (60 °C) : aucun effet de la température identifiable.",
               "Hausse transitoire du module vers 24 h dans les données (post-réticulation ?) : non modélisée."]),
    # PLA de Niaounakis 2011 (extraction tierce) : trois températures à 80 %HR. Modèle « cinetique » avec Arrhenius.
    "PLA_2002D": dict(
        nom="PLA NatureWorks 2002D (4.25 % D, moulé)", famille="pla", type="cinetique",
        source="Niaounakis et al. 2011, J. Appl. Polym. Sci., doi 10.1002/app.32644 (PDF non fourni)",
        verification="non_controle", propriete="traction",
        E0_MPa=None, E0_source="Tab. III (extraction tierce)", Tg0_C=None, Tg0_source="non extraite", h_ref_mm=2.0,
        T_essais=None, T_essai_module="non extraite", milieux={"vapeur"},
        # aw_cible=0,80 : une seule humidité mesurée ; T_domaine = plage mesurée ; T_refus = au-delà, prédiction
        # refusée (Tg du PLA).
        duree_module_max_h=3120, duree_sans_hydrolyse_h=None, aw_cible=0.80, T_domaine=(20.0, 50.0),
        T_refus=(10.0, 55.0),
        notes=["Valeurs NON recontrôlées sur le PDF (extraction tierce) ; incohérence signalée entre le texte "
               "(8 % de perte à 40 °C / 30 j) et le tableau (19 %).",
               "Une seule humidité (80 %) : l'effet de l'humidité n'est pas identifiable.",
               "Perte de module lente (mois) : hydrolyse et vieillissement physique, non limités par la diffusion "
               "dans 2 mm ; l'épaisseur n'est pas un paramètre du modèle.",
               "Au-dessus de la Tg du PLA (environ 55-60 °C), le mécanisme change : refusé."]),
    # PLA imprimé de Bergaliyeva 2022 (extraction tierce) : une seule condition. Modèle « cinetique » sans effet de T.
    "PLA_BQ_FFF": dict(
        nom="PLA filament BQ (impression FFF, 100 %)", famille="pla", type="cinetique",
        source="Bergaliyeva et al. 2022, Polymers 14:5256, doi 10.3390/polym14235256 (PDF non fourni)",
        verification="non_controle", propriete="traction",
        E0_MPa=None, E0_source="Tab. 3 (extraction tierce)", Tg0_C=None, Tg0_source="non extraite", h_ref_mm=None,
        T_essais=50.0, T_essai_module="23 °C après reconditionnement", milieux={"vapeur"},
        duree_module_max_h=1344, duree_sans_hydrolyse_h=None, aw_cible=0.70, T_tolerance=5.0,
        notes=["Valeurs NON recontrôlées sur le PDF (extraction tierce).",
               "Épaisseur non publiée : aucun effet d'épaisseur.",
               "Une seule condition (50 °C / 70 %HR) : ni la température ni l'humidité ne sont identifiables.",
               "Données très dispersées (E/E0 de 0.74 à 1.09) : le modèle ne capte qu'une tendance.",
               "Éprouvettes reconditionnées 88 h à 23 °C / 50 %HR avant essai : plastification en partie effacée."]),
    # PLA L130 (Goetjes 2024) : pas d'E0 ni de cinétique. Type « donnees » : affiché, jamais prédit.
    "PLA_L130": dict(
        nom="PLA Luminy L130 (injecté)", famille="pla", type="donnees",
        source="Goetjes et al. 2024, Materials 17:3683, doi 10.3390/ma17153683 (PDF non fourni)",
        verification="non_controle", propriete="traction", E0_MPa=None, h_ref_mm=None, milieux=set(),
        # raison : texte affiché à l'utilisateur pour expliquer l'absence de prédiction.
        raison="E0 non publié et une seule durée (168 h) : aucune évolution dans le temps calculable."),
    # PLA L130 + amidon (Goetjes 2024) : même situation.
    "PLA_L130_AMIDON": dict(
        nom="PLA Luminy L130 + 50 % amidon de pomme de terre", famille="pla_composite", type="donnees",
        source="Goetjes et al. 2024, Materials 17:3683, doi 10.3390/ma17153683 (PDF non fourni)",
        verification="non_controle", propriete="traction", E0_MPa=None, h_ref_mm=None, milieux=set(),
        raison="E0 non publié et une seule durée (168 h) : aucune évolution dans le temps calculable."),
}
# Sources fournies mais inexploitables, listées dans l'application avec la raison.
NON_EXPLOITABLES = {
    "PA6 Alyamid C2122 (Ksouri et Haddar 2018, doi 10.1007/s10965-018-1551-1)":
        "aucune valeur chiffrée dans l'extraction (Fig. 10 à digitaliser) et PDF non fourni.",
    "Nafion 117, PMMA, DGEBA-DDM / -TETA / -PACM / -IPDA / -POPA, HDDGE-IPDA (fichier « dataset_reel_..._complet »)":
        "valeurs et conditions introuvables dans les sources citées (DOI faux ou sans rapport) : exclues.",
}


# ---------- LOI DE FICK (diffusion de l'eau dans une plaque exposée sur ses deux faces)
# =========================================================================== Fick
# Prise en eau moyenne relative F = M(t)/M∞ en fonction du temps réduit tau = D·t/h² (D en m²/s, t en s, h en m).
def fraction_moyenne(tau, termes: int = 200) -> np.ndarray:
    """M(t)/M∞ pour une plaque (Crank éq. 4.18), tau = D t / h²."""
    # Indices k = 0, 1, 2... des termes de la série de Crank.
    k = np.arange(termes)
    # Coefficients 8/((2k+1)²π²) de la série (leur somme vaut 1).
    c = 8.0 / ((2 * k + 1) ** 2 * np.pi ** 2)
    # Convertit l'entrée (nombre ou liste) en tableau numpy à une dimension.
    tau = np.atleast_1d(np.asarray(tau, float))
    # Série de Crank (éq. 4.18) : F = 1 − Σ c_k·exp(−(2k+1)²π²·tau). tau[:, None] calcule tous les temps d'un coup.
    F = 1.0 - (c * np.exp(-((2 * k + 1) ** 2) * np.pi ** 2 * tau[:, None])).sum(1)
    # Repère les temps courts, où la série tronquée devient imprécise.
    court = tau < 0.01            # série tronquée imprécise aux temps courts : Crank éq. 4.20, F = 4 racine(tau/pi)
    # Aux temps courts, solution exacte du milieu semi-infini (Crank éq. 4.20) : F = 4·racine(tau/π). Vaut 0 à t = 0.
    F[court] = 4 * np.sqrt(np.clip(tau[court], 0, None) / np.pi)
    # Borne F entre 0 et 1 (protection contre les arrondis).
    return np.clip(F, 0.0, 1.0)


# Nombre de points dans l'épaisseur pour le profil d'eau.
_NX = 161
# Positions x/h de -0,5 à +0,5 ; la fonction sinus resserre les points près des faces, où le front d'eau est très
# raide aux temps courts.
_X = 0.5 * np.sin(np.pi / 2 * np.linspace(-1, 1, _NX))      # x/h, resserré près des faces
# Indices des 300 termes de la série du profil.
_K = np.arange(300)
# Cosinus précalculés de la série du profil pour chaque position (gain de temps : calculés une seule fois au
# chargement).
_COS = np.cos(np.outer(_X, (2 * _K + 1) * np.pi))
# Coefficients de la série du profil c(x,t) (Crank éq. 4.17).
_COEF = (4 / np.pi) * ((-1.0) ** _K) / (2 * _K + 1)
# Seuil de temps réduit sous lequel on utilise la solution erfc plutôt que la série.
TAU_COURT = 0.01


# Profil de teneur en eau dans l'épaisseur, c(x,t)/c∞, pour chaque temps (lignes) et chaque position (colonnes).
def profil(tau) -> np.ndarray:
    """c(x, t)/c∞ dans l'épaisseur (nt, nx) : série de Crank, erfc des deux faces aux temps courts."""
    # Tableau des temps réduits.
    tau = np.atleast_1d(np.asarray(tau, float))
    # Partie temporelle de la série : exp(−(2k+1)²π²·tau) multipliée par les coefficients ; produit externe temps x
    # termes.
    e = np.exp(-np.outer(tau, (2 * _K + 1) ** 2 * np.pi ** 2)) * _COEF
    # Somme de la série par produit matriciel (temps x termes) @ (termes x positions) : donne la concentration en
    # chaque point.
    c = 1.0 - e @ _COS.T
    # Repère les temps courts strictement positifs.
    court = (tau > 0) & (tau < TAU_COURT)
    # Seulement s'il y en a.
    if court.any():
        # Longueur de pénétration 2·racine(tau) (en unités de h) pour chaque temps court.
        r = 2 * np.sqrt(tau[court])[:, None]
        # Solution des deux faces indépendantes : erfc(distance à la face / longueur de pénétration), somme des deux
        # faces.
        c[court] = erfc((0.5 - _X) / r) + erfc((0.5 + _X) / r)
    # À t = 0 la plaque est sèche partout.
    c[tau <= 0] = 0.0
    # Borne entre 0 (sec) et 1 (saturé).
    return np.clip(c, 0.0, 1.0)


# ---------- STRUCTURE D'UN MODÈLE AJUSTÉ
# =========================================================================== structure
# Décorateur qui génère automatiquement le constructeur de la classe à partir des champs ci-dessous.
@dataclass
# Contient tout ce qui a été ajusté pour UN matériau.
class Modele:
    # Clé du matériau (ex. 'PA6_GOODFELLOW').
    cle: str
    # Fiche du matériau (copie de référence de MATERIAUX[cle]).
    meta: dict
    # Paramètres ajustés, pour affichage et rapport : nom -> (valeur, écart-type, unité, provenance).
    params: dict = field(default_factory=dict)      # affichage : nom -> (valeur, écart-type, unité, provenance)
    # Domaines de validité des données (plages de T, de a_w, de teneur en eau...), utilisés par les points de
    # vigilance.
    domaine: dict = field(default_factory=dict)
    # Liste de N_TIRAGES jeux de lois (un par tirage aléatoire des paramètres) : sert à l'intervalle de confiance.
    P: list = field(default_factory=list)           # un dict de lois par tirage
    # Jeu de lois central (meilleur ajustement) : sert à la courbe médiane.
    Pc: dict = field(default_factory=dict)          # lois centrales (meilleur ajustement)
    # Processus gaussien de correction (None s'il n'y a pas assez de mesures E(t)).
    gp: object | None = None
    # Informations sur le processus gaussien (nombre de mesures, amplitude...).
    gp_info: dict = field(default_factory=dict)
    # Écart de modèle relatif utilisé quand il n'y a pas de processus gaussien.
    sd_modele: float = BRUIT                        # écart de modèle relatif si pas de GP
    # Mesures E(t) utilisées pour l'ajustement (types apparent et cinetique).
    mesures: pd.DataFrame | None = None


# Charge les lignes de donnees_physiques.csv concernant un matériau.
def _donnees(cle: str) -> pd.DataFrame:
    # Lecture complète du CSV.
    d = pd.read_csv(DONNEES)
    # Filtre sur le matériau ; .copy() évite de modifier le tableau d'origine.
    return d[d["materiau"] == cle].copy()


# Tire n jeux de paramètres selon une loi normale multivariée (moyenne p, covariance cov) : propagation d'incertitude.
def _tirer(rng, p, cov, n, bas=None, haut=None):
    # Moyennes sous forme de tableau 1D.
    p = np.atleast_1d(np.asarray(p, float))
    # Covariance en 2D ; les valeurs non finies (paramètre non identifiable) sont remplacées par 0 (pas de
    # dispersion).
    cov = np.atleast_2d(np.where(np.isfinite(cov), cov, 0.0))
    # Tirage aléatoire ; check_valid='ignore' tolère les covariances presque singulières.
    s = rng.multivariate_normal(p, cov, size=n, check_valid="ignore")
    # Si des bornes physiques sont données...
    if bas is not None:
        # ... les tirages sont ramenés dans ces bornes (ex. perte entre 0 et 1).
        s = np.clip(s, bas, haut)
    # Renvoie un tableau (n tirages x nombre de paramètres).
    return s


# Interpolation monotone : rend d'abord les points monotones (régression isotone), puis interpole sans oscillation
# (PCHIP).
def _pchip_monotone(xs, ys, croissant):
    # Régression isotone croissante ou décroissante selon le besoin.
    iso = IsotonicRegression(increasing=croissant)
    # Points corrigés pour être monotones (ne change rien s'ils l'étaient déjà).
    yy = iso.fit_transform(xs, ys)
    # Interpolateur PCHIP construit sur les points corrigés.
    f = PchipInterpolator(xs, yy)
    # Renvoie une fonction qui bloque l'abscisse dans la plage mesurée (pas d'extrapolation), et les points corrigés.
    return (lambda v, f=f, a=xs[0], b=xs[-1]: f(np.clip(v, a, b))), yy


# ---------- AJUSTEMENT DU NIVEAU « CHAÎNE » : diffusion D, isotherme M∞, relation module/eau E(M), Tg(M)
# =========================================================================== ajustements « chaîne »
# Construit les lois d'un matériau de type « chaine » à partir de ses données d et d'un générateur aléatoire rng.
def _chaine(m: Modele, d: pd.DataFrame, rng) -> None:
    # Raccourcis : clé du matériau et nombre de tirages.
    cle, n = m.cle, N_TIRAGES
    # Petite fonction qui sélectionne les lignes d'une grandeur (ex. G('isotherme')).
    G = lambda g: d[d["grandeur"] == g]
    # Listes qui recevront, pour chaque tirage, la loi D, la loi M∞, la loi E(M) et la loi Tg.
    Dt, Mt, Gt, Tt = [None] * n, [None] * n, [None] * n, [None] * n      # lois par tirage

    # ----- 1. DIFFUSION
    # ---------------- diffusion
    # Le Gac : une seule cinétique d'absorption (25 °C, eau de mer, 2 mm).
    if cle == "PA6_GOODFELLOW":                     # une cinétique, 25 °C eau de mer, h = 2 mm
        # Points de prise en eau en fonction du temps.
        a = G("absorption")
        # Temps convertis en secondes ; teneurs en eau en %.
        t_s, M = a["x"].to_numpy() * 3600, a["y"].to_numpy()
        # Épaisseur de l'éprouvette d'absorption : 2 mm = 0,002 m.
        h = 2e-3
        # Modèle ajusté : M(t) = M∞·F(D·t/h²). D est ajusté via ln D (toujours positif, mieux conditionné).
        f = lambda t, lnD, Minf: Minf * fraction_moyenne(np.exp(lnD) * t / h ** 2)
        # Ajustement par moindres carrés ; p = [ln D, M∞], cov = matrice de covariance des deux paramètres.
        p, cov = curve_fit(f, t_s, M, p0=[np.log(1e-12), 10.0])
        # Enregistre D et son écart-type (propagé de ln D à D) pour l'affichage.
        m.params["D(25 °C)"] = (np.exp(p[0]), np.exp(p[0]) * np.sqrt(cov[0, 0]), "m²/s",
                                "Fick ajusté sur Le Gac Fig. 1 (25 °C, eau de mer, 2 mm)")
        # Enregistre M∞ ajusté à 25 °C en eau de mer.
        m.params["M∞(25 °C, eau de mer)"] = (p[1], np.sqrt(cov[1, 1]), "% masse", "idem")
        # Q (énergie d'activation) n'est pas calculable avec une seule température : affiché comme non identifiable.
        m.params["Q diffusion"] = (np.nan, np.nan, "kJ/mol", "NON IDENTIFIABLE : une seule température mesurée")
        # Tire n valeurs de ln D selon son incertitude.
        s = _tirer(rng, p[:1], cov[:1, :1], n)[:, 0]
        # Une loi D par tirage ; v=v fige la valeur dans la fonction (sinon toutes les fonctions partageraient la
        # dernière valeur).
        Dt = [lambda T, aw, v=v: np.exp(v) for v in s]
        # Loi D centrale (meilleur ajustement). D ne dépend ni de T ni de a_w ici (une seule mesure).
        Dc = lambda T, aw, v=p[0]: np.exp(v)
        # Domaine de validité de la diffusion : 25 °C, a_w 0,98 uniquement.
        m.domaine.update(T_diffusion=(25.0, 25.0), aw_diffusion=(0.98, 0.98))
        # M∞ à 25 °C conservé pour construire l'isotherme plus bas.
        msat25 = p[1]
    # Gibhardt : D publiés à 8, 40 et 50 °C, donc loi d'Arrhenius ajustable.
    elif cle == "EPOXY_HEXION_RIMR135":             # D publiés à 8, 40, 50 °C -> Arrhenius
        # Lignes de diffusivité.
        g = G("diffusivite")
        # Température de référence de la loi d'Arrhenius (milieu de la plage mesurée).
        Tref = 40.0
        # Abscisse d'Arrhenius : 1/T − 1/Tref (T en kelvins).
        xr = 1 / (g["x"].to_numpy() + KELVIN) - 1 / (Tref + KELVIN)
        # Ordonnée : ln D.
        y = np.log(g["y"].to_numpy())
        # Régression linéaire ln D = c1 + c0·(1/T − 1/Tref) ; cov='unscaled' renvoie la covariance brute.
        coef, cov = np.polyfit(xr, y, 1, cov="unscaled")
        # Mise à l'échelle de la covariance par la variance des résidus (incertitude réelle des coefficients).
        cov = cov * np.sum((y - np.polyval(coef, xr)) ** 2) / max(len(y) - 2, 1)
        # D à 40 °C et son écart-type.
        m.params["D(40 °C)"] = (np.exp(coef[1]), np.exp(coef[1]) * np.sqrt(cov[1, 1]), "m²/s",
                                "Arrhenius sur les D publiés (Gibhardt Fig. 5)")
        # Énergie d'activation Q = −pente × R (en kJ/mol) et son écart-type.
        m.params["Q diffusion"] = (-coef[0] * R_GAZ / 1000, np.sqrt(cov[0, 0]) * R_GAZ / 1000, "kJ/mol",
                                   "ajusté ; les auteurs donnent 47.2 kJ/mol")
        # Loi d'Arrhenius : D(T) = exp(c1 + c0·(1/T − 1/Tref)).
        arr = lambda T, c0, c1: np.exp(c1 + c0 * (1 / (T + KELVIN) - 1 / (Tref + KELVIN)))
        # Une loi D(T) par tirage des deux coefficients.
        Dt = [lambda T, aw, s=s: arr(T, *s) for s in _tirer(rng, coef, cov, n)]
        # Loi D(T) centrale.
        Dc = lambda T, aw: arr(T, *coef)
        # Domaine de validité : températures mesurées (8 à 50 °C), immersion.
        m.domaine.update(T_diffusion=(float(g["x"].min()), float(g["x"].max())), aw_diffusion=(1.0, 1.0))
    # Zarbali : une cinétique par humidité, température inconnue -> D dépend de a_w.
    else:                                           # Zarbali : D apparent par humidité (T inconnue)
        # M∞ publiés (Tab. 3) indexés par activité de l'eau.
        iso = G("isotherme").set_index("a_w")["y"]
        # Points de prise en eau (Fig. 1).
        a = G("absorption")
        # Listes : activités de l'eau, ln D ajustés, écarts-types.
        aws, lnDs, ses = [], [], []
        # Épaisseur des éprouvettes : 4 mm.
        h = 4e-3
        # Boucle sur chaque humidité (groupby regroupe les points de même a_w).
        for aw, g in a.groupby("a_w"):
            # M∞ publié pour l'humidité la plus proche ; il est fixé, seul D est ajusté.
            Minf = float(iso.loc[iso.index.to_series().sub(aw).abs().idxmin()])
            # Modèle de Fick à un seul paramètre (ln D).
            f = lambda t, lnD, Minf=Minf: Minf * fraction_moyenne(np.exp(lnD) * t / h ** 2)
            # Ajustement de ln D sur la cinétique de cette humidité.
            p, cov = curve_fit(f, g["x"].to_numpy() * 3600, g["y"].to_numpy(), p0=[np.log(1e-12)])
            # Stocke a_w, ln D et son écart-type.
            aws.append(aw); lnDs.append(p[0]); ses.append(np.sqrt(cov[0, 0]))
            # Enregistre D apparent de cette humidité pour l'affichage.
            m.params[f"D apparent ({aw * 100:.0f} %HR)"] = (np.exp(p[0]), np.exp(p[0]) * ses[-1], "m²/s",
                                                           "Fick (M∞ du Tab. 3) ajusté sur Zarbali Fig. 1, 4 mm")
        # Conversion des listes en tableaux numpy.
        aws, lnDs, ses = map(np.array, (aws, lnDs, ses))
        # Régression pondérée ln D = a + b·a_w (poids = 1/écart-type, plancher 0,05 pour ne pas surpondérer un point).
        coef, cov = np.polyfit(aws, lnDs, 1, w=1 / np.maximum(ses, 0.05), cov="unscaled")
        # Résidus de la régression.
        res = lnDs - np.polyval(coef, aws)
        # Mise à l'échelle de la covariance par le khi² réduit (au moins 1 : on ne réduit jamais l'incertitude).
        cov = cov * max(np.sum((res / np.maximum(ses, 0.05)) ** 2) / max(len(aws) - 2, 1), 1.0)
        # Enregistre la pente b.
        m.params["ln D = a + b·a_w : b"] = (coef[0], np.sqrt(cov[0, 0]), "-",
                                            "interpolation de ln D en fonction de l'activité de l'eau")
        # Loi ln D linéaire en a_w, avec a_w bloquée dans la plage mesurée (pas d'extrapolation).
        lin = lambda aw, c0, c1: np.exp(c1 + c0 * np.clip(aw, aws.min(), aws.max()))
        # Une loi D(a_w) par tirage.
        Dt = [lambda T, aw, s=s: lin(aw, *s) for s in _tirer(rng, coef, cov, n)]
        # Loi D(a_w) centrale.
        Dc = lambda T, aw: lin(aw, *coef)
        # Domaine : température inconnue, a_w de 0,36 à 0,999.
        m.domaine.update(T_diffusion=(np.nan, np.nan), aw_diffusion=(float(aws.min()), float(aws.max())))

    # ----- 2. ISOTHERME DE SORPTION : teneur en eau d'équilibre M∞ en fonction de T et de a_w
    # ---------------- isotherme
    # Lignes d'isotherme.
    iso = G("isotherme")
    # Le Gac : points à 80 °C (Tab. 2) + M∞ à 25 °C (absorption).
    if cle == "PA6_GOODFELLOW":
        # Loi puissance M∞ = Ms·a_wⁿ.
        f = lambda a, Ms, nn: Ms * a ** nn
        # Ajustement de Ms (80 °C) et n sur les 5 points du Tab. 2.
        p, cov = curve_fit(f, iso["x"].to_numpy(), iso["y"].to_numpy(), p0=[10, 1.5])
        # Ms à 25 °C déduit du M∞ en eau de mer (a_w = 0,98) et de n.
        ms25 = msat25 / 0.98 ** p[1]
        # Enregistre Ms(80 °C).
        m.params["Ms(80 °C)"] = (p[0], np.sqrt(cov[0, 0]), "% masse", "loi puissance sur Le Gac Tab. 2 (80 °C)")
        # Enregistre l'exposant n.
        m.params["n isotherme"] = (p[1], np.sqrt(cov[1, 1]), "-", "idem")
        # Enregistre Ms(25 °C).
        m.params["Ms(25 °C)"] = (ms25, np.nan, "% masse", "déduit de M∞(25 °C, Fig. 1) et de n")
        # Pente dMs/dT : HYPOTHÈSE d'une variation linéaire entre les deux seules températures.
        m.params["dMs/dT"] = ((p[0] - ms25) / 55, np.nan, "%/°C", "HYPOTHÈSE : Ms linéaire entre 25 et 80 °C")
        # Ms(T) linéaire entre 25 et 80 °C, bloqué aux bornes.
        Ms = lambda T, s80, s25: s25 + (s80 - s25) * (np.clip(T, 25, 80) - 25) / 55
        # Tirages de (Ms80, n) dans des bornes physiques.
        s = _tirer(rng, p, cov, n, [0.1, 0.3], [50, 5])
        # Une loi M∞(T, a_w) par tirage ; l'écart Ms80 − Ms25 est conservé.
        Mt = [lambda T, aw, v=v: Ms(T, v[0], v[0] - (p[0] - ms25)) * aw ** v[1] for v in s]
        # Loi M∞ centrale.
        Mc = lambda T, aw: Ms(T, p[0], ms25) * aw ** p[1]
        # Domaine : a_w de 0,2 à 0,98, T de 25 à 80 °C.
        m.domaine.update(aw_iso=(0.2, 0.98), T_iso=(25.0, 80.0))
    # Gibhardt : M∞ en immersion à 8, 40, 50 °C et en vapeur à 50 °C.
    elif cle == "EPOXY_HEXION_RIMR135":
        # Points en immersion (a_w = 1).
        eau = iso[iso["x"] == 1.0]
        # M∞(immersion) linéaire en T : régression et covariance brute.
        cT, cvT = np.polyfit(eau["T_C"].to_numpy(), eau["y"].to_numpy(), 1, cov="unscaled")
        # Mise à l'échelle de la covariance par la variance des résidus.
        cvT = cvT * np.sum((eau["y"] - np.polyval(cT, eau["T_C"])) ** 2) / max(len(eau) - 2, 1)
        # Points en vapeur (a_w < 1).
        vap = iso[iso["x"] < 1.0]
        # M∞ en immersion à 50 °C (même température que les points en vapeur).
        Ms50 = np.polyval(cT, 50.0)
        # Ajustement de n sur les points en vapeur à 50 °C : M∞ = M∞(immersion, 50 °C)·a_wⁿ.
        p, cov = curve_fit(lambda a, nn: Ms50 * a ** nn, vap["x"].to_numpy(), vap["y"].to_numpy(), p0=[2.0])
        # Enregistre la pente de M∞ en fonction de T.
        m.params["M∞ immersion : pente en T"] = (cT[0], np.sqrt(cvT[0, 0]), "%/°C", "Gibhardt Fig. 5A, 8-50 °C")
        # Enregistre n.
        m.params["n isotherme"] = (p[0], np.sqrt(cov[0, 0]), "-", "vapeur à 50 °C (Fig. 5B), forme supposée indépendante de T")
        # Tirages des coefficients M∞(T).
        sT = _tirer(rng, cT, cvT, n)
        # Tirages de n, bornés entre 0,3 et 6.
        sn = _tirer(rng, p, cov, n, [0.3], [6])[:, 0]
        # Une loi M∞(T, a_w) par tirage ; T bloquée dans 8-50 °C (pas d'augmentation mesurée au-delà).
        Mt = [lambda T, aw, a=a, b=b: np.polyval(a, np.clip(T, 8, 50)) * aw ** b for a, b in zip(sT, sn)]
        # Loi M∞ centrale.
        Mc = lambda T, aw: np.polyval(cT, np.clip(T, 8, 50)) * aw ** p[0]
        # Domaine : a_w de 0,5 à 1, T de 8 à 50 °C.
        m.domaine.update(aw_iso=(0.5, 1.0), T_iso=(8.0, 50.0))
    # Zarbali : la loi puissance représente mal ces points ; on interpole directement les mesures.
    else:                                           # Zarbali : interpolation monotone des points du Tab. 3
        # Abscisses : 0 ajouté (sec) + activités de l'eau du Tab. 3.
        xs = np.concatenate([[0.0], iso["x"].to_numpy()])
        # Ordonnées : 0 ajouté + M∞ du Tab. 3.
        ys = np.concatenate([[0.0], iso["y"].to_numpy()])
        # Ordre croissant des abscisses.
        o = np.argsort(xs)
        # Tri des deux tableaux.
        xs, ys = xs[o], ys[o]
        # Interpolation monotone croissante (plus d'humidité -> plus d'eau).
        f, _ = _pchip_monotone(xs, ys, True)
        # Loi M∞ centrale (T ignorée : inconnue).
        Mc = lambda T, aw, f=f: f(aw)
        # Lois par tirage.
        Mt = []
        # Pour chaque tirage, chaque point est perturbé de 5 % (incertitude SUPPOSÉE, aucune n'est publiée).
        for _ in range(n):                         # incertitude SUPPOSÉE : 5 % sur chaque point
            # Points perturbés.
            v = ys * (1 + rng.normal(0, 0.05, len(ys)))
            # Interpolation monotone des points perturbés.
            fi, _ = _pchip_monotone(xs, v, True)
            # Loi M∞ de ce tirage.
            Mt.append(lambda T, aw, fi=fi: fi(aw))
        # Paramètre affiché (pas de valeur numérique unique).
        m.params["isotherme"] = (np.nan, np.nan, "-", "interpolation monotone (PCHIP) des M∞ du Tab. 3, ±5 % supposé")
        # Domaine : a_w de 0,36 à 0,999, T inconnue.
        m.domaine.update(aw_iso=(0.36, 0.999), T_iso=(np.nan, np.nan))

    # ----- 3. RELATION MODULE / TENEUR EN EAU, mesurée sur éprouvettes à teneur homogène
    # ---------------- relation module / teneur en eau
    # Points module/teneur en eau, triés par teneur croissante.
    me = G("module_eau").sort_values("x")
    # Module de référence : moyenne des points secs (teneur < 0,05 %).
    E_ref = float(me.loc[me["x"] < 0.05, "y"].mean())
    # Teneurs en eau (négatifs de lecture ramenés à 0) et modules relatifs E/E_ref ; .copy() rend le tableau
    # modifiable.
    xs, r = me["x"].clip(lower=0).to_numpy().copy(), me["y"].to_numpy() / E_ref
    # Le premier point est exactement (0 ; 1) par définition de la référence.
    xs[0], r[0] = 0.0, 1.0
    # Gibhardt : deux points saturés seulement -> loi linéaire E/E0 = 1 − k·M.
    if cle == "EPOXY_HEXION_RIMR135":
        # Ajustement de la pente k.
        p, cov = curve_fit(lambda M, k: 1 - k * M, xs, r, p0=[0.02])
        # Enregistre k.
        m.params["E(M) : pente k"] = (p[0], np.sqrt(cov[0, 0]), "1/%", "linéaire sur Gibhardt §3.4 (2 points saturés)")
        # Une loi par tirage de k (borné entre 0 et 1).
        Gt = [lambda M, k=k: 1 - k * M for k in _tirer(rng, p, cov, n, [0], [1])[:, 0]]
        # Loi centrale.
        Gc = lambda M: 1 - p[0] * M
    # PA6 (Le Gac, Zarbali) : chute non linéaire (passage de Tg) -> interpolation monotone décroissante des points
    # mesurés.
    else:
        # Loi centrale et points corrigés.
        Gc, yy = _pchip_monotone(xs, r, False)
        # Lois par tirage.
        Gt = []
        # Pour chaque tirage...
        for _ in range(n):
            # ... chaque point est perturbé de BRUIT (3 %, hypothèse) ;
            v = r + rng.normal(0, BRUIT, len(r))
            # ... le point sec reste exactement 1 ;
            v[0] = 1.0
            # ... puis interpolation monotone (bornée entre 0,01 et 1,2).
            fi, _ = _pchip_monotone(xs, np.clip(v, 0.01, 1.2), False)
            # Ajoute la loi de ce tirage.
            Gt.append(fi)
        # Enregistre chaque point pour l'affichage.
        for xi, yi in zip(xs, yy):
            m.params[f"E/E0 à {xi:.2f} % d'eau"] = (yi, BRUIT, "-", "point mesuré (interpolation monotone PCHIP)")
    # Domaine de teneur en eau couvert par les mesures de module.
    m.domaine["M_module"] = (float(xs.min()), float(xs.max()))
    # Si E0 n'est pas dans la fiche, il prend la valeur du module sec mesuré.
    m.meta["E0_MPa"] = m.meta["E0_MPa"] or round(E_ref)

    # ----- 4. Tg EN FONCTION DE LA TENEUR EN EAU (pour l'alerte « au-dessus de la Tg humide »)
    # ---------------- Tg(M)
    # Points Tg/teneur en eau, triés.
    tg = G("tg_eau").sort_values("x")
    # Interpolation linéaire de Tg(M).
    Tgf = lambda M, xs=tg["x"].to_numpy(), ys=tg["y"].to_numpy(): np.interp(M, xs, ys)
    # Assemble les 4 lois de chaque tirage dans un dictionnaire.
    m.P = [dict(D=Dt[i], Minf=Mt[i], g=Gt[i], Tg=Tgf) for i in range(n)]
    # Assemble les 4 lois centrales.
    m.Pc = dict(D=Dc, Minf=Mc, g=Gc, Tg=Tgf)


# Calcule la courbe E/E0(t) d'un matériau « chaine » pour un jeu de lois P et des conditions (T, a_w, h, temps en h).
def _courbe_chaine(P, T, aw, h_mm, t_h):
    # Épaisseur en mètres.
    h = h_mm / 1000
    # Coefficient de diffusion dans ces conditions.
    D = float(P["D"](T, aw))
    # Teneur en eau d'équilibre dans ces conditions.
    Minf = float(P["Minf"](T, aw))
    # Temps réduits D·t/h² (temps convertis en secondes).
    tau = D * np.asarray(t_h, float) * 3600 / h ** 2
    # Module local g(teneur locale) en chaque point de l'épaisseur, puis moyenne sur l'épaisseur (intégrale par
    # trapèzes) : moyenne de Voigt, valable en traction.
    r = np.trapezoid(P["g"](Minf * profil(tau)), _X, axis=1)
    # Renvoie E/E0, la teneur en eau moyenne, M∞, D et la Tg humide à l'équilibre.
    return dict(r=r, M=Minf * fraction_moyenne(tau), Minf=Minf, D=D, Tg=float(P["Tg"](Minf)))


# ---------- AJUSTEMENT DU NIVEAU « APPARENT » : E/E0 = 1 − Δ∞(a_w)·g_b(F(D·t/h²))
# =========================================================================== ajustements « apparent »
# Construit les lois d'un matériau « apparent » (pas de relation E(M) mesurée).
def _apparent(m: Modele, d: pd.DataFrame, rng) -> None:
    # Raccourcis : fiche et nombre de tirages.
    meta, n = m.meta, N_TIRAGES
    # Mesures de module dans le temps.
    mt = d[d["grandeur"] == "module_temps"].copy()
    # E0 de chaque milieu : module à t = 0 du même milieu (chaque courbe est normalisée par sa propre référence).
    mt["E0"] = mt.groupby("a_w")["y"].transform(lambda s: s[mt.loc[s.index, "x"] == 0].iloc[0])
    # Module relatif E/E0.
    mt["r"] = mt["y"] / mt["E0"]
    # Si E0 n'est pas dans la fiche : moyenne des modules à t = 0.
    meta["E0_MPa"] = meta["E0_MPa"] or round(float(mt.loc[mt["x"] == 0, "y"].mean()))
    # Épaisseur des éprouvettes en mètres.
    h = meta["h_ref_mm"] / 1000
    # Points vieillis seulement (t > 0) : le point t = 0 vaut 1 par construction.
    data = mt[mt["x"] > 0]
    # Temps (s), activité de l'eau et E/E0 des points vieillis.
    t_s, aw, r = data["x"].to_numpy() * 3600, data["a_w"].to_numpy(), data["r"].to_numpy()
    # Vrai si plusieurs milieux ont été mesurés (Wang) : on peut alors ajuster l'effet de a_w.
    plusieurs_aw = data["a_w"].nunique() > 1
    # Teneur en eau d'équilibre : inconnue par défaut.
    Minf = None
    # Lignes d'isotherme éventuelles (Chilali publie M∞).
    iso = d[d["grandeur"] == "isotherme"]
    # Si M∞ est publié...
    if len(iso):
        # ... on le garde pour tracer la prise en eau.
        Minf = float(iso["y"].iloc[0])
        # ... et on l'affiche.
        m.params["M∞"] = (Minf, np.nan, "% masse", "publié (" + iso["source_fig"].iloc[0] + ")")

    # Chilali : D publié (gravimétrie). Seules l'amplitude Δ∞ et la forme b sont ajustées sur le module.
    if meta["D_mode"] == "publie":                 # D gravimétrique publié, amplitude et forme ajustées
        # D publié (Dzz du Tab. 4.1).
        Dpub = float(d.loc[d["grandeur"] == "diffusivite", "y"].iloc[0])
        # Modèle : E/E0 = 1 − Δ∞·g_b(F). X[0] = temps en secondes.
        f = lambda X, delta, b: 1 - delta * _g(fraction_moyenne(Dpub * X[0] / h ** 2), b)
        # Ajustement de Δ∞ (entre 0 et 1) et b (entre 0 et 20).
        par, cov = curve_fit(f, np.vstack([t_s, aw]), r, p0=[0.5, 1.0], bounds=([0, 0], [1, 20]))
        # Affiche D avec une incertitude de 6 % (ordre de grandeur des ± publiés au Tab. 4.1).
        m.params["D (épaisseur)"] = (Dpub, 0.06 * Dpub, "m²/s", "publié (Tab. 4.1, Dzz) ; non ajusté")
        # Affiche Δ∞.
        m.params["Δ∞ (perte à saturation)"] = (par[0], np.sqrt(cov[0, 0]), "-", "ajusté sur E(t)")
        # Affiche b.
        m.params["b (forme)"] = (par[1], np.sqrt(cov[1, 1]), "-",
                                 "ajusté sur E(t) : b > 0, le module chute plus vite que l'eau n'entre")
        # Tirages de (Δ∞, b).
        s = _tirer(rng, par, cov, n, [0, 0], [1, 20])
        # Tirages de D autour de la valeur publiée (6 % en relatif).
        sD = Dpub * np.exp(rng.normal(0, 0.06, n))
        # Une loi par tirage : D, Δ∞(a_w) (constant ici), b.
        m.P = [dict(D=lambda T, a, v=v: v, delta=lambda a, x=x: x[0], b=x[1]) for v, x in zip(sD, s)]
        # Lois centrales.
        m.Pc = dict(D=lambda T, a: Dpub, delta=lambda a: par[0], b=par[1])
        # Domaine : immersion seulement (a_w = 1).
        m.domaine["aw"] = (1.0, 1.0)
    # Takmid, Wang : D ajusté directement sur le module, forme linéaire (b = 0).
    else:                                           # D apparent ajusté sur E(t), p = 1
        # Plusieurs milieux (Wang) : Δ∞ dépend de a_w selon Δ∞ = Δ1·a_wⁿ.
        if plusieurs_aw:
            # Modèle à trois paramètres : ln D, Δ1, n. X[1] = activité de l'eau.
            f = lambda X, lnD, d1, nn: 1 - d1 * X[1] ** nn * fraction_moyenne(np.exp(lnD) * X[0] / h ** 2)
            # Valeurs initiales et bornes des trois paramètres.
            p0, lo, hi = [np.log(1e-12), 0.1, 1.0], [np.log(1e-16), 0, 0], [np.log(1e-8), 1, 20]
        # Un seul milieu (Takmid).
        else:
            # Modèle à deux paramètres : ln D, Δ∞.
            f = lambda X, lnD, d1: 1 - d1 * fraction_moyenne(np.exp(lnD) * X[0] / h ** 2)
            # Valeurs initiales et bornes.
            p0, lo, hi = [np.log(1e-12), 0.3], [np.log(1e-16), 0], [np.log(1e-8), 1]
        # Ajustement par moindres carrés bornés.
        par, cov = curve_fit(f, np.vstack([t_s, aw]), r, p0=p0, bounds=(lo, hi))
        # Affiche D apparent.
        m.params["D apparent"] = (np.exp(par[0]), np.exp(par[0]) * np.sqrt(cov[0, 0]), "m²/s",
                                  f"ajusté sur E(t) avec h = {meta['h_ref_mm']} mm")
        # Affiche Δ∞.
        m.params["Δ∞ (perte à saturation, a_w = 1)"] = (par[1], np.sqrt(cov[1, 1]), "-", "ajusté sur E(t)")
        # S'il y a plusieurs milieux...
        if plusieurs_aw:
            # ... affiche l'exposant n.
            m.params["n : Δ∞ ∝ a_wⁿ"] = (par[2], np.sqrt(cov[2, 2]), "-", "ajusté sur les 3 milieux")
        # Tirages des paramètres dans leurs bornes.
        s = _tirer(rng, par, cov, n, lo, hi)
        # Fabrique un jeu de lois à partir d'un vecteur de paramètres v.
        mk = lambda v: dict(D=lambda T, a, v=v: np.exp(v[0]),
                            delta=(lambda a, v=v: v[1] * a ** v[2]) if plusieurs_aw else (lambda a, v=v: v[1]), b=0.0)
        # Une loi par tirage.
        m.P = [mk(v) for v in s]
        # Lois centrales.
        m.Pc = mk(par)
        # Domaine d'activité de l'eau mesuré.
        m.domaine["aw"] = (float(aw.min()), float(aw.max()))
    # Ajoute M∞ (ou None) aux lois centrales...
    m.Pc["Minf"] = Minf
    # ... et à chaque tirage.
    for P in m.P:
        P["Minf"] = Minf
    # Prédictions du modèle central aux points mesurés.
    pred = np.array([_courbe_apparent(m.Pc, np.nan, a, meta["h_ref_mm"], [t])["r"][0] for t, a in zip(data["x"], aw)])
    # Écart de modèle : RMS des écarts relatifs ln(mesure/modèle), au moins 1 %.
    m.sd_modele = max(float(np.sqrt(np.mean(np.log(r / pred) ** 2))), 0.01)
    # Affiche cet écart.
    m.params["écart de modèle (RMS relatif)"] = (m.sd_modele, np.nan, "-", "résidus de l'ajustement sur E(t)")
    # Garde les mesures pour la validation.
    m.mesures = mt


# Forme g_b(F) : g = F si b ≈ 0 (module proportionnel à l'eau absorbée), sinon (1 − e^(−bF))/(1 − e^(−b)) : chute plus
# rapide au début, mais pente finie à t = 0.
def _g(F, b):
    # Tableau numpy.
    F = np.asarray(F, float)
    # Calcul de la forme.
    return F if b < 1e-6 else (1 - np.exp(-b * F)) / (1 - np.exp(-b))


# Calcule la courbe E/E0(t) d'un matériau « apparent ».
def _courbe_apparent(P, T, aw, h_mm, t_h):
    # Coefficient de diffusion.
    D = float(P["D"](T, aw))
    # Prise en eau moyenne relative F(D·t/h²).
    F = fraction_moyenne(D * np.asarray(t_h, float) * 3600 / (h_mm / 1000) ** 2)
    # E/E0 = 1 − Δ∞(a_w)·g_b(F).
    r = 1 - P["delta"](aw) * _g(F, P["b"])
    # M∞ publié (ou None).
    Minf = P["Minf"]
    # Renvoie E/E0, la teneur en eau moyenne si M∞ est connu, M∞, D ; pas de Tg pour ce niveau.
    return dict(r=r, M=None if Minf is None else Minf * F, Minf=Minf, D=D, Tg=None)


# ---------- AJUSTEMENT DU NIVEAU « CINÉTIQUE » : E/E0 = r∞ + (1 − r∞)·exp(−k(T)·t)
# =========================================================================== ajustements « cinétique »
# Construit les lois d'un matériau « cinetique » (dégradation lente, PLA).
def _cinetique(m: Modele, d: pd.DataFrame, rng) -> None:
    # Raccourcis.
    meta, n = m.meta, N_TIRAGES
    # Mesures de module dans le temps.
    mt = d[d["grandeur"] == "module_temps"].copy()
    # E0 : moyenne des modules à t = 0.
    E0 = float(mt.loc[mt["x"] == 0, "y"].mean())
    # E0 de la fiche, ou celui des données.
    meta["E0_MPa"] = meta["E0_MPa"] or round(E0)
    # Module relatif.
    mt["r"] = mt["y"] / E0
    # Points vieillis seulement.
    data = mt[mt["x"] > 0]
    # Temps (h), températures et E/E0.
    t, T, r = data["x"].to_numpy(), data["T_C"].to_numpy(), data["r"].to_numpy()
    # Plusieurs températures (Niaounakis) : k suit Arrhenius.
    if data["T_C"].nunique() > 1:
        # Température de référence.
        Tref = 40.0
        # Modèle à trois paramètres : plateau r∞, ln k(40 °C), énergie d'activation Ea (kJ/mol).
        f = lambda X, ri, lnk, Ea: ri + (1 - ri) * np.exp(-np.exp(lnk - Ea * 1000 / R_GAZ *
                                                                   (1 / (X[1] + KELVIN) - 1 / (Tref + KELVIN))) * X[0])
        # Ajustement borné (r∞ entre 0 et 0,99 ; Ea entre 1 et 300 kJ/mol).
        par, cov = curve_fit(f, np.vstack([t, T]), r, p0=[0.4, np.log(3e-4), 60], bounds=([0, -20, 1], [0.99, 0, 300]))
        # Affiche r∞.
        m.params["r∞ (plateau)"] = (par[0], np.sqrt(cov[0, 0]), "-", "ajusté sur E(t) à 20, 40 et 50 °C")
        # Affiche k(40 °C).
        m.params["k(40 °C)"] = (np.exp(par[1]), np.exp(par[1]) * np.sqrt(cov[1, 1]), "1/h", "idem")
        # Affiche Ea.
        m.params["Ea"] = (par[2], np.sqrt(cov[2, 2]), "kJ/mol", "idem (Arrhenius)")
        # Fabrique un jeu de lois (plateau, k(T)) à partir d'un vecteur de paramètres.
        mk = lambda v: dict(rinf=v[0], k=lambda TT, v=v: np.exp(v[1] - v[2] * 1000 / R_GAZ *
                                                               (1 / (TT + KELVIN) - 1 / (Tref + KELVIN))))
        # Tirages des paramètres.
        s = _tirer(rng, par, cov, n, [0, -20, 1], [0.99, 0, 300])
    # Une seule condition (Bergaliyeva) : pas d'effet de T.
    else:
        # Modèle à deux paramètres : r∞, ln k.
        f = lambda X, ri, lnk: ri + (1 - ri) * np.exp(-np.exp(lnk) * X[0])
        # Ajustement borné.
        par, cov = curve_fit(f, np.vstack([t, T]), r, p0=[0.85, np.log(0.01)], bounds=([0, -20], [0.99, 2]))
        # Affiche r∞.
        m.params["r∞ (plateau)"] = (par[0], np.sqrt(cov[0, 0]), "-", "ajusté sur E(t), une seule condition")
        # k est-il identifiable ? Non si sa variance est infinie/nulle ou s'il touche la borne (perte instantanée).
        ident = np.isfinite(cov[1, 1]) and cov[1, 1] > 0 and par[1] < 1.9
        # Affiche k, ou le déclare non identifiable.
        m.params["k"] = (np.exp(par[1]), np.exp(par[1]) * np.sqrt(cov[1, 1]) if ident else np.nan, "1/h",
                         "idem" if ident else "NON IDENTIFIABLE : la perte est atteinte avant la première mesure (24 h)")
        # Fabrique un jeu de lois (k constant).
        mk = lambda v: dict(rinf=v[0], k=lambda TT, v=v: np.exp(v[1]))
        # Tirages.
        s = _tirer(rng, par, cov, n, [0, -20], [0.99, 2])
    # Une loi par tirage.
    m.P = [mk(v) for v in s]
    # Lois centrales.
    m.Pc = mk(par)
    # Prédictions centrales aux points mesurés.
    pred = f(np.vstack([t, T]), *par)
    # Écart de modèle relatif (RMS, au moins 1 %).
    m.sd_modele = max(float(np.sqrt(np.mean(np.log(r / pred) ** 2))), 0.01)
    # Affiche cet écart.
    m.params["écart de modèle (RMS relatif)"] = (m.sd_modele, np.nan, "-", "résidus de l'ajustement sur E(t)")
    # Domaine de températures mesuré.
    m.domaine["T"] = (float(T.min()), float(T.max()))
    # Garde les mesures.
    m.mesures = mt


# Calcule la courbe E/E0(t) d'un matériau « cinetique ».
def _courbe_cinetique(P, T, aw, h_mm, t_h):
    # Constante de vitesse à la température demandée.
    k = float(P["k"](T))
    # E/E0 = r∞ + (1 − r∞)·exp(−k·t), t en heures.
    r = P["rinf"] + (1 - P["rinf"]) * np.exp(-k * np.asarray(t_h, float))
    # Renvoie E/E0 et k ; ni eau, ni D, ni Tg pour ce niveau.
    return dict(r=r, M=None, Minf=None, D=None, Tg=None, k=k)


# Table qui associe chaque niveau de modèle à sa fonction de calcul de courbe.
COURBE = {"chaine": _courbe_chaine, "apparent": _courbe_apparent, "cinetique": _courbe_cinetique}


# ---------- PROCESSUS GAUSSIEN (niveau « chaine ») : corrige l'écart physique/mesures et en estime l'incertitude
# =========================================================================== GP (type chaîne)
# Distances de corrélation a priori du processus gaussien : 0,5 décade de temps réduit, 10 °C, 0,15 d'activité de
# l'eau.
ECHELLES_GP = np.array([0.5, 10.0, 0.15])           # décade de D t / h², °C, a_w : choix a priori


# Avancement de la dégradation : 0 à t = 0, 1 quand la perte à l'équilibre est atteinte. Sert à annuler l'incertitude
# à t = 0.
def avancement(r_t, r_inf):
    """Part de la perte de module atteinte (0 à t = 0, 1 à l'équilibre) : pondère l'écart de modèle."""
    # Perte de module à l'équilibre.
    perte = 1.0 - float(r_inf)
    # Tableau numpy.
    r_t = np.asarray(r_t, float)
    # Avancement = perte actuelle / perte finale, borné entre 0 et 1 (1 partout si la perte finale est négligeable).
    return np.clip((1.0 - r_t) / perte, 0.0, 1.0) if perte > 1e-3 else np.ones_like(r_t)


# Mesures E(t) d'un matériau, normalisées et complétées (température et épaisseur de calcul).
def _mesures_temps(cle: str) -> pd.DataFrame:
    # Données du matériau.
    d = _donnees(cle)
    # Lignes de module dans le temps.
    d = d[d["grandeur"] == "module_temps"].copy()
    # Aucune mesure...
    if d.empty:
        # ... tableau vide.
        return d
    # E0 de la fiche.
    E0 = MATERIAUX[cle]["E0_MPa"]
    # Le Gac : la référence est le premier point de la Fig. 7a (même lecture que les autres points).
    if cle == "PA6_GOODFELLOW":
        # Valeur à t = 0.
        E0 = float(d.loc[d["x"] == 0, "y"].iloc[0])
    # Colonne de référence E0.
    d["E0_ref"] = E0
    # Pour les autres matériaux avec un point t = 0 par condition...
    if (d["x"] == 0).any() and cle not in ("PA6_GOODFELLOW", "EPOXY_HEXION_RIMR135"):
        # ... chaque condition est normalisée par son propre point t = 0.
        d["E0_ref"] = d.groupby(["a_w", "T_C"], dropna=False)["y"].transform(lambda s: s.iloc[0])
    # E/E0 mesuré.
    d["rapport"] = d["y"] / d["E0_ref"]
    # Température utilisée pour le calcul : celle de la mesure, sinon celle de la fiche, sinon 23 °C (sans effet pour
    # ces matériaux).
    d["T_calc"] = d["T_C"].fillna(MATERIAUX[cle].get("T_essais") or 23.0)
    # Épaisseur utilisée pour le calcul : celle de la mesure, sinon celle des éprouvettes.
    d["h_calc"] = d["h_mm"].fillna(MATERIAUX[cle]["h_ref_mm"] or 1.0)
    # Renvoie le tableau.
    return d


# Ajuste le processus gaussien sur les écarts entre la chaîne physique et les mesures E(t).
def _ajuster_gp(m: Modele) -> None:
    # Mesures du matériau.
    d = _mesures_temps(m.cle)
    # Points vieillis seulement.
    d = d[d["x"] > 0]
    # Moins de 3 mesures : pas de correction possible...
    if len(d) < 3:
        # ... on le note...
        m.gp_info = dict(actif=False, raison=f"{len(d)} mesure(s) E(t)")
        # ... et on sort.
        return
    # Entrées X et écarts res du processus gaussien.
    X, res = [], []
    # Pour chaque mesure...
    for _, q in d.iterrows():
        # ... courbe physique centrale au temps mesuré et à l'équilibre (1e9 h) ;
        c = _courbe_chaine(m.Pc, q["T_calc"], q["a_w"], q["h_calc"], [q["x"], 1e9])
        # ... avancement à ce temps (au moins 0,05 pour éviter une division par zéro) ;
        p_av = max(float(avancement(c["r"][:1], c["r"][1])[0]), 0.05)
        # ... temps réduit D·t/h² ;
        tau = c["D"] * q["x"] * 3600 / (q["h_calc"] / 1000) ** 2
        # ... entrées : log10 du temps réduit, température, activité de l'eau ;
        X.append([np.log10(tau + 1e-6), q["T_calc"], q["a_w"]])
        # ... écart relatif ln(mesure/physique) ramené à un avancement de 1.
        res.append(np.log(q["rapport"] / c["r"][0]) / p_av)
    # Conversion en tableaux.
    X, res = np.array(X), np.array(res)
    # Amplitude typique des écarts (RMS).
    amp = float(np.sqrt(np.mean(res ** 2)))
    # Noyau : amplitude (au moins 0,02) x similarité RBF + bruit de mesure. Paramètres FIXÉS (pas d'optimisation sur
    # si peu de points).
    noyau = (ConstantKernel(max(amp, 0.02) ** 2, "fixed") * RBF(ECHELLES_GP, "fixed")
             + WhiteKernel(BRUIT ** 2, "fixed"))
    # Ajustement du processus gaussien.
    m.gp = GaussianProcessRegressor(noyau, optimizer=None).fit(X, res)
    # Enregistre ses caractéristiques.
    m.gp_info = dict(actif=True, n=len(res), amplitude=amp, echelles=ECHELLES_GP.tolist(), bruit=BRUIT)


# ---------- CONSTRUCTION D'UN MODÈLE
# =========================================================================== construction
# Mémorise le résultat : chaque matériau n'est ajusté qu'une fois par session.
@lru_cache(maxsize=None)
# Ajuste et renvoie le modèle d'un matériau.
def modele(cle: str) -> Modele:
    # Fiche du matériau.
    meta = MATERIAUX[cle]
    # Type « donnees » : rien à ajuster...
    if meta["type"] == "donnees":
        # ... prédiction refusée avec la raison.
        raise NonPredictible(meta["raison"])
    # Objet modèle vide.
    m = Modele(cle=cle, meta=meta)
    # Générateur aléatoire avec graine fixe (0) : résultats reproductibles d'une exécution à l'autre.
    rng = np.random.default_rng(0)
    # Données du matériau.
    d = _donnees(cle)
    # Masque temporairement les avertissements numériques...
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ... pendant l'appel de la fonction d'ajustement correspondant au type (_chaine, _apparent ou _cinetique).
        {"chaine": _chaine, "apparent": _apparent, "cinetique": _cinetique}[meta["type"]](m, d, rng)
        # Pour une chaîne...
        if meta["type"] == "chaine":
            # ... écart de modèle par défaut ...
            m.sd_modele = BRUIT
            # ... et processus gaussien si des mesures E(t) existent.
            _ajuster_gp(m)
    # Renvoie le modèle ajusté.
    return m


# Liste des matériaux pour lesquels une prédiction est possible.
def predictibles() -> list[str]:
    # Tous sauf le type « donnees ».
    return [c for c, v in MATERIAUX.items() if v["type"] != "donnees"]


# ---------- PRÉDICTION
# =========================================================================== prédiction
# Activité de l'eau du cas : fixée par le milieu, ou HR/100 en vapeur.
def _aw(milieu, HR):
    # Calcul.
    return MILIEUX[milieu] if MILIEUX[milieu] is not None else HR / 100.0


# Contrôle qu'un cas est prédictible ; sinon lève NonPredictible avec un message destiné à l'utilisateur.
def verifier_cas(cle: str, T: float, HR: float, milieu: str, h_mm: float) -> None:
    """Lève NonPredictible si le cas sort de ce que les données permettent d'identifier."""
    # Fiche du matériau.
    meta = MATERIAUX[cle]
    # Type « donnees » : jamais prédictible.
    if meta["type"] == "donnees":
        # Refus.
        raise NonPredictible(meta["raison"])
    # Milieu jamais mesuré pour ce matériau...
    if milieu not in meta["milieux"]:
        # ... refus avec la liste des milieux mesurés.
        raise NonPredictible(f"Milieu « {milieu} » sans aucune donnée pour ce matériau "
                             f"(milieux mesurés : {', '.join(sorted(meta['milieux']))}).")
    # Activité de l'eau du cas.
    aw = _aw(milieu, HR)
    # Une seule humidité mesurée : écart de plus de 5 %HR refusé.
    if "aw_cible" in meta and abs(aw - meta["aw_cible"]) > 0.05:
        # Refus.
        raise NonPredictible(f"Une seule humidité mesurée ({meta['aw_cible'] * 100:.0f} %HR) : l'effet de "
                             f"l'humidité n'est pas identifiable (±5 %HR acceptés).")
    # Humidité sous le minimum mesuré...
    if "aw_min" in meta and aw < meta["aw_min"] - 1e-9:
        # ... refus.
        raise NonPredictible(f"Humidité sous le domaine mesuré (≥ {meta['aw_min'] * 100:.0f} %HR).")
    # Une seule température mesurée : écart supérieur à la tolérance refusé.
    if "T_tolerance" in meta and abs(T - meta["T_essais"]) > meta["T_tolerance"]:
        # Refus.
        raise NonPredictible(f"Une seule température mesurée ({meta['T_essais']:g} °C) : l'effet de la température "
                             f"n'est pas identifiable (±{meta['T_tolerance']:g} °C acceptés).")
    # Température hors de la plage où le mécanisme reste valable...
    if "T_refus" in meta and not (meta["T_refus"][0] <= T <= meta["T_refus"][1]):
        # ... refus.
        raise NonPredictible(f"Température hors du domaine où le mécanisme mesuré reste valable "
                             f"({meta['T_refus'][0]:g}-{meta['T_refus'][1]:g} °C).")
    # Épaisseur nulle ou négative...
    if h_mm <= 0:
        # ... refus.
        raise NonPredictible("Épaisseur invalide.")


# FONCTION PRINCIPALE : courbe E(t) avec intervalle de confiance et points de vigilance pour un cas donné.
def predire(cle: str, T: float, HR: float, milieu: str, h_mm: float, duree_h: float, n_points: int = 120) -> dict:
    # Refuse d'abord les cas non prédictibles.
    verifier_cas(cle, T, HR, milieu, h_mm)
    # Modèle ajusté (calculé une seule fois grâce au cache).
    m = modele(cle)
    # Fiche et niveau de modèle.
    meta, typ = m.meta, m.meta["type"]
    # Activité de l'eau du cas.
    aw = _aw(milieu, HR)
    # Grille de temps : 0, puis 120 temps espacés logarithmiquement de 0,1 h à la durée demandée, plus la durée
    # exacte.
    t = np.unique(np.concatenate([[0.0], np.geomspace(0.1, duree_h, n_points), [duree_h]]))
    # Fonction de courbe du niveau de modèle.
    courbe = COURBE[typ]
    # Épaisseur utilisée : celle demandée, sauf pour le niveau « cinetique » où elle n'intervient pas.
    h_calc = h_mm if typ != "cinetique" else (meta["h_ref_mm"] or h_mm)
    # Courbes E/E0 pour les 200 tirages de paramètres (tableau tirages x temps).
    tirs = np.array([courbe(P, T, aw, h_calc, t)["r"] for P in m.P])
    # Courbe centrale, prolongée d'un temps très long (1e9 h) pour obtenir l'équilibre.
    c = courbe(m.Pc, T, aw, h_calc, np.concatenate([t, [1e9]]))
    # Sépare la courbe centrale et sa valeur d'équilibre.
    centre, r_inf = c["r"][:-1], float(c["r"][-1])
    # Si un processus gaussien existe...
    if m.gp is not None:
        # ... temps réduits de la grille ;
        tau = c["D"] * t * 3600 / (h_mm / 1000) ** 2
        # ... correction moyenne mu et incertitude sd en chaque temps.
        mu, sd = m.gp.predict(np.column_stack([np.log10(tau + 1e-6), np.full_like(t, T), np.full_like(t, aw)]),
                              return_std=True)
    # Sinon...
    else:
        # ... pas de correction moyenne, incertitude = écart de modèle constant.
        mu, sd = np.zeros_like(t), np.full_like(t, m.sd_modele)
    # Avancement de la dégradation le long de la courbe centrale.
    p_av = avancement(centre, r_inf)
    # Correction et incertitude pondérées par l'avancement : nulles à t = 0, complètes à l'équilibre.
    mu, sd = mu * p_av, sd * p_av
    # Générateur aléatoire à graine fixe (résultats reproductibles).
    rng = np.random.default_rng(1)
    # Chaque tirage : courbe physique x exp(correction + écart aléatoire). Un seul écart par tirage : courbes lisses.
    total = tirs * np.exp(mu + rng.normal(0, 1, (len(tirs), 1)) * sd)
    # Contraintes physiques : E/E0 ≤ 1 et non croissant dans le temps (minimum cumulé).
    total = np.minimum.accumulate(np.minimum(total, 1.0), axis=1)
    # Courbe médiane : courbe centrale corrigée, avec les mêmes contraintes.
    centre_c = np.minimum.accumulate(np.minimum(centre * np.exp(mu), 1.0))
    # Module initial du matériau.
    E0 = meta["E0_MPa"]
    # Si une diffusion existe...
    if c["D"] is not None:
        # ... temps pour 95 % de la prise en eau : 0,2818·h²/D (en heures).
        t95 = TAU_95 * (h_mm / 1000) ** 2 / c["D"] / 3600
    # Sinon (cinétique)...
    else:
        # ... temps pour 95 % de la perte : ln(20)/k.
        t95 = np.log(20) / c["k"]
    # Assemble le résultat. E_bas et E_haut sont les percentiles 2,5 % et 97,5 % des tirages : intervalle de confiance
    # à 95 %.
    res = dict(materiau=cle, meta=meta, type=typ, T=T, HR=HR, a_w=aw, milieu=milieu, h_mm=h_mm, t=t, E0=E0,
               E_phys=centre * E0, E_med=centre_c * E0,
               E_bas=np.percentile(total, 2.5, 0) * E0, E_haut=np.percentile(total, 97.5, 0) * E0,
               M_moyen=c["M"][:-1] if c["M"] is not None else None, M_inf=c["Minf"], D=c["D"], t95_h=t95,
               Tg_humide=c["Tg"], E_inf=r_inf * E0)
    # Ajoute les points de vigilance du cas.
    res["vigilance"] = vigilance(m, res)
    # Renvoie le résultat.
    return res


# ---------- POINTS DE VIGILANCE
# =========================================================================== points de vigilance
# Liste de (niveau, message) ; niveau = critique, attention ou info.
def vigilance(m: Modele, r: dict) -> list[tuple[str, str]]:
    # Liste vide, fiche, domaines et niveau de modèle.
    v, meta, dom, typ = [], m.meta, m.domaine, m.meta["type"]
    # Conditions du cas et durée maximale demandée.
    T, aw, h, t_max = r["T"], r["a_w"], r["h_mm"], r["t"][-1]
    # Données non relues sur le PDF...
    if meta["verification"] == "non_controle":
        # ... alerte critique.
        v.append(("critique", "Données reprises d'une extraction tierce, NON recontrôlées sur le PDF source."))
    # Température non publiée par la source...
    if meta.get("T_inconnue"):
        # ... alerte critique.
        v.append(("critique", "Source sans température chiffrée (« ambiante ») : la température saisie n'est pas "
                              "prise en compte, la courbe vaut pour la température du laboratoire."))
    # PA6 Goodfellow hors de 25 °C : la vitesse de diffusion est extrapolée...
    if m.cle == "PA6_GOODFELLOW" and abs(T - 25) > 3:
        # ... alerte critique qui indique dans quel sens se trompe la courbe.
        v.append(("critique", f"Cinétique de diffusion mesurée à 25 °C seulement : D(T) non identifiable, la vitesse "
                              f"tracée est celle de 25 °C ({'trop lente' if T > 25 else 'trop rapide'} à {T:g} °C). "
                              f"L'équilibre tient compte de la température."))
    # Époxy Gibhardt :
    if m.cle == "EPOXY_HEXION_RIMR135":
        # plage de température de la diffusion ;
        lo, hi = dom["T_diffusion"]
        # hors de cette plage...
        if not lo <= T <= hi:
            # ... Arrhenius extrapolé.
            v.append(("attention", f"Arrhenius extrapolé : D mesuré de {lo:g} à {hi:g} °C, demandé {T:g} °C."))
        # Équilibre hors de 8-50 °C...
        if not dom["T_iso"][0] <= T <= dom["T_iso"][1]:
            # ... M∞ plafonné.
            v.append(("attention", "Équilibre hors de 8-50 °C : M∞ plafonné à la valeur mesurée la plus proche."))
    # Niveau « cinetique » hors de la plage de températures mesurée...
    if typ == "cinetique" and "T_domaine" in meta and not meta["T_domaine"][0] <= T <= meta["T_domaine"][1]:
        # ... Arrhenius extrapolé.
        v.append(("attention", f"Arrhenius extrapolé hors de {meta['T_domaine'][0]:g}-{meta['T_domaine'][1]:g} °C."))
    # Contrôles propres au niveau « chaine » :
    if typ == "chaine":
        # plage d'activité de l'eau de l'isotherme ;
        lo, hi = dom["aw_iso"]
        # hors de cette plage (avec une marge de 0,03)...
        if not lo - 0.03 <= aw <= hi + 0.03:
            # ... isotherme extrapolée.
            v.append(("attention", f"Activité de l'eau {aw:.2f} hors de l'isotherme mesurée ({lo:.2f}-{hi:.2f})."))
        # Teneur d'équilibre au-delà des mesures de module...
        if r["M_inf"] > dom["M_module"][1] * 1.03:
            # ... relation E(M) extrapolée.
            v.append(("attention", f"Teneur en eau à l'équilibre {r['M_inf']:.2f} % au-delà des mesures de module "
                                   f"({dom['M_module'][1]:.2f} %) : E(M) extrapolée."))
        # PA6 Akulon en immersion...
        if m.cle == "PA6_AKULON_F223D" and r["milieu"] != "vapeur":
            # ... assimilée à 99,9 %HR.
            v.append(("attention", "Immersion assimilée à l'atmosphère à 99.9 %HR (seule mesure proche)."))
        # Tg humide à l'équilibre.
        tg = r["Tg_humide"]
        # Si elle est connue :
        if tg is not None:
            # au-dessus et régime couvert par les mesures...
            if T >= tg and meta.get("regime_caoutchoutique_mesure"):
                # ... simple attention ;
                v.append(("attention", f"Au-dessus de la Tg humide ({tg:.0f} °C) : phase amorphe caoutchoutique ; "
                                       f"régime couvert par les mesures, diffusion non fickienne."))
            # au-dessus et régime NON couvert...
            elif T >= tg:
                # ... alerte critique ;
                v.append(("critique", f"Vieillissement AU-DESSUS de la Tg humide ({tg:.0f} °C) : changement de régime "
                                      f"non couvert par les mesures de module."))
            # à moins de 20 °C...
            elif T >= tg - 20:
                # ... attention.
                v.append(("attention", f"À moins de 20 °C de la Tg humide ({tg:.0f} °C)."))
    # Niveau « apparent » avec plusieurs milieux, humidité hors de la plage mesurée...
    if typ == "apparent" and m.domaine["aw"][0] < m.domaine["aw"][1] and not (
            m.domaine["aw"][0] <= aw <= m.domaine["aw"][1]):
        # ... loi en a_wⁿ extrapolée.
        v.append(("attention", "Humidité hors des milieux mesurés : loi Δ∞ ∝ a_wⁿ extrapolée."))
    # Épaisseur différente de plus de 10 % de celle des éprouvettes...
    if typ != "cinetique" and meta["h_ref_mm"] and abs(h - meta["h_ref_mm"]) / meta["h_ref_mm"] > 0.1:
        # ... effet calculé par Fick, non vérifié.
        v.append(("attention", f"Épaisseur {h:g} mm différente des éprouvettes ({meta['h_ref_mm']:g} mm) : effet "
                               f"calculé par la loi de Fick (temps ∝ h²), non vérifié pour ce matériau."))
    # Niveau « cinetique » :
    if typ == "cinetique":
        # ... l'épaisseur n'intervient pas.
        v.append(("info", "L'épaisseur n'intervient pas dans ce modèle (dégradation volumique lente)."))
    # Durée au-delà de la dernière mesure de module...
    if t_max > meta["duree_module_max_h"]:
        # ... partie tracée en pointillé.
        v.append(("attention", f"Durée au-delà des mesures de module ({meta['duree_module_max_h']:g} h) : "
                               f"partie tracée en pointillé."))
    # Au-delà de la durée sans hydrolyse démontrée...
    if meta.get("duree_sans_hydrolyse_h") and t_max > meta["duree_sans_hydrolyse_h"]:
        # ... alerte critique.
        v.append(("critique", f"Au-delà de {meta['duree_sans_hydrolyse_h']:g} h, l'absence d'hydrolyse n'est plus "
                              f"démontrée : la dégradation chimique irréversible n'est PAS modélisée."))
    # Sinon, pour les modèles fondés sur l'eau...
    elif typ in ("chaine", "apparent") and not meta.get("duree_sans_hydrolyse_h"):
        # ... rappel : hydrolyse non modélisée.
        v.append(("info", "Hydrolyse et dégradation chimique non modélisées."))
    # Équilibre non atteint dans la durée demandée...
    if r["t95_h"] > t_max:
        # ... information.
        v.append(("info", f"Équilibre non atteint sur la durée demandée (95 % de l'évolution à {r['t95_h']:.0f} h)."))
    # Module de flexion...
    if "FLEXION" in meta["propriete"]:
        # ... attention.
        v.append(("attention", "Module de FLEXION (pas de traction)."))
    # Rappel : module à la température d'essai des sources.
    v.append(("info", f"Module à la température d'essai des sources ({meta.get('T_essai_module', 'non précisée')}), "
                      f"pas à la température de service."))
    # Rappel : état initial sec.
    v.append(("info", "État initial supposé SEC (E0 = module avant exposition)."))
    # Ajoute les notes de la fiche du matériau.
    v.extend(("info", n) for n in meta.get("notes", []))
    # Renvoie la liste en retirant les messages vides.
    return [x for x in v if x[1]]


# ---------- VALIDATION ET RAPPORT
# =========================================================================== validation et rapport
# Compare le modèle central (sans correction GP) à chaque mesure E(t) d'un matériau.
def validation(cle: str) -> pd.DataFrame:
    # Modèle ajusté.
    m = modele(cle)
    # Mesures.
    d = _mesures_temps(cle)
    # Aucune mesure...
    if d.empty:
        # ... tableau vide.
        return pd.DataFrame()
    # Lignes du tableau de validation.
    lignes = []
    # Pour chaque mesure :
    for _, q in d.iterrows():
        # température de calcul ;
        T = q["T_C"] if pd.notna(q["T_C"]) else (m.meta.get("T_essais") or 23.0)
        # épaisseur de calcul ;
        h = q["h_mm"] if pd.notna(q["h_mm"]) else (m.meta["h_ref_mm"] or 1.0)
        # E/E0 mesuré ;
        rap = q["rapport"]
        # E/E0 prédit par le modèle central au même temps ;
        pred = COURBE[m.meta["type"]](m.Pc, T, q["a_w"], h, [q["x"]])["r"][0]
        # ajoute la ligne (résidu = mesuré − prédit ; indépendant = mesure jamais utilisée pour ajuster).
        lignes.append(dict(materiau=cle, T_C=q["T_C"], a_w=q["a_w"], h_mm=q["h_mm"], Temps_h=q["x"],
                           E_mesure_MPa=q["y"], rapport_mesure=rap, rapport_modele=pred, residu=rap - pred,
                           independant=bool(m.meta.get("validation_independante", False))))
    # Renvoie le tableau.
    return pd.DataFrame(lignes)


# Produit les tableaux et figures du rapport dans results/ et figures/.
def rapport(sortie: Path = RACINE) -> None:
    # Import de matplotlib seulement ici (inutile pour l'application).
    import matplotlib
    # Moteur de dessin sans fenêtre (fichiers PNG).
    matplotlib.use("Agg")
    # Interface de tracé.
    import matplotlib.pyplot as plt
    # Crée le dossier results/ s'il n'existe pas.
    (sortie / "results").mkdir(parents=True, exist_ok=True)
    # Crée le dossier figures/.
    (sortie / "figures").mkdir(parents=True, exist_ok=True)
    # Listes des paramètres et des tableaux de validation.
    rows, vals = [], []
    # Pour chaque matériau prédictible...
    for cle in predictibles():
        # ... modèle ajusté ;
        m = modele(cle)
        # ... chaque paramètre devient une ligne du tableau.
        for nom, (v, s, u, prov) in m.params.items():
            rows.append(dict(materiau=cle, type=m.meta["type"], parametre=nom, valeur=v, ecart_type=s, unite=u,
                             provenance=prov))
        # ... tableau de validation.
        vals.append(validation(cle))
    # Écrit results/hybride_parametres.csv.
    pd.DataFrame(rows).to_csv(sortie / "results" / "hybride_parametres.csv", index=False)
    # Assemble les validations non vides.
    val = pd.concat([v for v in vals if len(v)])
    # Écrit results/hybride_validation.csv.
    val.to_csv(sortie / "results" / "hybride_validation.csv", index=False)
    # En-tête du tableau imprimé dans le terminal.
    print(f"{'matériau':24s} {'type':10s} {'points E(t)':>11s} {'RMSE E/E0':>10s}  validation")
    # Pour chaque matériau :
    for cle in predictibles():
        # mesures vieillies ;
        g = val[(val["materiau"] == cle) & (val["Temps_h"] > 0)] if len(val) else pd.DataFrame()
        # RMSE en E/E0 (ou tiret s'il n'y a pas de mesure) ;
        rm = f"{np.sqrt(np.mean(g['residu'] ** 2)):.3f}" if len(g) else "-"
        # niveau de modèle ;
        typ = MATERIAUX[cle]["type"]
        # nature du chiffre : validation indépendante, ajustement, ou aucune mesure ;
        quoi = ("INDÉPENDANTE (jamais ajustée)" if len(g) and g["independant"].all()
                else "aucune mesure E(t)" if not len(g) else "ajustement (pas une validation)")
        # ligne imprimée.
        print(f"{cle:24s} {typ:10s} {len(g):>11d} {rm:>10s}  {quoi}")

    # ---- Planche : chaque matériau, modèle contre mesures.
    # ---- planche : chaque matériau, modèle contre mesures E(t)
    # Matériaux qui ont des mesures E(t).
    cles = [c for c in predictibles() if len(val) and (val["materiau"] == c).any()]
    # 3 graphiques par ligne.
    nc = 3
    # Grille de graphiques.
    fig, axs = plt.subplots(int(np.ceil(len(cles) / nc)), nc, figsize=(15, 4 * int(np.ceil(len(cles) / nc))))
    # Pour chaque matériau et son graphique :
    for ax, cle in zip(axs.flat, cles):
        # fiche ;
        meta = MATERIAUX[cle]
        # mesures du matériau ;
        g = val[val["materiau"] == cle]
        # pour chaque condition mesurée (température, activité de l'eau, épaisseur) :
        for i, ((T, aw, h), gg) in enumerate(g.groupby(["T_C", "a_w", "h_mm"], dropna=False)):
            # température de calcul ;
            Tq = T if pd.notna(T) else (meta.get("T_essais") or 23.0)
            # épaisseur de calcul ;
            hq = h if pd.notna(h) else (meta["h_ref_mm"] or 1.0)
            # milieu correspondant à l'activité de l'eau ;
            mil = "vapeur" if aw < 0.99 and cle != "PA6_GOODFELLOW" else (
                "immersion_eau_de_mer" if abs(aw - 0.98) < 1e-6 and cle == "PA6_GOODFELLOW" else "immersion_eau_douce")
            # calcul de la prédiction un peu au-delà de la dernière mesure...
            try:
                r = predire(cle, Tq, aw * 100, mil, hq, max(gg["Temps_h"].max() * 1.5, 50))
            # ... sauf si le cas est refusé ;
            except NonPredictible:
                continue
            # étiquette de la légende ;
            lab = (f"{T:g} °C, " if pd.notna(T) else "") + f"a_w {aw:.2f}"
            # bande d'incertitude ;
            ax.fill_between(r["t"][1:], r["E_bas"][1:] / r["E0"], r["E_haut"][1:] / r["E0"], color=f"C{i}", alpha=0.15)
            # courbe médiane ;
            ax.semilogx(r["t"][1:], r["E_med"][1:] / r["E0"], color=f"C{i}", label=lab)
            # mesures (t = 0 placé à 0,2 h pour l'échelle logarithmique).
            ax.plot(gg["Temps_h"].clip(lower=0.2), gg["rapport_mesure"], "o", color=f"C{i}", ms=4)
        # Titre et axes.
        ax.set(title=f"{cle} ({meta['type']})", xlabel="t (h)", ylabel="E/E0", ylim=(0, 1.1))
        # Légende.
        ax.legend(fontsize=7)
    # Cache les cases vides de la grille.
    for ax in list(axs.flat)[len(cles):]:
        ax.axis("off")
    # Enregistre la planche.
    fig.tight_layout(); fig.savefig(sortie / "figures" / "hybride_tous_materiaux.png", dpi=130); plt.close(fig)

    # ---- Figure des effets de l'épaisseur, de l'humidité et de la température.
    # ---- effets de h, HR et T (matériaux où ils sont identifiables)
    # Trois graphiques côte à côte.
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    # Effet de l'épaisseur (PA6 Goodfellow, eau de mer).
    for h in (1, 2, 4, 8):
        # Prédiction pour chaque épaisseur.
        r = predire("PA6_GOODFELLOW", 25, 100, "immersion_eau_de_mer", h, 20000)
        # Tracé.
        axs[0].semilogx(r["t"][1:], r["E_med"][1:], label=f"h = {h} mm")
    # Titre et axes.
    axs[0].set(title="PA6 Goodfellow, eau de mer 25 °C : épaisseur", xlabel="t (h)", ylabel="E (MPa)")
    # Effet de l'humidité (PA6 Akulon).
    for hr in (36, 57, 73, 99):
        # Prédiction pour chaque humidité.
        r = predire("PA6_AKULON_F223D", 23, hr, "vapeur", 4, 20000)
        # Tracé.
        axs[1].semilogx(r["t"][1:], r["E_med"][1:], label=f"HR = {hr} %")
    # Titre.
    axs[1].set(title="PA6 Akulon, 4 mm : humidité", xlabel="t (h)")
    # Effet de la température (époxy Gibhardt).
    for T in (8, 25, 40, 50):
        # Prédiction pour chaque température.
        r = predire("EPOXY_HEXION_RIMR135", T, 100, "immersion_eau_douce", 2, 20000)
        # Tracé.
        axs[2].semilogx(r["t"][1:], r["E_med"][1:], label=f"T = {T} °C")
    # Titre.
    axs[2].set(title="Époxy RIMR 135, immersion, 2 mm : température", xlabel="t (h)")
    # Légende de chaque graphique.
    for a in axs:
        a.legend(fontsize=8)
    # Enregistre la figure.
    fig.tight_layout(); fig.savefig(sortie / "figures" / "hybride_effets_T_HR_h.png", dpi=150); plt.close(fig)

    # ---- Validation indépendante du PA6 Goodfellow.
    # ---- validation indépendante PA6 Goodfellow
    # Prédiction dans les conditions de la Fig. 7a de Le Gac.
    r = predire("PA6_GOODFELLOW", 25, 100, "immersion_eau_de_mer", 2.0, 720)
    # Mesures correspondantes.
    g = val[val["materiau"] == "PA6_GOODFELLOW"]
    # Figure.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Chaîne physique seule (sans correction).
    ax.plot(r["t"] / 24, r["E_phys"], lw=2, label="chaîne physique seule")
    # Mesures jamais utilisées pour l'ajustement.
    ax.plot(g["Temps_h"] / 24, g["E_mesure_MPa"], "ks", label="Le Gac Fig. 7a (jamais ajusté)")
    # Axes et titre.
    ax.set(xlabel="Temps d'immersion (jours)", ylabel="E (MPa)", title="PA6 Goodfellow : validation indépendante")
    # Légende.
    ax.legend(fontsize=8)
    # Enregistre la figure.
    fig.tight_layout(); fig.savefig(sortie / "figures" / "hybride_validation_PA6.png", dpi=150); plt.close(fig)
    # Message de fin.
    print("\nÉcrit : results/hybride_parametres.csv, results/hybride_validation.csv, figures/hybride_*.png")


# ---------- UTILISATION EN LIGNE DE COMMANDE
def main(argv=None) -> int:
    # Analyseur d'options ; la description est la docstring du module.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --rapport : produit tableaux et figures.
    ap.add_argument("--rapport", action="store_true")
    # --materiau : clé du matériau, limitée à la liste existante.
    ap.add_argument("--materiau", choices=list(MATERIAUX))
    # --T : température en °C.
    ap.add_argument("--T", type=float)
    # --HR : humidité relative en % (100 par défaut).
    ap.add_argument("--HR", type=float, default=100)
    # --milieu : vapeur, immersion_eau_douce ou immersion_eau_de_mer.
    ap.add_argument("--milieu", choices=list(MILIEUX), default="vapeur")
    # --epaisseur : épaisseur en mm.
    ap.add_argument("--epaisseur", type=float)
    # --temps : durée d'exposition en heures.
    ap.add_argument("--temps", type=float)
    # Lecture des options.
    a = ap.parse_args(argv)
    # Mode rapport...
    if a.rapport:
        # ... exécution...
        rapport()
        # ... fin sans erreur.
        return 0
    # Options obligatoires manquantes...
    if None in (a.materiau, a.T, a.epaisseur, a.temps):
        # ... message d'erreur et arrêt.
        ap.error("--materiau, --T, --epaisseur et --temps sont requis (ou --rapport)")
    # Tente la prédiction...
    try:
        r = predire(a.materiau, a.T, a.HR, a.milieu, a.epaisseur, a.temps)
    # ... si le cas est refusé...
    except NonPredictible as e:
        # ... affiche la raison...
        print(f"NON PRÉDICTIBLE : {e}")
        # ... et renvoie le code d'erreur 1.
        return 1
    # Affiche le matériau, le niveau de modèle et E0.
    print(f"{r['meta']['nom']} [{TYPES[r['type']]}] ; E0 = {r['E0']} MPa ({r['meta']['E0_source']})")
    # Affiche le module à la durée demandée, son intervalle de confiance, l'équilibre et le temps à 95 %.
    print(f"À {a.temps:g} h : E = {r['E_med'][-1]:.0f} MPa (IC 95 % {r['E_bas'][-1]:.0f} - {r['E_haut'][-1]:.0f}) ; "
          f"équilibre {r['E_inf']:.0f} MPa ; 95 % de l'évolution à {r['t95_h']:.0f} h")
    # Affiche chaque point de vigilance.
    for niv, msg in r["vigilance"]:
        print(f"  [{niv.upper()}] {msg}")
    # Fin sans erreur.
    return 0


# Exécuté seulement quand le fichier est lancé directement (pas quand il est importé par l'application).
if __name__ == "__main__":
    # Lance main() et transmet son code de sortie au terminal.
    sys.exit(main())
