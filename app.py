# Déclare l'encodage UTF-8 du fichier (accents et symboles dans les textes affichés).
# -*- coding: utf-8 -*-
# Docstring du module : rôle de l'application et façon de la lancer.
"""
Application Streamlit - vieillissement humide d'une pièce polymère (modèle HYBRIDE, tous les matériaux).

Lancer depuis la racine du dépôt :
    streamlit run app.py          (ou : python app.py)

L'utilisateur choisit le matériau, le milieu, T, HR, l'épaisseur et la durée. Le module initial E0 est
celui du matériau (donnée publiée), jamais saisi. Tout le calcul est dans src/hybride.py.
La page « Comparaison XGBoost » (dossier pages/) garde le modèle purement data-driven.
"""
# Permet les annotations de type modernes sur toutes les versions de Python.
from __future__ import annotations

# sys : accès au chemin de recherche des modules (sys.path) et aux arguments de lancement.
import sys
# Path : manipulation des chemins de fichiers.
from pathlib import Path

# Dossier qui contient app.py, c'est-à-dire la racine du dépôt.
RACINE_APP = Path(__file__).resolve().parent
# Si ce dossier n'est pas dans le chemin de recherche des modules...
if str(RACINE_APP) not in sys.path:
    # ... on l'ajoute en tête, pour que « from src import hybride » fonctionne quel que soit le dossier de lancement.
    sys.path.insert(0, str(RACINE_APP))


# Renvoie Vrai si le script est exécuté par Streamlit, Faux s'il est lancé par « python app.py ».
def _sous_streamlit() -> bool:
    # Tente d'importer la fonction de contexte de Streamlit (emplacement des versions récentes)...
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    # ... sinon (anciennes versions de Streamlit)...
    except ImportError:
        # ... l'importe depuis son ancien emplacement.
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    # Un contexte d'exécution existe seulement sous Streamlit ; suppress_warning évite un avertissement inutile.
    return get_script_run_ctx(suppress_warning=True) is not None


# Lancé par « python app.py » ou le bouton Run de PyCharm (hors Streamlit) :
if __name__ == "__main__" and not _sous_streamlit():       # « python app.py » ou bouton Run de PyCharm
    # importe l'interface en ligne de commande de Streamlit ;
    from streamlit.web import cli as stcli
    # remplace les arguments par l'équivalent de « streamlit run app.py » ;
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    # lance Streamlit (qui rouvre ce fichier correctement) et termine ce processus.
    sys.exit(stcli.main())

# matplotlib : bibliothèque de graphiques (noqa : import volontairement après le bloc de relance).
import matplotlib  # noqa: E402
# Moteur de dessin sans fenêtre : les figures sont rendues en images dans la page web.
matplotlib.use("Agg")
# Interface de tracé de matplotlib.
import matplotlib.pyplot as plt  # noqa: E402
# numpy : calculs sur tableaux.
import numpy as np  # noqa: E402
# pandas : tableaux affichés et export CSV.
import pandas as pd  # noqa: E402
# streamlit : construction de l'interface web (widgets, onglets, graphiques).
import streamlit as st  # noqa: E402

# Moteur de calcul du projet (lois physiques, incertitudes, points de vigilance), importé sous le nom court H.
from src import hybride as H  # noqa: E402

# Réglages de la page : titre de l'onglet du navigateur, icône, mise en page pleine largeur.
st.set_page_config(page_title="Vieillissement humide - module", page_icon="💧", layout="wide")

# Libellés lisibles des trois milieux, affichés dans la barre latérale.
LIBELLES_MILIEU = {"vapeur": "Atmosphère humide (HR contrôlée)", "immersion_eau_douce": "Immersion, eau douce",
                   "immersion_eau_de_mer": "Immersion, eau de mer"}
# Échéances affichées dans le tableau de résultats (durée en heures, libellé).
JALONS = [(24, "1 jour"), (168, "1 semaine"), (720, "1 mois"), (4380, "6 mois"), (8760, "1 an"),
          (43800, "5 ans"), (87600, "10 ans")]
# Mention affichée selon que les données ont été relues sur le PDF ou non.
BADGE = {"controle_pdf": "✅ relu sur le PDF", "non_controle": "⚠️ extraction tierce non recontrôlée"}


# Mise en cache Streamlit d'une ressource : les modèles ne sont ajustés qu'une fois pour toute la session du serveur.
@st.cache_resource(show_spinner="Ajustement des lois sur les données sourcées (une fois)...")
# Ajuste les modèles de tous les matériaux prédictibles.
def modeles() -> dict:
    # Dictionnaire clé du matériau -> modèle ajusté.
    return {c: H.modele(c) for c in H.predictibles()}


# Mise en cache des résultats : un même cas (mêmes entrées) n'est calculé qu'une fois.
@st.cache_data(show_spinner="Calcul de la prédiction et des incertitudes...")
# Calcule la prédiction d'un cas.
def calcul(cle, T, HR, milieu, h, duree):
    # Délègue entièrement le calcul au moteur hybride.
    return H.predire(cle, T, HR, milieu, h, duree)


# Sélectionne les mesures publiées proches du cas demandé, pour les superposer à la courbe.
def mesures_proches(cle, T, a_w, h):
    """Mesures E(t) publiées dans des conditions proches (affichées pour comparaison)."""
    # Mesures E(t) du matériau, normalisées.
    d = H._mesures_temps(cle)
    # Aucune mesure...
    if d.empty:
        # ... tableau vide.
        return d
    # Fiche du matériau.
    meta = H.MATERIAUX[cle]
    # Garde les mesures dont l'activité de l'eau est à ±0,03 de celle du cas.
    ok = (d["a_w"] - a_w).abs() <= 0.03
    # Si la température est connue pour ce matériau...
    if not meta.get("T_inconnue") and d["T_C"].notna().any():
        # ... garde aussi seulement les mesures à ±3 °C.
        ok &= (d["T_C"] - T).abs() <= 3
    # Si l'épaisseur des mesures est connue...
    if d["h_mm"].notna().any():
        # ... garde les mesures à ±10 % d'épaisseur (ou d'épaisseur non renseignée).
        ok &= ((d["h_mm"] - h).abs() / h <= 0.1) | d["h_mm"].isna()
    # Renvoie les mesures retenues.
    return d[ok]


# Texte affiché dans la liste des matériaux.
def nom_materiau(c):
    # Fiche du matériau.
    m = H.MATERIAUX[c]
    # Nom et famille, suivis d'un symbole d'alerte si les données n'ont pas été relues sur le PDF.
    return f"{m['nom']}  ·  {m['famille']}" + ("" if m["verification"] == "controle_pdf" else "  ·  ⚠️")


# ========== BARRE LATÉRALE : SAISIE DU CAS
# ===================================================================================== saisie
# Titre de la page.
st.title("Vieillissement humide d'une pièce polymère : évolution du module d'Young")
# Texte d'introduction sous le titre.
st.caption("Chaque matériau est prédit avec le niveau de modèle que ses données publiées permettent : chaîne "
           "physique complète, cinétique apparente ou cinétique de dégradation. Les cas que les données ne "
           "permettent pas de prédire sont refusés avec leur raison, au lieu d'être extrapolés en silence.")
# Déclenche l'ajustement de tous les modèles au premier affichage (ensuite servis depuis le cache).
modeles()

# Tout ce bloc s'affiche dans la barre latérale.
with st.sidebar:
    # Titre de la barre latérale.
    st.header("Pièce et environnement")
    # Liste déroulante des 11 matériaux ; format_func choisit le texte affiché pour chaque clé.
    cle = st.selectbox("Matériau", list(H.MATERIAUX), format_func=nom_materiau)
    # Fiche du matériau choisi.
    meta = H.MATERIAUX[cle]
    # Affiche le niveau de modèle, le statut de vérification des données et la source.
    st.caption(f"**Modèle** : {H.TYPES[meta['type']]}  \n**Données** : {BADGE[meta['verification']]}  \n"
               f"**Source** : {meta['source']}")
    # Les champs de saisie ne s'affichent que pour un matériau prédictible.
    if meta["type"] != "donnees":
        # Affiche E0 (propre au matériau, jamais saisi), le type de module et la Tg sèche si elle est connue.
        st.caption(f"E0 = **{meta['E0_MPa']:.0f} MPa** ({meta['E0_source']}) ; module de {meta['propriete']}"
                   + (f" ; Tg sèche {meta['Tg0_C']} °C" if meta.get("Tg0_C") else ""))
        # Milieux proposés : seulement ceux mesurés pour ce matériau.
        milieux = [m for m in LIBELLES_MILIEU if m in meta["milieux"]]
        # Boutons radio de choix du milieu.
        milieu = st.radio("Milieu", milieux, format_func=LIBELLES_MILIEU.get)
        # Si la source ne publie pas la température...
        if meta.get("T_inconnue"):
            # ... avertissement immédiat.
            st.warning("Température non publiée par la source : la courbe vaut pour l'ambiante du laboratoire.")
        # Champ de température ; pré-rempli avec la température des essais quand elle est unique, sinon vide.
        T = st.number_input("Température (°C)", value=meta.get("T_essais"), min_value=-20.0, max_value=120.0,
                            step=1.0, placeholder="à renseigner")
        # En atmosphère humide...
        if milieu == "vapeur":
            # ... champ d'humidité relative, pré-rempli avec l'unique humidité mesurée s'il n'y en a qu'une.
            HR = st.number_input("Humidité relative (%)", value=meta["aw_cible"] * 100 if "aw_cible" in meta else None,
                                 min_value=1.0, max_value=100.0, step=1.0, placeholder="à renseigner")
        # En immersion...
        else:
            # ... HR n'a pas de sens : fixée à 100 (l'activité de l'eau vient du milieu).
            HR = 100.0
            # Affiche l'activité de l'eau du milieu.
            st.caption(f"Activité de l'eau = {H.MILIEUX[milieu]:.2f}")
        # Pour une cinétique de dégradation...
        if meta["type"] == "cinetique":
            # ... l'épaisseur n'est pas demandée (sans effet) : valeur des éprouvettes, ou 1 si inconnue.
            h = meta["h_ref_mm"] or 1.0
            # Explication affichée à la place du champ.
            st.caption("Épaisseur : sans effet dans ce modèle (dégradation volumique lente)"
                       + (f" ; éprouvettes de {meta['h_ref_mm']:g} mm." if meta["h_ref_mm"] else " ; non publiée."))
        # Sinon...
        else:
            # ... champ d'épaisseur, pré-rempli avec l'épaisseur des éprouvettes.
            h = st.number_input("Épaisseur de la pièce (mm)", value=meta["h_ref_mm"], min_value=0.05, max_value=50.0,
                                step=0.5, format="%.2f", placeholder="à renseigner",
                                help="Plaque exposée sur ses deux grandes faces. Valeur proposée = épaisseur des éprouvettes.")
        # Choix de l'unité de la durée.
        unite = st.radio("Unité de durée", ["heures", "jours", "années"], horizontal=True, index=1)
        # Champ de durée, laissé vide pour qu'aucune valeur par défaut ne passe pour un choix.
        duree = st.number_input(f"Durée d'exposition ({unite})", value=None, min_value=0.01, step=1.0,
                                placeholder="à renseigner")
        # Case à cocher : axe des temps logarithmique ou linéaire.
        echelle_log = st.checkbox("Axe des temps logarithmique", value=True)
    # Encadré repliable listant les sources inexploitables.
    with st.expander("Matériaux des sources non exploitables"):
        # Pour chacune...
        for nom, raison in H.NON_EXPLOITABLES.items():
            # ... nom et raison.
            st.markdown(f"- **{nom}** : {raison}")

# Trois onglets dans la page principale.
onglet_pred, onglet_modele, onglet_valid = st.tabs(["Prédiction", "Lois et paramètres du matériau",
                                                     "Tous les matériaux"])

# ========== ONGLET PRÉDICTION
# ===================================================================================== prédiction
# Contenu de l'onglet « Prédiction ».
with onglet_pred:
    # Matériau sans cinétique (Goetjes) :
    if meta["type"] == "donnees":
        # message d'explication ;
        st.error(f"Pas de prédiction dans le temps : {meta['raison']}")
        # données du matériau ;
        d = H._donnees(cle)
        # modules mesurés à 168 h ;
        d = d[d["grandeur"] == "module_168h"]
        # texte d'accompagnement ;
        st.markdown("Modules **mesurés** après 168 h (E0 non publié) : seule donnée disponible.")
        # tableau croisé : températures en lignes, humidités en colonnes ;
        tab = d.pivot_table(index="T_C", columns="a_w", values="y")
        # intitulés des colonnes en %HR ;
        tab.columns = [f"{c * 100:.0f} %HR" for c in tab.columns]
        # intitulé des lignes ;
        tab.index.name = "T (°C)"
        # affichage du tableau ;
        st.dataframe(tab.round(0))
        # rappel de la source ;
        st.caption(f"Source : {meta['source']} ({BADGE[meta['verification']]}).")
        # arrêt du script ici (rien d'autre à afficher).
        st.stop()
    # Liste des champs obligatoires encore vides.
    manquants = [n for n, v in (("température", T), ("humidité", HR), ("épaisseur", h), ("durée", duree)) if v is None]
    # S'il en manque...
    if manquants:
        # ... message...
        st.info("Renseigner : " + ", ".join(manquants) + " (barre latérale).")
        # ... et arrêt.
        st.stop()
    # Durée convertie en heures selon l'unité choisie.
    duree_h = duree * {"heures": 1, "jours": 24, "années": 8760}[unite]
    # Tente le calcul...
    try:
        # ... de la prédiction (mise en cache).
        r = calcul(cle, float(T), float(HR), milieu, float(h), float(duree_h))
    # Si le moteur refuse le cas...
    except H.NonPredictible as e:
        # ... affiche la raison du refus...
        st.error(f"Cas non prédictible avec les données de ce matériau : {e}")
        # ... et arrête.
        st.stop()
    # Grille de temps et E0.
    t, E0 = r["t"], r["E0"]
    # Quatre colonnes d'indicateurs.
    c1, c2, c3, c4 = st.columns(4)
    # Indicateur 1 : module à la durée demandée et variation par rapport à E0 (en rouge si baisse).
    c1.metric(f"E après {duree:g} {unite}", f"{r['E_med'][-1]:.0f} MPa",
              f"{100 * (r['E_med'][-1] / E0 - 1):+.0f} % de E0", delta_color="inverse")
    # Intervalle de confiance à 95 % sous l'indicateur.
    c1.caption(f"IC 95 % : {r['E_bas'][-1]:.0f} - {r['E_haut'][-1]:.0f} MPa")
    # Indicateur 2 : module à long terme.
    c2.metric("E à long terme (modèle)", f"{r['E_inf']:.0f} MPa")
    # Rappel de E0.
    c2.caption(f"E0 = {E0:.0f} MPa (donnée du matériau)")
    # Si la teneur en eau d'équilibre est connue...
    if r["M_inf"] is not None:
        # ... indicateur 3 : teneur en eau d'équilibre ;
        c3.metric("Teneur en eau à l'équilibre", f"{r['M_inf']:.2f} %")
    # sinon...
    else:
        # ... mention « non mesurée ».
        c3.metric("Teneur en eau", "non mesurée")
    # Temps pour atteindre 95 % de l'évolution.
    c3.caption(f"95 % de l'évolution après {r['t95_h']:.0f} h ({r['t95_h'] / 24:.0f} j)")
    # Si la Tg humide est calculable...
    if r["Tg_humide"] is not None:
        # ... indicateur 4 : Tg humide ;
        c4.metric("Tg humide à l'équilibre", f"{r['Tg_humide']:.0f} °C")
        # ... avec la Tg sèche et le coefficient de diffusion.
        c4.caption(f"Tg sèche {meta['Tg0_C']} °C" + (f" ; D = {r['D']:.2e} m²/s" if r["D"] else ""))
    # Sinon...
    else:
        # ... mention « non mesurée »...
        c4.metric("Tg humide", "non mesurée")
        # ... et D ou la nature du modèle.
        c4.caption(f"D = {r['D']:.2e} m²/s" if r["D"] else "dégradation non limitée par la diffusion")

    # Points de vigilance du cas.
    vig = r["vigilance"]
    # Alertes critiques en rouge, attentions en orange...
    for niveau, fonction in (("critique", st.error), ("attention", st.warning)):
        # ... une boîte par message...
        for _, msg in [v for v in vig if v[0] == niveau]:
            # ... avec l'icône correspondante.
            fonction(msg, icon="⛔" if niveau == "critique" else "⚠️")

    # Vrai si le modèle calcule une teneur en eau (on trace alors un second graphique).
    avec_eau = r["M_moyen"] is not None
    # Figure à un ou deux graphiques ; le premier est deux fois plus large.
    fig, axes = plt.subplots(1, 2 if avec_eau else 1, figsize=(12 if avec_eau else 9, 4.3),
                             gridspec_kw={"width_ratios": [2, 1]} if avec_eau else None)
    # Graphique principal.
    ax = axes[0] if avec_eau else axes
    # Temps strictement positifs (l'échelle logarithmique exclut 0).
    pos = t > 0
    # Temps couverts par des mesures de module.
    fiable = t <= meta["duree_module_max_h"]
    # Bande d'incertitude à 95 %.
    ax.fill_between(t[pos], r["E_bas"][pos], r["E_haut"][pos], color="C0", alpha=0.2,
                    label="IC 95 % (paramètres + écart au modèle)")
    # Courbe médiane, en trait plein là où des mesures existent.
    ax.plot(t[pos & fiable], r["E_med"][pos & fiable], "C0", lw=2.2, label="prédiction")
    # Temps au-delà des mesures.
    k = pos & ~fiable
    # S'il y en a...
    if k.any():
        # ... leurs positions ;
        j = np.flatnonzero(k)
        # ... on recule d'un point pour que le pointillé raccorde le trait plein ;
        j = np.concatenate([[j[0] - 1], j]) if j[0] > 0 else j
        # ... tracé en pointillé.
        ax.plot(t[j], r["E_med"][j], "C0", lw=2.2, ls=":", label=f"au-delà des mesures (> {meta['duree_module_max_h']:g} h)")
    # Pour une chaîne physique...
    if r["type"] == "chaine":
        # ... courbe physique seule, pour montrer l'effet de la correction.
        ax.plot(t[pos], r["E_phys"][pos], "k--", lw=1, label="physique seule (sans correction)")
    # Mesures publiées proches du cas.
    mes = mesures_proches(cle, float(T), r["a_w"], float(h))
    # S'il y en a...
    if len(mes):
        # ... on les trace (temps 0 décalé pour l'échelle logarithmique, ramenées au même E0).
        ax.plot(mes["x"].clip(lower=t[pos][0]), mes["y"] * E0 / mes["E0_ref"], "rs", ms=6,
                label="mesures publiées (conditions proches)")
    # Ligne horizontale du module à long terme.
    ax.axhline(r["E_inf"], color="0.5", lw=0.8, ls="-.", label="long terme")
    # Si l'option est cochée...
    if echelle_log:
        # ... axe des temps logarithmique.
        ax.set_xscale("log")
    # Étiquettes des axes, limites et titre (température non affichée si elle est inconnue).
    ax.set(xlabel="Temps (h)", ylabel="Module d'Young E (MPa)", ylim=(0, 1.08 * E0),
           title=f"{meta['nom']} - " + ("T ambiante" if meta.get("T_inconnue") else f"{T:g} °C") + ", "
                 + LIBELLES_MILIEU[milieu].lower() + (f" {HR:g} %HR" if milieu == "vapeur" else "")
                 + (f", {h:g} mm" if meta["type"] != "cinetique" else ""))
    # Légende.
    ax.legend(fontsize=7.5, loc="lower left")
    # Second graphique, si une teneur en eau est calculée :
    if avec_eau:
        # teneur en eau moyenne dans le temps ;
        axes[1].plot(t[pos], r["M_moyen"][pos], "C1", lw=2)
        # teneur d'équilibre ;
        axes[1].axhline(r["M_inf"], color="0.5", lw=0.8, ls="-.")
        # échelle logarithmique si demandée ;
        if echelle_log:
            axes[1].set_xscale("log")
        # étiquettes et titre.
        axes[1].set(xlabel="Temps (h)", ylabel="Teneur en eau moyenne (% masse)", title="Prise en eau (Fick)")
    # Ajuste les marges.
    fig.tight_layout()
    # Affiche la figure dans la page.
    st.pyplot(fig)
    # Libère la mémoire de la figure.
    plt.close(fig)

    # Lignes du tableau des échéances.
    lignes = []
    # Pour chaque échéance (et la durée choisie)...
    for th, lib in JALONS + [(duree_h, "durée choisie")]:
        # ... si elle est atteinte dans la durée demandée...
        if th <= duree_h:
            # ... indice du temps de la grille le plus proche ;
            i = int(np.argmin(np.abs(t - th)))
            # ... ligne : temps, module, intervalle de confiance, E/E0 ;
            ligne = {"échéance": lib, "temps (h)": round(float(t[i]), 1), "E (MPa)": round(float(r["E_med"][i])),
                     "IC 95 % bas": round(float(r["E_bas"][i])), "IC 95 % haut": round(float(r["E_haut"][i])),
                     "E/E0": round(float(r["E_med"][i] / E0), 3)}
            # ... teneur en eau si elle est calculée ;
            if avec_eau:
                ligne["eau (%)"] = round(float(r["M_moyen"][i]), 2)
            # ... indique si l'échéance est couverte par des mesures ;
            ligne["couvert par des mesures"] = "oui" if t[i] <= meta["duree_module_max_h"] else "non"
            # ... ajoute la ligne.
            lignes.append(ligne)
    # Affiche le tableau.
    st.dataframe(pd.DataFrame(lignes), hide_index=True)
    # Tableau exporté : courbe complète.
    export = pd.DataFrame({"Temps_h": t, "E_MPa": r["E_med"], "E_IC95_bas_MPa": r["E_bas"],
                           "E_IC95_haut_MPa": r["E_haut"], "E_sur_E0": r["E_med"] / E0})
    # Ajoute la teneur en eau si elle existe.
    if avec_eau:
        # Colonne teneur en eau.
        export["teneur_eau_pct"] = r["M_moyen"]
    # Bouton de téléchargement de la courbe en CSV.
    st.download_button("Télécharger la courbe (CSV)", export.to_csv(index=False).encode(),
                       file_name=f"prediction_{cle}_T{T:g}_HR{HR:g}_h{h:g}.csv", mime="text/csv")
    # Encadré repliable avec les points de vigilance de niveau « info ».
    with st.expander(f"Autres points de vigilance et hypothèses ({sum(v[0] == 'info' for v in vig)})"):
        # Pour chacun...
        for _, msg in [v for v in vig if v[0] == "info"]:
            # ... une puce.
            st.markdown(f"- {msg}")
    # Encadré repliable qui explique le calcul, selon le niveau de modèle du matériau.
    with st.expander("Comment est calculée cette courbe"):
        # Texte choisi dans un dictionnaire selon le type de modèle, suivi de l'explication de l'incertitude.
        st.markdown({
            "chaine": "**Chaîne physique.** Équilibre M∞ = isotherme(T, a_w) ; vitesse D(T, a_w) ; profil d'eau de Fick "
                      "dans la plaque (solution de Crank) ; chaque couche prend le module correspondant à sa teneur "
                      "en eau (relation E(M) mesurée), moyenne dans l'épaisseur ; correction par processus gaussien "
                      "sur les écarts aux mesures E(t) quand elles existent.",
            "apparent": "**Cinétique apparente.** E/E0 = 1 − Δ∞(a_w)·g(F), où F est la prise en eau moyenne de Fick "
                        "(temps ∝ h²/D) et g une forme ajustée ; Δ∞ (perte à saturation) ajusté sur les mesures E(t). "
                        "Pas de relation module / teneur en eau mesurée pour ce matériau.",
            "cinetique": "**Cinétique de dégradation.** E/E0 = r∞ + (1 − r∞)·exp(−k t), k suivant Arrhenius quand "
                         "plusieurs températures ont été mesurées. Dégradation lente (hydrolyse, vieillissement "
                         "physique), non limitée par la diffusion : l'épaisseur n'intervient pas.",
        }[r["type"]] + "\n\n**Incertitude** : 200 tirages des paramètres (covariances des ajustements) combinés à "
                       "l'écart de modèle ; intervalle à 95 %. Contraintes : E(t=0) = E0, E ≤ E0, E(t) non croissant.")

# ========== ONGLET LOIS ET PARAMÈTRES
# ===================================================================================== paramètres
# Contenu de l'onglet.
with onglet_modele:
    # Nom du matériau.
    st.subheader(meta["nom"])
    # Niveau de modèle, statut des données et source.
    st.markdown(f"**Modèle** : {H.TYPES[meta['type']]} — **Données** : {BADGE[meta['verification']]}  \n"
                f"**Source** : {meta['source']}")
    # Matériau sans cinétique...
    if meta["type"] == "donnees":
        # ... raison.
        st.info(meta["raison"])
    # Sinon...
    else:
        # ... modèle ajusté (depuis le cache) ;
        m = modeles()[cle]
        # ... tableau des paramètres ajustés (valeur, écart-type, unité, provenance) ;
        tab = pd.DataFrame([dict(paramètre=k, valeur=v, **{"écart-type": s}, unité=u, provenance=p)
                            for k, (v, s, u, p) in m.params.items()])
        # ... affichage avec un format numérique lisible.
        st.dataframe(tab, hide_index=True, column_config={
            "valeur": st.column_config.NumberColumn(format="%.4g"),
            "écart-type": st.column_config.NumberColumn(format="%.3g")})
        # Si un processus gaussien corrige le modèle...
        if m.gp_info.get("actif"):
            # ... nombre de mesures et amplitude de l'écart physique/mesures.
            st.markdown(f"**Correction GP** : apprise sur {m.gp_info['n']} mesures E(t) ; écart relatif RMS de la "
                        f"physique seule = {m.gp_info['amplitude']:.3f}.")
        # Sinon...
        else:
            # ... écart de modèle utilisé pour l'incertitude.
            st.markdown(f"**Écart de modèle** (relatif) : {m.sd_modele:.3f}.")
    # Encadré repliable : toutes les données du matériau avec leur source.
    with st.expander("Données utilisées (avec sources)"):
        # Tableau des données.
        st.dataframe(H._donnees(cle), hide_index=True)

# ========== ONGLET TOUS LES MATÉRIAUX
# ===================================================================================== synthèse
# Contenu de l'onglet.
with onglet_valid:
    # Texte qui distingue validation indépendante et qualité d'ajustement.
    st.markdown("Synthèse de **tous les matériaux** des sources fournies. Seule la ligne marquée *indépendante* "
                "valide le modèle sur des mesures qui n'ont servi à aucun ajustement ; les autres RMSE mesurent "
                "la qualité d'ajustement, pas la capacité de prédiction.")
    # Lignes du tableau de synthèse.
    lignes = []
    # Pour chaque matériau :
    for c, mm in H.MATERIAUX.items():
        # sans cinétique...
        if mm["type"] == "donnees":
            # ... ligne avec la raison...
            lignes.append({"matériau": mm["nom"], "modèle": H.TYPES[mm["type"]], "données": BADGE[mm["verification"]],
                           "E0 (MPa)": None, "points E(t)": 0, "RMSE E/E0": None, "statut": mm["raison"]})
            # ... et passage au suivant ;
            continue
        # comparaison modèle/mesures ;
        v = H.validation(c)
        # mesures vieillies seulement ;
        vi = v[v["Temps_h"] > 0] if len(v) else v
        # ligne : modèle, données, E0, nombre de points, RMSE, statut (validation indépendante, ajustement ou aucune
        # mesure).
        lignes.append({"matériau": mm["nom"], "modèle": H.TYPES[mm["type"]], "données": BADGE[mm["verification"]],
                       "E0 (MPa)": mm["E0_MPa"], "points E(t)": len(vi),
                       "RMSE E/E0": round(float(np.sqrt(np.mean(vi["residu"] ** 2))), 3) if len(vi) else None,
                       "statut": ("validation INDÉPENDANTE" if len(vi) and vi["independant"].all()
                                  else "ajustement" if len(vi) else "aucune mesure E(t) : non validé")})
    # Affiche le tableau.
    st.dataframe(pd.DataFrame(lignes), hide_index=True)
    # Chemin de la planche produite par « python src/hybride.py --rapport ».
    f = RACINE_APP / "figures" / "hybride_tous_materiaux.png"
    # Si elle existe...
    if f.exists():
        # ... elle est affichée ;
        st.image(str(f), caption="Modèle contre mesures E(t), tous les matériaux (python src/hybride.py --rapport)")
    # sinon...
    else:
        # ... on indique comment la produire.
        st.caption("Lancer `python src/hybride.py --rapport` pour générer la planche de tous les matériaux.")
