# -*- coding: utf-8 -*-
"""
Application Streamlit - vieillissement hygrothermique d'une pièce polymère.

Lancer depuis la racine du dépôt :
    streamlit run app.py

Pré-requis : un modèle entraîné dans models/ (python src/train.py --data ...).
Tout le calcul passe par src/predire_piece.py et src/train.py : l'application ne
contient aucune logique de prédiction propre (pas d'écart possible avec la ligne de commande).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

RACINE_APP = Path(__file__).resolve().parents[1]   # page Streamlit : racine du dépôt un niveau au-dessus
if str(RACINE_APP) not in sys.path:
    sys.path.insert(0, str(RACINE_APP))


def _sous_streamlit() -> bool:
    """Vrai si le script est exécuté par Streamlit (streamlit run, ou tests AppTest)."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    except ImportError:  # anciennes versions de Streamlit
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    return get_script_run_ctx(suppress_warning=True) is not None


# Lancé par "python app.py" (ou le bouton Run de PyCharm) : on relance proprement via
# "streamlit run app.py" au lieu de tourner en mode dégradé sans navigateur.
if __name__ == "__main__" and not _sous_streamlit():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    sys.exit(stcli.main())

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.predire_piece import (CAS, TOL_HR, TOL_T_C, courbes_voisines,  # noqa: E402
                               evaluer_cas, figure_cas)
from src.train import CIBLE, charger_modele, metriques, predire  # noqa: E402
from src.verifier_dataset import verifier  # noqa: E402

DOSSIER_MODELES = RACINE_APP / "models"
DOSSIER_RESULTATS = RACINE_APP / "results"
DOSSIER_FIGURES = RACINE_APP / "figures"

HYPOTHESES = """
**Ce que l'application calcule.** E/E0 prédit par un modèle XGBoost appris sur des courbes
publiées (digitalisées), puis E = (E/E0) × E0 avec un E0 **fourni** par l'utilisateur.
E0 n'est jamais prédit.

**Domaine de validité.** Un modèle à arbres n'extrapole pas : hors des conditions et des
durées couvertes par les données d'entraînement, il renvoie la valeur de sa dernière feuille
(une constante). Ces zones sont tracées en rouge : ce ne sont **pas** des prédictions.
Critère de couverture retenu (choix d'ingénieur) : au moins une courbe d'entraînement à
±{tol_t:g} °C et ±{tol_hr:g} %HR, avec le même milieu et le même état d'essai.

**Hypothèses.**
- T et HR constantes : les cycles réels (jour/nuit, saisons) et l'endommagement par
  gonflement/séchage alterné ne sont pas représentés.
- En immersion, HR = 100 % par convention : l'activité de l'eau d'une eau salée est
  légèrement inférieure à 1.
- La perte relative E/E0 mesurée sur éprouvettes est supposée transférable à la pièce réelle
  (procédé, cristallinité et orientation peuvent différer).
- État « humide » : rigidité en service, plastification comprise (réversible au séchage).
  État « reséché » : seule la part irréversible (hydrolyse, endommagement).

**Ce que le modèle ne prouve pas.** Une importance SHAP ou une bonne corrélation ne
démontrent ni une causalité, ni qu'une loi d'Arrhenius a été apprise.
"""

st.set_page_config(page_title="Comparaison XGBoost", page_icon="📊", layout="wide")


@st.cache_resource
def charger(dossier: str, empreinte: tuple):
    """L'empreinte (dates de modification) invalide le cache après un réentraînement."""
    return charger_modele(dossier)


def empreinte_modeles() -> tuple:
    return tuple((p.name, p.stat().st_mtime) for p in sorted(DOSSIER_MODELES.glob("*.json")))


def carte_domaine(meta: dict, cas: dict):
    """Conditions des courbes d'entraînement dans le plan (T, HR) et position du cas."""
    cond = pd.DataFrame(meta["domaine_entrainement"]["conditions_courbes"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    lt = np.log10(cond["t_max_h"].clip(lower=1.0))
    sc = None
    for milieu, mk in (("vapeur", "o"), ("immersion", "s")):
        c = cond[cond["milieu"] == milieu]
        if len(c):
            sc = ax.scatter(c["Temperature_C"], c["Humidite_HR"],
                            c=np.log10(c["t_max_h"].clip(lower=1.0)), marker=mk, cmap="viridis",
                            vmin=lt.min(), vmax=max(lt.max(), lt.min() + 1e-6), s=45,
                            edgecolor="k", linewidths=0.4, label=f"{milieu} ({len(c)} courbes)")
    T, HR = cas["Temperature_C"], cas["Humidite_HR"]
    ax.add_patch(Rectangle((T - TOL_T_C, HR - TOL_HR), 2 * TOL_T_C, 2 * TOL_HR,
                           fill=False, ls="--", ec="red", lw=1.2))
    ax.plot(T, HR, "r*", ms=15, label="cas étudié (± tolérance)")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="log10(durée max de la courbe, h)")
    ax.set(xlabel="Température de vieillissement (°C)", ylabel="HR (%)",
           title="Domaine des courbes d'entraînement")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


# ===================================================================================== en-tête
st.title("Comparaison : modèle purement data-driven (XGBoost)")
st.caption("Prédiction data-driven de la rigidité résiduelle E/E0. "
           "Tout ce qui sort du domaine des données est signalé en rouge.")

AIDE_ENTRAINEMENT = (
    "# avec le dataset réel :\n"
    "python src/verifier_dataset.py data/dataset_reel.csv\n"
    "python src/train.py --data data/dataset_reel.csv [--materiau <material_id>]\n\n"
    "# pour tester l'application AVANT d'avoir des données réelles (modèle synthétique) :\n"
    "python tests/generer_jeu_test.py\n"
    "python src/train.py --data tests/donnees/synthetique_valide.csv "
    "--autoriser-synthetique --repetitions 5")

try:
    modele, meta = charger(str(DOSSIER_MODELES), empreinte_modeles())
except (FileNotFoundError, ValueError) as e:
    st.error(f"Modèle inutilisable : {e}")
    st.code(AIDE_ENTRAINEMENT, language="bash")
    st.stop()
    raise SystemExit(0)  # garde-fou si st.stop() est sans effet (exécution hors Streamlit)
filtre = meta.get("filtre_entrainement", {})
if meta["dataset"].get("synthetique_autorise"):
    st.error("MODÈLE ENTRAÎNÉ SUR DES DONNÉES SYNTHÉTIQUES : aucune valeur physique.")

# ===================================================================================== saisie
with st.sidebar:
    st.header("Cas étudié")
    st.caption("Aucune valeur par défaut n'est une donnée : chaque champ vide est à renseigner.")
    milieu = st.selectbox("Milieu", ["vapeur", "immersion"],
                          index=["vapeur", "immersion"].index(CAS["milieu"]))
    T = st.number_input("Température (°C)", value=CAS["Temperature_C"], min_value=-60.0,
                        max_value=250.0, step=1.0, placeholder="à renseigner")
    if milieu == "immersion":
        HR = 100.0
        st.info("Immersion : HR = 100 % par convention du dataset (hypothèse).")
    else:
        HR = st.number_input("Humidité relative (%)", value=CAS["Humidite_HR"], min_value=0.0,
                             max_value=100.0, step=1.0, placeholder="à renseigner")
    unite = st.radio("Unité de durée", ["h", "jours", "années"], horizontal=True)
    duree = st.number_input(f"Durée d'exposition ({unite})", value=None, min_value=0.0,
                            step=1.0, placeholder="à renseigner")
    temps_h = None if duree is None else duree * {"h": 1, "jours": 24, "années": 8760}[unite]
    if temps_h:
        st.caption(f"soit {temps_h:g} h")
    epaisseur = st.number_input("Épaisseur de la pièce (mm)", value=CAS["Epaisseur_mm"],
                                min_value=0.0, step=0.1, format="%.2f",
                                placeholder="à renseigner")
    E0 = st.number_input("E0 de la pièce (MPa), optionnel", value=CAS["E0_MPa"],
                         min_value=0.0, step=100.0, placeholder="vide = E/E0 seul",
                         help="Module initial de TA pièce (fiche technique ou essai). "
                              "Donnée d'entrée, jamais prédite.")
    etats = ["humide", "reseche", "reconditionne", "inconnu"]
    etat = st.selectbox("État de la pièce quand sa rigidité compte", etats,
                        index=etats.index(CAS["etat_essai"]),
                        help="humide = en service (plastification incluse) ; "
                             "reseche = part irréversible seule ; reconditionne = "
                             "rééquilibrée à 23 °C / 50 %HR avant essai")
    propriete = st.selectbox("Module", ["E_traction", "E_flexion"],
                             index=["E_traction", "E_flexion"].index(CAS["propriete"]))
    if meta["jeu_features"] == "materiau":
        famille = st.selectbox("Famille de polymère", meta["categories"]["famille"])
        renfort = st.selectbox("Renfort", meta["categories"]["renfort"])
    else:
        famille, renfort = CAS["famille"], CAS["renfort"]
        st.caption("Matériau : " + (f"modèle spécifique à « {filtre['material_id']} »"
                                    if "material_id" in filtre
                                    else "non pris en compte par ce modèle"))
    seuil = None
    if st.checkbox("Critère de fin de vie"):
        seuil = st.slider("Seuil E/E0", 0.30, 1.00, 0.80, 0.01,
                          help="Choix de conception, pas une valeur issue des données")
    if st.button("Recharger le modèle"):
        st.cache_resource.clear()
        st.rerun()

cas = {"Temperature_C": T, "Humidite_HR": HR, "Temps_h": temps_h, "Epaisseur_mm": epaisseur,
       "E0_MPa": E0, "famille": famille, "renfort": renfort, "milieu": milieu,
       "etat_essai": etat, "propriete": propriete}

onglet_cas, onglet_test, onglet_modele = st.tabs(
    ["Prédiction du cas", "Tester avec des données expérimentales", "Modèle et traçabilité"])

# ===================================================================================== onglet 1
with onglet_cas:
    try:
        r = evaluer_cas(modele, meta, cas, seuil)
    except ValueError as e:
        st.info(f"Cas incomplet ou non calculable : {e}")
        r = None
    if r is not None:
        for niveau, msg in r["avertissements"]:
            (st.warning if niveau == "alerte" else st.info)(msg)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"E/E0 à {temps_h:g} h", f"{r['E_E0']:.3f}")
        c2.metric("E calculé (MPa)", "-" if r["E_MPa"] is None else f"{r['E_MPa']:.0f}",
                  help="E = (E/E0) × E0 ; E0 est une donnée de la pièce, pas une prédiction")
        c3.metric("Erreur typique (RMSE test)", f"±{r['rmse_test']:.3f}",
                  help="Calculée sur les 20 % de courbes test, hors t=0. "
                       "Ce n'est PAS un intervalle de confiance pour ce cas.")
        (st.success if r["verdict"] == "DOMAINE" else st.error)(r["message"])
        if seuil is not None:
            if r["t_seuil"] is None:
                st.write(f"E/E0 reste au-dessus de {seuil:.2f} jusqu'à {temps_h:g} h.")
            else:
                ou = "dans le" if r["t_seuil"] <= r["t_valide"] else "HORS du"
                st.write(f"E/E0 ≤ {seuil:.2f} atteint vers **{r['t_seuil']:g} h** "
                         f"({r['t_seuil'] / 8760:.2f} an), {ou} domaine.")
        gauche, droite = st.columns([3, 2])
        with gauche:
            fig = figure_cas(r, cas)
            st.pyplot(fig)
            plt.close(fig)
        with droite:
            fig = carte_domaine(meta, cas)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("La carte montre toutes les courbes d'entraînement ; le verdict exige en "
                       "plus le même milieu, le même état d'essai (et la même famille si le "
                       "modèle l'utilise).")
        st.dataframe(r["jalons"], hide_index=True)
        courbe = pd.DataFrame({"Temps_h": r["t"], "E_E0_predit": r["ratio"],
                               "E_MPa_calcule": r["ratio"] * E0 if E0 else np.nan,
                               "dans_domaine": r["dans"]})
        st.download_button("Télécharger la courbe (CSV)", courbe.to_csv(index=False).encode(),
                           file_name=f"prediction_T{T:g}_HR{HR:g}_h{epaisseur:g}_t{temps_h:g}.csv",
                           mime="text/csv")
        with st.expander("Courbes d'entraînement voisines"):
            st.dataframe(r["voisines"], hide_index=True)
    with st.expander("Hypothèses et limites"):
        st.markdown(HYPOTHESES.format(tol_t=TOL_T_C, tol_hr=TOL_HR))

# ===================================================================================== onglet 2
with onglet_test:
    st.markdown("Charge un CSV **au format du template** (`data/template_dataset_reel.csv`). "
                "Il est vérifié par `verifier_dataset.py` avant toute prédiction ; seules les "
                "courbes EXPLOITABLES sont évaluées.")
    fichier = st.file_uploader("CSV de validation", type=["csv"])
    autoriser = st.checkbox("Accepter source_doi = SYNTHETIQUE (tests du code uniquement)")
    if fichier is not None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(fichier.getvalue())
            chemin = f.name
        res = None
        try:
            res = verifier(chemin, autoriser_synthetique=autoriser)
        except (ValueError, KeyError) as e:
            st.error(f"Fichier illisible : {e}")
        finally:
            os.unlink(chemin)
        if res is not None:
            rap = res.rapport_courbes
            compte = rap["statut"].value_counts() if len(rap) else pd.Series(dtype=int)
            st.write(" | ".join(f"{s} : {int(compte.get(s, 0))}"
                                for s in ("EXPLOITABLE", "EXCLUE", "REJETEE")))
            if len(rap) and (rap["statut"] != "EXPLOITABLE").any():
                with st.expander("Courbes écartées et diagnostic"):
                    st.dataframe(rap[rap["statut"] != "EXPLOITABLE"][
                        ["curve_id", "statut", "diagnostic"]], hide_index=True)
            d = res.df_exploitable.copy()
            if d.empty:
                st.warning("Aucune courbe exploitable dans ce fichier.")
            else:
                vues = set(meta["split"]["curves_train"])
                if meta.get("refit_complet"):
                    vues |= set(meta["split"]["curves_test"])
                sources_vues = set(meta["domaine_entrainement"].get("sources", []))
                n_vues = d.loc[d["curve_id"].isin(vues), "curve_id"].nunique()
                n_src = d.loc[d["source_doi"].isin(sources_vues), "curve_id"].nunique()
                if n_vues:
                    st.error(f"{n_vues} courbe(s) ont servi à l'entraînement : le score ci-dessous "
                             f"n'est PAS une validation (fuite de données).")
                elif n_src:
                    st.warning(f"{n_src} courbe(s) viennent de sources vues à l'entraînement "
                               f"(même labo, même lot) : validation non indépendante.")
                else:
                    st.success("Courbes et sources jamais vues à l'entraînement.")
                if "material_id" in filtre and (d["material_id"] != filtre["material_id"]).any():
                    st.warning(f"Le modèle est spécifique à « {filtre['material_id']} » : "
                               f"les autres matériaux sont hors de son périmètre.")
                if meta["jeu_features"] == "materiau":
                    inconnues = sorted(set(d["famille"].replace("", "inconnu"))
                                       - set(meta["categories"]["famille"]))
                    if inconnues:
                        st.warning(f"Familles inconnues du modèle : {inconnues}")
                y = d[CIBLE].to_numpy(float)
                t = d["Temps_h"].to_numpy(float)
                p = predire(modele, meta, d)
                bornes = meta["domaine_entrainement"]["bornes"]
                dans = np.zeros(len(d), bool)
                for cid, g in d.groupby("curve_id"):
                    c = {"Temperature_C": g["Temperature_C"].iat[0],
                         "Humidite_HR": g["Humidite_HR"].iat[0], "milieu": g["milieu"].iat[0],
                         "etat_essai": g["etat_essai"].iat[0], "famille": g["famille"].iat[0]}
                    v = courbes_voisines(meta, c)
                    h_ok = bornes["Epaisseur_mm"][0] <= g["Epaisseur_mm"].iat[0] <= \
                        bornes["Epaisseur_mm"][1]
                    tv = float(v["t_max_h"].max()) if (len(v) and h_ok) else 0.0
                    dans[g.index.to_numpy()] = g["Temps_h"].to_numpy() <= tv
                m_tout = metriques(y, p, t)
                m_dom = metriques(y[dans], p[dans], t[dans]) if dans.any() else None
                c1, c2, c3 = st.columns(3)
                c1.metric("R² (t > 0)", f"{m_tout['R2']:.3f}")
                c2.metric("RMSE (t > 0)", f"{m_tout['RMSE']:.4f}")
                c3.metric("MAE (t > 0)", f"{m_tout['MAE']:.4f}")
                if m_dom is not None and np.isfinite(m_dom["RMSE"]):
                    st.caption(f"Points dans le domaine uniquement ({m_dom['n_points']} points, "
                               f"t > 0) : R² = {m_dom['R2']:.3f}, RMSE = {m_dom['RMSE']:.4f}")
                fig, ax = plt.subplots(figsize=(5.5, 5))
                pos = t > 0
                ax.scatter(y[pos & dans], p[pos & dans], s=16, label="dans le domaine")
                ax.scatter(y[pos & ~dans], p[pos & ~dans], s=16, c="red",
                           label="hors domaine")
                lo, hi = min(y.min(), p.min()) - 0.02, max(y.max(), p.max()) + 0.02
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="E/E0 mesuré (source)",
                       ylabel="E/E0 prédit", title="Parité, points t > 0")
                ax.legend(fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                d["prediction"], d["residu"], d["dans_domaine"] = p, p - y, dans
                par_courbe = d[d["Temps_h"] > 0].groupby("curve_id").agg(
                    material_id=("material_id", "first"), T=("Temperature_C", "first"),
                    HR=("Humidite_HR", "first"), n=("residu", "size"),
                    RMSE=("residu", lambda r: float(np.sqrt(np.mean(r ** 2)))),
                    part_dans_domaine=("dans_domaine", "mean")).reset_index()
                st.dataframe(par_courbe, hide_index=True)
                st.download_button("Télécharger les prédictions (CSV)",
                                   d.drop(columns="_ligne", errors="ignore")
                                   .to_csv(index=False).encode(),
                                   file_name="validation_predictions.csv", mime="text/csv")

# ===================================================================================== onglet 3
with onglet_modele:
    ds, sp = meta["dataset"], meta["split"]
    st.markdown(
        f"- **Dataset** : `{ds['fichier']}`, {ds['n_courbes']} courbes, {ds['n_points']} points  \n"
        f"- **SHA-256** : `{ds['sha256']}`  \n"
        f"- **Filtre d'entraînement** : {filtre or 'aucun (tous matériaux)'}  \n"
        f"- **Variables** ({meta['jeu_features']}) : {', '.join(meta['features'])}  \n"
        f"- **Split** : {sp['methode']}, test_size = {sp['test_size']}, "
        f"random_state = {sp['random_state']} → {len(sp['curves_train'])} courbes train, "
        f"{len(sp['curves_test'])} courbes test  \n"
        f"- **Modèle déployé réentraîné sur 100 %** : {'oui' if meta['refit_complet'] else 'non'}  \n"
        f"- **Pondération par courbe** : {'oui' if meta['pondere_par_courbe'] else 'non'}")
    mt, mt0 = meta["metriques_test_hors_t0"], meta["metriques_test_avec_t0"]
    st.dataframe(pd.DataFrame({
        "métrique": ["R²", "RMSE", "MAE"],
        "test, hors t=0 (référence)": [mt["R2"], mt["RMSE"], mt["MAE"]],
        "test, t=0 inclus (gonflé)": [mt0["R2"], mt0["RMSE"], mt0["MAE"]]}), hide_index=True)
    proto = DOSSIER_RESULTATS / "protocoles.csv"
    if proto.exists():
        st.subheader("Protocoles de validation")
        st.caption("A = témoin de fuite (ne pas utiliser comme score). B = référence "
                   "(moyenne ± écart-type sur les tirages). C/D = matériau / source inconnus.")
        st.dataframe(pd.read_csv(proto), hide_index=True)
    fig_test = DOSSIER_FIGURES / "parite_residus_test.png"
    if fig_test.exists():
        st.image(str(fig_test), caption="Split officiel : parité et résidus sur les courbes test")
    st.caption("Versions : " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()))
