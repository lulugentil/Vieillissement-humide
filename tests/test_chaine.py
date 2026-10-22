#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de bout en bout sur données SYNTHÉTIQUES : vérification -> entraînement -> prédiction.
Tout est écrit dans un dossier temporaire : models/ du dépôt n'est JAMAIS touché.

Usage (racine du dépôt) :   python tests/test_chaine.py
"""
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "tests"))

from generer_jeu_test import main as generer  # noqa: E402
from src import train  # noqa: E402
from src.predire_piece import evaluer_cas  # noqa: E402
from src.verifier_dataset import verifier  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        generer(["--sortie", str(tmp)])

        # 1) le vérificateur attrape chaque erreur injectée
        res = verifier(tmp / "synthetique_erreurs.csv", autoriser_synthetique=True)
        statut = dict(zip(res.rapport_courbes["curve_id"], res.rapport_courbes["statut"]))
        attendus_rejet = ["S01", "S02", "S03", "S04", "S05", "S06", "S08", "S09", "S11"]
        for cid in attendus_rejet:
            assert statut[cid] == "REJETEE", f"{cid} aurait dû être rejetée : {statut[cid]}"
        assert statut["S10"] == "EXCLUE", statut["S10"]
        assert statut["S07"] == "EXPLOITABLE", statut["S07"]
        print("[ok] vérification : 9 rejets, 1 exclusion, 1 avertissement détectés")

        # 2) les données synthétiques sont refusées par défaut
        res = verifier(tmp / "synthetique_valide.csv")
        assert res.df_exploitable.empty, "le synthétique doit être refusé sans option"
        print("[ok] données synthétiques refusées sans --autoriser-synthetique")

        # 3) entraînement complet, sorties dans le dossier temporaire
        code = train.main(["--data", str(tmp / "synthetique_valide.csv"), "--sortie", str(tmp),
                           "--autoriser-synthetique", "--repetitions", "2",
                           "--jeux", "contexte"])
        assert code == 0 and (tmp / "models" / "modele.json").exists()
        modele, meta = train.charger_modele(tmp / "models")
        assert not set(meta["split"]["curves_train"]) & set(meta["split"]["curves_test"])
        print("[ok] entraînement, split sans recouvrement de courbes, relecture JSON")

        # 4) prédiction : un cas couvert et un cas ambiant hors domaine
        cas = {"Temperature_C": 60.0, "Humidite_HR": 85.0, "Temps_h": 720.0,
               "Epaisseur_mm": 2.0, "E0_MPa": 3000.0, "famille": "", "renfort": "aucun",
               "milieu": "vapeur", "etat_essai": "humide", "propriete": "E_traction"}
        r = evaluer_cas(modele, meta, cas)
        assert 0 < r["E_E0"] < 1.5 and r["E_MPa"] is not None
        r2 = evaluer_cas(modele, meta, {**cas, "Temperature_C": 23.0, "Humidite_HR": 50.0,
                                        "Temps_h": 87600.0})
        assert r2["verdict"] == "HORS_CONDITIONS", r2["verdict"]
        print(f"[ok] prédiction : cas couvert '{r['verdict']}', cas 23 degC/50 %HR "
              f"'{r2['verdict']}'")
    print("\nTOUS LES TESTS PASSENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
