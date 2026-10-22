#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests de cohérence PHYSIQUE du modèle hybride, pour tous les matériaux (python tests/test_hybride.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import hybride as H  # noqa: E402

# un cas DANS le domaine de chaque matériau : (T, HR, milieu, h)
CAS = {"PA6_GOODFELLOW": (25, 100, "immersion_eau_de_mer", 2), "EPOXY_HEXION_RIMR135": (40, 100, "immersion_eau_douce", 2),
       "PA6_AKULON_F223D": (23, 73, "vapeur", 4), "LIN_EPOXY": (23, 100, "immersion_eau_douce", 3),
       "LIN_ELIUM": (23, 100, "immersion_eau_douce", 3), "PA6_TAKMID": (23, 100, "immersion_eau_douce", 7),
       "EPOXY_WANG_618": (60, 98, "vapeur", 3), "PLA_2002D": (40, 80, "vapeur", 2), "PLA_BQ_FFF": (50, 70, "vapeur", 2)}


def main() -> int:
    for tau in (1e-5, 1e-4, 1e-2, 0.1, 0.5):           # tau = D t / h² ; 1e-5 : front sur 0.6 % de h
        a = float(np.trapezoid(H.profil([tau])[0], H._X))
        b = float(H.fraction_moyenne(np.array([tau]))[0])
        assert abs(a - b) / b < 0.02, (tau, a, b)
    print("[ok] conservation de l'eau dans l'épaisseur (< 2 %)")
    assert set(CAS) == set(H.predictibles()), "chaque matériau prédictible doit avoir un cas de test"
    for cle, (T, HR, mil, h) in CAS.items():
        r = H.predire(cle, T, HR, mil, h, 1e5)
        E0 = r["E0"]
        assert E0 and abs(r["E_med"][0] - E0) < 1e-6 and r["E_bas"][0] == r["E_haut"][0] == E0, cle
        assert (r["E_bas"] > 0).all() and (r["E_haut"] <= E0 + 1e-6).all(), cle
        assert (np.diff(r["E_med"]) <= 1e-6).all(), cle
        assert (r["E_bas"] <= r["E_med"] + 1e-6).all() and (r["E_med"] <= r["E_haut"] + 1e-6).all(), cle
        if r["D"] is not None:          # équilibre indépendant de h, temps de saturation en h²
            r2 = H.predire(cle, T, HR, mil, 2 * h, 1e6)
            assert abs(r2["E_med"][-1] - H.predire(cle, T, HR, mil, h, 1e6)["E_med"][-1]) / E0 < 0.01, cle
            assert abs(r2["t95_h"] / r["t95_h"] - 4) < 1e-6, cle
        print(f"[ok] {cle:22s} E(0) = E0, 0 < E <= E0, E(t) non croissant, IC encadre la médiane"
              + (", équilibre indépendant de h, t95 en h²" if r["D"] is not None else ""))
    # effets identifiables
    a = H.predire("EPOXY_HEXION_RIMR135", 8, 100, "immersion_eau_douce", 2, 100)
    b = H.predire("EPOXY_HEXION_RIMR135", 50, 100, "immersion_eau_douce", 2, 100)
    assert b["D"] > a["D"]
    lo = H.predire("PA6_AKULON_F223D", 23, 36, "vapeur", 4, 1e5)
    hi = H.predire("PA6_AKULON_F223D", 23, 99, "vapeur", 4, 1e5)
    assert lo["M_inf"] < hi["M_inf"] and lo["E_inf"] > hi["E_inf"]
    p20 = H.predire("PLA_2002D", 20, 80, "vapeur", 2, 720)
    p50 = H.predire("PLA_2002D", 50, 80, "vapeur", 2, 720)
    assert p50["E_med"][-1] < p20["E_med"][-1]
    print("[ok] effets : T accélère la diffusion (époxy) et la dégradation (PLA) ; HR augmente l'eau et baisse E (PA6)")
    # refus hors domaine identifiable
    refus = [("LIN_EPOXY", 23, 70, "vapeur", 3), ("PA6_TAKMID", 23, 70, "vapeur", 7),
             ("EPOXY_WANG_618", 30, 98, "vapeur", 3), ("PLA_2002D", 40, 40, "vapeur", 2),
             ("PLA_2002D", 70, 80, "vapeur", 2), ("PLA_BQ_FFF", 23, 50, "vapeur", 2)]
    for args in refus:
        try:
            H.predire(*args, 100)
        except H.NonPredictible:
            continue
        raise AssertionError(f"aurait dû être refusé : {args}")
    for cle in ("PLA_L130", "PLA_L130_AMIDON"):
        try:
            H.predire(cle, 23, 50, "vapeur", 2, 100)
            raise AssertionError(cle)
        except H.NonPredictible:
            pass
    print(f"[ok] {len(refus) + 2} cas hors domaine refusés avec leur raison")
    c = H.predire("PA6_GOODFELLOW", 40, 70, "vapeur", 2, 100)
    assert any(n == "critique" and "25 °C" in m for n, m in c["vigilance"])
    c = H.predire("LIN_EPOXY", 40, 100, "immersion_eau_douce", 3, 100)
    assert any(n == "critique" and "température" in m for n, m in c["vigilance"])
    print("[ok] alertes critiques : cinétique extrapolée (PA6) et température non publiée (lin)")
    print("\nTESTS PHYSIQUES PASSÉS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
