"""Jede Zahl aus README und PRESET_HELP als Test. Die Verfahren sind deterministisch bei festem Seed; Bänder um die gerundeten Angaben, dazu Rangfolgen mit Abstand
(numpy-Versionen und Plattformen können den Zufallsstrom des HW-Fits um Rundung verschieben und damit die Modellwahl je Knoten kippen, feedback_ci_platform_robust_tests)."""

from functools import lru_cache

import numpy as np
import pytest

import hrc_constants as C
import hrc_evaluation as E
import hrc_presets as P

STD = "Standardfall: Modellwahl, 30 Depots, 5 Regionen"


@lru_cache(maxsize=None)
def _preset(name):
    v = P.PRESETS[name]
    return E.analyse(E.Settings(v["n_depots"], v["n_regions"], v["noise"], v["rho"], v["swing"], v["horizon"], v["base"], v["history"], v["seed"]))


def _gain(a, m):
    return 100.0 * (a.summary[m]["mean"] / a.summary["base"]["mean"] - 1.0)


def test_standard_preset():
    a = _preset(STD)
    s = a.summary
    assert s["base"]["mean"] == pytest.approx(0.834, abs=0.03) and s["wls_struct"]["mean"] == pytest.approx(0.814, abs=0.03) and s["td"]["mean"] == pytest.approx(1.026, abs=0.05)
    assert _gain(a, "wls_struct") == pytest.approx(-2.4, abs=1.5) and _gain(a, "ols") == pytest.approx(-1.2, abs=1.5) and _gain(a, "bu") == pytest.approx(2.0, abs=1.5) and _gain(a, "mint_shrink") == pytest.approx(1.5, abs=1.5)
    assert _gain(a, "td") == pytest.approx(23.1, abs=4) and _gain(a, "mo") == pytest.approx(21.3, abs=4)
    assert a.incoherence == pytest.approx(0.0145, abs=0.006) and int(np.sum(a.picks == 0)) + int(np.sum(a.picks == 1)) + int(np.sum(a.picks == 2)) == 36 and a.picks.max() <= 1
    assert s["td"]["mean"] > s["base"]["mean"] + 0.1 and s["mo"]["mean"] > s["base"]["mean"] + 0.1 and s["wls_struct"]["mean"] < s["bu"]["mean"]
    assert s["td"]["levels"][2] > 1.2 and s["mo"]["levels"][2] > 1.2 and s["td"]["levels"][0] == pytest.approx(s["base"]["levels"][0])


def test_strong_swing_preset():
    a = _preset("Starke gemeinsame Bewegung (Schwankung 0,12)")
    s = a.summary
    assert s["base"]["mean"] == pytest.approx(1.013, abs=0.05) and s["bu"]["mean"] == pytest.approx(0.971, abs=0.05) and s["mint_shrink"]["mean"] == pytest.approx(0.937, abs=0.05)
    assert _gain(a, "bu") == pytest.approx(-4.2, abs=2) and _gain(a, "wls_struct") == pytest.approx(-3.8, abs=2.5) and _gain(a, "wls_var") == pytest.approx(-6.3, abs=2.5) and _gain(a, "mint_shrink") == pytest.approx(-7.5, abs=2.5) and _gain(a, "mint_sample") == pytest.approx(-8.7, abs=3)
    assert a.incoherence == pytest.approx(0.040, abs=0.01) and _gain(a, "mint_shrink") < _gain(a, "bu") < 0


def test_holt_winters_preset():
    a = _preset("Nur Holt-Winters als Basis")
    s = a.summary
    assert s["base"]["mean"] == pytest.approx(0.871, abs=0.03) and a.incoherence == pytest.approx(0.0077, abs=0.004)
    for m in ("bu", "ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample"):
        assert -0.9 < _gain(a, m) < 1.5, m
    assert _gain(a, "td") == pytest.approx(26.7, abs=4) and _gain(a, "mo") == pytest.approx(18.0, abs=4) and s["td"]["mean"] == pytest.approx(1.104, abs=0.06) and s["mo"]["mean"] == pytest.approx(1.028, abs=0.06)


def test_weekly_mean_preset():
    a = _preset("Wochenmittel als Basis")
    s = a.summary
    assert a.incoherence < 1e-9 and s["base"]["mean"] == pytest.approx(0.941, abs=0.03)
    for m in ("bu", "ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample"):
        assert abs(_gain(a, m)) < 1e-6, m
    assert _gain(a, "td") == pytest.approx(20.1, abs=4) and _gain(a, "mo") == pytest.approx(14.2, abs=4) and s["td"]["mean"] == pytest.approx(1.131, abs=0.06) and s["mo"]["mean"] == pytest.approx(1.075, abs=0.06)


def test_many_depots_short_history_preset():
    a = _preset("Viele Depots, kurze Fehlerhistorie")
    s = a.summary
    assert a.hier.m == 106 and s["base"]["mean"] == pytest.approx(1.004, abs=0.05) and s["base"]["levels"][0] == pytest.approx(1.124, abs=0.06)
    assert s["mint_sample"]["mean"] > 100.0                                                       # Stichprobenkovarianz aus 60 Zeilen für 106 Knoten: singulär, das Ergebnis unbrauchbar
    assert s["mint_shrink"]["mean"] == pytest.approx(0.866, abs=0.05) and s["wls_var"]["mean"] == pytest.approx(0.908, abs=0.05) and s["bu"]["mean"] == pytest.approx(0.916, abs=0.05)
    assert _gain(a, "mint_shrink") == pytest.approx(-13.8, abs=3.5) and _gain(a, "wls_var") == pytest.approx(-9.6, abs=3) and _gain(a, "bu") == pytest.approx(-8.8, abs=3)
    assert a.shrink_lambda == pytest.approx(0.454, abs=0.1) and _gain(a, "mint_shrink") < _gain(a, "wls_var") < 0


def test_short_horizon_preset():
    a = _preset("Kurzer Horizont (1 Tag)")
    s = a.summary
    assert s["base"]["mean"] == pytest.approx(0.823, abs=0.03)
    assert _gain(a, "wls_struct") == pytest.approx(-2.5, abs=1.5) and _gain(a, "wls_var") == pytest.approx(-2.0, abs=1.5) and _gain(a, "ols") == pytest.approx(-1.2, abs=1.5) and _gain(a, "bu") == pytest.approx(-1.2, abs=1.5)
    assert _gain(a, "mint_shrink") == pytest.approx(-1.2, abs=1.5) and _gain(a, "mint_sample") == pytest.approx(0.6, abs=1.5) and _gain(a, "wls_struct") < 0


@lru_cache(maxsize=None)
def _structure():
    return E.structure_experiment(C.SWING_LEVELS, C.RHO_LEVELS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _horizon():
    return E.horizon_experiment(C.HORIZON_LEVELS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _history():
    return E.history_experiment(C.HISTORY_LEVELS, C.EXP_SEEDS)


def _rel_levels(r, m):
    avg = lambda k: float(np.mean([r["levels"][k][j][0] for j in range(3)]))
    return 100.0 * (avg(m) / avg("base") - 1.0)


def test_structure_experiment_swing():
    rows = {r["x"]: r for r in _structure()["swing"]}
    r0, r6, r12 = rows[0.0], rows[0.06], rows[0.12]
    assert np.mean([r12["levels"]["base"][j][0] for j in range(3)]) == pytest.approx(1.149, abs=0.06) and np.mean([r0["levels"]["base"][j][0] for j in range(3)]) == pytest.approx(0.750, abs=0.04)
    assert _rel_levels(r0, "bu") == pytest.approx(-3.4, abs=1.5) and _rel_levels(r0, "wls_var") == pytest.approx(-2.3, abs=1.5) and _rel_levels(r0, "mint_shrink") == pytest.approx(-1.9, abs=1.5)
    assert _rel_levels(r12, "mint_shrink") == pytest.approx(-6.3, abs=2.5) and _rel_levels(r12, "wls_struct") == pytest.approx(-4.6, abs=2.5) and _rel_levels(r12, "bu") == pytest.approx(-1.9, abs=2) and _rel_levels(r12, "mint_sample") == pytest.approx(-4.4, abs=3)
    assert _rel_levels(r12, "mint_shrink") < _rel_levels(r12, "bu") and _rel_levels(r0, "bu") < _rel_levels(r0, "mint_shrink") and abs(_rel_levels(r6, "bu")) < 1.5
    assert _rel_levels(r12, "td") > 5 and _rel_levels(r0, "td") == pytest.approx(42.6, abs=8)
    assert _rel_levels(r6, "wls_var") == pytest.approx(-0.7, abs=1.5) and _rel_levels(r12, "wls_var") == pytest.approx(-4.7, abs=2.5) and _rel_levels(r0, "wls_struct") == pytest.approx(-1.5, abs=1.5) and _rel_levels(r6, "wls_struct") == pytest.approx(-0.7, abs=1.5)
    assert _rel_levels(r6, "mint_shrink") == pytest.approx(-0.3, abs=1.5) and _rel_levels(r0, "mint_sample") == pytest.approx(0.1, abs=2.5) and _rel_levels(r6, "mint_sample") == pytest.approx(3.6, abs=2.5)
    assert _rel_levels(r6, "td") == pytest.approx(26.8, abs=6) and _rel_levels(r12, "td") == pytest.approx(15.1, abs=6)
    assert np.mean([r6['levels']['base'][j][0] for j in range(3)]) == pytest.approx(0.906, abs=0.05)


def test_structure_experiment_rho():
    rows = {r["x"]: r for r in _structure()["rho"]}
    for x in (0.0, 0.4, 0.8):
        assert all(-2.6 < _rel_levels(rows[x], m) <= 0.6 for m in ("bu", "wls_struct", "wls_var", "mint_shrink"))
        assert _rel_levels(rows[x], "ols") < 1.5
        assert _rel_levels(rows[x], "td") > 15
    assert _rel_levels(rows[0.0], "wls_var") == pytest.approx(-1.4, abs=1.2) and _rel_levels(rows[0.8], "wls_struct") == pytest.approx(-0.8, abs=1.2) and _rel_levels(rows[0.0], "mint_sample") == pytest.approx(3.5, abs=2.5)


def test_horizon_experiment():
    rows = {r["horizon"]: r for r in _horizon()}
    g = lambda h, m: 100.0 * (rows[h]["mean"][m][0] / rows[h]["mean"]["base"][0] - 1.0)
    assert rows[1]["mean"]["base"][0] == pytest.approx(0.889, abs=0.04) and rows[28]["mean"]["base"][0] == pytest.approx(0.908, abs=0.04)
    assert g(1, "wls_var") == pytest.approx(-1.7, abs=1.2) and g(1, "bu") == pytest.approx(-1.3, abs=1.2) and g(1, "mint_shrink") == pytest.approx(-0.7, abs=1.2)
    assert g(28, "wls_var") == pytest.approx(-0.6, abs=1.2) and g(28, "bu") == pytest.approx(-0.4, abs=1.2) and g(28, "mint_shrink") == pytest.approx(0.0, abs=1.2)
    assert g(1, "mint_sample") == pytest.approx(2.7, abs=2) and g(28, "mint_sample") == pytest.approx(4.1, abs=2) and g(1, "mint_sample") > 0 and g(28, "mint_sample") > 0
    assert g(1, "wls_var") < g(28, "wls_var") + 0.5 and g(1, "td") == pytest.approx(29.1, abs=6)
    assert g(7, "wls_var") == pytest.approx(-1.0, abs=1.2) and g(14, "wls_var") == pytest.approx(-0.7, abs=1.2) and g(7, "bu") == pytest.approx(-0.6, abs=1.2) and g(14, "bu") == pytest.approx(0.0, abs=1.2) and g(7, "mint_shrink") == pytest.approx(-0.8, abs=1.2) and g(14, "mint_shrink") == pytest.approx(-0.3, abs=1.2)
    assert g(7, "mint_sample") == pytest.approx(3.0, abs=2) and g(14, "mint_sample") == pytest.approx(3.6, abs=2)


def test_history_experiment():
    rows = {r["history"]: r for r in _history()}
    m = lambda h, k: rows[h]["mean"][k][0]
    assert m(60, "base") == pytest.approx(1.008, abs=0.05) and m(60, "bu") == pytest.approx(0.983, abs=0.05) and m(60, "mint_shrink") == pytest.approx(0.985, abs=0.05) and m(60, "wls_var") == pytest.approx(0.990, abs=0.05)
    assert m(60, "mint_sample") > 5.0 and m(120, "mint_sample") == pytest.approx(1.017, abs=0.08) and m(240, "mint_sample") == pytest.approx(1.015, abs=0.08) and m(600, "mint_sample") == pytest.approx(0.995, abs=0.06)
    assert rows[60]["lambda"][0] == pytest.approx(0.412, abs=0.1) and rows[600]["lambda"][0] == pytest.approx(0.064, abs=0.04) and rows[60]["lambda"][0] > rows[120]["lambda"][0] > rows[240]["lambda"][0] > rows[600]["lambda"][0]
    assert all(m(h, "mint_shrink") < 1.05 for h in rows) and m(60, "mint_sample") > 3 * m(60, "base")
    assert all(m(h, "wls_var") == pytest.approx(0.99, abs=0.04) for h in rows) and all(m(h, "mint_shrink") == pytest.approx(0.987, abs=0.04) for h in rows) and all(m(h, "bu") == pytest.approx(0.983, abs=0.04) for h in rows)
