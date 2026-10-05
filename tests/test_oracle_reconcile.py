"""Orakel-Test: Abstimmungsmatrizen gegen Cholesky-Weißung + lstsq und exakt rational (fractions); die Auswertungskette (Ein-Schritt-Fehler, Anteile, G, Beschneidung, RMSSE je Knoten und Ebene) gegen
eine unabhängige Skalar-/Schleifenrechnung auf Basisprognosen der Demo (Regression je Knoten)."""

import math
from fractions import Fraction

import numpy as np
import pytest

import hrc_constants as C
import hrc_evaluation as E
import hrc_reconcile as R
import hrc_scenario as S

sla = pytest.importorskip("scipy.linalg")


def _gls_fit(Sm, W, yhat):
    L = np.linalg.cholesky(W)
    A = sla.solve_triangular(L, Sm, lower=True)
    c = sla.solve_triangular(L, yhat, lower=True)
    return Sm @ np.linalg.lstsq(A, c, rcond=None)[0]


def test_gls_matrices_equal_whitened_least_squares_on_random_hierarchies():
    rng = np.random.default_rng(0)
    for _ in range(60):
        n = int(rng.integers(2, 9))
        region = np.sort(rng.permutation(np.arange(n) % int(rng.integers(1, max(2, n // 2) + 1))))
        Sm = S.summing_matrix(region)
        m = Sm.shape[0]
        yhat = rng.normal(size=m) * 10 + 50
        Tn = int(rng.integers(m + 5, 3 * m + 20))
        Ee = rng.normal(size=(Tn, m)) * (0.5 + rng.random(m) * 3)
        for G, W in ((R.ols(Sm), np.eye(m)), (R.wls_structure(Sm), np.diag(Sm.sum(axis=1))), (R.wls_variance(Sm, Ee), np.diag((Ee ** 2).mean(axis=0))),
                     (R.mint(Sm, Ee, False)[0], Ee.T @ Ee / Tn), (R.mint(Sm, Ee, True)[0], R.shrinkage_cov(Ee)[0])):
            assert np.allclose(Sm @ (G @ yhat), _gls_fit(Sm, W, yhat), atol=1e-7)
            assert np.allclose(G @ Sm, np.eye(n), atol=1e-8)


def _frac_solve(A, B):
    n = len(A)
    M = [[Fraction(x) for x in A[i]] + [Fraction(x) for x in B[i]] for i in range(n)]
    for c in range(n):
        p = next(i for i in range(c, n) if M[i][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    return [row[n:][0] for row in M]


def test_wls_structure_exactly_rational():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(2, 6))
        region = np.sort(rng.permutation(np.arange(n) % int(rng.integers(1, max(2, n // 2) + 1))))
        Sm = S.summing_matrix(region).astype(int)
        m = Sm.shape[0]
        w = Sm.sum(axis=1)
        yh = [int(v) for v in rng.integers(0, 100, size=m)]
        A = [[sum(Fraction(int(Sm[k, i]) * int(Sm[k, j]), int(w[k])) for k in range(m)) for j in range(n)] for i in range(n)]
        rhs = [[sum(Fraction(int(Sm[k, i]) * yh[k], int(w[k])) for k in range(m))] for i in range(n)]
        exact = [float(x) for x in _frac_solve(A, rhs)]
        assert np.allclose(R.wls_structure(Sm.astype(float)) @ np.array(yh, float), exact, atol=1e-9)


def test_evaluation_chain_matches_an_independent_loop():
    s = E.Settings(n_depots=20, n_regions=3, horizon=3, base="regression", history=150, seed=9)
    a = E.analyse(s)
    hier = a.hier
    F_all, org_all, _ = E._base(s.hierarchy_key, s.base, s.horizon)
    m, n, h = hier.m, hier.n, s.horizon
    Eerr = np.array([[hier.y[j, t] - F_all[j, t - C.FIRST_ORIGIN, 0] for j in range(m)] for t in range(C.FIRST_TEST - s.history, C.FIRST_TEST)])
    assert Eerr.shape == (s.history, m)
    pn, pr = np.zeros(n), np.zeros(n)
    for i in range(n):
        days = range(C.FIRST_ORIGIN, C.FIRST_TEST)
        pn[i] = sum(hier.bottom[i, d] / hier.y[0, d] for d in days) / len(days)
        pr[i] = sum(hier.bottom[i, d] / hier.y[1 + hier.region[i], d] for d in days) / len(days)
    Wi = lambda W: np.linalg.inv(W)
    gl = lambda W: np.linalg.inv(hier.S.T @ Wi(W) @ hier.S) @ hier.S.T @ Wi(W)
    Ws = Eerr.T @ Eerr / s.history
    sd = np.sqrt(np.diag(Ws))
    num = den = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                w = Eerr[:, i] / sd[i] * Eerr[:, j] / sd[j]
                r = w.mean()
                num += s.history / (s.history - 1.0) ** 3 * ((w - r) ** 2).sum()
                den += r * r
    lam = min(1.0, max(0.0, num / den))
    td, mo = np.zeros((n, m)), np.zeros((n, m))
    td[:, 0] = pn
    mo[np.arange(n), 1 + hier.region] = pr
    G = {"bu": np.hstack([np.zeros((n, m - n)), np.eye(n)]), "td": td, "mo": mo, "ols": gl(np.eye(m)), "wls_struct": gl(np.diag(hier.S.sum(axis=1))),
         "wls_var": gl(np.diag((Eerr ** 2).mean(axis=0))), "mint_sample": gl(Ws), "mint_shrink": gl(lam * np.diag(np.diag(Ws)) + (1 - lam) * Ws)}
    assert lam == pytest.approx(a.shrink_lambda, abs=1e-9)
    for k, g in G.items():
        assert np.allclose(g, a.G[k], rtol=1e-6, atol=1e-9), k
    scale = [math.sqrt(sum((hier.y[j, d] - hier.y[j, d - 7]) ** 2 for d in range(7, C.FIRST_TEST)) / (C.FIRST_TEST - 7)) for j in range(m)]
    test_org = range(C.FIRST_TEST, C.N_DAYS - h + 1)
    se = {k: np.zeros(m) for k in ["base"] + list(G)}
    for t in test_org:
        Fb = F_all[:, t - C.FIRST_ORIGIN, :]
        for k in se:
            f = Fb if k == "base" else np.maximum(hier.S @ (G[k] @ Fb), 0.0)
            se[k] += ((f - hier.y[:, t:t + h]) ** 2).sum(axis=1)
    for k in se:
        rm = np.sqrt(se[k] / (len(test_org) * h)) / np.array(scale)
        lv = [rm[hier.level_of == q].mean() for q in range(3)]
        assert a.summary[k]["levels"] == pytest.approx(lv, rel=1e-7) and a.summary[k]["mean"] == pytest.approx(np.mean(lv), rel=1e-7), k
