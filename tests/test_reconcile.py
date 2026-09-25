"""Die Abstimmungsmatrizen von Hand nachgerechnet (kleine Hierarchie), gegen unabhängige Rechnungen (kleinste Quadrate, Schleife, Zufallsstörungen) und auf ihre Eigenschaften."""

import numpy as np
import pytest

import hrc_reconcile as R
import hrc_scenario as S

S3 = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])                    # Netz = A + B


def test_summing_matrix_of_a_two_region_hierarchy():
    region = np.array([0, 0, 1, 1, 1])
    Sm = S.summing_matrix(region)
    assert Sm.shape == (1 + 2 + 5, 5) and np.array_equal(Sm[0], np.ones(5)) and np.array_equal(Sm[1], [1, 1, 0, 0, 0]) and np.array_equal(Sm[2], [0, 0, 1, 1, 1]) and np.array_equal(Sm[3:], np.eye(5))


def test_bottom_up_by_hand():
    G = R.bottom_up(S3)
    assert np.array_equal(G, [[0, 1, 0], [0, 0, 1]])
    assert np.allclose(R.apply(S3, G, np.array([10.0, 3.0, 4.0])), [7, 3, 4])                # Netz = 3 + 4


def test_top_down_and_middle_out_by_hand():
    Gt = R.top_down(S3, np.array([0.25, 0.75]))
    assert np.allclose(R.apply(S3, Gt, np.array([10.0, 3.0, 4.0])), [10, 2.5, 7.5])
    region = np.array([0, 0, 1])
    Sm = S.summing_matrix(region)                                                            # Netz, R1, R2, D1, D2, D3
    Gm = R.middle_out(Sm, region, np.array([0.4, 0.6, 1.0]))
    out = R.apply(Sm, Gm, np.array([99.0, 10.0, 5.0, 0, 0, 0]))
    assert np.allclose(out, [15, 10, 5, 4, 6, 5])


def test_ols_by_hand():
    G = R.ols(S3)
    assert np.allclose(G, [[1 / 3, 2 / 3, -1 / 3], [1 / 3, -1 / 3, 2 / 3]])
    out = R.apply(S3, G, np.array([10.0, 3.0, 4.0]))                                        # Depot A = (10 + 6 - 4)/3 = 4, B = (10 - 3 + 8)/3 = 5
    assert np.allclose(out, [9, 4, 5])


def test_wls_structure_by_hand():
    G = R.wls_structure(S3)                                                                  # W = diag(2, 1, 1)
    assert np.allclose(G, [[0.25, 0.75, -0.25], [0.25, -0.25, 0.75]])


def test_gls_is_the_weighted_least_squares_fit():
    rng = np.random.default_rng(0)
    Sm = S.summing_matrix(np.array([0, 0, 1, 1, 1]))
    A = rng.normal(size=(Sm.shape[0], Sm.shape[0]))
    W = A @ A.T + np.eye(Sm.shape[0])
    yhat = rng.normal(size=Sm.shape[0]) * 10 + 50
    b = R.gls(Sm, W) @ yhat
    L = np.linalg.cholesky(np.linalg.inv(W))                                                # Weißung: L' W^-1 L ... kleinste Quadrate im Maß W^-1
    b_ref = np.linalg.lstsq(L.T @ Sm, L.T @ yhat, rcond=None)[0]
    assert np.allclose(b, b_ref)


@pytest.mark.parametrize("make", ["bu", "ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample"])
def test_unbiasedness_condition_and_coherence(make):
    rng = np.random.default_rng(1)
    region = np.array([0, 0, 0, 1, 1, 1, 1])
    Sm = S.summing_matrix(region)
    E = rng.normal(size=(200, Sm.shape[0])) * (1.0 + np.arange(Sm.shape[0]))
    G = {"bu": R.bottom_up(Sm), "ols": R.ols(Sm), "wls_struct": R.wls_structure(Sm), "wls_var": R.wls_variance(Sm, E), "mint_shrink": R.mint(Sm, E, True)[0], "mint_sample": R.mint(Sm, E, False)[0]}[make]
    assert np.allclose(G @ Sm, np.eye(Sm.shape[1]))                                          # G S = I: unverzerrt
    F = rng.normal(size=(Sm.shape[0], 5, 3)) * 20 + 100
    out = R.apply(Sm, G, F)
    assert out.shape == F.shape
    assert np.allclose(out[0], out[3:].sum(axis=0)) and np.allclose(out[1], out[3:6].sum(axis=0)) and np.allclose(out[2], out[6:].sum(axis=0))


def test_mint_minimises_the_trace_of_the_forecast_error_covariance():
    rng = np.random.default_rng(2)
    Sm = S.summing_matrix(np.array([0, 0, 1, 1, 1]))
    m, n = Sm.shape
    A = rng.normal(size=(m, m))
    W = A @ A.T + np.eye(m)
    G = R.gls(Sm, W)
    base = np.trace(Sm @ G @ W @ G.T @ Sm.T)
    # Störungen N mit N S = 0 behalten die Unverzerrtheit (G + N) S = I bei; keine darf die Spur senken
    P = np.eye(m) - Sm @ np.linalg.pinv(Sm)
    for _ in range(50):
        N = rng.normal(size=(n, m)) @ P * 0.3
        assert np.allclose((G + N) @ Sm, np.eye(n))
        assert np.trace(Sm @ (G + N) @ W @ (G + N).T @ Sm.T) >= base - 1e-9


def test_wls_variance_uses_the_error_variances():
    rng = np.random.default_rng(3)
    Sm = S.summing_matrix(np.array([0, 0, 1, 1]))
    E = rng.normal(size=(300, Sm.shape[0])) * np.array([5.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(R.wls_variance(Sm, E), R.gls(Sm, np.diag((E ** 2).mean(axis=0))))


def _shrink_loop(E):
    """Schäfer/Strimmer in Schleifen (unabhängig von der vektorisierten Fassung)."""
    T, m = E.shape
    Ws = E.T @ E / T
    sd = np.sqrt(np.diag(Ws))
    num = den = 0.0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            w = (E[:, i] / sd[i]) * (E[:, j] / sd[j])
            r = w.mean()
            num += T / (T - 1.0) ** 3 * np.sum((w - r) ** 2)
            den += r ** 2
    lam = min(max(num / den, 0.0), 1.0)
    return lam * np.diag(np.diag(Ws)) + (1 - lam) * Ws, lam


def test_shrinkage_matches_a_loop():
    rng = np.random.default_rng(4)
    E = rng.normal(size=(120, 6)) @ rng.normal(size=(6, 6))
    W, lam = R.shrinkage_cov(E)
    W2, lam2 = _shrink_loop(E)
    assert lam == pytest.approx(lam2) and np.allclose(W, W2)


def test_shrinkage_limits():
    rng = np.random.default_rng(5)
    indep = rng.normal(size=(400, 8))
    W, lam_indep = R.shrinkage_cov(indep)
    corr = rng.normal(size=(400, 1)) + 0.05 * rng.normal(size=(400, 8))                      # fast perfekt korreliert
    _, lam_corr = R.shrinkage_cov(corr)
    assert lam_indep > 0.5 and lam_corr < 0.05 and lam_corr < lam_indep and 0.0 <= lam_indep <= 1.0
    assert np.allclose(W, W.T) and np.linalg.eigvalsh(W).min() > 0


def test_sample_covariance_is_singular_with_fewer_rows_than_nodes():
    rng = np.random.default_rng(6)
    E = rng.normal(size=(20, 30))
    assert np.linalg.matrix_rank(R.sample_cov(E)) <= 20 < 30
    assert np.linalg.eigvalsh(R.shrinkage_cov(E)[0]).min() > 0                              # geschrumpft: positiv definit
