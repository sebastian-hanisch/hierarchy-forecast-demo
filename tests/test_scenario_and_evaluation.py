"""Das Depot-Netz, die Basisprognosen mit Modellwahl, die Auswertung und ihre Kennzahlen; die Auswertung darf nichts aus der Zukunft des Ursprungs verwenden."""

import dataclasses

import numpy as np
import pytest

import hrc_constants as C
import hrc_evaluation as E
import hrc_reconcile as R
import hrc_scenario as S


def test_hierarchy_is_deterministic_seeded_and_aggregates():
    a, b, c = S.generate(20, 4, seed=1), S.generate(20, 4, seed=1), S.generate(20, 4, seed=2)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.y.shape == (1 + 4 + 20, C.N_DAYS)
    assert np.array_equal(a.y[0], a.bottom.sum(axis=0)) and all(np.array_equal(a.y[1 + r], a.bottom[a.region == r].sum(axis=0)) for r in range(4))
    assert np.array_equal(np.bincount(a.region), [5, 5, 5, 5]) and (np.diff(a.region) >= 0).all() and list(a.level_of[:6]) == [0, 1, 1, 1, 1, 2]
    assert a.names[0] == "Netz" and a.names[1] == "Region 1" and a.names[5].startswith("Depot 1")


def test_regions_are_limited_to_two_depots_each():
    h = S.generate(6, 5, seed=0)
    assert h.n_regions == 3 and np.bincount(h.region).min() >= 2


def _region_corr(h, r):
    d = h.bottom[h.region == r]
    z = np.log(d + 1.0)
    z = z - z.mean(axis=1, keepdims=True)
    c = np.corrcoef(z)
    return (c.sum() - len(c)) / (len(c) * (len(c) - 1))


def test_tagesschock_raises_the_correlation_within_a_region():
    low = S.generate(20, 4, rho=0.0, swing=0.0, seed=3)
    high = S.generate(20, 4, rho=0.8, swing=0.0, seed=3)
    assert np.mean([_region_corr(high, r) for r in range(4)]) > np.mean([_region_corr(low, r) for r in range(4)]) + 0.05


def test_swing_moves_regional_levels_by_about_its_size():
    # gleicher Seed: Die Zufallsströme und alle festen Bestandteile stimmen überein, nur die Niveauschwankungen sind verschieden gewichtet (swing 0 gegen 0,15)
    flat, wavy = S.generate(20, 4, rho=0.0, swing=0.0, noise_mean=0.04, seed=4), S.generate(20, 4, rho=0.0, swing=0.15, noise_mean=0.04, seed=4)
    d = np.log((wavy.y[1] + 1.0) / (flat.y[1] + 1.0))
    assert 0.08 < d.std() < 0.25 and np.array_equal(flat.region, wavy.region)
    same = np.log((flat.y[1] + 1.0) / (S.generate(20, 4, rho=0.0, swing=0.0, noise_mean=0.04, seed=4).y[1] + 1.0))
    assert same.std() == 0.0


def test_node_promo_share_by_hand():
    h = S.generate(20, 4, seed=5)
    share = E.node_promo_share(h)
    w = h.bottom[:, :C.FIRST_TEST].mean(axis=1)
    assert np.allclose(share[h.m - h.n + 3], h.promo[3]) and np.allclose(share[0], (w[:, None] * h.promo).sum(axis=0) / w.sum())
    r = 2
    assert np.allclose(share[1 + r], (w[h.region == r][:, None] * h.promo[h.region == r]).sum(axis=0) / w[h.region == r].sum())


def _settings(**kw):
    return E.Settings(n_depots=20, n_regions=4, horizon=7, **kw)


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(_settings())


def test_analysis_shapes(analysis):
    a = analysis
    T = C.N_DAYS - 7 - C.FIRST_TEST + 1
    assert a.actual.shape == (25, T, 7) and set(a.forecasts) == set(C.METHODS) and all(f.shape == a.actual.shape for f in a.forecasts.values())
    assert set(a.G) == set(C.METHODS[1:]) and all(g.shape == (20, 25) for g in a.G.values()) and a.picks.shape == (25,) and set(a.picks) <= {0, 1, 2}


def test_every_reconciled_forecast_is_coherent(analysis):
    a = analysis
    h = a.hier
    for m in C.METHODS[1:]:
        f = a.forecasts[m]
        pre_clip = R.apply(h.S, a.G[m], a.forecasts["base"])
        ok = (pre_clip >= 0).all(axis=0)                                                      # Zellen ohne Abschneiden bei 0
        assert ok.mean() > 0.5
        assert np.allclose(f[0][ok], f[h.m - h.n:].sum(axis=0)[ok]) and np.allclose(f[1][ok], f[1 + h.n_regions:][h.region == 0].sum(axis=0)[ok]), m
        assert np.allclose(np.maximum(pre_clip, 0.0), f)
    assert not np.allclose(a.forecasts["base"][0], a.forecasts["base"][h.m - h.n:].sum(axis=0))


def test_bottom_up_keeps_the_depot_forecasts_and_top_down_the_net_forecast(analysis):
    a = analysis
    n = a.hier.n
    assert np.allclose(a.forecasts["bu"][a.hier.m - n:], a.forecasts["base"][a.hier.m - n:])
    assert np.allclose(a.forecasts["td"][0], a.forecasts["base"][0]) and np.allclose(a.forecasts["mo"][1:1 + a.hier.n_regions], a.forecasts["base"][1:1 + a.hier.n_regions])


def test_rmsse_by_hand(analysis):
    a = analysis
    j = 7
    y = a.hier.y[j]
    scale = np.sqrt(np.mean((y[7:C.FIRST_TEST] - y[:C.FIRST_TEST - 7]) ** 2))
    assert a.scale[j] == pytest.approx(scale)
    assert a.rmsse["bu"][j] == pytest.approx(np.sqrt(np.mean((a.forecasts["bu"][j] - a.actual[j]) ** 2)) / scale)
    lv = [a.rmsse["ols"][a.hier.level_of == k].mean() for k in range(3)]
    assert a.summary["ols"]["levels"] == pytest.approx(lv) and a.summary["ols"]["mean"] == pytest.approx(np.mean(lv))


def test_auto_selection_picks_the_smallest_error_model():
    h = E._hierarchy(_settings().hierarchy_key)
    F, org, picks = E.base_forecasts(h, "auto", 7)
    Fs = [E.model_forecasts(h, m, 7)[0] for m in C.MODELS]
    idx = np.nonzero((org >= C.FIRST_TEST - C.SELECT_DAYS - 7) & (org <= C.FIRST_TEST - 7))[0]
    Y = h.y[:, org[idx][:, None] + np.arange(7)[None, :]]
    for j in (0, 3, 11):
        err = [np.mean((f[j, idx] - Y[j]) ** 2) for f in Fs]
        assert picks[j] == int(np.argmin(err)) and np.array_equal(F[j], Fs[picks[j]][j])
    assert F.shape == Fs[0].shape


def test_weekly_mean_is_exactly_coherent_and_reconciliation_leaves_it_alone():
    a = E.analyse(_settings(base="snaive_k"))
    assert a.incoherence < 1e-9
    for m in ("bu", "ols", "wls_struct", "wls_var", "mint_shrink"):
        assert np.allclose(a.forecasts[m], a.forecasts["base"], atol=1e-6), m


def test_forecasts_at_an_origin_ignore_the_future():
    s = _settings()
    h = E._hierarchy(s.hierarchy_key)
    F1, org, _ = E._base(s.hierarchy_key, s.base, s.horizon)
    t = 800
    b2 = h.bottom.copy()
    b2[:, t:] = np.random.default_rng(0).integers(0, 999, size=b2[:, t:].shape)
    h2 = dataclasses.replace(h, y=h.S @ b2)
    F2, _, _ = E.base_forecasts(h2, s.base, s.horizon)
    i = t - C.FIRST_ORIGIN
    assert np.allclose(F1[:, :i + 1], F2[:, :i + 1])
    a1 = E.assemble(h, s, F1, org, np.zeros(h.m, dtype=int))
    a2 = E.assemble(h2, s, F2, org, np.zeros(h.m, dtype=int))
    k = t - C.FIRST_TEST
    for m in C.METHODS:
        assert np.allclose(a1.forecasts[m][:, :k + 1], a2.forecasts[m][:, :k + 1]), m


def test_horizon_one_and_history_window():
    a = E.analyse(E.Settings(n_depots=20, n_regions=4, horizon=1))
    assert a.actual.shape[2] == 1 and np.isfinite(a.summary["mint_shrink"]["mean"])
    short, long = E.analyse(_settings(history=60)), E.analyse(_settings(history=600))
    assert 0.0 <= short.shrink_lambda <= 1.0 and not np.allclose(short.G["wls_var"], long.G["wls_var"]) and np.allclose(short.G["bu"], long.G["bu"])
    assert E.one_step_errors(short.hier, *E._base(short.settings.hierarchy_key, short.settings.base, 7)[:2], 60).shape == (60, 25)
