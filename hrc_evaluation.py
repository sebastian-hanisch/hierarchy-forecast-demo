"""Auswertung der hierarchischen Abstimmung: Basisprognosen je Knoten (Holt-Winters oder Wochenmittel, jeder Knoten für sich), neun Arten, sie kohärent zu machen, im Rolling-Origin-Vergleich über das Testjahr, dazu drei Experimente.

Kennzahl: **RMSSE** je Knoten (Wurzel des mittleren quadratischen Fehlers durch den saisonal naiven Trainingsfehler des Knotens), dann je Ebene (Netz, Regionen, Depots) gemittelt und über die drei Ebenen gemittelt ("Ø Ebenen").
Die Fehlerkovarianz für MinT und die Varianzgewichte stammen aus den Ein-Schritt-Fehlern der Basisprognosen auf den letzten `history` Trainingstagen."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import hrc_baselines as B
import hrc_constants as C
import hrc_reconcile as R
import hrc_scenario as S


@dataclass(frozen=True)
class Settings:
    n_depots: int = C.DEFAULT_DEPOTS
    n_regions: int = C.DEFAULT_REGIONS
    noise: float = C.DEFAULT_NOISE
    rho: float = C.DEFAULT_RHO
    swing: float = C.DEFAULT_SWING
    horizon: int = C.DEFAULT_HORIZON
    base: str = "auto"
    history: int = C.DEFAULT_HISTORY
    seed: int = 3

    @property
    def hierarchy_key(self):
        return (self.n_depots, self.n_regions, self.noise, self.rho, self.swing, self.seed)


@dataclass
class Analysis:
    settings: Settings
    hier: S.Hierarchy
    test_org: np.ndarray
    actual: np.ndarray             # (m, T, h)
    forecasts: dict                # Verfahren -> (m, T, h)
    scale: np.ndarray              # (m,) RMSSE-Nenner
    rmsse: dict                    # Verfahren -> (m,)
    summary: dict                  # Verfahren -> {"levels": (3,), "mean": Ø Ebenen}
    G: dict                        # Verfahren -> (n, m)
    shrink_lambda: float
    picks: np.ndarray              # (m,) gewähltes Modell je Knoten (Index in C.MODELS)
    incoherence: float             # mittlere Abweichung Netz-Prognose gegen Summe der Depot-Prognosen (relativ zur Netz-Prognose)


@lru_cache(maxsize=16)
def _hierarchy(key):
    n, R_, noise, rho, swing, seed = key
    return S.generate(n, R_, noise, rho, swing, seed)


def model_forecasts(hier, model, horizon):
    """Basisprognosen aller Knoten für alle Ursprünge ab FIRST_ORIGIN (Training und Test): (m, O_all, h) und die Ursprünge. Parameter aus den Tagen vor FIRST_TEST."""
    org = np.arange(C.FIRST_ORIGIN, C.N_DAYS - horizon + 1)
    out = []
    share = node_promo_share(hier)
    for j in range(hier.m):
        y = hier.y[j]
        if model == "hw_mult":
            out.append(B.hw_forecast(y, org, horizon))
        elif model == "regression":
            X = B.regression_design(hier.dow, hier.holiday, hier.after, share[j])
            out.append(B.regression_forecast(y, X, org, horizon))
        else:
            out.append(B.snaive_k(y, org, horizon))
    return np.stack(out), org


def base_forecasts(hier, model, horizon):
    """Basisprognosen (m, O_all, h), die Ursprünge und das gewählte Modell je Knoten. Bei "auto" wählt jeder Knoten das Modell mit dem kleinsten quadratischen Prognosefehler über alle Horizonte
    auf den letzten SELECT_DAYS Trainingstagen (Parameter aus allen Trainingstagen: die Auswahl ist nicht ehrlich außerhalb der Stichprobe, wie die Modellwahl in der Praxis oft)."""
    if model != "auto":
        F, org = model_forecasts(hier, model, horizon)
        return F, org, np.full(F.shape[0], C.MODELS.index(model))
    Fs = [model_forecasts(hier, b, horizon) for b in C.MODELS]
    org = Fs[0][1]
    idx = np.nonzero((org >= C.FIRST_TEST - C.SELECT_DAYS - horizon) & (org <= C.FIRST_TEST - horizon))[0]
    Y = hier.y[:, org[idx][:, None] + np.arange(horizon)[None, :]]
    err = np.stack([np.mean((F[:, idx] - Y) ** 2, axis=(1, 2)) for F, _ in Fs])
    pick = err.argmin(axis=0)
    return np.stack([Fs[pick[j]][0][j] for j in range(hier.m)]), org, pick


@lru_cache(maxsize=16)
def _base(key, model, horizon):
    return base_forecasts(_hierarchy(key), model, horizon)


def node_promo_share(hier):
    """Aktionsanteil je Knoten und Tag: das nach Depot-Größe (mittlere Aufträge der Trainingstage) gewichtete Mittel der Depot-Aktionen. Für ein Depot ist es sein eigener Aktionsplan: (m, T)."""
    w = hier.bottom[:, :C.FIRST_TEST].mean(axis=1)
    return (hier.S @ (w[:, None] * hier.promo)) / (hier.S @ w)[:, None]


def _scales(hier):
    return np.array([np.sqrt(np.mean((hier.y[j, 7:C.FIRST_TEST] - hier.y[j, :C.FIRST_TEST - 7]) ** 2)) for j in range(hier.m)])


def one_step_errors(hier, F_all, org_all, history):
    """Ein-Schritt-Fehler y - F der Basisprognosen auf den letzten `history` Trainingstagen: (T_res, m)."""
    keep = (org_all < C.FIRST_TEST) & (org_all >= C.FIRST_TEST - history)
    idx = np.nonzero(keep)[0]
    return (hier.y[:, org_all[idx]] - F_all[:, idx, 0]).T


def proportions(hier, first=C.FIRST_ORIGIN, last=C.FIRST_TEST):
    """Mittlere Tagesanteile der Depots am Netz und an ihrer Region auf den Trainingstagen: (p_net, p_region)."""
    b = hier.bottom[:, first:last]
    tot = hier.y[0, first:last]
    reg = hier.y[1 + hier.region][:, first:last]
    return (b / tot).mean(axis=1), (b / reg).mean(axis=1)


def reconciliation_matrices(hier, E):
    """Alle G-Matrizen (n, m) und das Schrumpfungs-λ von MinT."""
    p_net, p_reg = proportions(hier)
    G = {"bu": R.bottom_up(hier.S), "td": R.top_down(hier.S, p_net), "mo": R.middle_out(hier.S, hier.region, p_reg), "ols": R.ols(hier.S), "wls_struct": R.wls_structure(hier.S),
         "wls_var": R.wls_variance(hier.S, E)}
    G["mint_shrink"], lam = R.mint(hier.S, E, shrink=True)
    G["mint_sample"], _ = R.mint(hier.S, E, shrink=False)
    return G, lam


def _level_means(values, level_of):
    return np.array([values[level_of == k].mean() for k in range(3)])


def assemble(hier, s, F_all, org_all, picks):
    """Abstimmung, Auswertung und Kennzahlen aus den Basisprognosen aller Ursprünge."""
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - s.horizon + 1)
    F = F_all[:, test_org - C.FIRST_ORIGIN]
    days = test_org[:, None] + np.arange(s.horizon)[None, :]
    Y = hier.y[:, days]
    E = one_step_errors(hier, F_all, org_all, s.history)
    G, lam = reconciliation_matrices(hier, E)
    fc = {"base": F}
    for k, g in G.items():
        fc[k] = np.maximum(R.apply(hier.S, g, F), 0.0)
    scale = _scales(hier)
    rmsse = {k: np.sqrt(np.mean((f - Y) ** 2, axis=(1, 2))) / scale for k, f in fc.items()}
    summ = {}
    for k, v in rmsse.items():
        lv = _level_means(v, hier.level_of)
        summ[k] = {"levels": lv, "mean": float(lv.mean())}
    incoh = float(np.mean(np.abs(F[0] - F[hier.m - hier.n:].sum(axis=0)) / np.maximum(F[0], 1.0)))
    return Analysis(s, hier, test_org, Y, fc, scale, rmsse, summ, G, lam, picks, incoh)


@lru_cache(maxsize=6)
def analyse(s):
    return assemble(_hierarchy(s.hierarchy_key), s, *_base(s.hierarchy_key, s.base, s.horizon))


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _replace(base, **kw):
    d = {f: getattr(base, f) for f in base.__dataclass_fields__}
    d.update(kw)
    return Settings(**d)


def structure_experiment(swings=None, rhos=None, seeds=None, base=None):
    """RMSSE je Ebene und Verfahren bei wachsender gemeinsamer Bewegung: (a) regionale Niveauschwankung, (b) Tagesschock-Anteil."""
    swings = C.SWING_LEVELS if swings is None else swings
    rhos = C.RHO_LEVELS if rhos is None else rhos
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    out = {"swing": [], "rho": []}
    for key, levels in (("swing", swings), ("rho", rhos)):
        for x in levels:
            per = {m: [] for m in C.METHODS}
            for sd in seeds:
                a = analyse(_replace(base, seed=sd, **{key: x}))
                for m in C.METHODS:
                    per[m].append(a.summary[m]["levels"])
            out[key].append({"x": x, "n_seeds": len(seeds), "levels": {m: [_mean_se([v[k] for v in per[m]]) for k in range(3)] for m in C.METHODS}})
    return out


def history_experiment(levels=None, seeds=None, base=None):
    """Ø Ebenen und Depot-Ebene bei wachsender Zahl der Trainingstage für die Fehlerkovarianz (viele Knoten, wenig Fehler: die Stichprobenkovarianz wird instabil)."""
    levels = C.HISTORY_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_HISTORY_DEPOTS) if base is None else base
    rows = []
    for w in levels:
        per = {m: [] for m in C.METHODS}
        lam = []
        for sd in seeds:
            a = analyse(_replace(base, history=w, seed=sd))
            for m in C.METHODS:
                per[m].append((a.summary[m]["mean"], a.summary[m]["levels"][2]))
            lam.append(a.shrink_lambda)
        rows.append({"history": w, "n_seeds": len(seeds), "mean": {m: _mean_se([v[0] for v in per[m]]) for m in C.METHODS}, "depots": {m: _mean_se([v[1] for v in per[m]]) for m in C.METHODS},
                     "lambda": _mean_se(lam)})
    return rows


def horizon_experiment(levels=None, seeds=None, base=None):
    """Ø Ebenen je Verfahren bei wachsendem Prognosehorizont: die Fehlerkovarianz stammt aus Ein-Schritt-Fehlern, gilt also am besten für kurze Horizonte."""
    levels = C.HORIZON_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    rows = []
    for h in levels:
        per = {m: [] for m in C.METHODS}
        for sd in seeds:
            a = analyse(_replace(base, horizon=h, seed=sd))
            for m in C.METHODS:
                per[m].append(a.summary[m]["mean"])
        rows.append({"horizon": h, "n_seeds": len(seeds), "mean": {m: _mean_se(per[m]) for m in C.METHODS}})
    return rows
