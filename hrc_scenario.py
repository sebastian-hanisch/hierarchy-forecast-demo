"""Vehikel "Depot-Hierarchie": Netz -> Regionen -> Depots. Jedes Depot ist aufgebaut wie in den Vorgängern (Niveau, Trend, Wochen- und Jahresmuster, Feiertage, Aktionen, multiplikatives log-normales Rauschen);
dazu kommen zwei Quellen für gemeinsame Bewegungen innerhalb der Hierarchie:

  Niveauschwankungen   im Log mittelwertrückkehrende Zufallsgänge (AR(1), φ = 0,98) je Region (Stärke = Regler), je Netz (halb so stark) und je Depot (fest, klein)
  Tagesschocks         das Rauschen eines Depots besteht zu einem Anteil ρ aus einem Tagesschock, den sich alle Depots einer Region teilen (Wetter, Streik, Sperrung)

Reihen aggregieren durch Summen: die Region ist die Summe ihrer Depots, das Netz die Summe der Regionen."""

from dataclasses import dataclass

import numpy as np

import hrc_constants as C


@dataclass(frozen=True)
class Hierarchy:
    y: np.ndarray             # (m, T) Aufträge aller Knoten: Netz, Regionen, Depots (Summen)
    region: np.ndarray        # (n,) Region je Depot (aufsteigend sortiert)
    S: np.ndarray             # (m, n) Summationsmatrix
    names: list
    level_of: np.ndarray      # (m,) 0 = Netz, 1 = Region, 2 = Depot
    level: np.ndarray         # (n,) Ausgangsniveau der Depots
    noise: np.ndarray         # (n,) Streuung des Rauschens
    seed: int
    dow: np.ndarray           # (T,) Wochentag
    holiday: np.ndarray       # (T,) Feiertag
    after: np.ndarray         # (T,) Tag nach dem Feiertag
    promo: np.ndarray         # (n, T) Aktionstage der Depots

    @property
    def n(self):
        return self.S.shape[1]

    @property
    def m(self):
        return self.S.shape[0]

    @property
    def n_regions(self):
        return int(self.region.max()) + 1

    @property
    def bottom(self):
        return self.y[self.m - self.n:]


def summing_matrix(region):
    """S (m, n): Zeile 0 = Netz (alles), dann je Region, dann die Einheitsmatrix der Depots."""
    n, R = len(region), int(region.max()) + 1
    S = np.zeros((1 + R + n, n))
    S[0] = 1.0
    for r in range(R):
        S[1 + r, region == r] = 1.0
    S[1 + R:] = np.eye(n)
    return S


def calendar(n_days=C.N_DAYS):
    t = np.arange(n_days)
    holiday = np.isin(t % 365, C.HOLIDAY_DOY).astype(float)
    after = np.roll(holiday, 1)
    after[0] = 0.0
    return t % 7, holiday, after


def _ar1(rng, sd, n_series, n_days):
    """Mittelwertrückkehrende Zufallsgänge mit stationärer Streuung sd: (n_series, n_days)."""
    phi = C.AR_PHI
    innov = sd * np.sqrt(1.0 - phi ** 2)
    x = np.zeros((n_series, n_days))
    x[:, 0] = sd * rng.normal(size=n_series)
    e = innov * rng.normal(size=(n_series, n_days))
    for t in range(1, n_days):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def generate(n_depots=30, n_regions=5, noise_mean=C.DEFAULT_NOISE, rho=C.DEFAULT_RHO, swing=C.DEFAULT_SWING, seed=0, n_days=C.N_DAYS):
    """n_regions wird auf höchstens n_depots // 2 begrenzt (jede Region hat mindestens zwei Depots)."""
    R = max(1, min(int(n_regions), n_depots // 2))
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dow, holiday, after = calendar(n_days)
    pattern = np.array(C.WEEKLY_PATTERN)
    pattern = pattern / pattern.mean()
    region = np.sort(rng.permutation(np.arange(n_depots) % R))
    level = np.clip(C.LEVEL * np.exp(0.6 * rng.normal(size=n_depots)), 30.0, 500.0)
    noise = np.clip(noise_mean * np.exp(0.3 * rng.normal(size=n_depots)), 0.03, 0.6)
    weekly = rng.uniform(0.5, 1.5, size=n_depots)
    yearly = rng.uniform(0.0, 0.4, size=n_depots)
    trend = C.TREND_MEAN + 10.0 * rng.normal(size=n_depots)
    g_net = _ar1(rng, swing * C.NET_SHARE, 1, n_days)[0]
    g_reg = _ar1(rng, swing, R, n_days)
    g_dep = _ar1(rng, C.DEPOT_SWING, n_depots, n_days)
    shock = rng.normal(size=(R, n_days))
    y = np.zeros((n_depots, n_days))
    promos = np.zeros((n_depots, n_days))
    for i in range(n_depots):
        week_f = 1.0 + weekly[i] * (pattern[dow] - 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        year_f = 1.0 + yearly[i] * np.sin(2 * np.pi * (t % 365) / 365.0 + phase)
        trend_f = 1.0 + (trend[i] / 100.0) * t / 365.0
        holiday_f = 1.0 - C.EVENTS * C.HOLIDAY_DROP * holiday + C.EVENTS * C.HOLIDAY_REBOUND * after
        promo = np.zeros(n_days)
        for s in rng.choice(np.arange(30, n_days - C.PROMO_LENGTH), size=C.PROMO_PER_YEAR * (n_days // 365), replace=False):
            promo[s:s + C.PROMO_LENGTH] = 1.0
        promo_f = 1.0 + C.EVENTS * 0.5 * promo
        promos[i] = promo
        mu = level[i] * np.maximum(trend_f, 0.05) * week_f * year_f * holiday_f * promo_f * np.exp(g_net + g_reg[region[i]] + g_dep[i])
        xi = np.sqrt(1.0 - rho) * rng.normal(size=n_days) + np.sqrt(rho) * shock[region[i]]
        y[i] = np.maximum(np.rint(mu * np.exp(noise[i] * xi - 0.5 * noise[i] ** 2)), 0.0)
    S = summing_matrix(region)
    names = ["Netz"] + [f"Region {r + 1}" for r in range(R)] + [f"Depot {i + 1} (R{region[i] + 1})" for i in range(n_depots)]
    level_of = np.concatenate([[0], np.ones(R, dtype=int), np.full(n_depots, 2)])
    return Hierarchy(S @ y, region, S, names, level_of, level, noise, int(seed), dow, holiday, after, promos)
