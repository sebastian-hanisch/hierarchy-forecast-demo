"""Hierarchische Abstimmung: aus den Basisprognosen $\\hat y$ aller Knoten (Netz, Regionen, Depots) werden kohärente Prognosen $\\tilde y = S\\,G\\,\\hat y$, die sich exakt addieren. Alle Verfahren sind Wahlen der Matrix $G$ (n_b x m):

  Bottom-up          G = [0 | I]                              nur die Depot-Prognosen zählen, alles darüber ist ihre Summe
  Top-down           G = [p | 0]                              nur die Netz-Prognose zählt, Depots bekommen ihren mittleren Anteil p am Netz (Mittel der Tagesanteile auf den Trainingstagen)
  Middle-out         G[i, Region(i)] = q_i                    die Regions-Prognosen zählen, Depots bekommen ihren mittleren Anteil q an der Region; darüber Summe
  OLS                G = (S'S)^-1 S'                          alle Knoten gleich gewichtet
  WLS (Struktur)     G = (S'W^-1 S)^-1 S'W^-1, W = diag(S 1)  Gewicht = Zahl der Depots unter dem Knoten
  WLS (Fehlervarianz) W = diag(Fehlervarianz je Knoten)
  MinT               W = Kovarianz der Basis-Fehler (Wickramasuriya et al. 2019): Stichprobe oder nach Schäfer/Strimmer zur Diagonalen geschrumpft

Die letzten fünf sind die verallgemeinerte Kleinste-Quadrate-Lösung $G = (S' W^{-1} S)^{-1} S' W^{-1}$ mit verschiedenem $W$; erfüllt $G S = I$ (unverzerrt, wenn die Basisprognosen es sind)."""

import numpy as np


def bottom_up(S):
    m, n = S.shape
    G = np.zeros((n, m))
    G[:, m - n:] = np.eye(n)
    return G


def top_down(S, props):
    """props: (n,) Anteil jedes Depots am Netz."""
    m, n = S.shape
    G = np.zeros((n, m))
    G[:, 0] = props
    return G


def middle_out(S, region, props_in_region):
    """props_in_region: (n,) Anteil jedes Depots an seiner Region; Regions-Knoten stehen in den Zeilen 1..R."""
    m, n = S.shape
    G = np.zeros((n, m))
    G[np.arange(n), 1 + region] = props_in_region
    return G


def gls(S, W):
    """G = (S' W^-1 S)^-1 S' W^-1 für eine symmetrische, positiv definite W (m, m)."""
    Winv_S = np.linalg.solve(W, S)
    return np.linalg.solve(S.T @ Winv_S, Winv_S.T)


def ols(S):
    return np.linalg.solve(S.T @ S, S.T)


def wls_structure(S):
    return gls(S, np.diag(S.sum(axis=1)))


def wls_variance(S, E):
    """E: (T, m) Ein-Schritt-Fehler der Basisprognosen auf den Trainingstagen."""
    return gls(S, np.diag(np.mean(E ** 2, axis=0)))


def sample_cov(E):
    """Stichprobenkovarianz ohne Zentrierung (die Fehler haben den Mittelwert 0 als Erwartung): E'E / T."""
    return E.T @ E / E.shape[0]


def shrinkage_cov(E):
    """Zur Diagonalen geschrumpfte Kovarianz (Schäfer/Strimmer 2005): W = λ diag(W_s) + (1 - λ) W_s mit λ = Σ_{i≠j} var(r_ij) / Σ_{i≠j} r_ij², auf [0, 1] begrenzt.
    Rückgabe (W, λ)."""
    T = E.shape[0]
    Ws = sample_cov(E)
    sd = np.sqrt(np.maximum(np.diag(Ws), 1e-12))
    U = E / sd
    R = U.T @ U / T
    # Varianz der Korrelationsschätzung r_ij = mean_t(u_it u_jt): T / (T-1)^3 * Σ_t (u_it u_jt - r_ij)^2 = T^2 / (T-1)^3 * (mean(u_i^2 u_j^2) - r_ij^2)
    sq = (U ** 2).T @ (U ** 2) / T
    var_r = T ** 2 / (T - 1.0) ** 3 * (sq - R ** 2)
    off = ~np.eye(R.shape[0], dtype=bool)
    lam = float(np.clip(var_r[off].sum() / max((R[off] ** 2).sum(), 1e-12), 0.0, 1.0))
    W = lam * np.diag(np.diag(Ws)) + (1.0 - lam) * Ws
    return W, lam


def mint(S, E, shrink=True):
    """MinT-Matrix G und (bei Schrumpfung) das gewählte λ."""
    if shrink:
        W, lam = shrinkage_cov(E)
    else:
        W, lam = sample_cov(E), 0.0
    return gls(S, W), lam


def apply(S, G, F):
    """Kohärente Prognosen S G F für Basisprognosen F (m, ...) beliebiger Restform."""
    m = S.shape[0]
    shape = F.shape
    return (S @ (G @ F.reshape(m, -1))).reshape(shape)
