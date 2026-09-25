"""Plotly-Abbildungen der Hierarchie-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import hrc_constants as C

COLORS = {"base": "#7f7f7f", "bu": "#1f77b4", "td": "#d62728", "mo": "#e6550d", "ols": "#8c6bb1", "wls_struct": "#17becf", "wls_var": "#2ca02c", "mint_shrink": "#00897b", "mint_sample": "#a65628"}
ACTUAL = "#14233B"
WARN = "#f58518"
SHORT = C.METHOD_SHORT


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _tall(fig, height):
    """Für Diagramme mit vielen Legendeneinträgen: mehr Platz unter der Achse."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22, yanchor="top"), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_node(a, node, origin, methods):
    """42 Tage vor dem Ursprung und die nächsten h Tage eines Knotens: Ist und die Prognosen der gewählten Verfahren."""
    h = a.settings.horizon
    i = int(origin - a.test_org[0])
    y = a.hier.y[node]
    x_hist, x_fut = np.arange(origin - 42, origin), np.arange(origin, origin + h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=y[origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=y[origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=1), marker=dict(size=7, symbol="circle-open")))
    for m in methods:
        fig.add_trace(go.Scatter(x=x_fut, y=a.forecasts[m][node, i], mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=2.5 if m != "base" else 1.6, dash="dash" if m == "base" else "solid")))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_coherence(a, origin, method):
    """Die Basisprognosen des Netzes, der Regionen und der Depots am selben Ursprung als Summen über die Ebenen; dazu die abgestimmte Netz-Prognose."""
    h = a.settings.horizon
    i = int(origin - a.test_org[0])
    x = np.arange(origin, origin + h)
    F = a.forecasts["base"][:, i]                                                                # (m, h)
    hier = a.hier
    R = hier.n_regions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hier.y[0, origin:origin + h], mode="lines+markers", name="Netz: tatsächlich", line=dict(color=ACTUAL, width=1), marker=dict(size=6, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=x, y=F[0], mode="lines", name="Netz-Prognose (Basis)", line=dict(color="#d62728", width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=F[1:1 + R].sum(axis=0), mode="lines", name="Summe der Regions-Prognosen", line=dict(color="#e6550d", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=F[1 + R:].sum(axis=0), mode="lines", name="Summe der Depot-Prognosen", line=dict(color="#1f77b4", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=a.forecasts[method][0, i], mode="lines", name=f"Netz abgestimmt: {SHORT[method]}", line=dict(color=COLORS[method], width=3)))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag im Netz", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_levels(a):
    """RMSSE je Verfahren für Netz, Regionen, Depots und den Mittelwert der Ebenen."""
    xs = list(C.LEVELS) + ["Ø Ebenen"]
    fig = go.Figure()
    for m in C.METHODS:
        v = list(a.summary[m]["levels"]) + [a.summary[m]["mean"]]
        fig.add_trace(go.Bar(x=xs, y=v, name=SHORT[m], marker=dict(color=COLORS[m]), hovertemplate="%{x}, " + SHORT[m] + ": %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="RMSSE (kleiner ist besser)", rangemode="tozero")
    return _tall(fig, 420)


def build_relative(a):
    """Veränderung des RMSSE gegenüber der Basisprognose in Prozent, je Verfahren und Ebene."""
    xs = list(C.LEVELS) + ["Ø Ebenen"]
    base = list(a.summary["base"]["levels"]) + [a.summary["base"]["mean"]]
    fig = go.Figure()
    for m in C.METHODS[1:]:
        v = list(a.summary[m]["levels"]) + [a.summary[m]["mean"]]
        d = [100 * (x / b - 1) for x, b in zip(v, base)]
        fig.add_trace(go.Bar(x=xs, y=d, name=SHORT[m], marker=dict(color=COLORS[m]), hovertemplate="%{x}, " + SHORT[m] + ": %{y:+.1f} %<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#7f7f7f", width=1))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="RMSSE gegenüber der Basis (%, negativ ist besser)")
    return _tall(fig, 420)


def build_weights(a, depot_node, methods):
    """Gewichte, mit denen die Basisprognosen aller Knoten in die abgestimmte Prognose eines Depots eingehen (Zeile von G)."""
    hier = a.hier
    j = depot_node - (hier.m - hier.n)
    ms = [m for m in methods if m in a.G]
    z = np.array([a.G[m][j] for m in ms])
    fig = go.Figure(go.Heatmap(z=z, x=hier.names, y=[SHORT[m] for m in ms], colorscale="RdBu", zmid=0, colorbar=dict(title="Gewicht"), hovertemplate="%{y}, Basis von %{x}: %{z:.3f}<extra></extra>"))
    fig.update_xaxes(showticklabels=False, title_text=f"Basisprognosen der Knoten: Netz, {hier.n_regions} Regionen, {hier.n} Depots")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 60 + 45 * max(len(ms), 1))


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_sweep(rows, x_label, fmt, key_methods=("bu", "ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample")):
    """Ø Ebenen gegenüber der Basisprognose (Prozent) bei wachsender gemeinsamer Bewegung."""
    xs = [fmt(r["x"]) for r in rows]
    fig = go.Figure()
    for m in key_methods:
        d = []
        for r in rows:
            mean = np.mean([r["levels"][m][k][0] for k in range(3)])
            b = np.mean([r["levels"]["base"][k][0] for k in range(3)])
            d.append(100 * (mean / b - 1))
        fig.add_trace(go.Bar(x=xs, y=d, name=SHORT[m], marker=dict(color=COLORS[m]), text=[f"{x:+.1f}".replace(".", ",") for x in d], textposition="outside", textfont=dict(size=9)))
    fig.add_hline(y=0, line=dict(color="#7f7f7f", width=1))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text=x_label, type="category")
    fig.update_yaxes(title_text="Ø Ebenen gegenüber der Basis (%, negativ ist besser)")
    return _base(fig, 360)


def build_horizon(rows, methods=("bu", "ols", "wls_struct", "wls_var", "mint_shrink", "mint_sample")):
    xs = [str(r["horizon"]) for r in rows]
    fig = go.Figure()
    for m in methods:
        d = [100 * (r["mean"][m][0] / r["mean"]["base"][0] - 1) for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=d, mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.5)))
    fig.add_hline(y=0, line=dict(color="#7f7f7f", dash="dash"), annotation_text="Basisprognose", annotation_position="top left")
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", type="category")
    fig.update_yaxes(title_text="Ø Ebenen gegenüber der Basis (%, negativ ist besser)")
    return _base(fig, 360)


def build_history(rows, methods=("base", "bu", "wls_var", "mint_shrink", "mint_sample")):
    xs = [f"{r['history']} Tage" for r in rows]
    fig = go.Figure()
    for m in methods:
        fig.add_trace(go.Scatter(x=xs, y=[r["mean"][m][0] for r in rows], error_y=dict(type="data", array=[r["mean"][m][1] for r in rows]), mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.5)))
    fig.update_xaxes(title_text="Trainingstage für die Fehlerkovarianz", type="category")
    fig.update_yaxes(title_text="Ø Ebenen RMSSE (logarithmisch)", type="log")
    return _base(fig, 360)
