"""Hierarchische Abstimmung - Prognosen, die sich addieren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: ein Netz von Depots in Regionen, jeder Knoten mit eigener Prognose - und neun Arten, die Prognosen so abzustimmen, dass sie sich addieren
(Bottom-up, Top-down, Middle-out, OLS, WLS, MinT), gemessen je Ebene.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import hrc_constants as C
from hrc_evaluation import Settings, analyse, history_experiment, horizon_experiment, structure_experiment
from hrc_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from hrc_visualization import SHORT, build_coherence, build_history, build_horizon, build_levels, build_node, build_relative, build_sweep, build_weights

st.set_page_config(page_title="Hierarchische Abstimmung – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=1):
    return f"{de(x, digits)} %"


def rel(v, base):
    return 100.0 * (v / base - 1.0)


def spct(x, digits=1):
    """Prozent mit ausdrücklichem Vorzeichen (Minuszeichen U+2212)."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return ("+" if x > 0 else "−" if x < 0 else "±") + f"{abs(x):.{digits}f}".replace(".", ",") + " %"


@st.cache_data(show_spinner=False)
def _structure(swings, rhos, seeds):
    return structure_experiment(swings, rhos, seeds)


@st.cache_data(show_spinner=False)
def _horizon(levels, seeds):
    return horizon_experiment(levels, seeds)


@st.cache_data(show_spinner=False)
def _history(levels, seeds):
    return history_experiment(levels, seeds)


st.title("🌳 Hierarchische Abstimmung – Prognosen, die sich addieren")
st.markdown(
    """
Ein Netz von Depots, in Regionen gegliedert: **jeder Knoten bekommt seine eigene Prognose** - das Netz, jede Region, jedes Depot. Die Prognosen widersprechen sich: die Summe der Depot-Prognosen ist nicht die Prognose der Region, die Summe der Regionen nicht die des Netzes. Wer damit plant, muss sich entscheiden.
**Hierarchische Abstimmung** macht aus den Basisprognosen kohärente, die sich exakt addieren - **Bottom-up** (nur die Depots zählen), **Top-down** (nur das Netz zählt), **Middle-out** (die Regionen), und die Verfahren, die alle Knoten zusammen verwenden: **OLS, WLS und MinT** (kleinste Fehlervarianz nach Wickramasuriya et al.).
Die Demo misst auf einem erzeugten Depot-Netz, **was das an Genauigkeit bringt** - je Ebene, je Horizont, bei wachsender gemeinsamer Bewegung und bei wenig Fehlerhistorie - und was es kostet. Alle Daten sind erzeugt; die Rechnung ist in numpy geschrieben.
"""
)
st.caption(
    "Achtes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe. **Bezug zu OR:** Netzplanung braucht Zahlen, die zusammenpassen: die Kapazität der Region muss die Summe der Depots tragen; die Abstimmung liefert sie, "
    "und die Frage ist, welche Ebene dabei den Ton angibt."
)

with st.expander("So wird abgestimmt", expanded=True):
    st.markdown(
        """
1. **Die Hierarchie.** $m$ Knoten, $n$ Depots unten: $y = S\\,b$ mit der Summationsmatrix $S$ (Zeile Netz: lauter Einsen, Zeile Region: die Depots der Region, darunter die Einheitsmatrix). Jeder Knoten hat eine Reihe von Tagesaufträgen.
2. **Basisprognosen.** Jeder Knoten wird für sich prognostiziert ($\\hat y$): Holt-Winters, die Regression auf Kalender und Aktionsanteil, das Wochenmittel - oder das beste der drei je Knoten. Das Ergebnis ist **nicht kohärent**.
3. **Abstimmung.** $\\tilde y = S\\,G\\,\\hat y$: die Matrix $G$ ($n \\times m$) bildet alle Basisprognosen auf die Depots ab, $S$ addiert wieder hoch. **Bottom-up** nimmt nur die Depot-Zeilen, **Top-down** verteilt die Netz-Prognose nach mittleren Anteilen, **Middle-out** die Regions-Prognosen.
4. **OLS, WLS, MinT.** Alle drei sind $G = (S' W^{-1} S)^{-1} S' W^{-1}$ mit verschiedenem $W$: die Einheitsmatrix (OLS), die Zahl der Depots je Knoten oder die Fehlervarianz (WLS), die ganze **Fehlerkovarianz** (MinT). Bei vielen Knoten und wenig Fehlern ist die Stichprobenkovarianz instabil: **geschrumpft** (Schäfer/Strimmer) zur Diagonalen wird sie robust.
5. **Kennzahl.** Der RMSSE je Knoten (Wurzel des mittleren quadratischen Fehlers durch den saisonal naiven Trainingsfehler), je Ebene gemittelt und über die drei Ebenen (Netz, Regionen, Depots) gemittelt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Netz**")
    n_depots = st.slider("Zahl der Depots", *bounds("depots_slider"), key="depots_slider", step=C.DEPOTS_STEP, help="Wie viele Depots das Netz hat.")
    n_regions = st.slider("Zahl der Regionen", *bounds("regions_slider"), key="regions_slider", help="In wie viele Regionen die Depots gegliedert sind (gleich große Gruppen).")
    noise = st.slider("Rauschen (Mittel der Depots)", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Mittlere Streuung des multiplikativen Tagesrauschens.")
    rho = st.slider("Gemeinsamer Tagesschock der Region (Anteil)", *bounds("rho_slider"), key="rho_slider", step=C.RHO_STEP, help="Anteil des Rauschens, den sich alle Depots einer Region teilen (Wetter, Streik, Sperrung).")
    swing = st.slider("Gemeinsame Niveauschwankung (Region)", *bounds("swing_slider"), key="swing_slider", step=C.SWING_STEP, help="Stärke der langsamen, gemeinsamen Niveauschwankungen je Region (im Log; das Netz schwankt halb so stark).")
    st.markdown("**Prognose und Abstimmung**")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird.")
    base = st.selectbox("Basisprognose je Knoten", list(C.BASES), key="base_select", format_func=lambda k: C.BASE_NAMES[k], help="Womit jeder Knoten für sich prognostiziert wird.")
    history = st.slider("Trainingstage für die Fehlerkovarianz", *bounds("history_slider"), key="history_slider", step=C.HISTORY_STEP, help="Aus den Ein-Schritt-Fehlern der letzten so vielen Trainingstage werden die Gewichte von WLS und MinT geschätzt.")
    st.markdown("**Anzeige**")
    methods = st.multiselect("Verfahren in den Diagrammen", list(C.METHODS), key="methods_select", format_func=lambda m: C.METHOD_NAMES[m], help="Welche Prognosen im Diagramm eines Knotens gezeigt werden; ohne Auswahl zeigt die App die Basisprognose und Bottom-up.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt das ganze Netz fest.")
    st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed)

shown = [m for m in C.METHODS if m in methods] or ["base", "bu"]
sync_query_params({"depots_slider": int(n_depots), "regions_slider": int(n_regions), "noise_slider": round(float(noise), 2), "rho_slider": round(float(rho), 2), "swing_slider": round(float(swing), 2), "horizon_slider": int(horizon),
                   "base_select": base, "history_slider": int(history), "methods_select": shown, "seed_input": int(seed)})

settings = Settings(int(n_depots), int(n_regions), round(float(noise), 2), round(float(rho), 2), round(float(swing), 2), int(horizon), base, int(history), int(seed))
if not methods:
    st.warning("Kein Verfahren gewählt: die Diagramme zeigen die Basisprognose und Bottom-up.")
with st.spinner("Die Basisprognosen aller Knoten und die Abstimmung werden berechnet ..."):
    a = analyse(settings)
hier, sm = a.hier, a.summary

# --- Ein Knoten ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ein Knoten und die Prognosen an einem Ursprung")
st.session_state["node_select"] = min(hier.m - 1, max(0, st.session_state.get("node_select", 0)))
node = int(st.selectbox("Knoten", list(range(hier.m)), key="node_select", format_func=lambda j: hier.names[j], help="Welcher Knoten des Netzes gezeigt wird: das Netz, eine Region oder ein Depot."))
lo_o, hi_o = int(a.test_org[0]), int(a.test_org[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor. Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_node(a, node, origin, shown), width="stretch", key="node_chart")
st.caption(f"{hier.names[node]}, Ebene {C.LEVELS[hier.level_of[node]]}. Gestrichelt grau die Basisprognose dieses Knotens, farbig die abgestimmten Prognosen. Alle Verfahren bis auf Bottom-up, Top-down und Middle-out ändern auch die Prognose dieses Knotens, wenn sich die anderen Knoten ändern.")
coh_method = next((m for m in shown if m != "base"), "bu")
st.plotly_chart(build_coherence(a, origin, coh_method), width="stretch", key="coherence_chart")
gap = a.incoherence
st.caption(
    f"Die Basisprognosen widersprechen sich: im Mittel über alle Ursprünge und Horizonte weicht die Netz-Prognose um {pct(100 * gap)} von der Summe der Depot-Prognosen ab (im Diagramm ein Ursprung). Abgestimmt ({SHORT[coh_method]}) ist die Netz-Prognose exakt die Summe "
    f"der Regionen und der Depots."
)

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was bringt die Abstimmung?")
b_mean = sm["base"]["mean"]
rows = []
for m in C.METHODS:
    lv = sm[m]["levels"]
    rows.append({"Verfahren": C.METHOD_NAMES[m], "Netz": de(lv[0], 3), "Regionen": de(lv[1], 3), "Depots": de(lv[2], 3), "Ø Ebenen": de(sm[m]["mean"], 3), "gegenüber Basis": spct(rel(sm[m]["mean"], b_mean), 1) if m != "base" else "–"})
st.dataframe(rows, hide_index=True)
ranked = sorted(C.METHODS, key=lambda m: sm[m]["mean"])
best = ranked[0] if ranked[0] != "base" else ranked[1]
best_gain = rel(sm[best]["mean"], b_mean)
bu_gain = rel(sm["bu"]["mean"], b_mean)
if sm["base"]["mean"] <= sm[ranked[0]]["mean"] + 1e-12:
    st.info(f"Hier ist keine Abstimmung besser als die (nicht kohärente) Basisprognose ({de(b_mean, 3)}); das beste kohärente Verfahren ist {SHORT[best]} mit {de(sm[best]['mean'], 3)} ({spct(best_gain, 1)}). Bottom-up: {de(sm['bu']['mean'], 3)} ({spct(bu_gain, 1)}); "
            f"Top-down: {de(sm['td']['mean'], 3)} ({spct(rel(sm['td']['mean'], b_mean), 1)}).")
elif best_gain > -1.0:
    st.info(f"Die Abstimmung ändert wenig: das beste Verfahren ({SHORT[best]}) liegt bei {de(sm[best]['mean'], 3)} gegen {de(b_mean, 3)} der Basisprognose ({spct(best_gain, 1)}); Bottom-up {spct(bu_gain, 1)}. Der Gewinn ist hier die **Kohärenz**, nicht die Genauigkeit. "
            f"Top-down ({spct(rel(sm['td']['mean'], b_mean), 1)}) und Middle-out ({spct(rel(sm['mo']['mean'], b_mean), 1)}) verlieren dagegen deutlich.")
else:
    st.success(f"✅ {SHORT[best]} senkt den Ø-RMSSE von {de(b_mean, 3)} auf {de(sm[best]['mean'], 3)} ({spct(best_gain, 1)}); Bottom-up erreicht {spct(bu_gain, 1)}, Top-down {spct(rel(sm['td']['mean'], b_mean), 1)}.")
if base == "auto":
    lv_names = C.LEVELS
    pick_rows = [{"Ebene": lv_names[k], **{C.BASE_SHORT[mod]: int(np.sum((hier.level_of == k) & (a.picks == i))) for i, mod in enumerate(C.MODELS)}} for k in range(3)]
    st.caption("Modellwahl je Knoten (Anzahl der Knoten je Ebene, die das jeweilige Modell wählen):")
    st.dataframe(pick_rows, hide_index=True)
st.caption(
    f"RMSSE: Wurzel des mittleren quadratischen Fehlers über {len(a.test_org)} Ursprünge und {settings.horizon} Horizonte, durch den saisonal naiven Trainingsfehler des Knotens geteilt, je Ebene über die Knoten gemittelt ({hier.n} Depots, {hier.n_regions} Regionen, ein Netz). "
    f"Fehlerkovarianz aus den Ein-Schritt-Fehlern der letzten {settings.history} Trainingstage; MinT geschrumpft wählt λ = {de(a.shrink_lambda, 3)} (0 = Stichprobe, 1 = nur Diagonale). Negative Prognosen werden auf 0 gesetzt."
)
st.markdown("##### RMSSE je Ebene")
st.plotly_chart(build_levels(a), width="stretch", key="levels_chart")
st.markdown("##### Veränderung gegenüber der Basisprognose")
st.plotly_chart(build_relative(a), width="stretch", key="relative_chart")

st.markdown("---")

st.markdown("## 🎯 Wer bekommt wie viel Gewicht?")
n_dep = hier.n
st.session_state["depot_select"] = min(n_dep - 1, max(0, st.session_state.get("depot_select", 0)))
dep = int(st.selectbox("Depot", list(range(n_dep)), key="depot_select", format_func=lambda j: hier.names[hier.m - n_dep + j], help="Für welches Depot die Gewichte gezeigt werden."))
w_methods = [m for m in C.METHODS[1:]]
st.plotly_chart(build_weights(a, hier.m - n_dep + dep, w_methods), width="stretch", key="weights_chart")
st.caption(
    f"Die abgestimmte Prognose von {hier.names[hier.m - n_dep + dep]} ist die Summe der Basisprognosen aller Knoten, jeweils mal das Gewicht der Zeile von G (Farbe). Bottom-up gibt sich selbst das Gewicht 1; Top-down legt alles auf das Netz "
    f"(Anteil {de(a.G['td'][dep, 0], 3)}); Middle-out auf die Region; OLS, WLS und MinT verteilen es auf alle Knoten - auch mit negativen Gewichten."
)

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wann hilft die Abstimmung? Gemeinsame Bewegung im Netz")
st.caption(f"{C.EXP_DEPOTS} Depots, Modellwahl je Knoten, Horizont 14; (a) die gemeinsame Niveauschwankung der Regionen wächst auf {', '.join(str(x).replace('.', ',') for x in C.SWING_LEVELS)}, (b) der Anteil des gemeinsamen Tagesschocks auf {', '.join(str(x).replace('.', ',') for x in C.RHO_LEVELS)}. "
           f"Gezeigt wird der Ø-RMSSE gegenüber der Basisprognose. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Gemeinsame Bewegung durchrechnen", key="structure_start"):
    st.session_state["structure_on"] = True
if st.session_state.get("structure_on"):
    rs = _structure(C.SWING_LEVELS, C.RHO_LEVELS, C.EXP_SEEDS)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("##### Gemeinsame Niveauschwankung")
        st.plotly_chart(build_sweep(rs["swing"], "Niveauschwankung der Region", lambda x: de(x, 2)), width="stretch", key="swing_chart")
    with s2:
        st.markdown("##### Gemeinsamer Tagesschock")
        st.plotly_chart(build_sweep(rs["rho"], "Anteil des Tagesschocks", lambda x: de(x, 1)), width="stretch", key="rho_chart")
    sw0, sw1 = rs["swing"][0], rs["swing"][-1]

    def _avg(r, m):
        return float(np.mean([r["levels"][m][k][0] for k in range(3)]))

    st.warning(
        f"**Befund:** Ohne gemeinsame Niveauschwankung ist Bottom-up am besten ({spct(rel(_avg(sw0, 'bu'), _avg(sw0, 'base')), 1)} gegen die Basis; MinT geschrumpft {spct(rel(_avg(sw0, 'mint_shrink'), _avg(sw0, 'base')), 1)}). "
        f"Bei Schwankung {de(sw1['x'], 2)} gewinnen die Verfahren, die alle Ebenen verwenden: MinT geschrumpft {spct(rel(_avg(sw1, 'mint_shrink'), _avg(sw1, 'base')), 1)}, WLS (Struktur) {spct(rel(_avg(sw1, 'wls_struct'), _avg(sw1, 'base')), 1)}, Bottom-up nur {spct(rel(_avg(sw1, 'bu'), _avg(sw1, 'base')), 1)}. "
        "Der gemeinsame Tagesschock ändert an den Gewinnen wenig: er ist für alle Depots einer Region am selben Tag gleich, aber nicht vorhersagbar."
    )

st.markdown("---")

st.subheader("🔬 Kurzer und langer Horizont")
st.caption(f"{C.EXP_DEPOTS} Depots, Modellwahl je Knoten; Horizont {', '.join(str(h) for h in C.HORIZON_LEVELS)} Tage. Die Fehlerkovarianz stammt aus **Ein-Schritt**-Fehlern: sie beschreibt die Fehler bei kurzem Horizont besser. Gezeigt: Ø-RMSSE gegenüber der Basis. "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Horizonte durchrechnen", key="horizon_start"):
    st.session_state["horizon_on"] = True
if st.session_state.get("horizon_on"):
    rh = _horizon(C.HORIZON_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_horizon(rh), width="stretch", key="horizon_chart")
    h0, h1 = rh[0], rh[-1]
    st.warning(
        f"**Befund:** Bei Horizont {h0['horizon']} liegen WLS (Fehlervarianz) {spct(rel(h0['mean']['wls_var'][0], h0['mean']['base'][0]), 1)}, Bottom-up {spct(rel(h0['mean']['bu'][0], h0['mean']['base'][0]), 1)} und MinT geschrumpft {spct(rel(h0['mean']['mint_shrink'][0], h0['mean']['base'][0]), 1)} vor der Basis; "
        f"bei Horizont {h1['horizon']} sind es {spct(rel(h1['mean']['wls_var'][0], h1['mean']['base'][0]), 1)}, {spct(rel(h1['mean']['bu'][0], h1['mean']['base'][0]), 1)} und {spct(rel(h1['mean']['mint_shrink'][0], h1['mean']['base'][0]), 1)}: die Gewinne sind klein und werden mit dem Horizont kleiner. "
        f"MinT mit der Stichprobenkovarianz ist über alle Horizonte schlechter als die Basis ({spct(rel(h0['mean']['mint_sample'][0], h0['mean']['base'][0]), 1)} bei {h0['horizon']}, {spct(rel(h1['mean']['mint_sample'][0], h1['mean']['base'][0]), 1)} bei {h1['horizon']})."
    )

st.markdown("---")

st.subheader("🔬 Zu wenig Fehler für zu viele Knoten: die Kovarianz")
st.caption(f"{C.EXP_HISTORY_DEPOTS} Depots (also {C.EXP_HISTORY_DEPOTS + 6} Knoten), Modellwahl je Knoten; die Fehlerkovarianz wird aus den Ein-Schritt-Fehlern der letzten {', '.join(str(w) for w in C.HISTORY_LEVELS)} Trainingstage geschätzt. Gezeigt: Ø-RMSSE (logarithmisch). "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Fehlerhistorie durchrechnen", key="history_start"):
    st.session_state["history_on"] = True
if st.session_state.get("history_on"):
    rk = _history(C.HISTORY_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_history(rk), width="stretch", key="history_chart")
    k0, k1 = rk[0], rk[-1]
    st.warning(
        f"**Befund:** Mit {k0['history']} Tagen Fehlerhistorie für {C.EXP_HISTORY_DEPOTS + 6} Knoten hat die Stichprobenkovarianz weniger Fehlerzeilen als Knoten und ist singulär: MinT (Stichprobe) erreicht einen Ø-RMSSE von {de(k0['mean']['mint_sample'][0], 1)} gegen {de(k0['mean']['base'][0], 2)} der Basis. "
        f"Mit {k1['history']} Tagen sind es {de(k1['mean']['mint_sample'][0], 2)}. MinT (geschrumpft) bleibt bei {de(k0['mean']['mint_shrink'][0], 2)} beziehungsweise {de(k1['mean']['mint_shrink'][0], 2)}: die Schrumpfung (λ = {de(k0['lambda'][0], 2)} bei {k0['history']} Tagen) hält es stabil, "
        f"und WLS mit der Diagonalen hat das Problem nicht ({de(k0['mean']['wls_var'][0], 2)})."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Basisprognosen unterscheiden sich im Fehler** | Sind sie ohnehin fast kohärent (Wochenmittel: exakt; Holt-Winters je Knoten: fast), gibt es nichts zu gewinnen; die Abstimmung ändert kaum etwas. | – |
| **Die Anteile bleiben stabil** | Top-down und Middle-out verteilen nach mittleren Anteilen der Trainingstage; ändern sie sich (Trend, Schwankung, Aktionen), verliert das Depot-Ergebnis deutlich. | Anteile nach Prognose statt nach Historie, Bottom-up |
| **Die Fehlerkovarianz ist schätzbar** | Bei vielen Knoten und wenig Fehlern ist die Stichprobenkovarianz unbrauchbar; die Schrumpfung hilft, ersetzt aber nicht die Daten. | WLS, Schrumpfung, weniger Knoten |
| **Ein-Schritt-Fehler beschreiben die Fehler bei Horizont $h$** | Die Gewichte stammen aus Ein-Schritt-Fehlern; bei langem Horizont passen sie schlechter. | Fehler je Horizont schätzen |
| **Negative Prognosen sind kein Problem** | OLS, WLS und MinT können negative Werte liefern; hier werden sie abgeschnitten (das macht die Prognosen leicht inkohärent). | Nicht-negative Abstimmung (Wickramasuriya et al. 2020) |
| **Die Hierarchie ist fest und summierbar** | Nur Summen über eine Baumstruktur; gitterförmige (mehrere Gliederungen) und zeitliche Hierarchien sind nicht abgebildet. | Verallgemeinerte Summationsmatrizen |
| **Erzeugtes Netz, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal, Niveauschwankungen als AR(1)); echte Netze sind unordentlicher. Die Zahlen gelten für diese Netze. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu Croston, Boosting, Prognoseintervalle, **Hierarchie**, Kombination, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Hierarchie.** Depots $b \in \mathbb R^n$, alle Knoten $y = S b \in \mathbb R^m$ mit $S = \begin{pmatrix} \mathbf 1' \\ S_{\text{Regionen}} \\ I_n \end{pmatrix}$. Basisprognosen $\hat y_h \in \mathbb R^m$ (je Knoten für sich), abgestimmt $\tilde y_h = S\,G\,\hat y_h$ mit $G \in \mathbb R^{n \times m}$.

**Bottom-up:** $G = (0_{n \times (m-n)} \mid I_n)$. **Top-down:** $G = (p \mid 0)$ mit den mittleren Tagesanteilen $p_i = \overline{y_{i,t} / y_{\text{Netz},t}}$ der Trainingstage. **Middle-out:** $G_{i, r(i)} = q_i$ mit dem mittleren Anteil des Depots an seiner Region $r(i)$.

**Verallgemeinerte Kleinste Quadrate.** $\tilde y = S\,(S' W^{-1} S)^{-1} S' W^{-1}\,\hat y$ ist die Projektion auf den Raum kohärenter Prognosen im Maß $W^{-1}$; sie erfüllt $G S = I$ (unverzerrt, wenn die Basisprognosen es sind) und minimiert unter dieser Nebenbedingung die Spur der Fehlerkovarianz der abgestimmten Prognosen, wenn $W$ die Kovarianz der Basisfehler ist (MinT, Wickramasuriya/Athanasopoulos/Hyndman 2019).
**OLS:** $W = I$. **WLS (Struktur):** $W = \operatorname{diag}(S\mathbf 1)$, die Zahl der Depots unter dem Knoten. **WLS (Fehlervarianz):** $W = \operatorname{diag}(\hat\sigma_j^2)$. **MinT (Stichprobe):** $W = \tfrac1T E'E$ mit den Ein-Schritt-Fehlern $E$ ($T \times m$).
**MinT (geschrumpft):** $W = \lambda \operatorname{diag}(W_s) + (1-\lambda) W_s$ mit $\lambda = \sum_{i \ne j} \widehat{\operatorname{Var}}(r_{ij}) / \sum_{i \ne j} r_{ij}^2$ (Schäfer/Strimmer 2005; $r_{ij}$ die Korrelationen der Fehler, auf $[0,1]$ begrenzt).

**RMSSE.** $\sqrt{\tfrac1{TH}\sum (y - \tilde y)^2} \Big/ \sqrt{\tfrac1{N-7}\sum_{t} (y_t - y_{t-7})^2}$ (Nenner auf den Tagen vor 730), je Ebene über die Knoten gemittelt, "Ø Ebenen" der Mittelwert der drei Ebenen. Negative abgestimmte Werte werden auf 0 gesetzt.

**Vehikel.** Depot $i$ in Region $r$: $\mu_{i,t} = \ell_i\,(\text{Trend, Wochenmuster, Jahresmuster, Feiertag, Aktion})\,\exp(g_t + g_{r,t} + g_{i,t})$ mit mittelwertrückkehrenden AR(1)-Prozessen ($\varphi = 0{,}98$), Tagesrauschen $y = \mu\,\exp(\sigma\,(\sqrt{1-\rho}\,\varepsilon_{i,t} + \sqrt\rho\,\eta_{r,t}) - \sigma^2/2)$ mit dem regionalen Tagesschock $\eta_{r,t}$.

Implementiert in `hrc_reconcile.py` (die Matrizen $G$, Schrumpfung), `hrc_baselines.py`, `hrc_ets.py` (Basismodelle), `hrc_scenario.py` (das Netz), `hrc_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
