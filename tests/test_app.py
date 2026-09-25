"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Depot-/Ursprungs-Regler, Würfel-Knopf, Permalink-Grenzen, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hrc_constants as C
import hrc_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.get("plotly_chart")) == 5 and len(at.info) + len(at.success) + len(at.warning) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_node_depot_and_origin_selections_survive_a_smaller_network_and_longer_horizon():
    at = _run(node_select=105, depot_select=99, origin_slider=1094 - 14, depots_slider=100, regions_slider=5)
    _ok(at)
    at.slider(key="depots_slider").set_value(20).run()
    _ok(at)
    assert at.session_state["node_select"] <= 25 and at.session_state["depot_select"] <= 19
    at.slider(key="horizon_slider").set_value(28).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 28


def test_empty_method_selection_falls_back():
    at = _run(methods_select=[])
    _ok(at)
    assert any("Kein Verfahren gewählt" in w.value for w in at.warning)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Netz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["depots"] = "77"
    at.query_params["regions"] = "99"
    at.query_params["history"] = "10"
    at.query_params["base"] = "arima"
    at.query_params["show"] = "mint_shrink,base"
    at.query_params["swing"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["depots_slider"] == 80 and at.session_state["regions_slider"] == C.REGIONS_MAX and at.session_state["history_slider"] == C.HISTORY_MIN
    assert at.session_state["base_select"] == "auto" and at.session_state["methods_select"] == ["base", "mint_shrink"] and at.session_state["swing_slider"] == C.DEFAULT_SWING


@pytest.mark.parametrize("kw", [dict(depots_slider=C.DEPOTS_MIN, regions_slider=C.REGIONS_MAX, horizon_slider=1), dict(horizon_slider=C.HORIZON_MAX, history_slider=C.HISTORY_MIN, swing_slider=C.SWING_MAX),
                                dict(base_select="snaive_k", rho_slider=C.RHO_MAX, noise_slider=C.NOISE_MAX), dict(base_select="regression", swing_slider=0.0, rho_slider=0.0, noise_slider=C.NOISE_MIN),
                                dict(base_select="hw_mult", methods_select=["mint_sample"])])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    monkeypatch.setattr(C, "EXP_DEPOTS", 20)
    monkeypatch.setattr(C, "EXP_HISTORY_DEPOTS", 20)
    monkeypatch.setattr(C, "SWING_LEVELS", (0.0, 0.12))
    monkeypatch.setattr(C, "RHO_LEVELS", (0.0, 0.8))
    monkeypatch.setattr(C, "HORIZON_LEVELS", (1, 28))
    monkeypatch.setattr(C, "HISTORY_LEVELS", (60, 600))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_structure_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "structure_start")
    assert at.session_state["structure_on"] and any("die Verfahren, die alle Ebenen verwenden" in w.value for w in at.warning)


def test_horizon_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "horizon_start")
    assert at.session_state["horizon_on"] and any("die Gewinne sind klein" in w.value for w in at.warning)


def test_history_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "history_start")
    assert at.session_state["history_on"] and any("weniger Fehlerzeilen als Knoten" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
