"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. fi_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import hrc_constants as C


def _base(value):
    v = str(value).strip().lower()
    if v not in C.BASES:
        raise ValueError(value)
    return v


def _methods(value):
    parts = [str(v) for v in value] if isinstance(value, (list, tuple)) else [v.strip() for v in str(value).split(",") if v.strip()]
    if not parts or any(p not in C.METHODS for p in parts):
        raise ValueError(value)
    return [m for m in C.METHODS if m in parts]


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "depots_slider": SettingSpec("depots", int, C.DEFAULT_DEPOTS, C.DEPOTS_MIN, C.DEPOTS_MAX),
    "regions_slider": SettingSpec("regions", int, C.DEFAULT_REGIONS, C.REGIONS_MIN, C.REGIONS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "rho_slider": SettingSpec("rho", float, C.DEFAULT_RHO, C.RHO_MIN, C.RHO_MAX),
    "swing_slider": SettingSpec("swing", float, C.DEFAULT_SWING, C.SWING_MIN, C.SWING_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "base_select": SettingSpec("base", _base, "auto"),
    "history_slider": SettingSpec("history", int, C.DEFAULT_HISTORY, C.HISTORY_MIN, C.HISTORY_MAX),
    "methods_select": SettingSpec("show", _methods, ["base", "bu", "td", "mint_shrink"]),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n_depots": "depots_slider", "n_regions": "regions_slider", "noise": "noise_slider", "rho": "rho_slider", "swing": "swing_slider", "horizon": "horizon_slider", "base": "base_select", "history": "history_slider",
               "show": "methods_select", "seed": "seed_input"}
STEPS = {"depots_slider": C.DEPOTS_STEP, "noise_slider": C.NOISE_STEP, "rho_slider": C.RHO_STEP, "swing_slider": C.SWING_STEP, "history_slider": C.HISTORY_STEP}


def _p(**kw):
    base = {"n_depots": C.DEFAULT_DEPOTS, "n_regions": C.DEFAULT_REGIONS, "noise": C.DEFAULT_NOISE, "rho": C.DEFAULT_RHO, "swing": C.DEFAULT_SWING, "horizon": C.DEFAULT_HORIZON, "base": "auto",
            "history": C.DEFAULT_HISTORY, "show": ["base", "bu", "td", "mint_shrink"], "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: Modellwahl, 30 Depots, 5 Regionen": _p(),
    "Starke gemeinsame Bewegung (Schwankung 0,12)": _p(swing=0.12, show=["base", "bu", "wls_struct", "mint_shrink"]),
    "Nur Holt-Winters als Basis": _p(base="hw_mult"),
    "Wochenmittel als Basis": _p(base="snaive_k", show=["base", "bu", "td", "mo"]),
    "Viele Depots, kurze Fehlerhistorie": _p(n_depots=100, history=60, show=["base", "bu", "mint_shrink", "mint_sample"]),
    "Kurzer Horizont (1 Tag)": _p(horizon=1, show=["base", "bu", "wls_var", "mint_shrink"]),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = list(spec.default) if isinstance(spec.default, list) else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 3)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = ",".join(value) if isinstance(value, (list, tuple)) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        v = PRESETS[name][key]
        st.session_state[state_key] = list(v) if isinstance(v, list) else v


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: Modellwahl, 30 Depots, 5 Regionen": "36 Knoten (4 wählen Holt-Winters, 32 die Regression), Seed 3: Ø-RMSSE der Basis 0,834 (nicht kohärent, Netz gegen Summe der Depots 1,5 % Abweichung); WLS (Struktur) 0,814 (−2,4 %), OLS −1,2 %, Bottom-up +2,0 %, MinT geschrumpft +1,5 %, Top-down 1,026 (+23,1 %).",
    "Starke gemeinsame Bewegung (Schwankung 0,12)": "Regionale Niveauschwankung 0,12: Basis 1,013; Bottom-up 0,971 (−4,2 %), WLS (Struktur) −3,8 %, WLS (Fehlervarianz) −6,3 %, MinT geschrumpft 0,937 (−7,5 %), MinT (Stichprobe) −8,7 %; die Netz-Prognose weicht 4,0 % von der Summe der Depots ab.",
    "Nur Holt-Winters als Basis": "Holt-Winters je Knoten ist schon fast kohärent (Abweichung 0,8 %): alle Verfahren außer Top-down und Middle-out liegen zwischen −0,2 und +0,5 % der Basis (0,871); Top-down 1,104 (+26,7 %), Middle-out 1,028 (+18,0 %).",
    "Wochenmittel als Basis": "Das Wochenmittel ist linear und damit exakt kohärent (Abweichung 0): Bottom-up, OLS, WLS und MinT ändern nichts (0,941); Top-down 1,131 (+20,1 %), Middle-out 1,075 (+14,2 %).",
    "Viele Depots, kurze Fehlerhistorie": "100 Depots (106 Knoten), Fehlerkovarianz aus 60 Tagen: MinT (Stichprobe) bricht zusammen (Ø-RMSSE 2585, die Matrix ist singulär); MinT geschrumpft (λ = 0,45) 0,866 (−13,8 %), WLS (Fehlervarianz) 0,908 (−9,6 %), Bottom-up 0,916 (−8,8 %) gegen 1,004 der Basis, deren Netz-Prognose bei 1,124 liegt.",
    "Kurzer Horizont (1 Tag)": "Horizont 1: Basis 0,823; WLS (Struktur) 0,802 (−2,5 %), WLS (Fehlervarianz) −2,0 %, OLS, Bottom-up und MinT geschrumpft je −1,2 %, MinT (Stichprobe) +0,6 %.",
}
