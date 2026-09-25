"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import hrc_constants as C
import hrc_evaluation as E
import hrc_presets as P


def _settings(p):
    return E.Settings(p["n_depots"], p["n_regions"], p["noise"], p["rho"], p["swing"], p["horizon"], p["base"], p["history"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n_depots", "depots_slider"), ("noise", "noise_slider"), ("rho", "rho_slider"), ("swing", "swing_slider"), ("history", "history_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6
        assert p["base"] in C.BASES and p["show"] and set(p["show"]) <= set(C.METHODS)


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: Modellwahl, 30 Depots, 5 Regionen"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("history_slider") == (C.HISTORY_MIN, C.HISTORY_MAX) and P.bounds("regions_slider") == (C.REGIONS_MIN, C.REGIONS_MAX)
    assert set(P.STEPS) == {"depots_slider", "noise_slider", "rho_slider", "swing_slider", "history_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEPOTS_MIN // 2 >= C.REGIONS_MAX


def test_casters_reject_bad_values_and_canonicalise_lists():
    for caster, bad in ((P._base, "arima"), (P._methods, ""), (P._methods, "bu,quatsch")):
        try:
            caster(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert P._base(" AUTO ") == "auto" and P._methods("mint_shrink,base") == ["base", "mint_shrink"] and P._methods(["td", "bu"]) == ["bu", "td"]
