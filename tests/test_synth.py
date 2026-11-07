"""Tests for gb.synth."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gb.config import load_config
from gb.schema import validate
from gb.synth import (
    SynthWarning,
    build_params,
    duty_profile,
    full_day_plan_Wh,
    generate,
    main,
    nominal_pack_energy_Wh,
    outdoor_temp_C,
    sanity_table,
)
from gb.types import build_layout


@pytest.fixture(scope="module")
def zones_1d():
    cfg = load_config()
    df, truth = generate(cfg, days=1, seed=0, scenario="zones")
    return cfg, df, truth


def _layout(cfg):
    return build_layout(cfg["pack"]["n_modules"], cfg["pack"]["cells_per_module"])


# ---- build_params ---------------------------------------------------------------------------


def test_build_params_zones_causes(cfg) -> None:
    layout = _layout(cfg)
    params, cause = build_params(cfg, layout, 0, "zones")
    hot = np.isin(layout.module_idx, [6, 7])
    pcs = (layout.module_idx == 5) & (layout.slot_idx < 4)
    assert np.all(params.T_offset_K[hot] == 8.0)
    assert np.all(params.T_offset_K[pcs] == 5.0)
    assert np.all(params.T_offset_K[~(hot | pcs)] == 0.0)
    assert np.all(cause[hot | pcs] == "zone")
    assert cause[24] == "self_heat" and layout.module_idx[24] == 1 and layout.slot_idx[24] == 8
    assert (cause == "none").sum() == layout.n - 36 - 1
    # high_R factor applied on top of spread: R / R_nominal well above the 5% spread
    assert params.R_ref_ohm[24] / cfg["cell"]["R_dc_ref_ohm"] > 1.6
    assert np.allclose(params.C_JK, cfg["cell"]["mass_kg"] * cfg["cell"]["cp_J_kgK"])
    assert np.all(params.soh0 == 1.0)


def test_build_params_zones_light(cfg) -> None:
    layout = _layout(cfg)
    params, cause = build_params(cfg, layout, 0, "zones_light")
    assert (cause == "zone").sum() == 16 and np.all(layout.module_idx[cause == "zone"] == 7)
    assert np.flatnonzero(cause == "self_heat").tolist() == [24]
    assert np.all(params.T_offset_K[cause == "zone"] == 8.0)


def test_build_params_faulty_and_uniform(cfg) -> None:
    layout = _layout(cfg)
    p_u, c_u = build_params(cfg, layout, 3, "uniform")
    p_f, c_f = build_params(cfg, layout, 3, "faulty_cells")
    assert np.all(c_u == "none") and np.all(p_u.T_offset_K == 0)
    assert np.all(p_f.T_offset_K == 0)
    assert sorted(np.flatnonzero(c_f == "self_heat").tolist()) == [10, 30, 50, 70, 90, 110]
    r = p_f.R_ref_ohm / p_u.R_ref_ohm
    g = p_f.G_cool_WK / p_u.G_cool_WK
    assert np.allclose(r[[10, 50, 90, 70]], 2.5) and np.allclose(g[[30, 110, 70]], 0.2)
    others = np.setdiff1d(np.arange(layout.n), [10, 30, 50, 70, 90, 110])
    assert np.allclose(r[others], 1.0) and np.allclose(g[others], 1.0)


def test_build_params_spread_and_seed(cfg) -> None:
    layout = _layout(cfg)
    a, _ = build_params(cfg, layout, 7, "uniform")
    b, _ = build_params(cfg, layout, 7, "uniform")
    c, _ = build_params(cfg, layout, 8, "uniform")
    assert np.array_equal(a.R_ref_ohm, b.R_ref_ohm) and not np.array_equal(a.R_ref_ohm, c.R_ref_ohm)
    sp = cfg["scenarios"]["spread"]
    rel_R = np.std(a.R_ref_ohm / cfg["cell"]["R_dc_ref_ohm"])
    assert 0.5 * sp["R_rel_std"] < rel_R < 1.5 * sp["R_rel_std"]


def test_build_params_rejects_bad_scenarios(cfg, small_pack) -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        build_params(cfg, _layout(cfg), 0, "nope")
    with pytest.raises(ValueError, match="modules"):
        build_params(small_pack.cfg, small_pack.layout, 0, "zones")  # 2-module pack has no module 6
    bad = copy.deepcopy(cfg)
    bad["scenarios"]["catalog"]["uniform"]["defects"] = {"x": {"kind": "melted", "cell_id": 0}}
    with pytest.raises(ValueError, match="unknown kind"):
        build_params(bad, _layout(bad), 0, "uniform")


# ---- duty profile ---------------------------------------------------------------------------


def test_duty_profile(cfg) -> None:
    d = cfg["site"]["duty"]
    E = full_day_plan_Wh(cfg)
    rng = np.random.default_rng(0)
    fracs = []
    for k in range(200):
        P, win, plan = duty_profile(cfg, k, rng)
        fracs.append(plan / E)
        t = np.arange(0, 86400, 60.0)
        p = P(t)
        assert np.all(p[(t >= d["charge_start_h"] * 3600) & (t < d["charge_end_h"] * 3600)] == d["charge_P_W"])
        dis = p > 0
        assert np.all(p[dis] == d["discharge_P_W"])
        assert np.all(win(t)[dis])  # discharge only inside the dispatch window
        # constant power, energy = plan (to one step), clipped by the window length
        E_dis = dis.sum() * 60.0 / 3600 * d["discharge_P_W"]
        cap = (d["discharge_end_h"] - d["discharge_start_h"]) * d["discharge_P_W"]
        assert E_dis == pytest.approx(min(plan, cap), abs=d["discharge_P_W"] * 60 / 3600 + 1e-6)
        assert win(d["discharge_start_h"] * 3600) and not win(d["discharge_end_h"] * 3600)
    fracs = np.array(fracs)
    lo, hi = d["daily_energy_frac_range"]
    assert np.all((fracs >= lo) & (fracs <= hi))
    assert abs(np.mean(fracs == 1.0) - d["full_dod_day_frac"]) < 0.1


def test_full_day_plan_fits_usable_energy(cfg) -> None:
    from gb import plant
    from gb.synth import bol_params

    c = cfg["cell"]
    usable = plant.usable_energy_Wh(c["soc_min"], c["soc_max"], bol_params(cfg), cfg)
    plan = full_day_plan_Wh(cfg)
    d = cfg["site"]["duty"]
    assert plan == pytest.approx(usable * (1 - d["loss_frac"]) * d["full_day_reserve_frac"])
    assert plan < usable < nominal_pack_energy_Wh(cfg)


def test_outdoor_temp(cfg) -> None:
    h = cfg["site"]["hvac"]
    assert outdoor_temp_C(cfg, h["outdoor_peak_h"] * 3600) == pytest.approx(h["outdoor_mean_C"] + h["outdoor_amp_C"])
    t = np.arange(0, 86400, 60.0)
    assert outdoor_temp_C(cfg, t).min() == pytest.approx(h["outdoor_mean_C"] - h["outdoor_amp_C"], abs=1e-3)


# ---- generate -------------------------------------------------------------------------------


def test_row_count_and_order(zones_1d) -> None:
    cfg, df, _ = zones_1d
    n = cfg["pack"]["n_modules"] * cfg["pack"]["cells_per_module"]
    assert len(df) == 1 * 86400 / cfg["site"]["dt_s"] * n
    assert df["cell_id"].iloc[:n].tolist() == list(range(n))
    assert (df["room_temp_C"] == cfg["site"]["room_C"]).all()


def test_row_count_small_pack_multi_day(small_pack) -> None:
    df, _ = generate(small_pack.cfg, days=2, seed=0, scenario="uniform")
    assert len(df) == 2 * 86400 / small_pack.cfg["site"]["dt_s"] * small_pack.layout.n


def test_schema_validation_passes(zones_1d) -> None:
    cfg, df, _ = zones_1d
    rep = validate(df, n_cells=cfg["pack"]["n_modules"] * cfg["pack"]["cells_per_module"], cfg=cfg)
    assert rep.ok, rep.errors
    assert rep.warnings == []
    assert rep.nominal_dt_s == cfg["site"]["dt_s"]


def test_deterministic_with_seed(cfg) -> None:
    a, ta = generate(cfg, days=1, seed=5, scenario="faulty_cells")
    b, tb = generate(cfg, days=1, seed=5, scenario="faulty_cells")
    c, _ = generate(cfg, days=1, seed=6, scenario="faulty_cells")
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(ta["peak_T_C"], tb["peak_T_C"])
    assert not a["temp_C"].equals(c["temp_C"])


def test_discharge_and_charge_happen(zones_1d) -> None:
    cfg, df, _ = zones_1d
    I = df.groupby("timestamp", sort=True)["current_A"].first().to_numpy()
    assert I.max() > 50 and I.min() < -20  # 30 kW / ~370 V discharge, 15 kW charge
    soc = df["soc"].to_numpy()
    assert soc.max() <= 1.0 and soc.min() >= 0.0


def test_zone_cells_rest_at_offset(zones_1d) -> None:
    cfg, _, truth = zones_1d
    off = truth["params"].T_offset_K
    rest = truth["rest_above_room_K"]
    mod = truth["module_idx"]
    # whole hot modules have no cooler neighbour: rest = offset
    hot = np.isin(mod, [6, 7])
    assert np.allclose(rest[hot], 8.0, atol=0.2)
    # PCS hot spot: slot 3 conducts into the cooler slot 4, so allow for the edge
    pcs = (mod == 5) & (truth["slot_idx"] < 4)
    assert np.all(rest[pcs] > 0.6 * 5.0) and np.all(rest[pcs] < 5.0 + 0.2)
    assert np.mean(rest[pcs & (truth["slot_idx"] < 3)]) == pytest.approx(5.0, abs=0.6)
    zone = truth["cause"] == "zone"
    assert np.mean(rest[zone]) == pytest.approx(np.mean(off[zone]), abs=0.3)


def test_high_R_cell_rests_cool_but_rises_more(zones_1d) -> None:
    _, _, truth = zones_1d
    rest, rise, cause = truth["rest_above_room_K"], truth["peak_rise_K"], truth["cause"]
    assert abs(rest[24]) < 0.2
    healthy = np.median(rise[cause == "none"])
    assert 2.0 < healthy < 4.5
    # R x2 doubles electrochemical heat; switch heat and conduction to neighbours dilute the rise
    # to ~1.5x (measured), so accept 1.3-2.2x.
    assert 1.3 < rise[24] / healthy < 2.2


def test_sanity_table(zones_1d) -> None:
    cfg, _, truth = zones_1d
    tab = sanity_table(truth, cfg)
    assert list(tab.index) == ["zone", "self_heat", "none"]
    assert tab.loc["zone", "n"] == 36
    assert tab.loc["zone", "share_above_30C"] == 1.0
    # room 25 C: +8 K cells (32) sit above the 30 C edge at rest, +5 K cells (30) do not
    assert tab.loc["zone", "floor_above_band"] == pytest.approx(32 / 36)
    assert tab.loc["none", "floor_above_band"] == 0.0 and tab.loc["self_heat", "floor_above_band"] == 0.0
    cold = dict(truth, peak_T_C=np.full_like(truth["peak_T_C"], 20.0))
    with pytest.warns(SynthWarning, match="no zone cell crosses"):
        sanity_table(cold, cfg)


def test_cli_writes_csv_and_truth(tmp_path: Path, capsys) -> None:
    out = tmp_path / "s.csv"
    main(["--days", "1", "--seed", "0", "--scenario", "zones", "--out", str(out), "--room", "24"])
    df = pd.read_csv(out)
    truth = json.loads((tmp_path / "s_truth.json").read_text(encoding="utf-8"))
    assert len(df) == 1440 * 128
    assert (df["room_temp_C"] == 24).all()
    assert truth["cause"][24] == "self_heat" and len(truth["params"]["R_ref_ohm"]) == 128
    assert validate(df, n_cells=128).ok
    printed = capsys.readouterr().out
    assert "zone" in printed and "rest_above_room_K" in printed
