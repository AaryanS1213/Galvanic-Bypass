"""Unit tests for gb.controllers (one decide call at a time, true CellParams)."""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from gb import controllers as C
from gb import plant
from gb.config import load_config
from gb.simulate import initial_state, simulate
from gb.synth import bol_params, build_params, full_day_plan_Wh
from gb.types import PackState, build_layout
from gb.units import c_to_k


@pytest.fixture(scope="module")
def mcfg():
    return load_config()


@pytest.fixture(scope="module")
def layout(mcfg):
    return build_layout(int(mcfg["pack"]["n_modules"]), int(mcfg["pack"]["cells_per_module"]))


def _ctx(cfg, T_hist=None, plan=0.0, plan_rem=0.0, hours=9.0):
    return {
        "cfg": cfg,
        "t_of_day_s": 0.0,
        "in_dispatch_window": plan_rem > 0,
        "plan_today_Wh": plan,
        "plan_remaining_Wh": plan_rem,
        "hours_until_next_charge": hours,
        "T_hist_K": T_hist,
        "dt_s": float(cfg["site"]["dt_s"]),
    }


def _state(cfg, params, room_C=25.0, soc=0.6) -> PackState:
    s = initial_state(cfg, params, float(c_to_k(room_C)))
    s.soc = np.full(params.Q_Ah.size, soc)
    s.dwell_s = np.full(params.Q_Ah.size, 1e4)
    return s


# ---- zone_soc_cap / trend ---------------------------------------------------------------------


def test_zone_soc_cap_table(mcfg):
    caps = C.zone_soc_cap(np.array([20.0, 27.99, 28.0, 30.9, 31.0, 40.0]), mcfg)
    smax = float(mcfg["cell"]["soc_max"])
    np.testing.assert_allclose(caps, [smax, smax, 0.80, 0.80, 0.70, 0.70])
    assert caps.dtype == np.float64


def test_trend_slope_linear():
    t = np.arange(6)[:, None] * 60.0
    H = 300.0 + t * np.array([1e-3, 0.0, -2e-3])
    np.testing.assert_allclose(C.trend_slope_K_s(H, 60.0), [1e-3, 0.0, -2e-3], atol=1e-12)
    assert np.all(C.trend_slope_K_s(H[:1], 60.0) == 0.0)


# ---- baselines --------------------------------------------------------------------------------


def test_fixed_derate_scaling(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params)
    ctl = C.FixedStringDerate(mcfg)
    ctl.reset(layout.n)
    T0, T1 = float(mcfg["controllers"]["derate"]["T_start_C"]), float(mcfg["controllers"]["derate"]["T_stop_C"])
    for T_C, expect in [(T0 - 1, 1.0), (0.5 * (T0 + T1), 0.5), (T1 + 1, 0.0)]:
        s.T_K = np.full(layout.n, float(c_to_k(25.0)))
        s.T_K[3] = float(c_to_k(T_C))
        d = ctl.decide(s, params, layout, 30000.0, _ctx(mcfg))
        assert d.active.all()
        assert d.P_W == pytest.approx(30000.0 * expect)
    typ = C.FixedStringDerate(mcfg, t_start_C=float(mcfg["controllers"]["derate"]["variant_typical_T_start_C"]))
    typ.reset(layout.n)
    s.T_K[:] = float(c_to_k(40.0))
    assert typ.decide(s, params, layout, 30000.0, _ctx(mcfg)).P_W == 30000.0


def test_fixed_no_derate_only_at_crit(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params)
    ctl = C.FixedStringNoDerate(mcfg)
    ctl.reset(layout.n)
    s.T_K[0] = float(c_to_k(float(mcfg["controllers"]["T_crit_C"]) - 1))
    assert ctl.decide(s, params, layout, 30000.0, _ctx(mcfg)).P_W == 30000.0
    s.T_K[0] = float(c_to_k(float(mcfg["cell"]["T_lim_C"])))
    assert ctl.decide(s, params, layout, 30000.0, _ctx(mcfg)).P_W == 0.0


def test_threshold_hysteresis_and_label(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params)
    ctl = C.ThresholdBypass(mcfg)
    ctl.reset(layout.n)
    assert ctl.label == "Simple threshold bypass (assumed REON-like logic)"
    tb = mcfg["controllers"]["threshold_bypass"]
    s.T_K[5] = float(c_to_k(float(tb["T_on_C"]) + 0.5))
    assert not ctl.decide(s, params, layout, 30000.0, _ctx(mcfg)).active[5]
    s.T_K[5] = float(c_to_k(0.5 * (float(tb["T_on_C"]) + float(tb["T_off_C"]))))
    assert not ctl.decide(s, params, layout, 30000.0, _ctx(mcfg)).active[5]  # still inside hysteresis
    s.T_K[5] = float(c_to_k(float(tb["T_off_C"]) - 0.5))
    assert ctl.decide(s, params, layout, 30000.0, _ctx(mcfg)).active[5]


# ---- GalvanicBypass single decisions ------------------------------------------------------------


def test_galvanic_forced_out_at_T_crit_even_idle(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params)
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    s.T_K[7] = float(c_to_k(float(mcfg["controllers"]["T_crit_C"]) + 0.1))
    d = g.decide(s, params, layout, 0.0, _ctx(mcfg))
    assert not d.active[7] and d.active.sum() == layout.n - 1
    assert g.kind[7] == C.FORCED


def test_galvanic_idle_keeps_states(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params)
    s.active[[3, 9]] = False
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    d = g.decide(s, params, layout, 0.0, _ctx(mcfg))
    np.testing.assert_array_equal(d.active, s.active)
    assert d.P_W == 0.0


def test_galvanic_charge_gating_and_caps(mcfg, layout):
    cfg = copy.deepcopy(mcfg)
    params = bol_params(cfg)
    params.T_offset_K = np.zeros(layout.n)
    params.T_offset_K[-16:] = 8.0  # hot top module (local air 33 C at a 25 C room)
    s = _state(cfg, params, soc=0.5)
    s.T_K = 298.15 + params.T_offset_K
    s.T_K[0] = float(c_to_k(float(cfg["controllers"]["T_charge_min_C"]) - 1.0))
    s.soc[-1] = 0.75  # above the 0.70 hot-zone cap
    g = C.GalvanicBypass(cfg)
    g.reset(layout.n)
    d = g.decide(s, params, layout, -15000.0, _ctx(cfg, plan=0.5 * full_day_plan_Wh(cfg)))
    assert not d.active[0]  # too cold to charge
    assert not d.active[-1]  # at/above its cap
    assert d.active[-2]  # hot cell below its cap still charges
    assert g.caps[-1] == pytest.approx(0.70) and g.caps[0] == pytest.approx(float(cfg["cell"]["soc_max"]))
    assert not g.cap_events[-1].relaxed
    # catch-up: once warm again and below cap, the gated cell returns
    s.T_K[0] = 298.15
    d2 = g.decide(s, params, layout, -15000.0, _ctx(cfg, plan=0.5 * full_day_plan_Wh(cfg)))
    assert d2.active[0]


def test_galvanic_caps_relaxed_for_heavy_day(mcfg, layout):
    cfg = mcfg
    params = bol_params(cfg)
    params.T_offset_K = np.zeros(layout.n)
    params.T_offset_K[-32:] = 8.0
    params.T_offset_K[-36:-32] = 5.0
    s = _state(cfg, params, soc=0.5)
    g = C.GalvanicBypass(cfg)
    g.reset(layout.n)
    plan = full_day_plan_Wh(cfg)
    d = g.decide(s, params, layout, -15000.0, _ctx(cfg, plan=plan))
    ev = g.cap_events[-1]
    need = (1.0 + float(cfg["site"]["duty"]["plan_guard_margin"])) * plan
    assert ev.relaxed and ev.relaxed_Wh > 0.0
    assert plant.deliverable_energy_Wh(float(cfg["cell"]["soc_min"]), g.caps, params, cfg) >= need
    assert "caps relaxed for heavy day" in d.note
    # coolest floor first: the 30 C cells (offset 5) are relaxed before any 33 C cell
    smax = float(cfg["cell"]["soc_max"])
    assert np.all(g.caps[-36:-32] == smax)
    assert np.any(g.caps[-32:] < smax)


def _hot_top_module(cfg, layout, soc=0.8, rise_K=1.0):
    """16 hot-zone cells (offset 8 K) 1 K above their local air and rising; the rest at room."""
    params = bol_params(cfg)
    params.T_offset_K = np.zeros(layout.n)
    params.T_offset_K[-16:] = 8.0
    s = _state(cfg, params, soc=soc)
    s.T_K = 298.15 + params.T_offset_K
    s.T_K[-16:] += rise_K
    hist = s.T_K[None, :] + np.linspace(-0.3, 0.0, 6)[:, None] * (params.T_offset_K > 0)
    return params, s, hist


def test_galvanic_parks_all_hot_cells_when_slack(mcfg, layout):
    params, s, hist = _hot_top_module(mcfg, layout)
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    d = g.decide(s, params, layout, 30000.0, _ctx(mcfg, T_hist=hist, plan=20000.0, plan_rem=20000.0))
    hot = params.T_offset_K > 0
    assert np.all(~d.active[hot]) and np.all(d.active[~hot])
    assert np.all(g.kind[hot] == C.OPTIONAL)
    assert "out: 16/16 hot (parked, slack)" in d.note
    ev = g.share_events[-1]
    assert (ev.n_hot, ev.k, ev.binding) == (16, 16, "")
    assert d.P_W == 30000.0
    # still parked much later: no time limit when slack
    s.out_s[hot] = 10 * float(mcfg["controllers"]["share_period_s"])
    d2 = g.decide(s, params, layout, 30000.0, _ctx(mcfg, T_hist=hist, plan=20000.0, plan_rem=20000.0))
    assert np.all(~d2.active[hot])


def test_galvanic_shares_when_healthy_guard_binds(mcfg, layout):
    params, s, hist = _hot_top_module(mcfg, layout)
    guard = float(mcfg["controllers"]["healthy_guard_C"])
    s.T_K[: layout.n - 16] = float(c_to_k(guard - 0.6))  # healthy cells 4.4 K above air, near the guard
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    d = g.decide(s, params, layout, 30000.0, _ctx(mcfg, T_hist=hist, plan=20000.0, plan_rem=20000.0))
    ev = g.share_events[-1]
    assert 0 <= ev.k < ev.n_hot == 16
    assert ev.binding.startswith("healthy guard")
    assert f"out: {ev.k}/16 hot (sharing; binding: healthy guard" in d.note
    # the chosen set keeps every healthy cell inside the guard by the I^2 prediction
    n_new = int(d.active.sum())
    T_air = s.T_room_K + params.T_offset_K
    pred = T_air + (s.T_K - T_air) * (layout.n / n_new) ** 2
    assert np.all(pred[d.active & (T_air < c_to_k(guard))] <= c_to_k(guard) + 1e-9)


def test_galvanic_shares_fairly_after_share_period(mcfg, layout):
    params, s, hist = _hot_top_module(mcfg, layout)
    guard = float(mcfg["controllers"]["healthy_guard_C"])
    s.T_K[: layout.n - 16] = float(c_to_k(guard - 0.6))
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    ctx = _ctx(mcfg, T_hist=hist, plan=20000.0, plan_rem=20000.0)
    d = g.decide(s, params, layout, 30000.0, ctx)
    k = g.share_events[-1].k
    assert 0 < k < 16
    out1 = np.flatnonzero(~d.active)
    # those cells have now been out a full share period; the others have been in long enough
    s.active = d.active.copy()
    s.out_s = np.where(d.active, 0.0, float(mcfg["controllers"]["share_period_s"]))
    d2 = g.decide(s, params, layout, 30000.0, ctx)
    hot = params.T_offset_K > 0
    waiting = hot & d.active  # hot cells that were in the string
    k2 = g.share_events[-1].k
    assert 0 < k2 < 16
    # the slots go first to hot cells that were in the string; cells out a full period yield
    assert int((~d2.active & waiting).sum()) == min(k2, int(waiting.sum()))
    if k2 <= int(waiting.sum()):
        assert set(np.flatnonzero(~d2.active)).isdisjoint(out1)


def test_galvanic_charge_balance(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params, soc=0.2)
    s.soc[:60] = 0.85  # 60 cells far ahead: with everyone in, the charge would stop with many cells low
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    plan = full_day_plan_Wh(mcfg)
    d = g.decide(s, params, layout, -15000.0, _ctx(mcfg, plan=plan))
    assert "charge balance" in d.note
    assert np.all(~d.active[np.flatnonzero(g.kind == C.OPTIONAL)])
    out = ~d.active
    assert out.any() and np.all(s.soc[out] == 0.85)  # only cells ahead are bypassed
    I, V, _, reason = plant.string_operating_point(-15000.0, d.active, s, params, mcfg)
    assert reason == "" and V >= (1 + float(mcfg["controllers"]["v_margin_frac"])) * C.v_floor_V(-15000.0, mcfg)
    # balanced pack: nothing to do
    g2 = C.GalvanicBypass(mcfg)
    g2.reset(layout.n)
    d2 = g2.decide(_state(mcfg, params, soc=0.2), params, layout, -15000.0, _ctx(mcfg, plan=plan))
    assert d2.active.all()


def test_galvanic_energy_check_blocks_removal(mcfg, layout):
    params = bol_params(mcfg)
    params.T_offset_K = np.zeros(layout.n)
    params.T_offset_K[-32:] = 8.0
    s = _state(mcfg, params, soc=0.45)
    s.T_K = 298.15 + params.T_offset_K + 2.0
    hist = s.T_K[None, :] + np.linspace(-0.3, 0.0, 6)[:, None]
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    E_all = plant.deliverable_energy_Wh(float(mcfg["cell"]["soc_min"]), s.soc, params, mcfg)
    d = g.decide(s, params, layout, 30000.0, _ctx(mcfg, T_hist=hist, plan=E_all, plan_rem=E_all))
    assert d.active.all()


def test_galvanic_cold_cells_stay_in(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params, room_C=10.0, soc=0.45)
    hist = s.T_K[None, :] + np.linspace(-1.0, 0.0, 6)[:, None]
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    d = g.decide(s, params, layout, 30000.0, _ctx(mcfg, T_hist=hist, plan=30000.0, plan_rem=20000.0))
    assert d.active.all()


def test_galvanic_power_wins_alert_only_for_thermal(mcfg, layout):
    params = bol_params(mcfg)
    s = _state(mcfg, params, soc=0.45)
    s.T_K[:40] = float(c_to_k(float(mcfg["controllers"]["T_crit_C"]) + 1.0))  # 40 cells hard-limited
    g = C.GalvanicBypass(mcfg)
    g.reset(layout.n)
    d = g.decide(s, params, layout, 30000.0, _ctx(mcfg, plan=30000.0, plan_rem=30000.0))
    assert d.alert == C.ALERT_SERVICE
    assert d.P_W < 30000.0
    assert plant.string_operating_point(d.P_W, d.active, s, params, mcfg)[3] == ""


# ---- speed ------------------------------------------------------------------------------------


def _small_cfg(cfg):
    small = copy.deepcopy(cfg)
    n_mod, cpm = 2, 4
    scale = n_mod * cpm / (int(cfg["pack"]["n_modules"]) * int(cfg["pack"]["cells_per_module"]))
    small["pack"]["n_modules"], small["pack"]["cells_per_module"] = n_mod, cpm
    for key in ("V_pcs_min_V", "V_pcs_max_V", "P_rated_W"):
        small["pack"][key] = float(cfg["pack"][key]) * scale
    for key in ("charge_P_W", "discharge_P_W"):
        small["site"]["duty"][key] = float(cfg["site"]["duty"][key]) * scale
    small["site"]["hvac"]["UA_per_string_WK"] = float(cfg["site"]["hvac"]["UA_per_string_WK"]) * scale
    return small, build_layout(n_mod, cpm)


def test_all_controllers_small_pack_fast(mcfg):
    cfg, lay = _small_cfg(mcfg)
    params, _ = build_params(cfg, lay, 0, "uniform")
    ctls = [C.FixedStringDerate(cfg), C.FixedStringNoDerate(cfg), C.ThresholdBypass(cfg), C.GalvanicBypass(cfg)]
    t0 = time.perf_counter()
    for ctl in ctls:
        r = simulate(params, lay, cfg, ctl, days=2, seed=0)
        assert r.T_C.shape == (2 * 1440, lay.n)
        assert r.delivered_Wh() > 0.0
    assert time.perf_counter() - t0 < 10.0
