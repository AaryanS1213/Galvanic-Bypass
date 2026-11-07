"""Tests for gb.plant."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gb import plant
from gb.config import Config
from gb.types import CellParams, Decision, Layout, PackState, build_layout
from gb.units import c_to_k


def make_params(cfg: Config, n: int) -> CellParams:
    c, p = cfg["cell"], cfg["pack"]
    one = np.ones(n)
    return CellParams(
        Q_Ah=one * c["Q_nom_Ah"],
        R_ref_ohm=one * c["R_dc_ref_ohm"],
        G_cool_WK=one * p["G_cool_WK"],
        C_JK=one * c["mass_kg"] * c["cp_J_kgK"],
        T_offset_K=np.zeros(n),
        soh0=one.copy(),
    )


def make_state(cfg: Config, n: int, soc: float = 0.5, T_C: float | None = None) -> PackState:
    T_room = float(c_to_k(cfg["site"]["room_C"]))
    T = T_room if T_C is None else float(c_to_k(T_C))
    return PackState(
        t_s=0.0,
        soc=np.full(n, soc),
        T_K=np.full(n, T),
        q_cal=np.zeros(n),
        q_cyc=np.zeros(n),
        active=np.ones(n, dtype=bool),
        dwell_s=np.zeros(n),
        out_s=np.zeros(n),
        ah_throughput=np.zeros(n),
        T_room_K=T_room,
    )


def full_pack(cfg: Config) -> tuple[Layout, CellParams, PackState]:
    layout = build_layout(cfg["pack"]["n_modules"], cfg["pack"]["cells_per_module"])
    return layout, make_params(cfg, layout.n), make_state(cfg, layout.n)


def test_usable_energy_bol_full_window(cfg) -> None:
    layout, params, _ = full_pack(cfg)
    c = cfg["cell"]
    E = plant.usable_energy_Wh(c["soc_min"], c["soc_max"], params, cfg)
    assert E == pytest.approx(53867.0, rel=1e-3)
    # matches a fine numerical integral of the OCV curve
    s = np.linspace(c["soc_min"], c["soc_max"], 20001)
    v = plant.ocv(s, cfg)
    ref = layout.n * c["Q_nom_Ah"] * float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(s)))
    assert E == pytest.approx(ref, rel=1e-6)
    d = cfg["site"]["duty"]
    assert plant.deliverable_energy_Wh(c["soc_min"], c["soc_max"], params, cfg) == pytest.approx(E * (1 - d["loss_frac"]))


def test_usable_energy_per_cell_caps_and_active(cfg) -> None:
    layout, params, _ = full_pack(cfg)
    c = cfg["cell"]
    lo, hi = c["soc_min"], c["soc_max"]
    E = plant.usable_energy_Wh(lo, hi, params, cfg)
    caps = np.full(layout.n, hi)
    capped = np.arange(layout.n) < 32
    caps[capped] = 0.70
    Ec = plant.usable_energy_Wh(lo, caps, params, cfg)
    one_cell_lost = plant.usable_energy_Wh(0.70, hi, params, cfg) / layout.n
    assert Ec == pytest.approx(E - 32 * one_cell_lost, rel=1e-9)
    assert Ec < E
    active = ~capped
    assert plant.usable_energy_Wh(lo, caps, params, cfg, active) == pytest.approx(E * (layout.n - 32) / layout.n)
    # inverted bounds contribute nothing; SOH scales linearly
    assert plant.usable_energy_Wh(0.9, 0.2, params, cfg) == 0.0
    import dataclasses

    half = dataclasses.replace(params, soh0=params.soh0 * 0.5)
    assert plant.usable_energy_Wh(lo, hi, half, cfg) == pytest.approx(0.5 * E)


def test_power_equals_I_times_V(cfg: Config) -> None:
    _, params, state = full_pack(cfg)
    for P in (30000.0, 10000.0, -15000.0):
        I, V, V_cell, reason = plant.string_operating_point(P, state.active, state, params, cfg)
        assert reason == ""
        assert abs(I * V - P) < 1e-9
        assert np.sign(I) == np.sign(P)
    I, V, _, reason = plant.string_operating_point(0.0, state.active, state, params, cfg)
    assert I == 0.0 and reason == ""


def test_resistance_falls_with_temperature(cfg: Config) -> None:
    _, params, state = full_pack(cfg)
    hot = state.copy()
    hot.T_K = hot.T_K + 10.0
    assert np.all(plant.R_cell(hot, params, cfg) < plant.R_cell(state, params, cfg))
    assert np.allclose(plant.R_cell(state, params, cfg), cfg["cell"]["R_dc_ref_ohm"])  # T_ref, SOC 0.5, SOH 1


@pytest.mark.parametrize("k", [4, 8, 12])
def test_bypass_heat_scaling(cfg: Config, k: int) -> None:
    _, params, state = full_pack(cfg)
    n = state.soc.size
    P = float(cfg["pack"]["P_rated_W"])
    R_i = plant.R_cell(state, params, cfg)
    all_on = np.ones(n, dtype=bool)
    some_off = all_on.copy()
    some_off[:k] = False
    I0, _, _, r0 = plant.string_operating_point(P, all_on, state, params, cfg)
    Ik, _, _, rk = plant.string_operating_point(P, some_off, state, params, cfg)
    assert r0 == "" and rk == ""
    e0 = np.where(all_on, I0**2 * R_i, 0.0)
    ek = np.where(some_off, Ik**2 * R_i, 0.0)
    ratio = n / (n - k)
    assert ek[k] / e0[k] == pytest.approx(ratio**2, rel=0.01)
    assert ek.sum() / e0.sum() == pytest.approx(ratio, rel=0.01)


def test_bypassed_cell_has_no_electrochemical_heat(cfg: Config) -> None:
    n = 8
    p = cfg["pack"]
    active = np.ones(n, dtype=bool)
    active[3] = False
    R_i = np.full(n, 1e-3)
    I = 100.0
    Q, sw = plant.heat(I, active, R_i, cfg)
    frac = p["switch_heat_to_cell_frac"]
    assert Q[3] == pytest.approx(frac * I**2 * p["R_on_bypass_ohm"])
    assert Q[0] == pytest.approx(I**2 * R_i[0] + frac * I**2 * p["R_on_series_ohm"])
    assert sw == pytest.approx(I**2 * (7 * p["R_on_series_ohm"] + p["R_on_bypass_ohm"]))


def test_single_cell_adiabatic_heating(cfg: Config) -> None:
    c = copy.deepcopy(cfg)
    c["pack"]["G_ij_WK"] = 0.0
    layout = build_layout(1, 1)
    params = make_params(c, 1)
    params.G_cool_WK = np.zeros(1)
    T = np.array([float(c_to_k(25.0))])
    I, R = 82.0, 0.56e-3
    Q = np.array([I**2 * R])
    dt = 60.0
    for _ in range(int(7200 / dt)):
        T = T + dt * plant.thermal_derivative(T, Q, params, layout, float(c_to_k(25.0)), c)
    dT = float(T[0] - c_to_k(25.0))
    assert dT == pytest.approx(5.6, rel=0.02)


def test_thermal_neighbors_conserve_heat(small_pack) -> None:
    cfg, layout = small_pack.cfg, small_pack.layout
    params = make_params(cfg, layout.n)
    params.G_cool_WK = np.zeros(layout.n)
    T = np.linspace(295.0, 310.0, layout.n)
    d = plant.thermal_derivative(T, np.zeros(layout.n), params, layout, 298.15, cfg)
    assert abs(np.sum(d * params.C_JK)) < 1e-9
    assert d[0] > 0 and d[-1] < 0


def test_arrhenius_factor(cfg: Config) -> None:
    T_ref = float(c_to_k(25.0))
    assert plant.arrhenius_factor(T_ref, 28500.0, T_ref) == pytest.approx(1.0)
    r = plant.arrhenius_factor(c_to_k(35.0), 28500.0, T_ref) / plant.arrhenius_factor(c_to_k(25.0), 28500.0, T_ref)
    assert r == pytest.approx(1.45, abs=0.01)


def test_calendar_aging_sqrt_t(cfg: Config) -> None:
    n = 3
    params = make_params(cfg, n)
    T = np.full(n, float(c_to_k(cfg["cell"]["T_ref_C"])))
    soc = np.array([0.95, 0.5, 0.05])
    A = cfg["aging"]["A_cal_per_day_z"]
    Af = A * plant.f_soc(soc, cfg)
    t0_d = 10.0
    q = Af * np.sqrt(t0_d)
    dt_s = 3600.0
    days = np.array([100.0, 400.0, 1000.0])
    out = []
    t_d = t0_d
    for target in days:
        while t_d < target - 1e-9:
            dq, _ = plant.aging_rates(T, soc, 0.0, np.ones(n, dtype=bool), q, params, cfg)
            q = q + dq * dt_s
            t_d += dt_s / 86400.0
        out.append(q.copy())
    for qq, t in zip(out, days):
        np.testing.assert_allclose(qq, Af * np.sqrt(t), rtol=1e-3)
    np.testing.assert_allclose(out[1] / out[0], 2.0, rtol=1e-3)


def test_cycle_aging_only_active(cfg: Config) -> None:
    n = 4
    params = make_params(cfg, n)
    T = np.full(n, float(c_to_k(25.0)))
    active = np.array([True, True, False, True])
    _, dq_cyc = plant.aging_rates(T, np.full(n, 0.5), 81.0, active, np.zeros(n), params, cfg)
    assert dq_cyc[2] == 0.0
    assert dq_cyc[0] == pytest.approx(cfg["aging"]["B_cyc"] * 81.0 / (3600 * cfg["cell"]["Q_nom_Ah"]))


def _stored_energy_J(state: PackState, params: CellParams, cfg: Config) -> float:
    """E = sum_i Q_i * 3600 * SOH_i * integral_0^soc_i ocv(s) ds [J]."""
    grid = np.linspace(0.0, 1.0, 20001)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (plant.ocv(grid[1:], cfg) + plant.ocv(grid[:-1], cfg)) * np.diff(grid))])
    integ = np.interp(state.soc, grid, cum)
    return float(np.sum(params.Q_Ah * 3600.0 * state.soh * integ))


def test_energy_balance_discharge(cfg: Config) -> None:
    layout, params, state = full_pack(cfg)
    state.soc[:] = 0.9
    n = layout.n
    active = np.ones(n, dtype=bool)
    active[::16] = False  # one bypassed cell per module
    dt = float(cfg["site"]["dt_s"])
    P = float(cfg["pack"]["P_rated_W"])
    E0 = _stored_energy_J(state, params, cfg)
    delivered = losses = 0.0
    s = state
    for _ in range(int(3600 / dt)):
        R_i = plant.R_cell(s, params, cfg)
        s_new, info = plant.step(s, params, layout, cfg, Decision(active=active, P_W=P), dt)
        assert info.infeasible_reason == ""
        resistive = float(np.sum(np.where(active, info.I_A**2 * R_i, 0.0)))
        delivered += info.P_del_W * dt
        losses += (resistive + info.switch_loss_W) * dt
        s = s_new
    E1 = _stored_energy_J(s, params, cfg)
    assert delivered + losses == pytest.approx(E0 - E1, rel=0.02)
    assert np.all(s.soc[~active] == 0.9)
    assert s.T_K[~active].max() < s.T_K[active].mean()


def test_step_does_not_mutate_and_idle_switching(cfg: Config) -> None:
    layout, params, state = full_pack(cfg)
    before = state.copy()
    active = state.active.copy()
    active[5] = False
    s1, info = plant.step(state, params, layout, cfg, Decision(active=active, P_W=0.0), 60.0)
    for name in ("soc", "T_K", "q_cal", "active", "dwell_s"):
        np.testing.assert_array_equal(getattr(state, name), getattr(before, name))
    assert info.I_A == 0.0 and np.all(info.Q_cell_W == 0.0)
    assert not s1.active[5]
    np.testing.assert_array_equal(s1.soc, state.soc)
    assert s1.dwell_s[5] == 60.0 and s1.dwell_s[0] == 60.0 and s1.out_s[5] == 60.0
    assert s1.t_s == 60.0
    s2, _ = plant.step(s1, params, layout, cfg, Decision(active=active, P_W=0.0), 60.0)
    assert s2.dwell_s[5] == 120.0 and s2.out_s[5] == 120.0 and s2.out_s[0] == 0.0


def test_step_rejects_unstable_dt(cfg: Config) -> None:
    layout, params, state = full_pack(cfg)
    with pytest.raises(ValueError, match="unstable"):
        plant.step(state, params, layout, cfg, Decision(active=state.active, P_W=0.0), 1e5)


def test_k_max_smaller_near_empty(cfg: Config) -> None:
    _, params, state = full_pack(cfg)
    P = float(cfg["pack"]["P_rated_W"])
    mid = plant.k_max_voltage(P, state, params, cfg, state.active)
    low = state.copy()
    low.soc[:] = cfg["cell"]["soc_min"] + 0.01
    empty = plant.k_max_voltage(P, low, params, cfg, low.active)
    assert 0 <= empty < mid
    # The estimate is confirmed by the exact operating point at mid SOC.
    active = state.active.copy()
    active[:mid] = False
    assert plant.string_operating_point(P, active, state, params, cfg)[3] == ""


def test_max_feasible_power_when_infeasible(cfg: Config) -> None:
    _, params, state = full_pack(cfg)
    active = state.active.copy()
    active[:40] = False  # 88 cells * 2.9 V ~ 255 V: 30 kW breaks V_pcs_min
    P_req = float(cfg["pack"]["P_rated_W"])
    assert plant.string_operating_point(P_req, active, state, params, cfg)[3] != ""
    P = plant.max_feasible_power(P_req, active, state, params, cfg)
    assert 0.0 < P < P_req
    assert plant.string_operating_point(P, active, state, params, cfg)[3] == ""
    # Charging at soc_max: nothing is feasible.
    full = state.copy()
    full.soc[:] = cfg["cell"]["soc_max"]
    assert plant.max_feasible_power(-15000.0, full.active, full, params, cfg) == 0.0


def test_step_reports_shortfall(cfg: Config) -> None:
    layout, params, state = full_pack(cfg)
    active = state.active.copy()
    active[:40] = False
    P_req = float(cfg["pack"]["P_rated_W"])
    _, info = plant.step(state, params, layout, cfg, Decision(active=active, P_W=P_req), 60.0)
    assert info.shortfall_W > 0 and info.infeasible_reason != ""
    assert info.shortfall_W == pytest.approx(P_req - info.P_del_W)


def test_hvac_power_rises_as_room_colder(cfg: Config) -> None:
    T_out = 35.0
    p = [plant.hvac_power(500.0, T_room, T_out, cfg) for T_room in (30.0, 25.0, 20.0, 15.0)]
    assert all(b > a for a, b in zip(p, p[1:]))
    assert plant.hvac_power(0.0, 25.0, 10.0, cfg) == 0.0


# --- exact calendar ageing (aging_step) -------------------------------------------------


def _calendar_case(cfg: Config) -> tuple[CellParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3 cells at 35 C (AF != 1), three SOCs; returns params, T, soc, A*f, AF_cal."""
    n = 3
    params = make_params(cfg, n)
    T = np.full(n, float(c_to_k(35.0)))
    soc = np.array([0.95, 0.5, 0.05])
    Af = cfg["aging"]["A_cal_per_day_z"] * plant.f_soc(soc, cfg)
    af = plant.arrhenius_factor(T, cfg["aging"]["Ea_cal_J_mol"], float(c_to_k(cfg["cell"]["T_ref_C"])))
    return params, T, soc, Af, af


def _integrate_calendar(cfg: Config, dt_s: float, days: float) -> np.ndarray:
    params, T, soc, _, _ = _calendar_case(cfg)
    q = np.zeros(3)
    active = np.ones(3, dtype=bool)
    for _ in range(int(round(days * 86400.0 / dt_s))):
        q, _ = plant.aging_step(q, T, soc, 0.0, active, dt_s, params, cfg)
    return q


def test_aging_step_independent_of_dt(cfg: Config) -> None:
    q_min = _integrate_calendar(cfg, 60.0, 365.0)
    q_day = _integrate_calendar(cfg, 86400.0, 365.0)
    np.testing.assert_allclose(q_min, q_day, rtol=5e-3)


def test_aging_step_exact_from_zero(cfg: Config) -> None:
    params, T, soc, Af, af = _calendar_case(cfg)
    z = cfg["aging"]["z"]
    active = np.ones(3, dtype=bool)
    q = np.zeros(3)
    t_d = 0.0
    for dt_d in (0.5, 3.0, 30.0, 365.0):  # uneven steps, all from q = 0
        q, _ = plant.aging_step(q, T, soc, 0.0, active, dt_d * 86400.0, params, cfg)
        t_d += dt_d
        np.testing.assert_allclose(q, Af * (af * t_d) ** z, rtol=1e-12)


def test_aging_step_cycle_term(cfg: Config) -> None:
    params, T, soc, _, _ = _calendar_case(cfg)
    active = np.array([True, False, True])
    dt = 60.0
    _, dq_cyc = plant.aging_step(np.zeros(3), T, soc, 81.0, active, dt, params, cfg)
    _, rate = plant.aging_rates(T, soc, 81.0, active, np.zeros(3), params, cfg)
    np.testing.assert_allclose(dq_cyc, rate * dt, rtol=1e-12)
    assert dq_cyc[1] == 0.0


def test_aging_rates_finite_at_zero(cfg: Config) -> None:
    params, T, soc, Af, af = _calendar_case(cfg)
    z = cfg["aging"]["z"]
    dq, _ = plant.aging_rates(T, soc, 0.0, np.ones(3, dtype=bool), np.zeros(3), params, cfg)
    assert np.all(np.isfinite(dq)) and np.all(dq > 0)
    q1 = Af * af**z  # state after 1 day
    expected = z * Af ** (1 / z) * af * q1 ** (1 - 1 / z) / 86400.0
    np.testing.assert_allclose(dq, expected, rtol=1e-12)


def test_step_uses_exact_calendar(cfg: Config) -> None:
    layout, params, state = full_pack(cfg)
    dt = 60.0
    new, _ = plant.step(state, params, layout, cfg, Decision(active=state.active, P_W=0.0), dt)
    exp, _ = plant.aging_step(state.q_cal, state.T_K, state.soc, 0.0, state.active, dt, params, cfg)
    np.testing.assert_allclose(new.q_cal, exp, rtol=1e-12)
    assert np.all(new.q_cal > 0)


def test_R_cell_independent_of_eol_threshold(cfg: Config) -> None:
    _, params, state = full_pack(cfg)
    state.q_cyc[:] = 0.1  # SOH 0.9, so the growth term matters
    R0 = plant.R_cell(state, params, cfg)
    c = copy.deepcopy(cfg)
    c["aging"]["eol_soh_fixed"] = 0.8
    np.testing.assert_array_equal(plant.R_cell(state, params, c), R0)
    c["cell"]["r_growth_ref_loss"] = 0.2
    assert np.all(plant.R_cell(state, params, c) > R0)
