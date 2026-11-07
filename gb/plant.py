"""Electro-thermal plant model of one string: OCV, resistance, cell and switch heat, thermal network.

Single source of truth for the physics; synth and simulate both call it. All functions are pure,
vectorized over cells, SI internally (s, K, ohm, A, W, J). Sign: P > 0 / I > 0 is discharge.
Every physical parameter is read from cfg (see config/*.yaml); only physical/unit constants live here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gb.types import CellParams, Decision, Layout, PackState, StepInfo
from gb.units import c_to_k

RG_J_MOLK: float = 8.314  # universal gas constant [J/(mol K)]
S_PER_H: float = 3600.0
S_PER_DAY: float = 86400.0
Q_CAL_TINY: float = 1e-6  # below this q, aging_rates evaluates the calendar rate at the 1-day state (q^(1-1/z) singular at 0)
BISECT_ITERS: int = 30
EULER_STABILITY_LIMIT: float = 0.5

Cfg = dict[str, Any]
FArr = NDArray[np.float64]
BArr = NDArray[np.bool_]


# --------------------------------------------------------------------------------------------
# Cell electrical model
# --------------------------------------------------------------------------------------------


def ocv(soc: ArrayLike, cfg: Cfg) -> FArr:
    """Open-circuit voltage [V]: U_oc(soc) = interp(soc, cell.ocv_table.soc, cell.ocv_table.ocv).

    soc [1]; clamps outside the table range.
    """
    tab = cfg["cell"]["ocv_table"]
    return np.interp(np.asarray(soc, dtype=np.float64), tab["soc"], tab["ocv"])


def ocv_integral(soc: ArrayLike, cfg: Cfg) -> FArr:
    """F(soc) = integral_{soc_0}^{soc} U_oc(s) ds  [V * 1], soc_0 = first OCV-table point.

    Exact for the piecewise-linear OCV table; outside the table range U_oc is clamped (as in
    ocv), so F extends linearly there. Use differences F(b) - F(a) only.
    """
    tab = cfg["cell"]["ocv_table"]
    xs = np.asarray(tab["soc"], dtype=np.float64)
    ys = np.asarray(tab["ocv"], dtype=np.float64)
    # cumulative integral at the knots, measured from xs[0]
    F_knots = np.concatenate([[0.0], np.cumsum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs))])
    s = np.asarray(soc, dtype=np.float64)
    sc = np.clip(s, xs[0], xs[-1])
    k = np.clip(np.searchsorted(xs, sc, side="right") - 1, 0, xs.size - 2)
    F_in = F_knots[k] + 0.5 * (ys[k] + ocv(sc, cfg)) * (sc - xs[k])
    return F_in + (s - sc) * np.where(s < xs[0], ys[0], ys[-1])


def usable_energy_Wh(
    soc_lo: ArrayLike, soc_hi: ArrayLike, params: CellParams, cfg: Cfg, active: BArr | None = None
) -> float:
    """Open-circuit energy between two SOC bounds, summed over cells [Wh].

    E = sum_i Q_Ah_i * SOH0_i * integral_{soc_lo_i}^{soc_hi_i} U_oc(s) ds   (only active cells if
    `active` is given). soc_lo / soc_hi are scalars or per-cell (n,) arrays; a cell with
    soc_hi_i <= soc_lo_i contributes 0. Loss-free; see deliverable_energy_Wh.
    """
    n = params.Q_Ah.size
    lo = np.broadcast_to(np.asarray(soc_lo, dtype=np.float64), (n,))
    hi = np.broadcast_to(np.asarray(soc_hi, dtype=np.float64), (n,))
    e = params.Q_Ah * params.soh0 * np.maximum(ocv_integral(hi, cfg) - ocv_integral(lo, cfg), 0.0)
    if active is not None:
        e = e[np.asarray(active, dtype=bool)]
    return float(e.sum())


def deliverable_energy_Wh(
    soc_lo: ArrayLike, soc_hi: ArrayLike, params: CellParams, cfg: Cfg, active: BArr | None = None
) -> float:
    """E_del = usable_energy_Wh(...) * (1 - site.duty.loss_frac)  [Wh] (I^2 R in cells and switches)."""
    return usable_energy_Wh(soc_lo, soc_hi, params, cfg, active) * (1.0 - float(cfg["site"]["duty"]["loss_frac"]))


def r_soc_factor(soc: ArrayLike, cfg: Cfg) -> FArr:
    """Resistance SOC factor [1]: f_R(soc) = interp(soc, cell.r_soc_table.soc, .factor)."""
    tab = cfg["cell"]["r_soc_table"]
    return np.interp(np.asarray(soc, dtype=np.float64), tab["soc"], tab["factor"])


def R_cell(state: PackState, params: CellParams, cfg: Cfg) -> FArr:
    """Cell DC resistance [ohm], shape (n,).

    R_i = R_ref_i * exp[(Ea_R / Rg) * (1/T_i - 1/T_ref)] * f_R(soc_i) * (1 + k_g * (1 - SOH_i) / dq_ref)

    T in K, Ea_R [J/mol] = cell.Ea_R_J_mol, Rg = 8.314 J/(mol K), T_ref = cell.T_ref_C in K,
    k_g = cell.r_growth_k, dq_ref = cell.r_growth_ref_loss (capacity loss at which R grows by
    a factor (1 + k_g); independent of the end-of-life threshold). Resistance falls as
    temperature rises.
    """
    c = cfg["cell"]
    T_ref = float(c_to_k(c["T_ref_C"]))
    arr = np.exp((float(c["Ea_R_J_mol"]) / RG_J_MOLK) * (1.0 / state.T_K - 1.0 / T_ref))
    dq_ref = float(c["r_growth_ref_loss"])
    growth = 1.0 + float(c["r_growth_k"]) * (1.0 - state.soh) / dq_ref
    return params.R_ref_ohm * arr * r_soc_factor(state.soc, cfg) * growth


def _string_R(active: BArr, R_i: FArr, cfg: Cfg) -> float:
    """R_str = sum(active * (R_i + R_on_series)) + sum(~active * R_on_bypass) [ohm]."""
    p = cfg["pack"]
    r_s = float(p["R_on_series_ohm"])
    r_b = float(p["R_on_bypass_ohm"])
    return float(np.sum(np.where(active, R_i + r_s, r_b)))


def _operating_point(
    P_W: float, active: BArr, soc: FArr, ocv_i: FArr, R_i: FArr, cfg: Cfg
) -> tuple[float, float, FArr, str]:
    """Core of string_operating_point with precomputed ocv_i and R_i (see that docstring)."""
    c, p = cfg["cell"], cfg["pack"]
    P = float(P_W)
    V_oc = float(np.sum(np.where(active, ocv_i, 0.0)))
    if P == 0.0:
        return 0.0, V_oc, np.where(active, ocv_i, 0.0), ""

    R_str = _string_R(active, R_i, cfg)
    disc = V_oc * V_oc - 4.0 * R_str * P
    if disc < 0.0:
        return float("nan"), float("nan"), np.full_like(ocv_i, np.nan), "power exceeds string maximum (discriminant < 0)"
    denom = V_oc + np.sqrt(disc)
    if denom <= 0.0:
        return float("nan"), float("nan"), np.full_like(ocv_i, np.nan), "no active cells"
    # I = (V_oc - sqrt(disc)) / (2 R_str), written in the cancellation-free form 2P / (V_oc + sqrt(disc)).
    I = 2.0 * P / float(denom)
    V_str = P / I
    V_cell = np.where(active, ocv_i - I * R_i, 0.0)

    I_max = float(c["I_max_A"])
    V_floor = max(float(p["V_pcs_min_V"]), abs(P) / I_max)
    V_pcs_max = float(p["V_pcs_max_V"])
    if V_str < V_floor:
        return I, V_str, V_cell, f"string voltage {V_str:.1f} V below floor {V_floor:.1f} V"
    if V_str > V_pcs_max:
        return I, V_str, V_cell, f"string voltage {V_str:.1f} V above PCS max {V_pcs_max:.1f} V"
    if abs(I) > I_max:
        return I, V_str, V_cell, f"current {abs(I):.1f} A above I_max {I_max:.1f} A"
    V_act = V_cell[active]
    if V_act.size and (V_act.min() < float(c["V_min"]) or V_act.max() > float(c["V_max"])):
        return I, V_str, V_cell, "active cell terminal voltage outside [V_min, V_max]"
    soc_act = soc[active]
    if I > 0.0 and soc_act.size and soc_act.min() <= float(c["soc_min"]):
        return I, V_str, V_cell, "discharging with an active cell at or below soc_min"
    if I < 0.0 and soc_act.size and soc_act.max() >= float(c["soc_max"]):
        return I, V_str, V_cell, "charging with an active cell at or above soc_max"
    return I, V_str, V_cell, ""


def string_operating_point(
    P_W: float, active: BArr, state: PackState, params: CellParams, cfg: Cfg
) -> tuple[float, float, FArr, str]:
    """Steady operating point of the series string at DC power P_W [W].

    V_oc  = sum(active * U_oc_i)                                   [V]
    R_str = sum(active * (R_i + R_on_series)) + sum(~active * R_on_bypass)  [ohm]
    P = I * (V_oc - I * R_str)  ->  I = (V_oc - sqrt(V_oc^2 - 4 R_str P)) / (2 R_str)  [A]
    V_str = P / I = V_oc - I * R_str [V] (V_str = V_oc at P = 0); V_cell_i = U_oc_i - I * R_i [V]
    (V_cell = 0 for bypassed cells). Works for charge: P < 0 gives I < 0.

    Returns (I_A, V_str_V, V_cell (n,), reason); reason is "" if feasible, else the first failing
    check in order: discriminant < 0; V_str outside [max(V_pcs_min, |P|/I_max), V_pcs_max];
    |I| > I_max; an active cell's V_cell outside [V_min, V_max]; discharging with an active cell
    soc <= soc_min; charging with an active cell soc >= soc_max. P = 0 (idle) is always feasible.
    """
    active = np.asarray(active, dtype=bool)
    return _operating_point(P_W, active, state.soc, ocv(state.soc, cfg), R_cell(state, params, cfg), cfg)


def _max_feasible(P_req: float, active: BArr, soc: FArr, ocv_i: FArr, R_i: FArr, cfg: Cfg) -> float:
    if _operating_point(P_req, active, soc, ocv_i, R_i, cfg)[3] == "":
        return float(P_req)
    sign = 1.0 if P_req > 0 else -1.0
    lo, hi = 0.0, abs(float(P_req))
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if _operating_point(sign * mid, active, soc, ocv_i, R_i, cfg)[3] == "":
            lo = mid
        else:
            hi = mid
    return sign * lo


def max_feasible_power(
    P_req: float, active: BArr, state: PackState, params: CellParams, cfg: Cfg
) -> float:
    """Largest deliverable power [W] with the sign of P_req and |P| <= |P_req|.

    Returns P_req if string_operating_point(P_req) is feasible; otherwise bisects |P| on
    [0, |P_req|] for 30 iterations, keeping the feasible end. Returns 0.0 if nothing is feasible.
    """
    active = np.asarray(active, dtype=bool)
    return _max_feasible(P_req, active, state.soc, ocv(state.soc, cfg), R_cell(state, params, cfg), cfg)


def k_max_voltage(
    P_W: float, state: PackState, params: CellParams, cfg: Cfg, candidate_active: BArr
) -> int:
    """Fast estimate of how many of the candidate-active cells can be bypassed at power P_W.

    V_floor = max(V_pcs_min, |P| / I_max)                       [V]
    I_est   = P / V_floor                                       [A]
    n_min   = ceil(V_floor / (mean U_oc - mean (R_i + R_on_series) * I_est))  over candidate cells
    k_max   = max(0, n_candidate - n_min); 0 if the denominator <= 0 or no candidates.
    Controllers must confirm with string_operating_point.
    """
    cand = np.asarray(candidate_active, dtype=bool)
    n_cand = int(cand.sum())
    if n_cand == 0:
        return 0
    c, p = cfg["cell"], cfg["pack"]
    V_floor = max(float(p["V_pcs_min_V"]), abs(float(P_W)) / float(c["I_max_A"]))
    R_i = R_cell(state, params, cfg) + float(p["R_on_series_ohm"])
    mean_ocv = float(np.mean(ocv(state.soc[cand], cfg)))
    mean_R = float(np.mean(R_i[cand]))
    denom = mean_ocv - mean_R * float(P_W) / V_floor
    if denom <= 0.0:
        return 0
    n_min = int(np.ceil(V_floor / denom))
    return max(0, n_cand - n_min)


# --------------------------------------------------------------------------------------------
# Heat and thermal network
# --------------------------------------------------------------------------------------------


def heat(I: float, active: BArr, R_i: FArr, cfg: Cfg) -> tuple[FArr, float]:
    """Heat deposited in each cell node [W] and total switch loss [W] at string current I [A].

    Q_cell_i = active_i * I^2 * R_i + frac * (active_i * I^2 * R_on_series + ~active_i * I^2 * R_on_bypass)
    switch_loss = I^2 * (n_active * R_on_series + n_bypassed * R_on_bypass)
    frac = pack.switch_heat_to_cell_frac; the remaining (1 - frac) of switch loss goes to the room air.
    A bypassed cell gets no electrochemical heat, only frac * I^2 * R_on_bypass from its bypass switch.
    """
    p = cfg["pack"]
    active = np.asarray(active, dtype=bool)
    r_s = float(p["R_on_series_ohm"])
    r_b = float(p["R_on_bypass_ohm"])
    frac = float(p["switch_heat_to_cell_frac"])
    I2 = float(I) ** 2
    q_echem = np.where(active, I2 * R_i, 0.0)
    q_sw = np.where(active, I2 * r_s, I2 * r_b)
    n_act = int(active.sum())
    switch_loss = I2 * (n_act * r_s + (active.size - n_act) * r_b)
    return q_echem + frac * q_sw, float(switch_loss)


def thermal_derivative(
    T_K: FArr, Q_cell: FArr, params: CellParams, layout: Layout, T_room_K: float, cfg: Cfg
) -> FArr:
    """Cell temperature rate [K/s].

    C_i dT_i/dt = Q_i - sum_j G_ij (T_i - T_j) - G_cool_i (T_i - (T_room + T_offset_i))
    C [J/K], Q [W], G_ij = pack.G_ij_WK [W/K] uniform over layout.neighbors, G_cool [W/K], T [K].
    """
    G_ij = float(cfg["pack"]["G_ij_WK"])
    flow = np.zeros_like(T_K, dtype=np.float64)  # net conduction out of each cell [W]
    nb = layout.neighbors
    if nb.size:
        a, b = nb[:, 0], nb[:, 1]
        f_ab = G_ij * (T_K[a] - T_K[b])
        np.add.at(flow, a, f_ab)
        np.add.at(flow, b, -f_ab)
    cool = params.G_cool_WK * (T_K - (T_room_K + params.T_offset_K))
    return (Q_cell - flow - cool) / params.C_JK


def _euler_stability_number(dt_s: float, params: CellParams, layout: Layout, cfg: Cfg) -> float:
    """dt * max_i((G_cool_i + deg_i * G_ij) / C_i) [1]; explicit Euler needs this < 0.5."""
    n = params.C_JK.size
    nb = layout.neighbors
    deg = np.bincount(nb.ravel(), minlength=n).astype(np.float64) if nb.size else np.zeros(n)
    G_tot = params.G_cool_WK + deg * float(cfg["pack"]["G_ij_WK"])
    return float(dt_s * np.max(G_tot / params.C_JK))


# --------------------------------------------------------------------------------------------
# Ageing
# --------------------------------------------------------------------------------------------


def arrhenius_factor(T_K: ArrayLike, Ea_J_mol: float, T_ref_K: float) -> FArr:
    """Arrhenius acceleration factor [1]: AF = exp[(Ea / Rg) * (1/T_ref - 1/T)], T in K, Ea in J/mol."""
    T = np.asarray(T_K, dtype=np.float64)
    return np.exp((float(Ea_J_mol) / RG_J_MOLK) * (1.0 / float(T_ref_K) - 1.0 / T))


def f_soc(soc: ArrayLike, cfg: Cfg) -> FArr:
    """Calendar SOC stress factor [1]: f_min for soc <= soc_lo, linear to 1 at soc_hi, 1 above."""
    fs = cfg["aging"]["f_soc"]
    return np.interp(
        np.asarray(soc, dtype=np.float64), [float(fs["soc_lo"]), float(fs["soc_hi"])], [float(fs["f_min"]), 1.0]
    )


def aging_rates(
    T_K: FArr, soc: FArr, I: float, active: BArr, q_cal: FArr, params: CellParams, cfg: Cfg
) -> tuple[FArr, FArr]:
    """Instantaneous capacity-loss rates (dq_cal/dt, dq_cyc/dt) [1/s], shape (n,). For controllers;
    the time step itself uses the exact `aging_step`.

    Calendar (time-scaled state form of q = A f(SOC) (AF t)^z, t in days):
      dq_cal/dt_day = z * (A f)^(1/z) * AF_cal * q^(1 - 1/z); divided by 86400 -> 1/s.
      The rate is infinite at q = 0 (for z < 1), so where q_cal < 1e-6 it is evaluated at the
      state reached after one day at the current drive, q = A f (AF * 1 day)^z, i.e. the
      instantaneous rate at t = 1 day, which keeps it finite. Cells with A f = 0 get rate 0.
      Applies to every cell, bypassed or not.
    Cycle: dq_cyc/dt = B * AF_cyc * active * |I| / (3600 * Q_Ah) [1/s].
    AF = arrhenius_factor(T, Ea, T_ref); A = aging.A_cal_per_day_z [day^-z], B = aging.B_cyc [1],
    z = aging.z, Ea_cal/Ea_cyc [J/mol], T_ref = cell.T_ref_C in K, Q_Ah = params.Q_Ah.
    """
    ag = cfg["aging"]
    T_ref = float(c_to_k(cfg["cell"]["T_ref_C"]))
    z = float(ag["z"])
    Af = float(ag["A_cal_per_day_z"]) * f_soc(soc, cfg)
    af_cal = arrhenius_factor(T_K, float(ag["Ea_cal_J_mol"]), T_ref)
    af_cyc = arrhenius_factor(T_K, float(ag["Ea_cyc_J_mol"]), T_ref)
    q_cal = np.asarray(q_cal, dtype=np.float64)
    q = np.where(q_cal < Q_CAL_TINY, Af * af_cal**z, q_cal)  # (AF * 1 day)^z with t in days
    with np.errstate(divide="ignore", invalid="ignore"):
        dq_cal_day = np.where(Af > 0.0, z * Af ** (1.0 / z) * af_cal * q ** (1.0 - 1.0 / z), 0.0)
    active = np.asarray(active, dtype=bool)
    dq_cyc = float(ag["B_cyc"]) * af_cyc * np.where(active, abs(float(I)), 0.0) / (S_PER_H * params.Q_Ah)
    return dq_cal_day / S_PER_DAY, dq_cyc


def aging_step(
    q_cal: FArr, T_K: FArr, soc: FArr, I: float, active: BArr, dt_s: float, params: CellParams, cfg: Cfg
) -> tuple[FArr, FArr]:
    """Advance capacity loss over dt_s [s] at constant drive (T, soc, I held over the step).

    Calendar, exact for q = A f (AF t)^z (z > 0), t in days:
      q_cal_new = ( q_cal^(1/z) + (A f(soc))^(1/z) * AF_cal(T) * dt_days )^z,  dt_days = dt_s / 86400
      (z = 0.5: sqrt(q^2 + (A f)^2 AF dt_days)). Exact from q = 0; independent of step size for
      constant drive. Applies to every cell, bypassed or not.
    Cycle: dq_cyc = B * AF_cyc(T) * active * |I| * dt_s / (3600 * Q_Ah)  [1].
    Returns (q_cal_new, dq_cyc), shape (n,). Symbols as in aging_rates.
    """
    ag = cfg["aging"]
    T_ref = float(c_to_k(cfg["cell"]["T_ref_C"]))
    z = float(ag["z"])
    Af = float(ag["A_cal_per_day_z"]) * f_soc(soc, cfg)
    af_cal = arrhenius_factor(T_K, float(ag["Ea_cal_J_mol"]), T_ref)
    af_cyc = arrhenius_factor(T_K, float(ag["Ea_cyc_J_mol"]), T_ref)
    q = np.maximum(np.asarray(q_cal, dtype=np.float64), 0.0)
    q_new = (q ** (1.0 / z) + Af ** (1.0 / z) * af_cal * (float(dt_s) / S_PER_DAY)) ** z
    active = np.asarray(active, dtype=bool)
    dq_cyc = (
        float(ag["B_cyc"]) * af_cyc * np.where(active, abs(float(I)), 0.0) * float(dt_s) / (S_PER_H * params.Q_Ah)
    )
    return q_new, dq_cyc


# --------------------------------------------------------------------------------------------
# Time step
# --------------------------------------------------------------------------------------------


def step(
    state: PackState, params: CellParams, layout: Layout, cfg: Cfg, decision: Decision, dt_s: float
) -> tuple[PackState, StepInfo]:
    """Advance the string by dt_s [s] with the requested switch pattern and power (explicit Euler).

    1. P_del = max_feasible_power(decision.P_W, decision.active)            [W]
    2. I, V_str at P_del; Q_cell, switch loss from heat(); T += dt * dT/dt   [K]
       (raises ValueError unless dt * max((G_cool + sum_j G_ij) / C) < 0.5)
    3. soc -= active * I * dt / (3600 * Q_Ah * SOH)
    4. q_cal, dq_cyc from aging_step (exact calendar integration at start-of-step T, soc, I);
       ah_throughput += active * |I| * dt / 3600 [Ah]; dwell_s = dt where active changed else +dt;
       out_s = 0 where active else +dt; t_s += dt.
    StepInfo: shortfall = |P_req| - |P_del|; Q_pack = sum(Q_cell) + (1 - frac) * switch_loss [W];
    infeasible_reason = the reason the requested P was infeasible ("" if delivered in full).
    The input state is not mutated. With P = 0 there is no current and no heat; the requested
    active pattern is still applied.
    """
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt}")
    stab = _euler_stability_number(dt, params, layout, cfg)
    if not stab < EULER_STABILITY_LIMIT:
        raise ValueError(
            f"explicit Euler thermal step unstable: dt * max((G_cool + sum G_ij) / C) = {stab:.3g} "
            f">= {EULER_STABILITY_LIMIT}; reduce dt_s"
        )

    active = np.asarray(decision.active, dtype=bool)
    if active.shape != state.soc.shape:
        raise ValueError(f"decision.active shape {active.shape} != state shape {state.soc.shape}")
    P_req = float(decision.P_W)

    ocv_i = ocv(state.soc, cfg)
    R_i = R_cell(state, params, cfg)
    req_reason = _operating_point(P_req, active, state.soc, ocv_i, R_i, cfg)[3]
    P_del = P_req if req_reason == "" else _max_feasible(P_req, active, state.soc, ocv_i, R_i, cfg)
    I, V_str, _, _ = _operating_point(P_del, active, state.soc, ocv_i, R_i, cfg)

    Q_cell, switch_loss = heat(I, active, R_i, cfg)
    frac = float(cfg["pack"]["switch_heat_to_cell_frac"])

    new = state.copy()
    new.T_K = state.T_K + dt * thermal_derivative(state.T_K, Q_cell, params, layout, state.T_room_K, cfg)
    I_cell = np.where(active, I, 0.0)
    new.soc = state.soc - I_cell * dt / (S_PER_H * params.Q_Ah * state.soh)

    new.q_cal, dq_cyc = aging_step(state.q_cal, state.T_K, state.soc, I, active, dt, params, cfg)
    new.q_cyc = state.q_cyc + dq_cyc
    new.ah_throughput = state.ah_throughput + np.abs(I_cell) * dt / S_PER_H
    changed = active != np.asarray(state.active, dtype=bool)
    new.dwell_s = np.where(changed, dt, state.dwell_s + dt)
    new.out_s = np.where(active, 0.0, state.out_s + dt)
    new.active = active.copy()
    new.t_s = state.t_s + dt

    info = StepInfo(
        I_A=float(I),
        V_str_V=float(V_str),
        P_req_W=P_req,
        P_del_W=float(P_del),
        shortfall_W=abs(P_req) - abs(P_del),
        n_active=int(active.sum()),
        Q_cell_W=Q_cell,
        Q_pack_W=float(np.sum(Q_cell) + (1.0 - frac) * switch_loss),
        switch_loss_W=switch_loss,
        infeasible_reason=req_reason,
    )
    return new, info


# --------------------------------------------------------------------------------------------
# HVAC (cost accounting only; never controlled)
# --------------------------------------------------------------------------------------------


def hvac_power(Q_pack_W: float, T_room_C: float, T_out_C: float, cfg: Cfg) -> float:
    """Electric power the HVAC needs to hold the room [W] (cost accounting only).

    COP = max(1, cop_ref * (1 + cop_slope_per_K * (T_room_C - cop_ref_room_C)))
    P_hvac = max(0, Q_pack + UA_per_string * (T_out - T_room)) / COP
    Q_pack [W], temperatures in degC (only differences used), UA [W/K].
    """
    h = cfg["site"]["hvac"]
    cop = max(1.0, float(h["cop_ref"]) * (1.0 + float(h["cop_slope_per_K"]) * (T_room_C - float(h["cop_ref_room_C"]))))
    load = max(0.0, float(Q_pack_W) + float(h["UA_per_string_WK"]) * (float(T_out_C) - float(T_room_C)))
    return load / cop
