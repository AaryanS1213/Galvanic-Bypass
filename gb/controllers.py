"""Bypass controllers: predictive bypass plus baselines (threshold bypass, whole-pack derate).

Four controllers implement gb.types.Controller:
  FixedStringDerate    conventional BMS + whole-pack derate between derate.T_start_C and T_stop_C
  FixedStringNoDerate  conventional BMS, derate only above T_crit_C (protection = colder room)
  ThresholdBypass      "Simple threshold bypass (assumed REON-like logic)": T_on / T_off hysteresis
  GalvanicBypass       ours: park hot cells when slack, share bypass slots when constrained

Every controller exposes `kind` (n,) int8 after each decide: 0 = in string, 1 = forced out,
2 = optional out, so the simulation can tell them apart. Temperatures are handled in kelvin; the
degC thresholds from config are converted once at the boundary.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gb import plant
from gb.types import CellParams, Decision, Layout, PackState
from gb.units import c_to_k, k_to_c

Cfg = dict[str, Any]
FArr = NDArray[np.float64]
BArr = NDArray[np.bool_]

S_PER_H: float = 3600.0
WH_PER_KWH: float = 1000.0

IN, FORCED, OPTIONAL = np.int8(0), np.int8(1), np.int8(2)

# Fixed-point iterations for the end-of-window SOC projection (numerical, not physical).
PROJECTION_ITERS: int = 3

LABEL_THRESHOLD: str = "Simple threshold bypass (assumed REON-like logic)"
ALERT_SERVICE: str = "too many cells out of range: service needed"
NOTE_CATCHUP: str = "catch-up limited by inverter V floor"


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------


def zone_soc_cap(T_air_C: ArrayLike, cfg: Cfg) -> FArr:
    """Per-cell SOC cap [1] keyed on local air temperature T_air_C [degC], shape (n,).

    cap_i = caps[k_i], k_i = number of thresholds <= T_air_i (controllers.zone_soc_cap.air_thresholds_C,
    ascending; caps has one more entry than thresholds). With the default config: T_air < 28 C ->
    soc_max, 28 <= T_air < 31 C -> 0.80, T_air >= 31 C -> 0.70.
    """
    zc = cfg["controllers"]["zone_soc_cap"]
    thr = np.asarray(zc["air_thresholds_C"], dtype=np.float64)
    caps = np.asarray(zc["caps"], dtype=np.float64)
    if caps.size != thr.size + 1:
        raise ValueError("zone_soc_cap: caps must have len(air_thresholds_C) + 1 entries")
    k = np.searchsorted(thr, np.asarray(T_air_C, dtype=np.float64), side="right")
    return caps[k]


def trend_slope_K_s(T_hist_K: ArrayLike, dt_s: float) -> FArr:
    """Least-squares slope of each column of T_hist_K (k, n) sampled every dt_s [s] -> [K/s], (n,).

    slope_i = sum_t (t - t_mean)(T_ti - T_mean_i) / sum_t (t - t_mean)^2; zeros if k < 2.
    """
    H = np.asarray(T_hist_K, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 2:
        return np.zeros(H.shape[-1] if H.ndim else 0)
    t = np.arange(H.shape[0], dtype=np.float64) * float(dt_s)
    tc = t - t.mean()
    return (tc @ (H - H.mean(axis=0))) / float(tc @ tc)


def _approx_current_A(P_W: float, active: BArr, state: PackState, cfg: Cfg) -> float:
    """Open-circuit current estimate I ~ P / sum(active * U_oc) [A] (for SOC look-ahead only)."""
    V_oc = float(np.sum(np.where(active, plant.ocv(state.soc, cfg), 0.0)))
    return float(P_W) / V_oc if V_oc > 0.0 else 0.0


def _soc_step(I_A: float, state: PackState, params: CellParams, horizon_s: float) -> FArr:
    """SOC moved in horizon_s at |I|: dsoc_i = |I| * horizon / (3600 * Q_i * SOH_i)  [1], (n,)."""
    return abs(float(I_A)) * float(horizon_s) / (S_PER_H * params.Q_Ah * state.soh)


def soc_limit_mask(P_W: float, active: BArr, state: PackState, params: CellParams, cfg: Cfg, horizon_s: float) -> BArr:
    """Cells that would cross a SOC limit within horizon_s at the requested power [bool, (n,)].

    discharge (P > 0): soc_i - dsoc_i <= soc_min;  charge (P < 0): soc_i + dsoc_i >= soc_max;
    dsoc_i from _soc_step with I ~ P / V_oc. Idle: no cell.
    """
    c = cfg["cell"]
    if P_W == 0.0:
        return np.zeros(state.soc.size, dtype=bool)
    dsoc = _soc_step(_approx_current_A(P_W, active, state, cfg), state, params, horizon_s)
    if P_W > 0.0:
        return state.soc - dsoc <= float(c["soc_min"])
    return state.soc + dsoc >= float(c["soc_max"])


def v_floor_V(P_W: float, cfg: Cfg) -> float:
    """Inverter DC voltage floor at power P [V]: V_floor = max(V_pcs_min, |P| / I_max)."""
    return max(float(cfg["pack"]["V_pcs_min_V"]), abs(float(P_W)) / float(cfg["cell"]["I_max_A"]))


def voltage_limit_mask(P_W: float, active: BArr, state: PackState, params: CellParams, cfg: Cfg) -> BArr:
    """Active cells whose terminal voltage V_i = U_oc_i - I R_i at P_W is outside [V_min, V_max]."""
    c, p = cfg["cell"], cfg["pack"]
    I, V_str, V_cell, _ = plant.string_operating_point(P_W, active, state, params, cfg)
    none = np.zeros(state.soc.size, dtype=bool)
    if not np.isfinite(I) or P_W == 0.0:
        return none
    # Only a cell-level limit counts: if the string itself cannot carry P (current or PCS voltage),
    # the implied current is fictitious and "power wins" derates instead.
    if abs(I) > float(c["I_max_A"]) or not v_floor_V(P_W, cfg) <= V_str <= float(p["V_pcs_max_V"]):
        return none
    return active & ((V_cell < float(c["V_min"])) | (V_cell > float(c["V_max"])))


class _Base:
    """Common bookkeeping: name, label, kind (n,) int8."""

    name: str = "base"
    label: str = "base"

    def __init__(self) -> None:
        self.kind: NDArray[np.int8] = np.zeros(0, dtype=np.int8)

    def reset(self, n: int) -> None:
        self.kind = np.zeros(n, dtype=np.int8)

    def _ensure(self, n: int) -> None:
        if self.kind.size != n:
            self.reset(n)


# --------------------------------------------------------------------------------------------
# 1) Conventional BMS + whole-pack derate
# --------------------------------------------------------------------------------------------


class FixedStringDerate(_Base):
    """All cells active, no bypass. P = P_req * clip((T_stop - max T) / (T_stop - T_start), 0, 1).

    T_start = derate.T_start_C (matched: = T_band_edge_C) or the constructor's t_start_C (typical
    variant: derate.variant_typical_T_start_C); T_stop = derate.T_stop_C (= T_crit_C). Temperatures
    in K internally. The plant's feasibility checks stop the string at the weakest cell's limit.
    """

    def __init__(self, cfg: Cfg, t_start_C: float | None = None) -> None:
        super().__init__()
        d = cfg["controllers"]["derate"]
        self.T_start_K = float(c_to_k(d["T_start_C"] if t_start_C is None else t_start_C))
        self.T_stop_K = float(c_to_k(d["T_stop_C"]))
        if not self.T_stop_K > self.T_start_K:
            raise ValueError("derate T_stop_C must exceed T_start_C")
        t0 = float(k_to_c(self.T_start_K))
        self.name = f"fixed_derate_{t0:g}C"
        self.label = f"Fixed string + derate from {t0:g} C"

    def decide(
        self, state: PackState, params: CellParams, layout: Layout, P_request_W: float, ctx: dict[str, Any]
    ) -> Decision:
        n = state.soc.size
        self._ensure(n)
        T_max = float(np.max(state.T_K))
        scale = float(np.clip((self.T_stop_K - T_max) / (self.T_stop_K - self.T_start_K), 0.0, 1.0))
        note = f"derate {scale:.2f} (max T {k_to_c(T_max):.1f} C)" if scale < 1.0 and P_request_W != 0 else ""
        return Decision(active=np.ones(n, dtype=bool), P_W=float(P_request_W) * scale, note=note)


# --------------------------------------------------------------------------------------------
# 2) Conventional BMS + extra cooling (colder room set by the caller)
# --------------------------------------------------------------------------------------------


class FixedStringNoDerate(_Base):
    """All cells active, no bypass. Derate only at the critical temperature:
    P = P_req * clip((T_lim - max T) / (T_lim - T_crit), 0, 1), T_lim = cell.T_lim_C, T_crit = T_crit_C.
    Thermal protection comes from a colder room (room_C passed to simulate)."""

    name = "fixed_no_derate"
    label = "Fixed string + extra cooling"

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self.T_crit_K = float(c_to_k(cfg["controllers"]["T_crit_C"]))
        self.T_lim_K = float(c_to_k(cfg["cell"]["T_lim_C"]))

    def decide(
        self, state: PackState, params: CellParams, layout: Layout, P_request_W: float, ctx: dict[str, Any]
    ) -> Decision:
        n = state.soc.size
        self._ensure(n)
        T_max = float(np.max(state.T_K))
        scale = float(np.clip((self.T_lim_K - T_max) / (self.T_lim_K - self.T_crit_K), 0.0, 1.0))
        return Decision(active=np.ones(n, dtype=bool), P_W=float(P_request_W) * scale)


# --------------------------------------------------------------------------------------------
# 3) Simple threshold bypass (assumed REON-like logic)
# --------------------------------------------------------------------------------------------


class ThresholdBypass(_Base):
    """Simple threshold bypass (assumed REON-like logic).

    Per cell: out when T_i > T_on, back in when T_i < T_off (hysteresis on measured T only);
    also out while at a SOC limit (soc_limit_mask over soc_lookahead_s) or voltage limit.
    No trend, no power awareness, no SOC cap, no rotation. If the set is infeasible the plant derates.
    """

    name = "threshold_bypass"
    label = LABEL_THRESHOLD

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        tb = cfg["controllers"]["threshold_bypass"]
        self.T_on_K = float(c_to_k(tb["T_on_C"]))
        self.T_off_K = float(c_to_k(tb["T_off_C"]))
        self.soc_lookahead_s = float(cfg["controllers"]["soc_lookahead_s"])
        self.hot = np.zeros(0, dtype=bool)

    def reset(self, n: int) -> None:
        super().reset(n)
        self.hot = np.zeros(n, dtype=bool)

    def decide(
        self, state: PackState, params: CellParams, layout: Layout, P_request_W: float, ctx: dict[str, Any]
    ) -> Decision:
        cfg = ctx["cfg"]
        n = state.soc.size
        self._ensure(n)
        T = state.T_K
        self.hot = (self.hot | (T > self.T_on_K)) & ~(T < self.T_off_K)
        P = float(P_request_W)
        limit = soc_limit_mask(P, ~self.hot, state, params, cfg, self.soc_lookahead_s)
        active = ~self.hot & ~limit
        active &= ~voltage_limit_mask(P, active, state, params, cfg)
        self.kind = np.where(active, IN, np.where(self.hot, OPTIONAL, FORCED)).astype(np.int8)
        n_out = int((~active).sum())
        return Decision(active=active, P_W=P, note=f"out: {n_out} (threshold)" if n_out else "")


# --------------------------------------------------------------------------------------------
# 4) Galvanic Bypass (ours): park when slack, share when constrained
# --------------------------------------------------------------------------------------------


@dataclass
class CapEvent:
    """SOC caps set at the start of a charge window (relaxed_Wh > 0 if the energy guard raised caps)."""

    t_s: float
    caps: FArr
    relaxed: bool
    relaxed_Wh: float
    deliverable_Wh: float
    need_Wh: float


@dataclass
class CatchupEvent:
    """Charging stopped by the inverter's minimum DC voltage while cells were below their target.
    lost_Wh = deliverable_energy_Wh(soc, min(cap, soc_max)) over those cells [Wh]."""

    t_s: float
    lost_Wh: float
    n_below: int


@dataclass
class ShareEvent:
    """One discharge decision: n_hot cells in the hot list, k of them bypassed; binding = the
    constraint that failed at k + 1 ("" when k = n_hot, i.e. parked)."""

    t_s: float
    n_hot: int
    k: int
    binding: str


class GalvanicBypass(_Base):
    """Galvanic Bypass: park hot cells when there is slack, share bypass slots when constrained.

    Every control step, from measured state and one-step plant physics:
      forced   T_i >= T_crit; SOC limit (plant rule); voltage limit; charging and
               (T_i >= T_charge_max - charge_T_margin_K or T_i < T_charge_min); charging and soc_i >= cap_i
      caps     cap_i = zone_soc_cap(T_air_i), T_air = T_room + T_offset; at the start of each charge,
               raise the caps of the coolest-floor capped cells to soc_max until
               deliverable_energy_Wh(soc_min, caps) >= (1 + plan_guard_margin) * plan_today_Wh
      trigger  T_trig_i = T_i + t_lead * max(0, slope_i), slope from a linear fit over T_hist  [K]
      hot list H (discharge): T_trig_i >= thr_i = max(T_band_edge, T_air_i + air_margin_K), or already
               optionally out with T_trig_i >= thr_i - hysteresis_K; never if T_i < T_opt.
               priority p_i = w_i (r_cal(T_i, soc_i) + r_cyc(T_i, |I|)), w_i = mean(SOH) / SOH_i [1/s]
      k        the largest k (searched from |H| down) such that bypassing forced + the first k of H
               (ordered for fair sharing, see _order) satisfies _check: power now, power at the end of
               the dispatch window, deliverable energy, healthy guard.
      park     k = |H|: all hot cells out, no time limit.
      share    k < |H|: a cell out for share_period_s yields its slot to the highest-priority
               in-string hot cell; min_dwell_s is respected in both directions.
      power    if forced cells alone are infeasible at P: P = plant.max_feasible_power, alert if a
               thermally forced cell is out.
    During charging every non-forced cell is in (bypass is a discharge action). When the inverter
    voltage floor stops charging with cells below target, NOTE_CATCHUP is logged with the kWh lost.
    """

    name = "galvanic_bypass"
    label = "Galvanic Bypass (ours)"

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        c, k = cfg["cell"], cfg["controllers"]
        self.t_lead_s = float(k["t_lead_s"])
        self.hyst_K = float(k["hysteresis_K"])
        self.min_dwell_s = float(k["min_dwell_s"])
        self.share_period_s = float(k["share_period_s"])
        self.band_K = float(c_to_k(k["T_band_edge_C"]))
        self.T_crit_K = float(c_to_k(k["T_crit_C"]))
        self.T_opt_K = float(c_to_k(k["T_opt_C"]))
        self.T_chg_min_K = float(c_to_k(k["T_charge_min_C"]))
        self.T_chg_max_K = float(c_to_k(c["T_charge_max_C"])) - float(k["charge_T_margin_K"])
        self.air_margin_K = float(k["air_margin_K"])
        self.guard_C = float(k["healthy_guard_C"])
        self.guard_K = float(c_to_k(self.guard_C))
        self.v_margin = float(k["v_margin_frac"])
        self.e_margin = float(k["energy_margin_frac"])
        self.soc_lookahead_s = float(k["soc_lookahead_s"])
        self.balance_min_frac = float(k["charge_balance_min_frac"])
        self.soc_min = float(c["soc_min"])
        self.soc_max = float(c["soc_max"])
        self.caps = np.zeros(0)
        self.thermal_forced = np.zeros(0, dtype=bool)
        self.hot = np.zeros(0, dtype=bool)
        self._charging = False
        self._catchup_logged = False
        self.cap_events: list[CapEvent] = []
        self.catchup_events: list[CatchupEvent] = []
        self.share_events: list[ShareEvent] = []

    def reset(self, n: int) -> None:
        super().reset(n)
        self.caps = np.full(n, self.soc_max)
        self.thermal_forced = np.zeros(n, dtype=bool)
        self.hot = np.zeros(n, dtype=bool)
        self._charging = False
        self._catchup_logged = False
        self.cap_events = []
        self.catchup_events = []
        self.share_events = []

    # ---- caps ------------------------------------------------------------------------------

    def _set_caps(self, state: PackState, params: CellParams, cfg: Cfg, plan_Wh: float) -> str:
        """cap_i = zone_soc_cap(T_air_i); raise coolest-floor capped cells to soc_max (one cell per
        step, ascending T_air, ties by index) until both hold, need = (1 + plan_guard_margin) * plan_Wh:
          energy: deliverable_energy_Wh(soc_min, caps) >= need  [Wh]
          series: starting all cells at their caps, the string can deliver `need` at discharge_P_W
                  and still meet the voltage floor at the end (_end_power_ok): capped cells empty
                  first and drop out, which aggregate energy alone does not see.
        Returns a note fragment."""
        T_air_K = state.T_room_K + params.T_offset_K
        caps = zone_soc_cap(k_to_c(T_air_K), cfg)
        caps = np.minimum(caps, self.soc_max)
        need = (1.0 + float(cfg["site"]["duty"]["plan_guard_margin"])) * float(plan_Wh)
        P_d = float(cfg["site"]["duty"]["discharge_P_W"])
        all_in = np.ones(caps.size, dtype=bool)

        def fits(cp: FArr) -> tuple[bool, float]:
            E_c = plant.deliverable_energy_Wh(self.soc_min, cp, params, cfg)
            ok = E_c >= need and (need <= 0.0 or self._end_power_ok(all_in, cp, P_d, state, params, cfg, need) == "")
            return ok, E_c

        ok, E0 = fits(caps)
        E = E0
        if not ok:
            for i in np.argsort(T_air_K, kind="stable"):
                if caps[i] >= self.soc_max:
                    continue
                caps[i] = self.soc_max
                ok, E = fits(caps)
                if ok:
                    break
        relaxed = E > E0
        self.caps = caps
        self.cap_events.append(CapEvent(state.t_s, caps.copy(), relaxed, E - E0, E, need))
        capped = caps < self.soc_max
        frag = ""
        if capped.any():
            frag = ", ".join(f"cap {v:.2f} x{int((caps == v).sum())}" for v in np.unique(caps[capped]))
        if relaxed:
            frag += f"; caps relaxed for heavy day (+{(E - E0) / WH_PER_KWH:.2f} kWh)"
        return frag

    # ---- constraints -------------------------------------------------------------------------

    def _project(
        self, act: BArr, soc: FArr, state: PackState, params: CellParams, cfg: Cfg, E_Wh: float
    ) -> tuple[FArr, BArr, bool]:
        """End state if the cells in `act` (starting at `soc`) deliver E_Wh [Wh] in series.

        Every active cell passes the same charge x [Ah] until it reaches soc_min, where it is
        forced out and the rest carry on (series string with per-cell bypass):
          a_i = act_i (soc_i - soc_min) Q_i SOH_i   [Ah available]
          sum_i min(x, a_i) U_oc(soc_mid_i) = E / (1 - loss_frac),  soc_mid = soc - min(x, a_i) / (2 Q_i SOH_i)
        solved for x by bisection (PROJECTION_ITERS outer updates of soc_mid).
        Returns (soc_end, act_end = act & soc_end > soc_min, enough = the cells hold E).
        """
        cap_Ah = params.Q_Ah * state.soh
        E = float(E_Wh) / (1.0 - float(cfg["site"]["duty"]["loss_frac"]))
        a = np.where(act, np.maximum(soc - self.soc_min, 0.0) * cap_Ah, 0.0)
        V = plant.ocv(soc, cfg)
        x = 0.0
        enough = True
        for _ in range(PROJECTION_ITERS):
            if float(np.sum(a * V)) < E:
                x, enough = float(a.max()) if a.size else 0.0, False
            else:
                lo, hi = 0.0, float(a.max())
                for _ in range(plant.BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    if float(np.sum(np.minimum(mid, a) * V)) < E:
                        lo = mid
                    else:
                        hi = mid
                x = hi
            V = plant.ocv(soc - 0.5 * np.minimum(x, a) / cap_Ah, cfg)
        soc_end = soc - np.minimum(x, a) / cap_Ah
        act_end = act & (soc_end > self.soc_min + 1e-9)
        return soc_end, act_end, enough

    def _end_power_ok(
        self, act: BArr, soc: FArr, P: float, state: PackState, params: CellParams, cfg: Cfg, E_Wh: float
    ) -> str:
        """"" if after delivering E_Wh (see _project) the surviving cells still carry P with
        V_str >= (1 + v_margin) V_floor; else the reason."""
        soc_end, act_end, enough = self._project(act, soc, state, params, cfg, E_Wh)
        if not enough:
            return "cells run empty before the plan is met"
        later = dataclasses.replace(state, soc=soc_end)
        _, V2, _, reason = plant.string_operating_point(P, act_end, later, params, cfg)
        if reason:
            return f"{reason} ({int(act_end.sum())} cells left)"
        V_need = (1.0 + self.v_margin) * v_floor_V(P, cfg)
        if V2 < V_need:
            return f"V {V2:.0f}/{V_need:.0f} V ({int(act_end.sum())} cells left)"
        return ""

    def _check(
        self,
        act: BArr,
        base: BArr,
        P: float,
        state: PackState,
        params: CellParams,
        cfg: Cfg,
        plan_rem_Wh: float,
        in_window: bool,
        n_now: int,
        T_trig: FArr,
        T_air: FArr,
    ) -> str:
        """"" if bypassing ~act is allowed, else the first failing (binding) constraint:
          power now:   string_operating_point(P, act) feasible and V_str >= (1 + v_margin) V_floor
          power later: the trial set runs for min_dwell_s (soc1 = soc - act I h / (3600 Q SOH)), then
                       every non-forced cell (`base`) is readmitted and must deliver the rest of
                       plan_remaining with V_str >= (1 + v_margin) V_floor at the end, cells that
                       reach soc_min dropping out (_end_power_ok; inside the window only)
          energy:      deliverable_energy_Wh(soc_min, soc, act) >= (1 + energy_margin) plan_remaining
          healthy:     for active j with T_air_j < healthy_guard (hot-listed or not; a floor above
                       the guard cannot be helped by switching),
                       T_air_j + (T_trig_j - T_air_j) (n_now / n_new)^2 <= healthy_guard
                       (self-heating scales with I^2 ~ (1 / n_active)^2 at fixed power; T_trig adds
                       the measured trend over t_lead_s, as for the hot list)
        """
        V_need = (1.0 + self.v_margin) * v_floor_V(P, cfg)
        I, V, _, reason = plant.string_operating_point(P, act, state, params, cfg)
        if reason:
            return f"power now: {reason}"
        if V < V_need:
            return f"power now: V {V:.0f}/{V_need:.0f} V"
        if in_window and plan_rem_Wh > 0.0:
            # The trial set is committed for min_dwell_s; after that every non-forced cell can be
            # readmitted (parked cells keep their charge), so the end of the window is projected
            # from soc after the commitment with all of them in.
            h = self.min_dwell_s
            soc1 = state.soc - np.where(act, I * h / (S_PER_H * params.Q_Ah * state.soh), 0.0)
            rest = plan_rem_Wh - P * h / S_PER_H
            if rest > 0.0:
                why = self._end_power_ok(base, soc1, P, state, params, cfg, rest)
                if why:
                    return f"power later: {why} at window end"
        if plan_rem_Wh > 0.0:
            E = plant.deliverable_energy_Wh(self.soc_min, state.soc, params, cfg, act)
            need = (1.0 + self.e_margin) * plan_rem_Wh
            if E < need:
                return f"energy {E / WH_PER_KWH:.1f}/{need / WH_PER_KWH:.1f} kWh"
        n_new = int(act.sum())
        # Every in-string cell whose floor is inside the guard is protected, hot-listed or not: near
        # the end of a heavy window healthy cells trend toward the band and join the hot list, and
        # must not then be pushed over it. Floors above the guard cannot be helped by switching.
        healthy = act & (T_air < self.guard_K)
        if healthy.any() and n_new > 0:
            T_pred = T_air[healthy] + (T_trig[healthy] - T_air[healthy]) * (n_now / n_new) ** 2
            T_max = float(T_pred.max())
            if T_max > self.guard_K:
                return f"healthy guard {k_to_c(T_max):.1f}/{self.guard_C:.1f} C"
        return ""

    def _balance_charge(
        self, base: BArr, prev_opt: BArr, P: float, state: PackState, params: CellParams, cfg: Cfg, plan_Wh: float
    ) -> tuple[BArr, str]:
        """Charge balancing: bypass the cells closest to their target so every cell finishes together.

        need_i = max(0, min(cap_i, soc_max) - soc_i) Q_i SOH_i [Ah]. At fixed charge power the string
        passes about the same total cell-Ah per hour whatever the in-string count, so bypassing cells
        that are ahead costs no throughput. Cells reaching target must leave, and the inverter needs
        n_req cells in (V_str >= (1 + v_margin) V_floor), so with everyone in the charge stops when
        m - n_req + 1 of the m cells are full:
          predicted loss = sum over the n_req - 1 unfinished cells of (need_j - s*) * U_oc(target) [Wh],
          s* = the need of the (m - n_req + 1)-th smallest.
        If it exceeds L = charge_balance_min_frac * plan_Wh, bypass only the cells that are ahead by
        more than the tolerance that would keep the loss under L:
          tol = L / ((n_req - 1) * mean U_oc(target))  [Ah];  bypass need_i < max_in_need - tol,
        smallest need first, at most k_max cells (V_str >= (1 + v_margin) V_floor, bisection on k).
        Bypassing only what is ahead keeps the string current (and heat) as low as possible.
        min_dwell_s is respected: cells out for less than min_dwell_s stay out first; cells in for
        less than min_dwell_s stay in. Returns (bypass mask, note fragment).
        """
        none = np.zeros(base.size, dtype=bool)
        target = np.minimum(self.caps, self.soc_max)
        cap_Ah = params.Q_Ah * state.soh
        need = np.where(base, np.maximum(target - state.soc, 0.0) * cap_Ah, 0.0)
        idx = np.flatnonzero(base)
        if idx.size == 0:
            return none, ""
        V_need = (1.0 + self.v_margin) * v_floor_V(P, cfg)

        def ok(sel: NDArray[np.int64]) -> bool:
            trial = base.copy()
            trial[sel] = False
            _, V, _, reason = plant.string_operating_point(P, trial, state, params, cfg)
            return reason == "" and V >= V_need

        by_need = idx[np.lexsort((idx, need[idx]))]
        if not ok(by_need[:0]):
            return none, ""
        lo, hi = 0, by_need.size
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if ok(by_need[:mid]) else (lo, mid - 1)
        n_req = by_need.size - lo
        s = need[by_need]
        if n_req <= 1:
            return none, ""
        s_star = s[s.size - n_req]
        V_t = plant.ocv(target[by_need], cfg)
        loss_Wh = float(np.sum((s[s.size - n_req + 1 :] - s_star) * V_t[s.size - n_req + 1 :]))
        L = self.balance_min_frac * float(plan_Wh)
        if loss_Wh <= L:
            return none, ""
        tol_Ah = L / ((n_req - 1) * float(np.mean(V_t)))

        dwell_ok = state.dwell_s >= self.min_dwell_s
        locked_out = prev_opt & ~dwell_ok
        locked_in = base & ~prev_opt & ~dwell_ok
        ahead = need < float(s.max()) - tol_Ah
        first = by_need[locked_out[by_need]]
        rest = by_need[~locked_out[by_need] & ~locked_in[by_need] & ahead[by_need]]
        order = np.concatenate([first, rest])
        lo, hi = 0, order.size
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if ok(order[:mid]) else (lo, mid - 1)
        mask = none.copy()
        mask[order[:lo]] = True
        return mask, f"charge balance: out {lo} (catch-up loss {loss_Wh / WH_PER_KWH:.2f} kWh predicted)"

    def _order(self, H: BArr, prev_opt: BArr, prio: FArr, state: PackState) -> tuple[NDArray[np.int64], int]:
        """Bypass order over the hot list for fair sharing; returns (order, n_locked_in).

        rank 0: out, dwell < min_dwell (cannot come back yet)
        rank 1: out for less than share_period_s (keeps its slot)
        rank 2: in-string, dwell >= min_dwell (may take a slot), by priority
        rank 3: out for share_period_s or more (yields its slot)
        Within a rank: descending priority, ties by index. In-string hot cells with dwell < min_dwell
        cannot leave yet and are excluded (n_locked_in of them).
        """
        dwell_ok = state.dwell_s >= self.min_dwell_s
        rank = np.where(
            prev_opt,
            np.where(~dwell_ok, 0, np.where(state.out_s < self.share_period_s, 1, 3)),
            np.where(dwell_ok, 2, -1),
        )
        idx = np.flatnonzero(H & (rank >= 0))
        order = idx[np.lexsort((idx, -prio[idx], rank[idx]))]
        return order.astype(np.int64), int((H & (rank < 0)).sum())

    # ---- main --------------------------------------------------------------------------------

    def decide(
        self, state: PackState, params: CellParams, layout: Layout, P_request_W: float, ctx: dict[str, Any]
    ) -> Decision:
        cfg = ctx["cfg"]
        n = state.soc.size
        self._ensure(n)
        P = float(P_request_W)
        T = state.T_K
        T_air = state.T_room_K + params.T_offset_K
        cur = np.asarray(state.active, dtype=bool).copy()
        prev_kind = np.where(cur, IN, np.where(self.kind == IN, OPTIONAL, self.kind)).astype(np.int8)
        notes: list[str] = []
        alert = ""

        dt_hist = float(ctx.get("dt_s", cfg["site"]["dt_s"]))
        hist = ctx.get("T_hist_K")
        slope = np.maximum(0.0, trend_slope_K_s(T[None, :] if hist is None else hist, dt_hist))
        T_trig = T + self.t_lead_s * slope
        thr = np.maximum(self.band_K, T_air + self.air_margin_K)

        charging, discharging = P < 0.0, P > 0.0
        if charging and not self._charging:
            self._catchup_logged = False
            frag = self._set_caps(state, params, cfg, float(ctx.get("plan_today_Wh", 0.0)))
            if frag:
                notes.append(frag)
        self._charging = charging

        # forced outs
        thermal = T >= self.T_crit_K
        if charging:
            thermal = thermal | (T >= self.T_chg_max_K) | (T < self.T_chg_min_K)
        forced = thermal.copy()
        if charging:
            forced |= state.soc >= self.caps
        forced |= soc_limit_mask(P, cur & ~forced, state, params, cfg, self.soc_lookahead_s)

        # idle: keep states except forced changes
        if P == 0.0:
            newly = forced & cur
            cur &= ~forced
            prev_kind[newly] = FORCED
            self.kind = prev_kind
            self.thermal_forced = thermal & ~cur
            note = f"idle; forced out {np.flatnonzero(newly).tolist()}" if newly.any() else ""
            return Decision(active=cur, P_W=0.0, note=note)

        base = ~forced
        v_lim = voltage_limit_mask(P, base, state, params, cfg)
        forced |= v_lim
        base &= ~v_lim
        active = base.copy()
        kind = np.where(base, IN, FORCED).astype(np.int8)
        self.hot = np.zeros(n, dtype=bool)

        if charging:
            prev_opt = (prev_kind == OPTIONAL) & base
            bal, frag = self._balance_charge(base, prev_opt, P, state, params, cfg, float(ctx.get("plan_today_Wh", 0.0)))
            active &= ~bal
            kind[bal] = OPTIONAL
            if frag:
                notes.append(frag)

        if discharging:
            prev_opt = (prev_kind == OPTIONAL) & base
            hold = prev_opt & (state.dwell_s < self.min_dwell_s)  # dwell: cannot come back yet
            H = ((T_trig >= thr) | (prev_opt & (T_trig >= thr - self.hyst_K))) & (T >= self.T_opt_K)
            H = (H | hold) & base
            self.hot = H
            n_hot = int(H.sum())
            if n_hot:
                Iabs = abs(float(plant.string_operating_point(P, base, state, params, cfg)[0]))
                if not np.isfinite(Iabs):
                    Iabs = abs(_approx_current_A(P, base, state, cfg))
                soh = state.soh
                rc, ry = plant.aging_rates(T, state.soc, Iabs, np.ones(n, dtype=bool), state.q_cal, params, cfg)
                prio = (float(np.mean(soh)) / soh) * (rc + ry)
                order, n_locked = self._order(H, prev_opt, prio, state)
                plan_rem = float(ctx.get("plan_remaining_Wh", 0.0))
                in_win = bool(ctx.get("in_dispatch_window", False))
                n_now = int(cur.sum())
                k_sel, binding = 0, ""
                for k in range(order.size, -1, -1):
                    trial = base.copy()
                    trial[order[:k]] = False
                    why = self._check(trial, base, P, state, params, cfg, plan_rem, in_win, n_now, T_trig, T_air)
                    if not why:
                        k_sel = k
                        break
                    binding = why
                else:
                    k_sel = 0
                if not binding and n_locked:
                    binding = "min dwell"
                active[order[:k_sel]] = False
                kind[order[:k_sel]] = OPTIONAL
                self.share_events.append(ShareEvent(state.t_s, n_hot, k_sel, binding if k_sel < n_hot else ""))
                if k_sel == n_hot:
                    notes.insert(0, f"out: {k_sel}/{n_hot} hot (parked, slack)")
                else:
                    notes.insert(0, f"out: {k_sel}/{n_hot} hot (sharing; binding: {binding})")

        # power wins: only forced outs can make the set infeasible here
        P_out = P
        I_op, V_op, _, reason = plant.string_operating_point(P, active, state, params, cfg)
        if reason:
            opt_idx = np.flatnonzero(kind == OPTIONAL)
            for i in opt_idx:  # defensive: the k search already kept the set feasible
                active[i] = True
                kind[i] = IN
                if plant.string_operating_point(P, active, state, params, cfg)[3] == "":
                    reason = ""
                    break
        if reason:
            P_out = plant.max_feasible_power(P, active, state, params, cfg)
            if (thermal & ~active).any():
                alert = ALERT_SERVICE
            notes.append(f"derated to {P_out / WH_PER_KWH:.1f} kW ({reason})")
            if charging and np.isfinite(V_op) and V_op < v_floor_V(P, cfg) and not self._catchup_logged:
                target = np.minimum(self.caps, self.soc_max)
                below = state.soc < target - 1e-6
                lost = plant.deliverable_energy_Wh(state.soc, target, params, cfg, below)
                self.catchup_events.append(CatchupEvent(state.t_s, lost, int(below.sum())))
                self._catchup_logged = True
                notes.append(f"{NOTE_CATCHUP} ({lost / WH_PER_KWH:.2f} kWh not recovered, {int(below.sum())} cells)")

        self.kind = kind
        self.thermal_forced = thermal & ~active
        n_forced = int((kind == FORCED).sum())
        if n_forced:
            notes.append(f"forced out {n_forced}")
        if P_out == P and discharging and notes:
            notes.append("power held")
        return Decision(active=active, P_W=P_out, note="; ".join(notes), alert=alert)


def all_controllers(cfg: Cfg) -> list[_Base]:
    """The four controllers with default settings (derate in its matched-protection variant)."""
    return [FixedStringDerate(cfg), FixedStringNoDerate(cfg), ThresholdBypass(cfg), GalvanicBypass(cfg)]
