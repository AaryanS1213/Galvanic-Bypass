"""Time loop: duty cycle -> controller -> plant -> state, with step diagnostics.

simulate() runs one string for `days` with the room held constant at room_C (HVAC holds it; its
electric power is only accounted for). The operator request comes from gb.synth.duty_profile with
the same seed stream as gb.synth.generate, so every controller sees an identical day and load.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gb import plant
from gb.synth import duty_profile, outdoor_temp_C
from gb.types import CellParams, Controller, Decision, Layout, PackState
from gb.units import c_to_k, k_to_c

Cfg = dict[str, Any]
S_PER_H: float = 3600.0
S_PER_DAY: float = 86400.0


@dataclass
class SimResult:
    """Simulation record. Row k holds the state at the start of step k and the quantities applied
    over [t_k, t_k + dt). Per-cell arrays are (steps, n). Units in the names (W, A, V, C, s)."""

    controller: str
    label: str
    room_C: float
    dt_s: float
    t_s: np.ndarray
    P_req: np.ndarray  # operator request [W]
    P_del: np.ndarray  # delivered DC power [W]
    I: np.ndarray  # string current [A]
    V_str: np.ndarray  # string voltage [V]
    Q_pack: np.ndarray  # heat into the room [W]
    HVAC_W: np.ndarray  # HVAC electric power to hold the room [W] (accounting only)
    switch_loss_W: np.ndarray
    in_window: np.ndarray  # bool, inside the dispatch window
    T_C: np.ndarray
    soc: np.ndarray
    active: np.ndarray
    Q_cell: np.ndarray
    kind: np.ndarray  # int8: 0 in, 1 forced out, 2 optional out (controller.kind; 0/1 if absent)
    shortfall_W: np.ndarray  # (|P_req| - |P_decision|) + StepInfo.shortfall_W [W]
    plan_Wh: list[float]
    final_state: PackState
    notes: list[tuple[float, str]] = field(default_factory=list)
    alerts: list[tuple[float, str]] = field(default_factory=list)

    def window_shortfall_Wh(self, mask: np.ndarray | None = None) -> float:
        """Energy not delivered inside the dispatch window [Wh] = sum(shortfall * in_window) dt / 3600
        (optionally restricted further by a boolean step mask)."""
        m = self.in_window if mask is None else self.in_window & mask
        return float(np.sum(self.shortfall_W[m]) * self.dt_s / S_PER_H)

    def delivered_Wh(self) -> float:
        """Discharged energy [Wh] = sum(max(P_del, 0)) dt / 3600."""
        return float(np.sum(np.maximum(self.P_del, 0.0)) * self.dt_s / S_PER_H)


def initial_state(cfg: Cfg, params: CellParams, T_room_K: float) -> PackState:
    """Start state: soc = scenarios.initial_soc, every cell at rest in its local air
    T_i = T_room + T_offset_i [K], q_cal = 1 - soh0, all cells active, t = 0."""
    n = params.Q_Ah.size
    return PackState(
        t_s=0.0,
        soc=np.full(n, float(cfg["scenarios"]["initial_soc"])),
        T_K=T_room_K + params.T_offset_K,
        q_cal=1.0 - params.soh0,
        q_cyc=np.zeros(n),
        active=np.ones(n, dtype=bool),
        dwell_s=np.zeros(n),
        out_s=np.zeros(n),
        ah_throughput=np.zeros(n),
        T_room_K=T_room_K,
    )


def hours_until_next_charge(t_of_day_s: float, cfg: Cfg) -> float:
    """Hours from t_of_day_s to the next charge_start_h (today if still ahead, else tomorrow) [h]."""
    c0 = float(cfg["site"]["duty"]["charge_start_h"]) * S_PER_H
    dt = c0 - t_of_day_s
    if dt <= 0.0:
        dt += S_PER_DAY
    return dt / S_PER_H


def simulate(
    params: CellParams,
    layout: Layout,
    cfg: Cfg,
    controller: Controller,
    days: int,
    seed: int,
    room_C: float | None = None,
    init_state: PackState | None = None,
) -> SimResult:
    """Run `controller` on one string for `days` days at a constant room temperature.

    Loop at dt = site.dt_s; controller.decide every controllers.control_dt_s (held in between) with
    ctx = {cfg, t_of_day_s, in_dispatch_window, plan_today_Wh, plan_remaining_Wh,
    hours_until_next_charge, T_hist_K (last trend_window_s of cell T, (k, n)), dt_s}.
    plan_remaining_Wh = max(0, plan_today - delivered discharge energy today) [Wh].
    Duty: synth.duty_profile per day on SeedSequence(seed).spawn(3)[1] (same stream as synth).
    HVAC_W = plant.hvac_power(Q_pack, room_C, T_out(t), cfg) each step. Shortfall from
    StepInfo.shortfall_W only.
    """
    dt = float(cfg["site"]["dt_s"])
    ctrl_dt = float(cfg["controllers"]["control_dt_s"])
    steps_per_day = S_PER_DAY / dt
    every = ctrl_dt / dt
    if days < 1 or abs(steps_per_day - round(steps_per_day)) > 1e-9:
        raise ValueError(f"need days >= 1 and 86400 / dt_s integer (dt_s = {dt})")
    if every < 1.0 or abs(every - round(every)) > 1e-9:
        raise ValueError(f"control_dt_s ({ctrl_dt}) must be a positive integer multiple of dt_s ({dt})")
    spd, every = int(round(steps_per_day)), int(round(every))
    N = days * spd
    room = float(cfg["site"]["room_C"]) if room_C is None else float(room_C)
    T_room_K = float(c_to_k(room))
    n = layout.n

    rng_duty = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1])
    t_s = np.arange(N) * dt
    tod = t_s - np.floor(t_s / S_PER_DAY) * S_PER_DAY
    P_req = np.empty(N)
    in_win = np.empty(N, dtype=bool)
    plans: list[float] = []
    for d in range(days):
        P_fn, win_fn, plan = duty_profile(cfg, d, rng_duty)
        sl = slice(d * spd, (d + 1) * spd)
        P_req[sl] = P_fn(tod[sl])
        in_win[sl] = win_fn(tod[sl])
        plans.append(float(plan))  # duty_profile aligns the discharge to the dt grid: exact
    T_out = np.asarray(outdoor_temp_C(cfg, tod), dtype=np.float64)

    state = init_state.copy() if init_state is not None else initial_state(cfg, params, T_room_K)
    state.T_room_K = T_room_K
    controller.reset(n)
    k_hist = int(round(float(cfg["controllers"]["trend_window_s"]) / dt)) + 1
    hist: deque[np.ndarray] = deque([state.T_K.copy()] * k_hist, maxlen=k_hist)

    out = {
        "P_del": np.empty(N), "I": np.empty(N), "V_str": np.empty(N), "Q_pack": np.empty(N),
        "HVAC_W": np.empty(N), "sw": np.empty(N), "short": np.empty(N),
    }
    T_C = np.empty((N, n))
    soc = np.empty((N, n))
    act = np.empty((N, n), dtype=bool)
    Q_cell = np.empty((N, n))
    kind = np.empty((N, n), dtype=np.int8)
    notes: list[tuple[float, str]] = []
    alerts: list[tuple[float, str]] = []

    decision = Decision(active=np.ones(n, dtype=bool), P_W=0.0)
    delivered_today = 0.0
    for k in range(N):
        day = k // spd
        if k % spd == 0:
            delivered_today = 0.0
        if k % every == 0:
            ctx = {
                "cfg": cfg,
                "t_of_day_s": float(tod[k]),
                "in_dispatch_window": bool(in_win[k]),
                "plan_today_Wh": plans[day],
                "plan_remaining_Wh": max(0.0, plans[day] - delivered_today),
                "hours_until_next_charge": hours_until_next_charge(float(tod[k]), cfg),
                "T_hist_K": np.asarray(hist),
                "dt_s": dt,
            }
            decision = controller.decide(state, params, layout, float(P_req[k]), ctx)
            if decision.note:
                notes.append((float(t_s[k]), decision.note))
            if decision.alert:
                alerts.append((float(t_s[k]), decision.alert))
        T_C[k] = k_to_c(state.T_K)
        soc[k] = state.soc
        kd = getattr(controller, "kind", None)
        a = np.asarray(decision.active, dtype=bool)
        kind[k] = kd if kd is not None and np.size(kd) == n else np.where(a, 0, 1)
        act[k] = a
        state, info = plant.step(state, params, layout, cfg, decision, dt)
        hist.append(state.T_K.copy())

        out["P_del"][k] = info.P_del_W
        out["I"][k] = info.I_A
        out["V_str"][k] = info.V_str_V
        out["Q_pack"][k] = info.Q_pack_W
        out["sw"][k] = info.switch_loss_W
        # controller derate (request vs decision) plus the plant's own shortfall from StepInfo
        out["short"][k] = (abs(float(P_req[k])) - abs(float(decision.P_W))) + info.shortfall_W
        out["HVAC_W"][k] = plant.hvac_power(info.Q_pack_W, room, float(T_out[k]), cfg)
        Q_cell[k] = info.Q_cell_W
        if info.P_del_W > 0.0:
            delivered_today += info.P_del_W * dt / S_PER_H

    res = SimResult(
        controller=getattr(controller, "name", type(controller).__name__),
        label=getattr(controller, "label", type(controller).__name__),
        room_C=room,
        dt_s=dt,
        t_s=t_s,
        P_req=P_req,
        P_del=out["P_del"],
        I=out["I"],
        V_str=out["V_str"],
        Q_pack=out["Q_pack"],
        HVAC_W=out["HVAC_W"],
        switch_loss_W=out["sw"],
        in_window=in_win,
        T_C=T_C,
        soc=soc,
        active=act,
        Q_cell=Q_cell,
        kind=kind,
        shortfall_W=out["short"],
        plan_Wh=plans,
        final_state=state,
        notes=notes,
        alerts=alerts,
    )
    return res

