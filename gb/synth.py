"""Synthetic REON-style telemetry from the plant model, with known ground truth.

Deterministic given a seed. All physics comes from gb.plant; every parameter from config/*.yaml
(scenario definitions, spread and sensor noise in config/scenarios.yaml).

    python -m gb.synth --days 7 --seed 0 --scenario zones --out data/synth_zones_7d.csv [--room 25]

writes the long-format CSV (gb.schema columns) plus <out stem>_truth.json, and prints a sanity
table per cause group.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gb import plant
from gb.config import load_config
from gb.types import CellParams, Decision, Layout, PackState, build_layout
from gb.units import c_to_k, k_to_c

Cfg = dict[str, Any]
S_PER_H: float = 3600.0
S_PER_DAY: float = 86400.0
WH_PER_KWH: float = 1000.0

CAUSES: tuple[str, ...] = ("zone", "self_heat", "both", "none")
DEFECT_KINDS: tuple[str, ...] = ("zone", "high_R", "poor_cooling")
# Window after the discharge window ends in which the peak temperature is still searched [s]
# (analysis window, not a physical parameter).
PEAK_TAIL_S: float = 3600.0


class SynthWarning(UserWarning):
    """Sanity check on the synthetic data failed (e.g. no zone cell crosses the band edge)."""


# --------------------------------------------------------------------------------------------
# 1) Parameters with known ground truth
# --------------------------------------------------------------------------------------------


def scenario_defects(cfg: Cfg, scenario: str) -> list[dict[str, Any]]:
    """Defect list of a scenario from cfg["scenarios"]["catalog"], each with a "name" key added."""
    catalog = cfg["scenarios"]["catalog"]
    if scenario not in catalog:
        raise ValueError(f"unknown scenario {scenario!r}; known: {sorted(catalog)}")
    defects = catalog[scenario].get("defects", {})
    return [{"name": name, **d} for name, d in defects.items()]


def build_params(
    cfg: Cfg, layout: Layout, seed: int | np.random.Generator, scenario: str = "zones"
) -> tuple[CellParams, np.ndarray]:
    """Per-cell parameters with manufacturing spread and the scenario's defects.

    Spread (z ~ N(0,1) per cell, drawn in the order Q, R, G):
      Q_Ah   = Q_nom_Ah     * (1 + Q_rel_std * z_Q)        [Ah]
      R_ref  = R_dc_ref_ohm * (1 + R_rel_std * z_R)        [ohm]
      G_cool = pack.G_cool  * (1 + G_cool_rel_std * z_G)   [W/K]
      C      = mass_kg * cp_J_kgK                          [J/K]; soh0 = 1; T_offset = 0 [K]
    Defects (applied after spread):
      zone:         T_offset += offset_K on cells in `modules` (and `slots`, if given)  [K]
      high_R:       R_ref  *= factor on cell_id
      poor_cooling: G_cool *= factor on cell_id
    Cause per cell: "zone" if T_offset > 0 only, "self_heat" if a high_R/poor_cooling defect only,
    "both" if both, "none" otherwise (manufacturing spread alone is "none").
    Raises ValueError for unknown kinds or modules/slots/cell_ids outside the layout.
    """
    sc = cfg["scenarios"]["spread"]
    c, p = cfg["cell"], cfg["pack"]
    n = layout.n
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    z_Q, z_R, z_G = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    Q = float(c["Q_nom_Ah"]) * (1.0 + float(sc["Q_rel_std"]) * z_Q)
    R = float(c["R_dc_ref_ohm"]) * (1.0 + float(sc["R_rel_std"]) * z_R)
    G = float(p["G_cool_WK"]) * (1.0 + float(sc["G_cool_rel_std"]) * z_G)
    if np.any(Q <= 0) or np.any(R <= 0) or np.any(G <= 0):
        raise ValueError("manufacturing spread produced a non-positive Q, R or G_cool; reduce the spread")
    offset = np.zeros(n)
    self_heat = np.zeros(n, dtype=bool)

    for d in scenario_defects(cfg, scenario):
        kind = d["kind"]
        if kind == "zone":
            mods = np.asarray(d["modules"], dtype=np.int64)
            if np.any((mods < 0) | (mods >= layout.n_modules)):
                raise ValueError(f"defect {d['name']}: modules {mods.tolist()} outside 0..{layout.n_modules - 1}")
            mask = np.isin(layout.module_idx, mods)
            if d.get("slots") is not None:
                slots = np.asarray(d["slots"], dtype=np.int64)
                if np.any((slots < 0) | (slots >= layout.cells_per_module)):
                    raise ValueError(
                        f"defect {d['name']}: slots {slots.tolist()} outside 0..{layout.cells_per_module - 1}"
                    )
                mask &= np.isin(layout.slot_idx, slots)
            offset[mask] += float(d["offset_K"])
        elif kind in ("high_R", "poor_cooling"):
            i = int(d["cell_id"])
            if not 0 <= i < n:
                raise ValueError(f"defect {d['name']}: cell_id {i} outside 0..{n - 1}")
            if kind == "high_R":
                R[i] *= float(d["factor"])
            else:
                G[i] *= float(d["factor"])
            self_heat[i] = True
        else:
            raise ValueError(f"defect {d['name']}: unknown kind {kind!r}; known: {DEFECT_KINDS}")

    zone = offset > 0.0
    cause = np.where(zone & self_heat, "both", np.where(zone, "zone", np.where(self_heat, "self_heat", "none")))
    params = CellParams(
        Q_Ah=Q,
        R_ref_ohm=R,
        G_cool_WK=G,
        C_JK=np.full(n, float(c["mass_kg"]) * float(c["cp_J_kgK"])),
        T_offset_K=offset,
        soh0=np.ones(n),
    )
    return params, cause


# --------------------------------------------------------------------------------------------
# 2) Duty cycle and outdoor temperature
# --------------------------------------------------------------------------------------------


def nominal_pack_energy_Wh(cfg: Cfg) -> float:
    """E_nom = n_modules * cells_per_module * Q_nom_Ah * V_nom  [Wh]."""
    c, p = cfg["cell"], cfg["pack"]
    return float(p["n_modules"]) * float(p["cells_per_module"]) * float(c["Q_nom_Ah"]) * float(c["V_nom"])


def bol_params(cfg: Cfg) -> CellParams:
    """Nominal beginning-of-life CellParams for the configured pack (no spread, no defects, SOH 1)."""
    c, p = cfg["cell"], cfg["pack"]
    n = int(p["n_modules"]) * int(p["cells_per_module"])
    one = np.ones(n)
    return CellParams(
        Q_Ah=one * float(c["Q_nom_Ah"]),
        R_ref_ohm=one * float(c["R_dc_ref_ohm"]),
        G_cool_WK=one * float(p["G_cool_WK"]),
        C_JK=one * float(c["mass_kg"]) * float(c["cp_J_kgK"]),
        T_offset_K=np.zeros(n),
        soh0=one.copy(),
    )


def full_day_plan_Wh(cfg: Cfg) -> float:
    """Full-depth day plan [Wh]:
    E_full = full_day_reserve_frac * deliverable_energy_Wh(soc_min, soc_max, BOL params, all active)."""
    c = cfg["cell"]
    base_Wh = plant.deliverable_energy_Wh(float(c["soc_min"]), float(c["soc_max"]), bol_params(cfg), cfg)
    return float(cfg["site"]["duty"]["full_day_reserve_frac"]) * base_Wh


def duty_profile(
    cfg: Cfg, day_index: int, rng: np.random.Generator
) -> tuple[Callable[[Any], np.ndarray], Callable[[Any], np.ndarray], float]:
    """One day's operator request. Returns (P_request(t), in_dispatch_window(t), plan_Wh).

    t is time of day [s] (scalar or array). Day energy fraction: frac = 1 with probability
    full_dod_day_frac, else frac ~ U(daily_energy_frac_range). E_full = full_day_plan_Wh(cfg)
    (deliverable BOL energy soc_min..soc_max times the reserve). The discharge lasts
    n = round(frac * E_full / P_d / dt) steps of site.dt_s (clipped to the window), so
    plan_Wh = n * dt * P_d / 3600 [Wh] exactly equals the energy requested on the dt grid
    (|plan_Wh - frac * E_full| <= P_d * dt / 7200).
      P_request = charge_P_W     for charge_start_h <= t/3600 < charge_end_h  (the plant stops
                                 charging when the first cell reaches soc_max: "until full")
                = discharge_P_W  for discharge_start <= t < discharge_start + n * dt
                                 (constant power; light days end early)
                = 0 otherwise                                                   [W]
      in_dispatch_window = discharge_start_h <= t/3600 < discharge_end_h
    day_index is not used by the physics; it is kept for callers' bookkeeping.
    """
    del day_index
    d = cfg["site"]["duty"]
    if rng.random() < float(d["full_dod_day_frac"]):
        frac = 1.0
    else:
        lo, hi = (float(x) for x in d["daily_energy_frac_range"])
        frac = float(rng.uniform(lo, hi))
    c0, c1 = float(d["charge_start_h"]) * S_PER_H, float(d["charge_end_h"]) * S_PER_H
    d0, d1 = float(d["discharge_start_h"]) * S_PER_H, float(d["discharge_end_h"]) * S_PER_H
    P_c, P_d = float(d["charge_P_W"]), float(d["discharge_P_W"])
    dt = float(cfg["site"]["dt_s"])
    if P_d > 0:
        # discharge length on the dt grid: round(frac * E_full / P / dt) steps, clipped to the window
        n_steps = min(int(round(frac * full_day_plan_Wh(cfg) / P_d * S_PER_H / dt)), int(np.floor((d1 - d0) / dt)))
        d_stop = d0 + n_steps * dt
        plan_Wh = n_steps * dt * P_d / S_PER_H
    else:
        d_stop, plan_Wh = d0, 0.0

    def P_request(t: Any) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return np.where((t >= c0) & (t < c1), P_c, np.where((t >= d0) & (t < d_stop), P_d, 0.0))

    def in_dispatch_window(t: Any) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return (t >= d0) & (t < d1)

    return P_request, in_dispatch_window, plan_Wh


def outdoor_temp_C(cfg: Cfg, t_of_day_s: Any) -> np.ndarray:
    """Outdoor air [degC] for HVAC cost accounting only (never affects the room):
    T_out = outdoor_mean_C + outdoor_amp_C * cos(2 pi (t - outdoor_peak_h * 3600) / 86400)."""
    h = cfg["site"]["hvac"]
    t = np.asarray(t_of_day_s, dtype=np.float64)
    phase = 2.0 * np.pi * (t - float(h["outdoor_peak_h"]) * S_PER_H) / S_PER_DAY
    return float(h["outdoor_mean_C"]) + float(h["outdoor_amp_C"]) * np.cos(phase)


# --------------------------------------------------------------------------------------------
# 3) Generate telemetry
# --------------------------------------------------------------------------------------------


def _initial_state(cfg: Cfg, params: CellParams, T_room_K: float) -> PackState:
    n = params.Q_Ah.size
    return PackState(
        t_s=0.0,
        soc=np.full(n, float(cfg["scenarios"]["initial_soc"])),
        T_K=T_room_K + params.T_offset_K,  # at rest in local air
        q_cal=1.0 - params.soh0,
        q_cyc=np.zeros(n),
        active=np.ones(n, dtype=bool),
        dwell_s=np.zeros(n),
        out_s=np.zeros(n),
        ah_throughput=np.zeros(n),
        T_room_K=T_room_K,
    )


def _cell_stats(T_K: np.ndarray, t_s: np.ndarray, days: int, cfg: Cfg, T_room_K: float) -> dict[str, np.ndarray]:
    """Per-cell analysis of true temperatures T_K (N, n) on grid t_s (N,).

    rest_above_room_K: mean over days of T - T_room one step before the discharge window [K]
    peak_rise_K:       max over days of max(T in [d0, d1 + 1 h)) - T(d0) [K]
    peak_T_C:          max T over the run [degC]
    hours_above_band:  time with T > T_band_edge_C [h]
    """
    d = cfg["site"]["duty"]
    dt = float(t_s[1] - t_s[0]) if t_s.size > 1 else float(cfg["site"]["dt_s"])
    d0, d1 = float(d["discharge_start_h"]) * S_PER_H, float(d["discharge_end_h"]) * S_PER_H
    tod = t_s - np.floor(t_s / S_PER_DAY) * S_PER_DAY
    day = (t_s // S_PER_DAY).astype(np.int64)
    rest, rise = [], []
    for k in range(days):
        pre = np.flatnonzero((day == k) & (tod >= d0 - dt) & (tod < d0))
        start = np.flatnonzero((day == k) & (tod >= d0))
        win = (day == k) & (tod >= d0) & (tod < d1 + PEAK_TAIL_S)
        if pre.size:
            rest.append(T_K[pre[0]] - T_room_K)
        if start.size and win.any():
            rise.append(T_K[win].max(axis=0) - T_K[start[0]])
    n = T_K.shape[1]
    band_K = float(c_to_k(cfg["controllers"]["T_band_edge_C"]))
    return {
        "rest_above_room_K": np.mean(rest, axis=0) if rest else np.full(n, np.nan),
        "peak_rise_K": np.max(rise, axis=0) if rise else np.full(n, np.nan),
        "peak_T_C": np.asarray(k_to_c(T_K.max(axis=0))),
        "hours_above_band": (T_K > band_K).sum(axis=0) * dt / S_PER_H,
    }


def generate(
    cfg: Cfg, days: int = 7, seed: int = 0, scenario: str = "zones", room_C: float | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate `days` of operation with every cell active and a constant room, then add sensor noise.

    Plant: gb.plant.step at site.dt_s with Decision(all active, P_request). The room is held at
    room_C (default site.room_C). Recorded at each t_k = k dt (state at the start of the step, with
    the current applied over the step):
      temp_C    = T_i - 273.15 + N(0, T_K)          voltage_V = U_oc(soc_i) - I R_i + N(0, V_V)
      current_A = I + N(0, I_A) (one draw per timestamp, string current sensor)
      soc, soh  = true values; bypass_state = 0; room_temp_C = room_C
    Rows = days * 86400 / dt * n, ordered by timestamp then cell_id.

    Seeds: SeedSequence(seed).spawn(3) -> independent streams for params, duty, noise.
    truth: params (CellParams), cause (n,), scenario, defects, seed, days, dt_s, room_C, and
    per-cell rest_above_room_K, peak_rise_K, peak_T_C, hours_above_band (true, noise-free).
    """
    p = cfg["pack"]
    layout = build_layout(int(p["n_modules"]), int(p["cells_per_module"]))
    n = layout.n
    dt = float(cfg["site"]["dt_s"])
    steps_per_day = S_PER_DAY / dt
    if days < 1 or abs(steps_per_day - round(steps_per_day)) > 1e-9:
        raise ValueError(f"need days >= 1 and 86400 / dt_s integer (dt_s = {dt})")
    N = int(days * round(steps_per_day))
    room = float(cfg["site"]["room_C"]) if room_C is None else float(room_C)
    T_room_K = float(c_to_k(room))

    rng_params, rng_duty, rng_noise = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    params, cause = build_params(cfg, layout, rng_params, scenario)

    t_s = np.arange(N) * dt
    tod = t_s - np.floor(t_s / S_PER_DAY) * S_PER_DAY
    P_req = np.empty(N)
    plan_Wh = []
    for k in range(days):
        P_fn, _, plan = duty_profile(cfg, k, rng_duty)
        sl = slice(int(k * steps_per_day), int((k + 1) * steps_per_day))
        P_req[sl] = P_fn(tod[sl])
        plan_Wh.append(plan)

    T = np.empty((N, n))
    V = np.empty((N, n))
    soc = np.empty((N, n))
    soh = np.empty((N, n))
    I = np.empty(N)
    all_on = np.ones(n, dtype=bool)
    state = _initial_state(cfg, params, T_room_K)
    for k in range(N):
        T[k], soc[k], soh[k] = state.T_K, state.soc, state.soh
        R_i = plant.R_cell(state, params, cfg)
        ocv_i = plant.ocv(state.soc, cfg)
        state, info = plant.step(state, params, layout, cfg, Decision(active=all_on, P_W=float(P_req[k])), dt)
        I[k] = info.I_A
        V[k] = ocv_i - info.I_A * R_i

    nz = cfg["scenarios"]["sensor_noise"]
    T_meas = np.asarray(k_to_c(T)) + rng_noise.normal(0.0, float(nz["T_K"]), (N, n))
    V_meas = V + rng_noise.normal(0.0, float(nz["V_V"]), (N, n))
    I_meas = I + rng_noise.normal(0.0, float(nz["I_A"]), N)

    t0 = pd.Timestamp(cfg["scenarios"]["start_timestamp"])
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    stamps = (t0 + pd.to_timedelta(t_s, unit="s")).strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(stamps, n),
            "cell_id": np.tile(np.arange(n, dtype=np.int64), N),
            "voltage_V": V_meas.ravel(),
            "current_A": np.repeat(I_meas, n),
            "temp_C": T_meas.ravel(),
            "soc": np.clip(soc, 0.0, 1.0).ravel(),
            "soh": soh.ravel(),
            "bypass_state": np.zeros(N * n, dtype=np.int64),
            "room_temp_C": np.full(N * n, room),
        }
    )
    truth: dict[str, Any] = {
        "scenario": scenario,
        "defects": scenario_defects(cfg, scenario),
        "seed": int(seed),
        "days": int(days),
        "dt_s": dt,
        "room_C": room,
        "plan_Wh": plan_Wh,
        "params": params,
        "cause": cause,
        "module_idx": layout.module_idx.copy(),
        "slot_idx": layout.slot_idx.copy(),
        **_cell_stats(T, t_s, days, cfg, T_room_K),
    }
    return df, truth


# --------------------------------------------------------------------------------------------
# 4/5) Truth export and sanity table
# --------------------------------------------------------------------------------------------


def truth_to_json(truth: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable copy of truth (arrays -> lists, CellParams -> dict of lists)."""

    def conv(v: Any) -> Any:
        if isinstance(v, CellParams):
            return {k: conv(x) for k, x in asdict(v).items()}
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v

    return conv(truth)


def sanity_table(truth: dict[str, Any], cfg: Cfg) -> pd.DataFrame:
    """Per cause group: n cells, mean zone offset [K], mean/min/max rest temperature above room [K],
    median/max peak rise during discharge [K], share of cells whose true T ever exceeds
    T_band_edge_C, mean hours above it, and floor_above_band = share of cells whose rest floor
    room_C + T_offset_K exceeds T_band_edge_C (switching cannot bring these below the band). Warns (SynthWarning) if zone cells exist and none crosses."""
    band = float(cfg["controllers"]["T_band_edge_C"])
    cell = pd.DataFrame(
        {
            "cause": truth["cause"],
            "offset_K": truth["params"].T_offset_K,
            "rest": truth["rest_above_room_K"],
            "rise": truth["peak_rise_K"],
            "crossed": truth["peak_T_C"] > band,
            "h_above": truth["hours_above_band"],
            "floor_above": float(truth["room_C"]) + truth["params"].T_offset_K > band,
        }
    )
    g = cell.groupby("cause", sort=False)
    tab = pd.DataFrame(
        {
            "n": g.size(),
            "offset_K": g["offset_K"].mean(),
            "rest_above_room_K": g["rest"].mean(),
            "rest_min_K": g["rest"].min(),
            "rest_max_K": g["rest"].max(),
            "peak_rise_med_K": g["rise"].median(),
            "peak_rise_max_K": g["rise"].max(),
            f"share_above_{band:g}C": g["crossed"].mean(),
            "h_above_mean": g["h_above"].mean(),
            "floor_above_band": g["floor_above"].mean(),
        }
    ).reindex([c for c in CAUSES if c in set(cell["cause"])])
    zone_cells = cell["cause"].isin(["zone", "both"])
    if zone_cells.any() and not cell.loc[zone_cells, "crossed"].any():
        warnings.warn(
            f"no zone cell crosses T_band_edge_C = {band:g} C; the zone scenario does not exercise the controller",
            SynthWarning,
            stacklevel=2,
        )
    return tab


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic REON per-cell telemetry with ground truth.")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="zones")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--room", type=float, default=None, help="room temperature [degC] (default site.room_C)")
    ap.add_argument("--config", type=Path, default=None, help="config directory (default config/)")
    a = ap.parse_args(argv)

    cfg = load_config() if a.config is None else load_config(a.config)
    df, truth = generate(cfg, days=a.days, seed=a.seed, scenario=a.scenario, room_C=a.room)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.6g")
    truth_path = a.out.with_name(a.out.stem + "_truth.json")
    truth_path.write_text(json.dumps(truth_to_json(truth), indent=1), encoding="utf-8")
    print(f"wrote {len(df):,} rows to {a.out} and truth to {truth_path}")
    print(f"scenario {a.scenario}, seed {a.seed}, {a.days} d, room {truth['room_C']:g} C")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SynthWarning)
        tab = sanity_table(truth, cfg)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(tab)
    for w in caught:
        print(f"WARNING: {w.message}")


if __name__ == "__main__":
    main()
