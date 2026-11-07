"""Learn per-cell local air temperature, resistance and cooling conductance from telemetry.

This is how the tool finds hot zones from REON per-cell data without CFD:

    params, report = identify("telemetry.csv", cfg)

1. fit_resistance: DC + fast-polarisation resistance from current steps, normalised to T_ref, SOC 0.5.
2. fit_thermal:    per-cell heat capacity, cooling conductance and local-air offset by least squares
                   on the plant's own thermal network equation.
3. risk_report:    local air temperature (the cell's floor), predicted self-heating rise over one
                   rated discharge, and the cause of each hot cell (zone / self_heat / both / none).

All physics (R(T, SOC), heat, conduction, time step) comes from gb.plant. SI units internally.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from gb import plant
from gb.ingest import column_map_from_config, load_csv, to_arrays
from gb.types import CellParams, Decision, Layout, PackState, build_layout
from gb.units import c_to_k, k_to_c

Cfg = dict[str, Any]
FArr = NDArray[np.float64]

# Analysis thresholds live in config/identify.yaml (cfg["identify"]).
SOC_NORM: float = 0.5  # SOC at which R_ref is defined (CellParams contract: "R at T_ref, SOC 0.5")
MAD_TO_SIGMA: float = 1.4826  # normal-consistent MAD scale (statistical constant)

CAUSES: tuple[str, ...] = ("zone", "self_heat", "both", "none")


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _layout(cfg: Cfg) -> Layout:
    p = cfg["pack"]
    return build_layout(int(p["n_modules"]), int(p["cells_per_module"]))


def _state(T_K: FArr, soc: FArr, soh: FArr) -> PackState:
    """A PackState carrying only what plant.R_cell reads (T_K, soc, soh); arrays may be 2-D."""
    z = np.zeros_like(T_K)
    return PackState(
        t_s=0.0,
        soc=soc,
        T_K=T_K,
        q_cal=1.0 - soh,
        q_cyc=z,
        active=np.ones(T_K.shape, dtype=bool),
        dwell_s=z,
        out_s=z,
        ah_throughput=z,
        T_room_K=0.0,
    )


def _unit_params(shape: tuple[int, ...]) -> CellParams:
    """CellParams with R_ref = 1 ohm, used to evaluate plant.R_cell's dimensionless factors."""
    one = np.ones(shape)
    return CellParams(Q_Ah=one, R_ref_ohm=one, G_cool_WK=one, C_JK=one, T_offset_K=0.0 * one, soh0=one)


def r_factor(T_K: FArr, soc: FArr, soh: FArr, cfg: Cfg) -> FArr:
    """Dimensionless resistance factor of plant.R_cell, relative to (T_ref, SOC 0.5, SOH 1) [1].

    f = R_cell(T, soc, soh; R_ref = 1) / R_cell(T_ref, 0.5, 1; R_ref = 1)
      = exp[(Ea_R/Rg)(1/T - 1/T_ref)] * f_R(soc) / f_R(0.5) * growth(soh). T in K. Any shape.
    """
    T_K, soc, soh = (np.asarray(a, dtype=np.float64) for a in (T_K, soc, soh))
    f = plant.R_cell(_state(T_K, soc, soh), _unit_params(T_K.shape), cfg)
    T_ref = np.full(1, float(c_to_k(cfg["cell"]["T_ref_C"])))
    f0 = plant.R_cell(_state(T_ref, np.full(1, SOC_NORM), np.ones(1)), _unit_params((1,)), cfg)
    return f / float(f0[0])


def _active(arrays: dict[str, Any]) -> NDArray[np.bool_]:
    act = arrays.get("active")
    return np.ones(arrays["T_K"].shape, dtype=bool) if act is None else np.asarray(act, dtype=bool)


def _robust_z(x: FArr) -> FArr:
    """Robust z-score [1]: z = (x - median) / (1.4826 * MAD); 0 where MAD = 0."""
    med = np.median(x)
    mad = MAD_TO_SIGMA * np.median(np.abs(x - med))
    return (x - med) / mad if mad > 0 else np.zeros_like(x)


# --------------------------------------------------------------------------------------------
# 1) Resistance
# --------------------------------------------------------------------------------------------


def fit_resistance(arrays: dict[str, Any], cfg: Cfg) -> tuple[FArr, dict[str, Any]]:
    """Per-cell reference resistance R_ref [ohm], shape (n,), from current steps.

    Steps: consecutive samples k-1 -> k with |I_k - I_{k-1}| > 20 A, the cell in the string at both.
      R_step_i = -(dV_i - dU_oc_i) / dI                                        [ohm]
    dV = V_k - V_{k-1} [V]; dU_oc = U_oc(soc_k) - U_oc(soc_{k-1}) [V] (plant.ocv) removes the
    open-circuit drift over the step (about 10% of I*R at one-minute sampling). At 60 s sampling
    this is DC resistance plus fast polarisation, which is what heat depends on.
    Normalised: R_ref_i = median_steps( R_step_i / f ), f = mean over k-1, k of r_factor(T, soc, soh)
    (plant.R_cell's temperature, SOC and growth factors, relative to T_ref, SOC 0.5).
    Cells without a valid step get cell.R_dc_ref_ohm.

    quality: n_steps (n,) int, rel_mad (n,) = MAD / median of the normalised steps,
             fallback (n,) bool, step_idx (indices k of the steps used).
    """
    I = np.asarray(arrays["I_A"], dtype=np.float64)
    V, T, soc, soh = (np.asarray(arrays[k], dtype=np.float64) for k in ("V", "T_K", "soc", "soh"))
    n = V.shape[1]
    act = _active(arrays)
    dI = np.diff(I)
    k = np.flatnonzero(np.abs(dI) > float(cfg["identify"]["step_dI_min_A"])) + 1  # sample index after the step
    if k.size == 0:
        R = np.full(n, float(cfg["cell"]["R_dc_ref_ohm"]))
        q = {"n_steps": np.zeros(n, dtype=np.int64), "rel_mad": np.full(n, np.nan),
             "fallback": np.ones(n, dtype=bool), "step_idx": k}
        return R, q

    dV = V[k] - V[k - 1]
    dU = plant.ocv(soc[k], cfg) - plant.ocv(soc[k - 1], cfg)
    R_step = -(dV - dU) / dI[k - 1][:, None]
    f = 0.5 * (r_factor(T[k], soc[k], soh[k], cfg) + r_factor(T[k - 1], soc[k - 1], soh[k - 1], cfg))
    R_norm = R_step / f
    valid = act[k] & act[k - 1] & np.isfinite(R_norm)
    R_norm = np.where(valid, R_norm, np.nan)
    n_steps = valid.sum(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> NaN, handled below
        med = np.nanmedian(R_norm, axis=0)
        mad = np.nanmedian(np.abs(R_norm - med), axis=0)
    fallback = (n_steps == 0) | ~(med > 0)
    R = np.where(fallback, float(cfg["cell"]["R_dc_ref_ohm"]), med)
    quality = {"n_steps": n_steps, "rel_mad": mad / med, "fallback": fallback, "step_idx": k}
    return R, quality


# --------------------------------------------------------------------------------------------
# 2) Thermal parameters
# --------------------------------------------------------------------------------------------


def _conduction_out(T_K: FArr, layout: Layout, cfg: Cfg) -> FArr:
    """Net conduction out of each cell, sum_j G_ij (T_i - T_j) [W], for T_K of shape (N, n).

    Evaluated with plant.thermal_derivative (Q = 0, G_cool = 0, C = 1) on the transposed array.
    """
    Tt = T_K.T  # (n, N): plant indexes cells along axis 0
    ncell = Tt.shape[0]
    p = CellParams(
        Q_Ah=np.ones((ncell, 1)),
        R_ref_ohm=np.ones((ncell, 1)),
        G_cool_WK=np.zeros((ncell, 1)),
        C_JK=np.ones((ncell, 1)),
        T_offset_K=np.zeros((ncell, 1)),
        soh0=np.ones((ncell, 1)),
    )
    return -plant.thermal_derivative(Tt, np.zeros_like(Tt), p, layout, 0.0, cfg).T


def cell_heat(arrays: dict[str, Any], R_ref_ohm: FArr, cfg: Cfg) -> FArr:
    """Heat into each cell node [W], shape (N, n), from plant.heat at every sample.

    Q_k = plant.heat(I_k, active_k, R_cell(T_k, soc_k, soh_k; R_ref)) (cell I^2 R plus its
    share of switch loss). R_i,k = R_ref_i * r_factor(T, soc, soh).
    """
    I = np.asarray(arrays["I_A"], dtype=np.float64)
    R = R_ref_ohm[None, :] * r_factor(arrays["T_K"], arrays["soc"], arrays["soh"], cfg)
    act = _active(arrays)
    Q = np.empty_like(R)
    for t in range(I.size):  # time loop; vectorised across cells inside plant.heat
        Q[t] = plant.heat(float(I[t]), act[t], R[t], cfg)[0]
    return Q


def fit_thermal(
    arrays: dict[str, Any], R_ohm: FArr, layout: Layout, cfg: Cfg
) -> tuple[FArr, FArr, FArr, FArr, dict[str, Any]]:
    """Per-cell heat capacity C [J/K], cooling conductance G_cool [W/K], local-air offset [K], r2.

    Plant model: C dT_i/dt = Q_i - sum_j G_ij (T_i - T_j) - G_cool_i (T_i - T_room - offset_i).
    Rewritten per cell as a linear regression on y = dT/dt [K/s]:
        y = th1 * u - th2 * (T_i - T_room) + th3,   u = Q_i - sum_j G_ij (T_i - T_j)  [W]
        C = 1/th1,  G_cool = th2/th1,  offset = th3/th2.
    Q from cell_heat (plant.heat), G_ij = pack.G_ij_WK. T is smoothed with a Savitzky-Golay filter
    (window SG_WINDOW_S, order SG_POLYORDER) and y is its forward difference (T_{k+1} - T_k)/dt_k,
    matching the plant's explicit step; u is smoothed with the same filter so both sides of the
    equation see the same linear filter. Half a window at each end is dropped.
    Fits with C outside [0.3, 3] x mass*cp or G_cool <= 0 are rejected: C and G_cool fall back to
    mass*cp and pack.G_cool_WK, and offset is refit with those fixed
    (offset = mean(y - u/C + (G/C)(T - T_room)) * C / G).

    Returns (C_JK, G_cool_WK, T_offset_K, r2, info) with info = {"fit_ok": (n,) bool, "theta": (n, 3)}.
    """
    T = np.asarray(arrays["T_K"], dtype=np.float64)
    t = np.asarray(arrays["t_s"], dtype=np.float64)
    T_room = float(arrays["T_room_K"])
    N, n = T.shape
    idc = cfg["identify"]
    order = int(idc["sg_polyorder"])
    dt_med = float(np.median(np.diff(t)))
    win = int(round(float(idc["sg_window_s"]) / dt_med)) | 1  # odd
    win = max(win, order + 2 + (order % 2 == 0))
    if N < 2 * win:
        raise ValueError(f"need at least {2 * win} samples for the thermal fit, got {N}")

    Ts = savgol_filter(T, win, order, axis=0)
    u = cell_heat(arrays, R_ohm, cfg) - _conduction_out(T, layout, cfg)
    us = savgol_filter(u, win, order, axis=0)

    h = win // 2
    sl = slice(h, N - 1 - h)
    y = (np.diff(Ts, axis=0) / np.diff(t)[:, None])[sl]  # (M, n)
    x_u = us[:-1][sl]
    x_T = -(Ts[:-1][sl] - T_room)
    X = np.stack([x_u, x_T, np.ones_like(x_u)], axis=-1).transpose(1, 0, 2)  # (n, M, 3)
    Y = y.T  # (n, M)
    XtX = np.einsum("imk,iml->ikl", X, X)
    Xty = np.einsum("imk,im->ik", X, Y)
    theta = (np.linalg.pinv(XtX) @ Xty[..., None])[..., 0]  # (n, 3); pinv tolerates degenerate data
    resid = Y - np.einsum("imk,ik->im", X, theta)
    ss_res = np.sum(resid**2, axis=1)
    ss_tot = np.sum((Y - Y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    th1, th2, th3 = theta.T
    with np.errstate(divide="ignore", invalid="ignore"):
        C = 1.0 / th1
        G = th2 / th1
        off = th3 / th2
    C_cfg = float(cfg["cell"]["mass_kg"]) * float(cfg["cell"]["cp_J_kgK"])
    G_cfg = float(cfg["pack"]["G_cool_WK"])
    ok = (
        np.isfinite(C) & np.isfinite(G) & np.isfinite(off)
        & (C >= float(idc["C_fit_min_rel"]) * C_cfg) & (C <= float(idc["C_fit_max_rel"]) * C_cfg) & (G > 0)
    )
    # Fallback: C, G from config; offset from the mean residual.
    th3_fb = np.mean(Y - X[..., 0] / C_cfg - (G_cfg / C_cfg) * X[..., 1], axis=1)
    off_fb = th3_fb * C_cfg / G_cfg
    C = np.where(ok, C, C_cfg)
    G = np.where(ok, G, G_cfg)
    off = np.where(ok, off, off_fb)
    return C, G, off, r2, {"fit_ok": ok, "theta": theta}


# --------------------------------------------------------------------------------------------
# 3) Risk report
# --------------------------------------------------------------------------------------------


def self_rise(params: CellParams, cfg: Cfg, T_room_K: float, layout: Layout | None = None) -> FArr:
    """Predicted self-heating rise [K], shape (n,), over one standard discharge at rated power.

    Two plant.step runs from rest (T = T_room + offset, soc = soc_max, all cells active) over the
    dispatch window (site.duty.discharge_end_h - discharge_start_h) at dt = site.dt_s:
    one at P = pack.P_rated_W, one idle (P = 0). The network is linear in T for given Q, so
        self_rise_i = max_t (T_i,run(t) - T_i,idle(t))
    is the rise due to the cell's (and its neighbours') own heat, independent of the zone offset.
    dt is halved until the explicit step is stable.
    """
    layout = _layout(cfg) if layout is None else layout
    n = params.R_ref_ohm.size
    d = cfg["site"]["duty"]
    duration = (float(d["discharge_end_h"]) - float(d["discharge_start_h"])) * plant.S_PER_H
    dt = float(cfg["site"]["dt_s"])
    while plant._euler_stability_number(dt, params, layout, cfg) >= plant.EULER_STABILITY_LIMIT:
        dt /= 2.0
    steps = int(np.ceil(duration / dt))
    s0 = PackState(
        t_s=0.0,
        soc=np.full(n, float(cfg["cell"]["soc_max"])),
        T_K=T_room_K + params.T_offset_K,
        q_cal=1.0 - params.soh0,
        q_cyc=np.zeros(n),
        active=np.ones(n, dtype=bool),
        dwell_s=np.zeros(n),
        out_s=np.zeros(n),
        ah_throughput=np.zeros(n),
        T_room_K=T_room_K,
    )
    on = np.ones(n, dtype=bool)
    P = float(cfg["pack"]["P_rated_W"])
    run, idle = s0, s0.copy()
    rise = np.zeros(n)
    for _ in range(steps):
        run, _info = plant.step(run, params, layout, cfg, Decision(active=on, P_W=P), dt)
        idle, _info = plant.step(idle, params, layout, cfg, Decision(active=on, P_W=0.0), dt)
        rise = np.maximum(rise, run.T_K - idle.T_K)
    return rise


def classify_cause(offset_K: FArr, rise_K: FArr, cfg: Cfg) -> FArr:
    """Why a cell runs hot. med = median self_rise over non-zone cells (offset <= z; all if none).

    z = identify.zone_offset_min_K (2 K), b = identify.zone_self_rise_max_rel (1.5),
    s = identify.self_heat_rel (1.3):
    zone      : offset > z and self_rise <= b med
    both      : offset > z and self_rise >  b med
    self_heat : offset <= z and self_rise > s med
    none      : otherwise
    (1.3x because neighbour conduction caps a 2x-R cell at about 1.5x rise.)
    """
    idc = cfg["identify"]
    b, s = float(idc["zone_self_rise_max_rel"]), float(idc["self_heat_rel"])
    zone = offset_K > float(idc["zone_offset_min_K"])
    base = rise_K[~zone] if np.any(~zone) else rise_K
    med = float(np.median(base))
    out = np.full(offset_K.shape, "none", dtype=object)
    out[zone & (rise_K <= b * med)] = "zone"
    out[zone & (rise_K > b * med)] = "both"
    out[~zone & (rise_K > s * med)] = "self_heat"
    return out


def risk_report(
    params: CellParams, cfg: Cfg, T_room_C: float | None = None, layout: Layout | None = None
) -> pd.DataFrame:
    """Per-cell risk table (index cell_id).

    Columns: R_ref_ohm, G_cool_WK, C_JK, T_offset_K,
      T_air_C          = T_room_C + offset (the cell's floor: bypass cannot hold it below this) [degC]
      floor_above_band = T_air_C > controllers.T_band_edge_C
      self_rise_K      = self_rise() over one rated discharge [K]
      predicted_peak_C = T_air_C + self_rise_K [degC]
      z_R, z_inv_G     = robust z-scores of R_ref and 1/G_cool [1]
      cause            = classify_cause(offset, self_rise)
      flag             = predicted_peak_C >= T_band_edge_C or cause != "none"
      soc_cap          = controllers.zone_soc_cap(T_air_C, cfg) [1]
    T_room_C defaults to site.room_C.
    """
    from gb import controllers  # lazy: controllers is developed separately

    room = float(cfg["site"]["room_C"]) if T_room_C is None else float(T_room_C)
    band = float(cfg["controllers"]["T_band_edge_C"])
    T_air = room + params.T_offset_K
    rise = self_rise(params, cfg, float(c_to_k(room)), layout)
    peak = T_air + rise
    cause = classify_cause(params.T_offset_K, rise, cfg)
    cap = np.asarray(controllers.zone_soc_cap(T_air, cfg), dtype=np.float64)
    df = pd.DataFrame(
        {
            "R_ref_ohm": params.R_ref_ohm,
            "G_cool_WK": params.G_cool_WK,
            "C_JK": params.C_JK,
            "T_offset_K": params.T_offset_K,
            "T_air_C": T_air,
            "floor_above_band": T_air > band,
            "self_rise_K": rise,
            "predicted_peak_C": peak,
            "z_R": _robust_z(params.R_ref_ohm),
            "z_inv_G": _robust_z(1.0 / params.G_cool_WK),
            "cause": cause.astype(str),
            "flag": (peak >= band) | (cause != "none"),
            "soc_cap": np.broadcast_to(cap, T_air.shape),
        }
    )
    df.index.name = "cell_id"
    return df


# --------------------------------------------------------------------------------------------
# 4) End to end
# --------------------------------------------------------------------------------------------


def identify_arrays(arrays: dict[str, Any], cfg: Cfg) -> tuple[CellParams, pd.DataFrame]:
    """identify() on already-ingested arrays (gb.ingest.to_arrays output)."""
    layout = _layout(cfg)
    R, rq = fit_resistance(arrays, cfg)
    C, G, off, r2, info = fit_thermal(arrays, R, layout, cfg)
    n = R.size
    soh = np.asarray(arrays["soh"], dtype=np.float64)
    params = CellParams(
        Q_Ah=np.full(n, float(cfg["cell"]["Q_nom_Ah"])),  # capacity is not identified here
        R_ref_ohm=R,
        G_cool_WK=G,
        C_JK=C,
        T_offset_K=off,
        soh0=soh[-1].copy(),
    )
    rep = risk_report(params, cfg, T_room_C=float(k_to_c(arrays["T_room_K"])), layout=layout)
    rep["R_n_steps"] = rq["n_steps"]
    rep["R_fallback"] = rq["fallback"]
    rep["thermal_r2"] = r2
    rep["thermal_fit_ok"] = info["fit_ok"]
    return params, rep


def identify(csv_path: str | Path, cfg: Cfg) -> tuple[CellParams, pd.DataFrame]:
    """Read a REON telemetry CSV (column map from config) and return (CellParams, risk report)."""
    df = load_csv(csv_path, column_map_from_config(cfg), cfg=cfg)
    arrays = to_arrays(df, cfg)
    return identify_arrays(arrays, cfg)
