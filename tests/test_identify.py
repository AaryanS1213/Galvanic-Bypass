"""Tests for gb.identify on synthetic telemetry with known ground truth (gb.synth, 2 days, seed 1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from gb import identify as idf
from gb.config import load_config
from gb.ingest import column_map_from_config, load_csv, to_arrays
from gb.synth import generate, scenario_defects
from gb.types import CellParams, build_layout
from gb.units import c_to_k

SCENARIOS = ("zones", "zones_light")
DAYS, SEED = 2, 1


def _whole_module_zone_cells(cfg: Any, scenario: str, truth: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(mask of cells in whole-module zones, their zone offset [K])."""
    mod = truth["module_idx"]
    mask = np.zeros(mod.size, dtype=bool)
    off = np.zeros(mod.size)
    for d in scenario_defects(cfg, scenario):
        if d["kind"] == "zone" and d.get("slots") is None:
            m = np.isin(mod, d["modules"])
            mask |= m
            off[m] += float(d["offset_K"])
    return mask, off


def _zone_edge_cells(truth: dict[str, Any]) -> np.ndarray:
    """Cells whose in-module neighbour has a different true zone offset."""
    off = truth["params"].T_offset_K
    layout = build_layout(int(truth["module_idx"].max()) + 1, int(truth["slot_idx"].max()) + 1)
    a, b = layout.neighbors[:, 0], layout.neighbors[:, 1]
    diff = off[a] != off[b]
    edge = np.zeros(off.size, dtype=bool)
    edge[a[diff]] = True
    edge[b[diff]] = True
    return edge


@pytest.fixture(scope="module", params=SCENARIOS)
def run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Synthetic CSV on disk, its ingested arrays, the fits, and the truth."""
    cfg = load_config()
    scenario = request.param
    df, truth = generate(cfg, days=DAYS, seed=SEED, scenario=scenario)
    path = tmp_path_factory.mktemp(scenario) / f"synth_{scenario}.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    arrays = to_arrays(load_csv(path, column_map_from_config(cfg), cfg=cfg), cfg)
    layout = build_layout(int(cfg["pack"]["n_modules"]), int(cfg["pack"]["cells_per_module"]))
    R, rq = idf.fit_resistance(arrays, cfg)
    C, G, off, r2, info = idf.fit_thermal(arrays, R, layout, cfg)
    params = CellParams(
        Q_Ah=np.full(R.size, float(cfg["cell"]["Q_nom_Ah"])), R_ref_ohm=R, G_cool_WK=G, C_JK=C,
        T_offset_K=off, soh0=np.ones(R.size),
    )
    rise = idf.self_rise(params, cfg, arrays["T_room_K"], layout)
    return dict(cfg=cfg, scenario=scenario, path=path, arrays=arrays, truth=truth, R=R, rq=rq, C=C, G=G,
                off=off, r2=r2, info=info, rise=rise, params=params, layout=layout)


# ---- accuracy on synthetic data ---------------------------------------------------------------


def test_resistance_within_10pct(run: dict[str, Any]) -> None:
    err = np.abs(run["R"] / run["truth"]["params"].R_ref_ohm - 1.0)
    assert np.mean(err <= 0.10) >= 0.90, f"{run['scenario']}: median |err| {np.median(err):.3f}"
    assert run["rq"]["n_steps"].min() >= 4
    assert not run["rq"]["fallback"].any()


def test_g_cool_within_20pct(run: dict[str, Any]) -> None:
    err = np.abs(run["G"] / run["truth"]["params"].G_cool_WK - 1.0)
    assert np.mean(err <= 0.20) >= 0.90, f"{run['scenario']}: median |err| {np.median(err):.3f}"
    assert run["info"]["fit_ok"].mean() >= 0.90


def test_heat_capacity_plausible(run: dict[str, Any]) -> None:
    err = np.abs(run["C"] / run["truth"]["params"].C_JK - 1.0)
    assert np.median(err) <= 0.10


def test_offset_within_1K(run: dict[str, Any]) -> None:
    truth = run["truth"]
    err = np.abs(run["off"] - truth["params"].T_offset_K)
    edge = _zone_edge_cells(truth)
    assert np.mean(err <= 1.0) >= 0.90
    assert np.all(err[edge] <= 2.0)
    assert np.mean(err[~edge] <= 1.0) >= 0.90


def test_cause_accuracy(run: dict[str, Any]) -> None:
    cfg, truth = run["cfg"], run["truth"]
    cause = idf.classify_cause(run["off"], run["rise"], cfg)
    assert np.mean(cause == truth["cause"]) >= 0.90
    high_R = [d["cell_id"] for d in scenario_defects(cfg, run["scenario"]) if d["kind"] == "high_R"]
    assert high_R and all(cause[i] == "self_heat" for i in high_R)
    zmask, _ = _whole_module_zone_cells(cfg, run["scenario"], truth)
    assert zmask.any()
    assert np.all(np.isin(cause[zmask], ["zone", "both"]))


def test_self_rise_high_R_about_1p5x(run: dict[str, Any]) -> None:
    """Neighbour conduction caps a 2x-R cell at well under 2x the median rise."""
    i = 24
    rel = run["rise"][i] / np.median(run["rise"])
    assert float(run["cfg"]["identify"]["self_heat_rel"]) < rel < 2.0


# ---- end to end, risk report (needs gb.controllers.zone_soc_cap) ------------------------------


@pytest.fixture(scope="module")
def ident(run: dict[str, Any]) -> tuple[CellParams, pd.DataFrame]:
    return idf.identify(run["path"], run["cfg"])


def test_identify_matches_fits(run: dict[str, Any], ident: tuple[CellParams, pd.DataFrame]) -> None:
    params, rep = ident
    np.testing.assert_allclose(params.R_ref_ohm, run["R"])
    np.testing.assert_allclose(params.G_cool_WK, run["G"])
    np.testing.assert_allclose(params.T_offset_K, run["off"])
    assert params.Q_Ah.shape == params.soh0.shape == (run["R"].size,)
    assert len(rep) == run["R"].size


def test_report_columns_and_logic(run: dict[str, Any], ident: tuple[CellParams, pd.DataFrame]) -> None:
    cfg = run["cfg"]
    _, rep = ident
    for col in ("R_ref_ohm", "G_cool_WK", "C_JK", "T_offset_K", "T_air_C", "floor_above_band", "self_rise_K",
                "predicted_peak_C", "z_R", "z_inv_G", "cause", "flag", "soc_cap"):
        assert col in rep.columns
    room = float(cfg["site"]["room_C"])
    band = float(cfg["controllers"]["T_band_edge_C"])
    np.testing.assert_allclose(rep["T_air_C"], room + rep["T_offset_K"])
    np.testing.assert_allclose(rep["predicted_peak_C"], rep["T_air_C"] + rep["self_rise_K"])
    np.testing.assert_array_equal(rep["floor_above_band"], rep["T_air_C"] > band)
    np.testing.assert_array_equal(rep["flag"], (rep["predicted_peak_C"] >= band) | (rep["cause"] != "none"))
    assert set(rep["cause"]) <= set(idf.CAUSES)
    assert np.mean(rep["cause"].to_numpy() == run["truth"]["cause"]) >= 0.90
    # high-R cell stands out in resistance z-score
    assert rep["z_R"].iloc[24] > 3.0


def test_soc_cap_hot_modules(run: dict[str, Any], ident: tuple[CellParams, pd.DataFrame]) -> None:
    cfg = run["cfg"]
    _, rep = ident
    zmask, off = _whole_module_zone_cells(cfg, run["scenario"], run["truth"])
    hot8 = zmask & np.isclose(off, 8.0)
    assert hot8.any()
    np.testing.assert_allclose(rep["soc_cap"].to_numpy()[hot8], 0.70)
    cool = run["truth"]["params"].T_offset_K == 0.0
    assert np.all(rep["soc_cap"].to_numpy()[cool] > 0.70)


# ---- unit tests ---------------------------------------------------------------------------------


def test_r_factor_is_one_at_reference(cfg: Any) -> None:
    T_ref = float(c_to_k(cfg["cell"]["T_ref_C"]))
    f = idf.r_factor(np.array([T_ref, T_ref + 10.0]), np.array([0.5, 0.5]), np.ones(2), cfg)
    assert f[0] == pytest.approx(1.0)
    assert f[1] < 1.0  # resistance falls as temperature rises


def test_classify_cause_rules(cfg: Any) -> None:
    off = np.array([0.0, 0.0, 0.0, 8.0, 8.0, 3.0, 0.0])
    rise = np.array([1.0, 1.0, 1.5, 1.0, 1.6, 1.4, 1.2])
    # median over non-zone cells (offset <= 2): [1, 1, 1.5, 1.2] -> 1.1; 1.3x = 1.43, 1.5x = 1.65
    cause = idf.classify_cause(off, rise, cfg)
    assert list(cause) == ["none", "none", "self_heat", "zone", "zone", "zone", "none"]
    cause = idf.classify_cause(off, np.array([1.0, 1.0, 1.0, 1.0, 1.7, 1.0, 1.0]), cfg)
    assert cause[4] == "both"


def _flat_arrays(cfg: Any, n: int, N: int = 200, rise_K: float = 3.0) -> dict[str, Any]:
    T_room = float(c_to_k(cfg["site"]["room_C"]))
    dt = float(cfg["site"]["dt_s"])
    return {
        "t_s": np.arange(N) * dt,
        "I_A": np.zeros(N),
        "V": np.full((N, n), 2.9),
        "T_K": np.full((N, n), T_room + rise_K),
        "soc": np.full((N, n), 0.5),
        "soh": np.ones((N, n)),
        "T_room_K": T_room,
    }


def test_fallbacks_without_excitation(small_pack: Any) -> None:
    """No current steps and no thermal transients: R, C, G fall back to config and are flagged;
    the offset is still read from the rest temperature."""
    cfg, layout = small_pack.cfg, small_pack.layout
    arrays = _flat_arrays(cfg, layout.n)
    R, q = idf.fit_resistance(arrays, cfg)
    assert np.all(q["fallback"])
    np.testing.assert_allclose(R, cfg["cell"]["R_dc_ref_ohm"])
    C, G, off, _r2, info = idf.fit_thermal(arrays, R, layout, cfg)
    assert not info["fit_ok"].any()
    np.testing.assert_allclose(C, cfg["cell"]["mass_kg"] * cfg["cell"]["cp_J_kgK"])
    np.testing.assert_allclose(G, cfg["pack"]["G_cool_WK"])
    np.testing.assert_allclose(off, 3.0, atol=1e-9)


def test_resistance_step_exact(small_pack: Any) -> None:
    """Noise-free step at T_ref, SOC 0.5: R = -dV/dI exactly."""
    cfg, n = small_pack.cfg, small_pack.layout.n
    arrays = _flat_arrays(cfg, n, N=10, rise_K=0.0)
    arrays["T_K"][:] = float(c_to_k(cfg["cell"]["T_ref_C"]))
    R_true = np.linspace(0.5e-3, 1.0e-3, n)
    arrays["I_A"][5:] = 80.0
    arrays["V"] = arrays["V"] - arrays["I_A"][:, None] * R_true[None, :]
    R, q = idf.fit_resistance(arrays, cfg)
    np.testing.assert_allclose(R, R_true, rtol=1e-9)
    assert np.all(q["n_steps"] == 1)
    # a bypassed cell gives no valid step and falls back
    arrays["active"] = np.ones((10, n), dtype=bool)
    arrays["active"][:, 0] = False
    R, q = idf.fit_resistance(arrays, cfg)
    assert q["fallback"][0] and not q["fallback"][1:].any()
