"""Read REON per-cell telemetry CSV, map columns (config/column_map.yaml), convert to SI.

Pipeline:
    column_map = column_map_from_config(cfg)          # {canonical: source column}
    df = load_csv(path, column_map)                     # canonical long DataFrame
    arrays = to_arrays(df, cfg)                         # SI numpy arrays, (n_steps, n_cells)

A different REON export format only needs a change to config/column_map.yaml. Wide
exports (one column per cell, e.g. temp_C_017) are read with `load_csv(..., wide=True)`.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from gb.schema import (
    CANONICAL_COLUMNS,
    VALUE_COLUMNS,
    TelemetryWarning,
    ingest_settings,
    parse_timestamps,
    to_epoch_seconds,
    validate,
)
from gb.units import c_to_k

__all__ = ["TelemetryWarning", "column_map_from_config", "load_csv", "to_arrays", "wide_to_long"]

# Unit conversion (definitional).
OHM_PER_MOHM: float = 1e-3
# Reader chunk sizes.
DEFAULT_CHUNK_ROWS: int = 1_000_000
DEFAULT_BLOCK_BYTES: int = 64 << 20

WIDE_COLUMN_RE = re.compile(r"^(?P<base>.+)_(?P<cell>\d+)$")


def column_map_from_config(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Extract {canonical_name: source_column} from cfg["column_map"]["columns"].

    Keys that are not canonical column names (see gb.schema.CANONICAL_COLUMNS) are
    ignored silently, so placeholder entries do no harm. Missing section -> {} (identity).
    """
    cols = cfg.get("column_map", {}).get("columns", {}) or {}
    return {k: str(v) for k, v in cols.items() if k in CANONICAL_COLUMNS and v is not None}


def _rename_plan(
    header: list[str], column_map: Mapping[str, str] | None, wide: bool
) -> dict[str, str]:
    """Map each source column we keep to its canonical (long) or canonical-prefixed (wide) name.

    Canonical names absent from column_map map to themselves. Source columns that do not
    map to a canonical name are dropped.
    """
    inverse: dict[str, str] = {}
    for canon, src in (column_map or {}).items():
        if canon not in CANONICAL_COLUMNS:
            continue
        if src in inverse and inverse[src] != canon:
            raise ValueError(f"column_map maps both {inverse[src]!r} and {canon!r} to source {src!r}")
        inverse[src] = canon
    mapped_canon = set(inverse.values())

    def canon_of(name: str) -> str | None:
        if name in inverse:
            return inverse[name]
        # identity fallback, unless that canonical name was mapped to a different source
        if name in CANONICAL_COLUMNS and name not in mapped_canon:
            return name
        return None

    plan: dict[str, str] = {}
    for col in header:
        c = canon_of(col)
        if c is not None:
            plan[col] = c
            continue
        if wide:
            m = WIDE_COLUMN_RE.match(col)
            if m:
                base = canon_of(m["base"])
                if base in VALUE_COLUMNS:
                    plan[col] = f"{base}_{m['cell']}"
    targets = list(plan.values())
    dups = sorted({t for t in targets if targets.count(t) > 1})
    if dups:
        raise ValueError(f"several source columns map to the same canonical column(s): {dups}")
    return plan


def _read_pyarrow(path: Path, columns: list[str], ts_col: str | None, block_bytes: int) -> pd.DataFrame:
    """Stream the CSV with pyarrow in blocks of ~block_bytes, keeping only `columns`."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    types = {c: (pa.string() if c == ts_col else pa.float64()) for c in columns}
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_bytes),
        convert_options=pacsv.ConvertOptions(
            include_columns=columns, column_types=types, strings_can_be_null=True
        ),
    )
    frames = [batch.to_pandas() for batch in reader]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    return pd.concat(frames, ignore_index=True)


def _read_pandas(path: Path, columns: list[str], ts_col: str | None, chunk_rows: int) -> pd.DataFrame:
    """Read the CSV with pandas in chunks of chunk_rows rows, keeping only `columns`."""
    dtypes = {c: (str if c == ts_col else "float64") for c in columns}
    chunks = pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=chunk_rows)
    frames = list(chunks)
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    return pd.concat(frames, ignore_index=True)


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide DataFrame (canonical names) to canonical long format.

    Per-cell columns are named `<value column>_<cell id>` (e.g. temp_C_017, soc_3). Every
    other column (e.g. timestamp, current_A, room_temp_C) is shared by all cells of that
    row and is broadcast. Rows where all of a cell's per-cell values are NaN are dropped,
    so they show up as missing rows. Output is sorted by (timestamp row order, cell_id).
    """
    per_cell: dict[str, tuple[str, int]] = {}
    for col in df.columns:
        m = WIDE_COLUMN_RE.match(str(col))
        if m and m["base"] in VALUE_COLUMNS:
            per_cell[col] = (m["base"], int(m["cell"]))
    if not per_cell:
        raise ValueError("wide format: no per-cell columns like 'temp_C_017' found")
    shared = [c for c in df.columns if c not in per_cell]
    clash = sorted({b for b, _ in per_cell.values()} & set(shared))
    if clash:
        raise ValueError(f"wide format: columns {clash} are both shared and per-cell")

    src = df.reset_index(drop=True)
    wide = src[list(per_cell)].astype("float64")
    wide.columns = pd.MultiIndex.from_tuples(list(per_cell.values()), names=["field", "cell_id"])
    long = wide.stack(level="cell_id")  # index (row, cell_id), columns = fields
    long = long.dropna(how="all")
    long = long.reset_index(level="cell_id")
    long = long.join(src[shared], how="left")
    long = long.reset_index(drop=True)
    long["cell_id"] = long["cell_id"].astype("int64")
    order = [c for c in CANONICAL_COLUMNS if c in long.columns]
    return long[order + [c for c in long.columns if c not in order]]


def load_csv(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    *,
    wide: bool = False,
    engine: Literal["auto", "pyarrow", "pandas"] = "auto",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    cfg: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Read a telemetry CSV into a canonical long DataFrame.

    column_map: {canonical_name: source_column_name}; canonical names not in it are
        expected under their own name (identity). Non-canonical keys are ignored.
        In wide mode the mapping applies to the per-cell prefix (e.g. {"temp_C": "T"}
        reads T_017 as cell 17's temp_C).
    wide: the CSV has one column per cell per field (see `wide_to_long`).
    engine: "pyarrow" streams in blocks of block_bytes, "pandas" reads chunk_rows rows at
        a time; "auto" uses pyarrow if importable.

    cfg: full config, used for ingest.epoch_ms_threshold; None loads the default config.

    Only canonical columns are kept. timestamp is parsed to tz-aware UTC datetimes
    (ISO 8601, naive = UTC, epoch seconds, or epoch milliseconds detected by median with a
    TelemetryWarning); value columns are float64; cell_id is int64 when it has no missing
    values. No unit checks here: call gb.schema.validate.
    """
    path = Path(path)
    header = [str(c) for c in pd.read_csv(path, nrows=0).columns]
    plan = _rename_plan(header, column_map, wide)
    ts_src = next((s for s, c in plan.items() if c == "timestamp"), None)
    keep = list(plan)

    use_arrow = engine == "pyarrow"
    if engine == "auto":
        try:
            import pyarrow.csv  # noqa: F401

            use_arrow = True
        except ImportError:
            use_arrow = False
    if not keep:
        raw = pd.DataFrame()
    elif use_arrow:
        raw = _read_pyarrow(path, keep, ts_src, block_bytes)
    else:
        raw = _read_pandas(path, keep, ts_src, chunk_rows)
    df = raw.rename(columns=plan)
    if wide:
        df = wide_to_long(df)

    if "timestamp" in df.columns:
        df["timestamp"] = parse_timestamps(df["timestamp"], cfg)
    if "cell_id" in df.columns and not df["cell_id"].isna().any():
        cid = df["cell_id"].to_numpy(dtype=np.float64)
        if np.all(np.mod(cid, 1.0) == 0.0):
            df["cell_id"] = cid.astype(np.int64)
    order = [c for c in CANONICAL_COLUMNS if c in df.columns]
    return df[order].reset_index(drop=True)


def _fill_single_gaps(a: np.ndarray, name: str) -> np.ndarray:
    """Fill isolated NaNs in a (n_steps, n_cells) array; error on runs of 2+.

    A NaN at step t > 0 takes the value at t-1 (forward fill); a NaN at step 0 takes the
    value at step 1. Two or more consecutive NaNs in a cell (or a NaN when n_steps == 1)
    raise ValueError. Returns a new array.
    """
    m = np.isnan(a)
    if not m.any():
        return a
    if a.shape[0] < 2:
        bad = np.flatnonzero(m.any(axis=0))
        raise ValueError(f"{name}: missing samples with only one timestep, cells {bad[:10].tolist()}")
    run = m[1:] & m[:-1]
    if run.any():
        bad = np.flatnonzero(run.any(axis=0))
        raise ValueError(
            f"{name}: gaps of 2+ consecutive missing samples for {len(bad)} cells "
            f"(first: {bad[:10].tolist()}); only single missing samples are filled"
        )
    out = a.copy()
    rows, cols = np.nonzero(m)
    src_rows = np.where(rows > 0, rows - 1, 1)
    out[rows, cols] = a[src_rows, cols]
    return out


def to_arrays(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    current_mismatch: Literal["warn", "raise", "ignore"] = "warn",
    current_abs_tol_A: float | None = None,
    current_rel_tol: float | None = None,
) -> dict[str, Any]:
    """Pivot canonical long telemetry into SI numpy arrays.

    n_cells = cfg["pack"]["n_modules"] * cfg["pack"]["cells_per_module"]; cell_id must be
    in 0..n_cells-1. The DataFrame is validated first (gb.schema.validate); errors raise
    ValueError, warnings are emitted as TelemetryWarning. Pivot is timestamp x cell_id.
    Isolated missing samples are filled (see `_fill_single_gaps`); longer gaps raise.

    Returns (n_steps = number of unique timestamps, n = n_cells):
        t_s        (n_steps,) float64 seconds since the first timestamp
        t0_epoch_s float, epoch seconds (UTC) of the first timestamp
        I_A        (n_steps,) float64 string current [A], + = discharge: median over the
                   non-bypassed cells' readings (all cells if every cell is bypassed)
        V          (n_steps, n) float64 cell voltage [V]
        T_K        (n_steps, n) float64 cell temperature [K] (T_K = temp_C + 273.15)
        soc, soh   (n_steps, n) float64 [1], 0..1
        bypass     (n_steps, n) bool, True = bypassed   (only if bypass_state present)
        active     (n_steps, n) bool, = ~bypass          (only if bypass_state present)
        R_ohm      (n_steps, n) float64 [ohm]            (only if resistance_mOhm present)
        T_room_K   float, median room_temp_C over all rows if present, else
                   cfg["site"]["room_C"], converted to K

    Current consistency: at each timestamp, a non-bypassed reading differing from the
    median by more than max(current_abs_tol_A, current_rel_tol * |median|) triggers a
    TelemetryWarning ("warn", default), a ValueError ("raise"), or nothing ("ignore").
    The tolerances default to ingest.current_agree_abs_A and ingest.current_agree_rel.
    """
    ing = ingest_settings(cfg)
    if current_abs_tol_A is None:
        current_abs_tol_A = float(ing["current_agree_abs_A"])
    if current_rel_tol is None:
        current_rel_tol = float(ing["current_agree_rel"])
    n = int(cfg["pack"]["n_modules"]) * int(cfg["pack"]["cells_per_module"])
    rep = validate(df, n_cells=n, cfg=cfg)
    rep.raise_if_errors()
    for w in rep.warnings:
        warnings.warn(w, TelemetryWarning, stacklevel=2)
    d = rep.df
    if len(d) == 0:
        raise ValueError("telemetry is empty")

    optional = [c for c in ("bypass_state", "resistance_mOhm") if c in d.columns and d[c].notna().any()]
    cols = ["voltage_V", "current_A", "temp_C", "soc", "soh"] + optional
    piv = d.set_index(["timestamp", "cell_id"])[cols].unstack("cell_id").sort_index()
    piv = piv.reindex(columns=pd.MultiIndex.from_product([cols, range(n)]))

    t_epoch = to_epoch_seconds(piv.index)
    arr = {c: _fill_single_gaps(piv[c].to_numpy(dtype=np.float64), c) for c in cols}

    out: dict[str, Any] = {
        "t_s": t_epoch - t_epoch[0],
        "t0_epoch_s": float(t_epoch[0]),
        "V": arr["voltage_V"],
        "T_K": np.asarray(c_to_k(arr["temp_C"]), dtype=np.float64),
        "soc": arr["soc"],
        "soh": arr["soh"],
    }
    I_cells = arr["current_A"]
    if "bypass_state" in arr:
        bypass = arr["bypass_state"] > 0.5
        out["bypass"] = bypass
        out["active"] = ~bypass
        all_out = bypass.all(axis=1, keepdims=True)
        I_use = np.where(bypass & ~all_out, np.nan, I_cells)
    else:
        I_use = I_cells
    I_A = np.nanmedian(I_use, axis=1)
    out["I_A"] = I_A
    if current_mismatch != "ignore":
        tol = np.maximum(current_abs_tol_A, current_rel_tol * np.abs(I_A))
        dev = np.nanmax(np.abs(I_use - I_A[:, None]), axis=1)
        bad = np.flatnonzero(dev > tol)
        if bad.size:
            msg = (
                f"current_A disagrees across cells at {bad.size} timestamps "
                f"(max deviation {float(dev[bad].max()):.3g} A); using the median"
            )
            if current_mismatch == "raise":
                raise ValueError(msg)
            warnings.warn(msg, TelemetryWarning, stacklevel=2)
    if "resistance_mOhm" in arr:
        out["R_ohm"] = arr["resistance_mOhm"] * OHM_PER_MOHM

    if "room_temp_C" in d.columns and d["room_temp_C"].notna().any():
        out["T_room_K"] = float(c_to_k(float(np.nanmedian(d["room_temp_C"].to_numpy(dtype=np.float64)))))
    else:
        out["T_room_K"] = float(c_to_k(float(cfg["site"]["room_C"])))
    return out
