"""Shared contract between gb modules. Do not change without flagging it (see CLAUDE.md).

Conventions:
- SI units everywhere (s, K, ohm, A, W, J); field names carry the unit suffix.
- P > 0 and I > 0 mean discharge.
- active[i] = True means cell i is in the string (series switch closed, bypass open).
- Per-cell arrays are numpy float64 / bool with shape (n,).
- Cell index i = module_idx[i] * cells_per_module + slot_idx[i].
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Layout:
    """Physical arrangement of n = n_modules * cells_per_module series cells."""

    n_modules: int
    cells_per_module: int
    module_idx: np.ndarray  # (n,) int
    slot_idx: np.ndarray  # (n,) int, position within module
    neighbors: np.ndarray  # (m, 2) int pairs i<j, adjacent slots k and k+1 in the same module

    @property
    def n(self) -> int:
        return self.n_modules * self.cells_per_module


def build_layout(n_modules: int, cells_per_module: int) -> Layout:
    """Build a Layout. Neighbor pairs: (i, i+1) for adjacent slots within a module,
    so m = n_modules * (cells_per_module - 1). Arrays are read-only."""
    if n_modules < 1 or cells_per_module < 1:
        raise ValueError("n_modules and cells_per_module must be >= 1")
    n = n_modules * cells_per_module
    idx = np.arange(n, dtype=np.int64)
    module_idx = idx // cells_per_module
    slot_idx = idx % cells_per_module
    left = idx[slot_idx < cells_per_module - 1]
    neighbors = np.stack([left, left + 1], axis=1).reshape(-1, 2)
    for a in (module_idx, slot_idx, neighbors):
        a.flags.writeable = False
    return Layout(n_modules, cells_per_module, module_idx, slot_idx, neighbors)


@dataclass
class CellParams:
    """Per-cell physical parameters, all shape (n,)."""

    Q_Ah: np.ndarray  # beginning-of-life capacity [Ah]
    R_ref_ohm: np.ndarray  # DC resistance at T_ref, SOC 0.5, SOH 1 [ohm]
    G_cool_WK: np.ndarray  # thermal conductance to local air [W/K]
    C_JK: np.ndarray  # heat capacity [J/K]
    T_offset_K: np.ndarray  # local air temperature above room (hot zones) [K], default 0
    soh0: np.ndarray  # initial state of health [1]


@dataclass
class PackState:
    """Per-cell dynamic state of one string. Arrays shape (n,).

    q_cal is initialised to 1 - soh0 so that soh starts at soh0.
    """

    t_s: float
    soc: np.ndarray  # 0..1
    T_K: np.ndarray  # cell temperature [K]
    q_cal: np.ndarray  # fractional capacity loss, calendar
    q_cyc: np.ndarray  # fractional capacity loss, cycling
    active: np.ndarray  # bool, True = in string
    dwell_s: np.ndarray  # seconds since this cell last switched
    out_s: np.ndarray  # seconds continuously bypassed (0 if active)
    ah_throughput: np.ndarray  # cumulative |I| dt [Ah]
    T_room_K: float  # constant within a run

    @property
    def soh(self) -> np.ndarray:
        """SOH = 1 - q_cal - q_cyc [1]."""
        return 1.0 - self.q_cal - self.q_cyc

    def copy(self) -> PackState:
        """Deep copy: every array is copied, scalars shared."""
        arrays = {
            f.name: getattr(self, f.name).copy()
            for f in fields(self)
            if isinstance(getattr(self, f.name), np.ndarray)
        }
        return replace(self, **arrays)


@dataclass
class StepInfo:
    """Diagnostics for one plant step."""

    I_A: float
    V_str_V: float
    P_req_W: float
    P_del_W: float
    shortfall_W: float
    n_active: int
    Q_cell_W: np.ndarray  # (n,) heat deposited in each cell node
    Q_pack_W: float  # total heat into the room (cells + switches)
    switch_loss_W: float
    infeasible_reason: str  # "" if fine


@dataclass
class Decision:
    """Controller output for one control step."""

    active: np.ndarray  # (n,) bool requested switch state
    P_W: float  # power request after any controller derate
    note: str = ""
    alert: str = ""


class Controller(Protocol):
    """A bypass controller.

    ctx carries: cfg, t_of_day_s, in_dispatch_window (bool), plan_remaining_Wh,
    hours_until_next_charge, T_hist_K (last trend_window_s of per-cell temperatures,
    shape (k, n)).
    """

    name: str

    def reset(self, n: int) -> None: ...

    def decide(
        self,
        state: PackState,
        params: CellParams,
        layout: Layout,
        P_request_W: float,
        ctx: dict[str, Any],
    ) -> Decision: ...
