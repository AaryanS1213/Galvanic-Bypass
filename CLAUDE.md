# Galvanic Bypass: project rules

You are helping build a Python tool for the REON challenge at MIT Energy & Climate Hack 2026. Deadline Sunday 9 AM ET. Prefer simple, correct, tested code. Work only on the task in the current message.

## PRODUCT
REON battery systems have a sensor and a bypass switch at every cell. Some cells run hotter than the rest because of where they sit (hot air zones) or because of higher resistance. Conventionally, an operator either cools the whole room further so the hottest cell stays in range (expensive HVAC), or derates the whole pack (lost power).
Our tool reads REON per-cell telemetry (CSV) and learns each cell's local air temperature, resistance and cooling. Every minute it uses the bypass switches to take a cell out early, when its temperature plus a short measured trend approaches a 30 C band edge, and shifts that cell's share onto cells with thermal headroom. It holds the requested pack power through the dispatch window, caps how full hot-zone cells get, gates charging on temperature, and rotates cells so none is parked.
Goal: no cell spends long hot, the room never needs extra cooling for a few hot cells, and pack life is longer.
The room temperature is constant within a run (HVAC holds it). We never control the HVAC. We only compute how much extra cooling the conventional approach would need.

## PHYSICS TO RESPECT
- A cell can never be held below its local air temperature. Bypass removes only that cell's own heat.
- At fixed power, bypassing k of n cells raises current and total pack heat by n/(n-k).
- Switching does nothing while the pack is idle.
- Near empty, sodium-ion voltage falls steeply, so few cells can be bypassed without breaking the inverter's minimum DC voltage. Power wins: readmit hot cells unless one is near its hard limit.

## PHYSICAL SYSTEM (defaults live in config/*.yaml; never hard-code numbers in gb/)
- Demo unit: one REON EBS-sized string, 128 sodium-ion cells in series (8 modules x 16), about 58.5 kWh, 30 kW, 2 h at 0.5C.
- Value is also reported at CESS container scale (2.5 MW / 5 MWh, also 0.5C) by multiplying by the number of strings. This is a flagged assumption.
- Cell: Hithium N162Ah, 162 Ah, 2.82 V nominal, 1.5-3.3 V, 4.6 kg, charge -10..60 C, discharge -40..60 C, >=20,000 cycles to 70% SOH at 25 C.
- Each cell has a series switch (cell in the string) and a bypass switch (cell out; string current flows through the bypass switch).

## CONVENTIONS
- SI units internally: s, K, ohm, A, W, J. Convert C<->K only at boundaries. All Arrhenius math in kelvin.
- Sign: P > 0 is discharge, I > 0 is discharge. active[i] = True means the cell is in the string.
- Per-cell arrays are numpy float64 or bool with shape (n,). Vectorize across cells; never loop over cells inside the time loop.
- Every tunable parameter lives in config YAML as {value: ..., source: "..."} where source is one of "datasheet:<name>", "paper:<short cite>", "derived:<how>", "assumed", "TBD-REON". The app prints an assumptions table from these.

## CODE RULES
- Python 3.11. Allowed deps: numpy, pandas, scipy, pyyaml, pydantic>=2, streamlit, plotly, pytest, pyarrow. Ask before adding anything else.
- Package name gb. Type hints everywhere. Each physics function's docstring states its equation and units.
- Pure functions for physics, no global state, deterministic given a seed.
- Every module ships with pytest tests in tests/. Run pytest and show the result before saying a task is done.
- Never invent physical parameters. If one is missing, add it to config with source "assumed" plus a TODO, and tell me.
- gb/types.py is a shared contract. Do not change it without telling me exactly what changed and why.
- Do not edit files outside the ones the current task names, except to fix an import.
