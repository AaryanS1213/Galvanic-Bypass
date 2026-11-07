# Galvanic Bypass

Per-cell bypass thermal management for REON sodium-ion battery strings (MIT Energy & Climate Hack 2026, REON challenge).

The tool reads REON per-cell telemetry, learns each cell's local air temperature, resistance and cooling, and uses the per-cell bypass switches to keep every cell inside its thermal band while holding the requested pack power. It reports how much extra HVAC cooling, or how much derate, the conventional approach would need instead.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
streamlit run app.py
```

## Layout

- `config/`: every parameter as `{value, source}`; see the assumptions table in the app.
- `gb/`: the package (`types.py` is the shared contract).
- `tests/`: pytest suite.

Project rules for contributors and AI assistants are in `CLAUDE.md`.
