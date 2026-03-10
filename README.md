# Vitals Visualiser

A real-time vitals visualization tool built with Streamlit.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

## SpO2 calibration

The app computes:

- `R = (AC_red/DC_red) / (AC_ir/DC_ir)`

and then applies a calibration polynomial:

- `SpO2 = a*R^2 + b*R + c`

Sidebar presets include:

- `Sensor-Specific Quadratic (Recommended)` for ADPD slot mapping `slotA=IR`, `slotB=Red`

### ADPD sensor constants (our setup)

For our ADPD mapping (`slotA=IR`, `slotB=Red`), the app is locked to:

- `a = -0.0088722746`
- `b = 0.4259296021`
- `c = 94.845`
- Equation: `SpO2 = -0.0088722746*R^2 + 0.4259296021*R + 94.845`

## Development

```bash
# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Update dependencies
uv sync --upgrade
```
