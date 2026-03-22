# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the app
uv run streamlit run app.py

# Run the mock WebSocket server (for testing live streaming)
uv run python mock_ws_server.py
uv run python mock_ws_server.py --file export_data/my_file.xlsx --rate 250 --loop

# Add / remove dependencies
uv add <package>
uv remove <package>
```

**Never use `pip` directly — always use `uv`.**

## Architecture

Single-page Streamlit app (`app.py`) with a sidebar-driven data source and a vertically-stacked analysis area. All chart building and signal processing lives in `utils/`.

### Data flow

1. **Source selection** (sidebar) produces a `data` dict: `{"df": pd.DataFrame, "metadata": dict}`.
   - `load_adpd_export()` — reads multi-sheet ppg_afe `.xlsx` exports (sheets: `Data`, `Board`, `Version`, `ExportTime`, `Reference`).
   - `load_generic()` — reads plain CSV or single-sheet Excel.
   - WebSocket mode builds the dict from a live `DataBuffer` snapshot.

2. **Column detection** (`detect_columns`) auto-identifies the timestamp column and signal channels by regex patterns on column names. Users can override via the Advanced expander.

3. **Analysis window slider** slices the full DataFrame to a user-defined time window. All downstream analysis (HRV, SpO2) operates on this slice.

4. **HRV pipeline** (inline in `app.py`): `heartpy.filter_signal` → `heartpy.process` → metrics + four charts (`plot_signals`, `plot_peaks`, `plot_rr_tachogram`, `plot_poincare`). Both the raw signal chart and the peak chart support Plotly box-select to zoom the analysis window.

5. **SpO2 pipeline** (`utils/spo2.py`): ratio-of-ratios method. `extract_ac_dc` → `calculate_r_value` → `estimate_spo2`. `analyze_spo2` runs a sliding window (default 256-sample window, 128-sample step) and returns time-series values for plotting. Default calibration curve: `SpO2 = 110 - 25 * R`.

### WebSocket streaming (`utils/streaming.py`)

Three classes work together, stored in `st.session_state["_ws_stream"]` via `get_or_create_stream()`:

- `DataBuffer` — thread-safe `deque`-backed ring buffer; `snapshot()` returns a DataFrame copy.
- `WebSocketClient` — runs an `asyncio` event loop in a daemon thread; batches rows (flush at 50 or on 1s timeout); calls a pluggable `decoder` (default: JSON object or array of objects).
- `StreamRecorder` — appends rows to a Parquet file via PyArrow `ParquetWriter`; also exports filtered CSVs. Recordings land in `recordings/`.

The live chart is a `@st.fragment(run_every=1.0)` that re-renders independently of the rest of the page.

### Mock server (`mock_ws_server.py`)

Replays an `.xlsx` or `.csv` file row-by-row over WebSocket at a configurable rate (default 100 Hz, default port 8765). Connects to the app at `ws://localhost:8765`.

### `utils/plotting.py`

All charts use `plotly.graph_objects` with `Scattergl` for performance. `_COMMON_LAYOUT` is shared across all figures. `plot_signals` is used for both the static raw chart and the live streaming fragment.

## Data Files

Sample ppg_afe exports go in `export_data/` (scanned automatically on startup). Recordings from WebSocket sessions are saved to `recordings/` as `.parquet`.

## ppg_afe xlsx Format

The expected multi-sheet structure:
| Sheet | Content |
|-------|---------|
| `Data` | Sensor samples (timestamp + slot columns) |
| `Board` | Board identifier string |
| `Version` | Firmware version string |
| `ExportTime` | `startExportTime` / `stopExportTime` columns |
| `Reference` | `ref_sbp` / `ref_dbp` columns (optional blood pressure reference) |
