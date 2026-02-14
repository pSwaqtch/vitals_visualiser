"""ADPD7000 Sensor Data Visualization Tool."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.loader import load_adpd_export, load_generic, detect_columns
from utils.plotting import plot_signals, plot_peaks, plot_poincare, plot_rr_tachogram, COLORS

EXPORT_DIR = Path(__file__).parent / "export_data"

_HRV_DEFAULTS = dict(bp_lo=0.7, bp_hi=3.5, bpm_min=40, bpm_max=180)

# ── Page config & CSS ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ADPD7000", page_icon="~", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .app-header {
        display: flex; align-items: baseline; gap: 0.75rem;
        border-bottom: 2px solid #636EFA; padding-bottom: 0.5rem; margin-bottom: 0.5rem;
    }
    .app-header h1 { font-size: 1.5rem; margin: 0; }
    .app-header .subtitle { color: #888; font-size: 0.85rem; }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2); border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    div[data-testid="stMetric"] label { font-size: 0.7rem; opacity: 0.6; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.3rem; }

    .channel-header { font-size: 0.75rem; color: #888; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-bottom: 0.25rem; }

    .stDownloadButton > button { width: 100%; }

    .file-badge {
        background: #636EFA; color: white; padding: 0.3rem 0.6rem;
        border-radius: 4px; font-size: 0.8rem; font-family: monospace;
        display: inline-block; margin: 0.3rem 0 0.5rem 0; word-break: break-all;
    }
    .meta-row { font-size: 0.78rem; color: #666; line-height: 1.7; }
    .section-label { font-size: 0.8rem; font-weight: 600; opacity: 0.5;
                     text-transform: uppercase; letter-spacing: 0.06em;
                     margin: 0.8rem 0 0.3rem 0; }
    .window-info { font-size: 0.78rem; opacity: 0.6; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>ADPD7000</h1>
    <span class="subtitle">Sensor Data Viewer</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Data source ────────────────────────────────────────────────────

data = None
filename = ""

with st.sidebar:
    source = st.radio("Source", ["Sample Data", "Upload"], horizontal=True,
                      label_visibility="collapsed")

    if source == "Sample Data":
        sample_files = sorted(EXPORT_DIR.glob("*.xlsx")) + sorted(EXPORT_DIR.glob("*.csv"))
        if not sample_files:
            st.warning("No files in export_data/")
        else:
            chosen = st.selectbox("File", sample_files, format_func=lambda p: p.name,
                                  label_visibility="collapsed")
            if chosen:
                filename = chosen.name
                data = load_adpd_export(chosen)
    else:
        uploaded = st.file_uploader("Upload", type=["csv", "xlsx"],
                                    label_visibility="collapsed")
        if uploaded:
            filename = uploaded.name
            if filename.endswith(".csv"):
                data = load_generic(uploaded, filename)
            else:
                try:
                    data = load_adpd_export(uploaded)
                    if data["df"].empty:
                        uploaded.seek(0)
                        data = load_generic(uploaded, filename)
                except Exception:
                    uploaded.seek(0)
                    data = load_generic(uploaded, filename)

# ── No data state ────────────────────────────────────────────────────────────

if data is None or data["df"].empty:
    st.markdown("""
    <div style="text-align:center; padding: 6rem 2rem; color: #aaa;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">&#x1f4c8;</div>
        <div style="font-size: 1.1rem;">Open the sidebar to select a sample file or upload your own</div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">Supports ADPD7000 .xlsx exports and generic CSV files</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data loaded — compute globals ────────────────────────────────────────────

df_full = data["df"]
detected = detect_columns(df_full)
all_cols = list(df_full.columns)
signal_candidates = [c for c in all_cols if c != detected["timestamp"]]

timestamp_col = detected["timestamp"]
sample_rate_full = None
duration_s_full = None

if timestamp_col and timestamp_col in df_full.columns:
    ts_full = df_full[timestamp_col]
    duration_ms_full = float(ts_full.iloc[-1] - ts_full.iloc[0])
    if duration_ms_full > 0:
        sample_rate_full = (len(ts_full) - 1) / (duration_ms_full / 1000)
        duration_s_full = duration_ms_full / 1000

# ── Sidebar: File info + controls ────────────────────────────────────────────

with st.sidebar:
    st.markdown(f'<div class="file-badge">{filename}</div>', unsafe_allow_html=True)

    parts = [f"{len(df_full):,} samples"]
    if duration_s_full:
        parts.append(f"{duration_s_full:.1f}s")
    if sample_rate_full:
        parts.append(f"~{sample_rate_full:.0f} Hz")
    st.caption(" / ".join(parts))

    meta = data["metadata"]
    if meta:
        lines = []
        if "board" in meta:
            lines.append(f"<b>Board</b> {meta['board'].strip()}")
        if "version" in meta:
            lines.append(f"<b>Version</b> {meta['version'].strip()}")
        if "export_start" in meta:
            lines.append(f"<b>Window</b> {meta['export_start']} &rarr; {meta.get('export_stop', '')}")
        if meta.get("ref_sbp") or meta.get("ref_dbp"):
            lines.append(f"<b>Ref BP</b> {meta.get('ref_sbp', '-')}/{meta.get('ref_dbp', '-')} mmHg")
        st.markdown('<div class="meta-row">' + "<br>".join(lines) + "</div>",
                    unsafe_allow_html=True)

    st.divider()

    # ── Channels ─────────────────────────────────────────────────────────
    st.markdown('<div class="channel-header">Channels</div>', unsafe_allow_html=True)

    active_signals = []
    for col in signal_candidates:
        default_on = col in detected["signals"]
        if st.checkbox(col, value=default_on, key=f"ch_{col}"):
            active_signals.append(col)

    st.divider()

    # ── HRV Parameters ───────────────────────────────────────────────────
    st.markdown('<div class="channel-header">HRV Parameters</div>', unsafe_allow_html=True)

    # Apply queued HRV reset before widgets render
    if "_pending_hrv_reset" in st.session_state:
        for k, v in st.session_state.pop("_pending_hrv_reset").items():
            st.session_state[k] = v

    if active_signals:
        hrv_col = st.selectbox("Channel for HRV", active_signals, key="hrv_ch")
    else:
        hrv_col = None

    hc1, hc2 = st.columns(2)
    with hc1:
        bp_low = st.number_input("BP Low (Hz)", value=_HRV_DEFAULTS["bp_lo"],
                                 step=0.1, format="%.1f", key="bp_lo")
    with hc2:
        bp_high = st.number_input("BP High (Hz)", value=_HRV_DEFAULTS["bp_hi"],
                                  step=0.1, format="%.1f", key="bp_hi")

    hc3, hc4 = st.columns(2)
    with hc3:
        bpm_min = st.number_input("BPM min", value=_HRV_DEFAULTS["bpm_min"],
                                  step=5, key="bpm_min")
    with hc4:
        bpm_max = st.number_input("BPM max", value=_HRV_DEFAULTS["bpm_max"],
                                  step=5, key="bpm_max")

    if st.button("Reset to defaults", use_container_width=True):
        st.session_state._pending_hrv_reset = dict(_HRV_DEFAULTS)
        st.rerun()

    # ── Timestamp override ───────────────────────────────────────────────
    st.divider()
    with st.expander("Advanced"):
        ts_options = [None] + all_cols
        ts_idx = ts_options.index(detected["timestamp"]) if detected["timestamp"] in ts_options else 0
        timestamp_col = st.selectbox("Timestamp column", ts_options, index=ts_idx,
                                     format_func=lambda x: "(use index)" if x is None else x)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

if not active_signals:
    st.info("Open the sidebar and enable at least one channel.")
    st.stop()

# ── Analysis Window slider — drives all charts + HRV ─────────────────────────

if timestamp_col:
    x_full = df_full[timestamp_col]
else:
    x_full = pd.Series(df_full.index, name="index")

x_min, x_max = float(x_full.iloc[0]), float(x_full.iloc[-1])


def _extract_box_x(event):
    """Extract x-range from a Plotly box selection event."""
    try:
        if event and event.selection and event.selection.box:
            box = event.selection.box[0]
            if box.get("x"):
                return float(box["x"][0]), float(box["x"][1])
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    return None


# Apply any queued window update from a previous box-selection (before slider renders)
if "_pending_window" in st.session_state:
    st.session_state.analysis_window = st.session_state.pop("_pending_window")

st.markdown('<div class="section-label">Analysis Window</div>', unsafe_allow_html=True)

wc1, wc2 = st.columns([11, 1])
with wc1:
    window = st.slider(
        "Analysis window", min_value=x_min, max_value=x_max,
        value=(x_min, x_max), key="analysis_window",
        label_visibility="collapsed",
    )
with wc2:
    if st.button("Reset", key="reset_window", use_container_width=True):
        st.session_state._pending_window = (x_min, x_max)
        st.rerun()

# Slice data to window
mask = (x_full >= window[0]) & (x_full <= window[1])
df = df_full.loc[mask].reset_index(drop=True)

# Compute windowed sample rate
sample_rate = None
if timestamp_col and timestamp_col in df.columns and len(df) > 1:
    ts_w = df[timestamp_col]
    dur_ms = float(ts_w.iloc[-1] - ts_w.iloc[0])
    if dur_ms > 0:
        sample_rate = (len(ts_w) - 1) / (dur_ms / 1000)

# Window info line
win_dur = (window[1] - window[0]) / 1000 if timestamp_col else mask.sum()
win_label = f"{win_dur:.1f}s" if timestamp_col else f"{mask.sum()} pts"
sr_label = f" / ~{sample_rate:.0f} Hz" if sample_rate else ""
st.markdown(
    f'<div class="window-info">{mask.sum():,} samples / {win_label}{sr_label}</div>',
    unsafe_allow_html=True,
)

# ── 1. Raw Signal ────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Raw Signal</div>', unsafe_allow_html=True)
st.caption("Box-select a region to zoom the analysis window")
fig_raw = plot_signals(df, timestamp_col, active_signals)
raw_event = st.plotly_chart(fig_raw, use_container_width=True, key="raw_chart",
                            on_select="rerun", selection_mode="box")

# If user box-selected on raw chart, queue window update for next rerun
raw_box = _extract_box_x(raw_event)
if raw_box:
    new_lo = max(x_min, min(raw_box))
    new_hi = min(x_max, max(raw_box))
    if new_hi - new_lo > 0 and (new_lo, new_hi) != tuple(window):
        st.session_state._pending_window = (new_lo, new_hi)
        st.rerun()

# ── 2. Export (behind checkbox) ──────────────────────────────────────────────

show_export = st.checkbox("Export this window as CSV", value=False, key="show_export")

if show_export:
    export_cols = ([timestamp_col] + active_signals) if timestamp_col else active_signals
    export_df = df[export_cols]
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    ec1, ec2, ec3 = st.columns([1, 1, 2])
    with ec1:
        st.metric("Rows", f"{len(export_df):,}")
    with ec2:
        st.metric("Duration", win_label)
    with ec3:
        st.download_button(
            "Download CSV", csv_data,
            file_name=f"window_{filename.rsplit('.', 1)[0]}.csv",
            mime="text/csv", use_container_width=True,
        )

# ── 3. HRV Analysis (computed on windowed data) ─────────────────────────────

st.divider()

if sample_rate is None:
    st.caption("HRV analysis unavailable — need a timestamp column and enough data.")
elif hrv_col is None:
    st.caption("HRV analysis unavailable — no channel selected.")
elif len(df) < 30:
    st.caption("HRV analysis unavailable — window too small (need at least ~0.5s of data).")
else:
    import heartpy as hp

    signal = df[hrv_col].values.astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = hp.filter_signal(signal, cutoff=[bp_low, bp_high],
                                        sample_rate=sample_rate, order=3,
                                        filtertype="bandpass")
            wd, m = hp.process(filtered, sample_rate=sample_rate,
                               bpmmin=bpm_min, bpmmax=bpm_max)

        # ── Metrics ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">HRV Metrics</div>', unsafe_allow_html=True)

        mc = st.columns(5)
        mc[0].metric("HR", f"{m['bpm']:.0f} bpm")
        mc[1].metric("IBI", f"{m['ibi']:.0f} ms")
        mc[2].metric("RMSSD", f"{m['rmssd']:.1f} ms")
        mc[3].metric("SDNN", f"{m['sdnn']:.1f} ms")
        mc[4].metric("Resp Rate", f"{m['breathingrate']:.2f} Hz")

        mc2 = st.columns(5)
        mc2[0].metric("pNN50", f"{m['pnn50']:.1%}")
        mc2[1].metric("pNN20", f"{m['pnn20']:.1%}")
        mc2[2].metric("SD1", f"{m.get('sd1', 0):.1f} ms")
        mc2[3].metric("SD2", f"{m.get('sd2', 0):.1f} ms")
        mc2[4].metric("SD1/SD2", f"{m['sd1/sd2']:.2f}")

        # ── Peak Detection (full width) ─────────────────────────────────
        st.markdown('<div class="section-label">Peak Detection</div>', unsafe_allow_html=True)
        st.caption("Box-select a region to zoom the analysis window")
        fig_peaks = plot_peaks(wd, m, sample_rate)
        peaks_event = st.plotly_chart(fig_peaks, use_container_width=True, key="peaks_chart",
                                      on_select="rerun", selection_mode="box")

        # Peak chart x-axis is in seconds relative to window start — convert back
        peaks_box = _extract_box_x(peaks_event)
        if peaks_box and timestamp_col:
            win_start = window[0]
            new_lo = max(x_min, win_start + min(peaks_box) * 1000)
            new_hi = min(x_max, win_start + max(peaks_box) * 1000)
            if new_hi - new_lo > 0 and (new_lo, new_hi) != tuple(window):
                st.session_state._pending_window = (new_lo, new_hi)
                st.rerun()

        # ── RR Tachogram + Poincare (side by side) ───────────────────────
        col_tacho, col_poincare = st.columns(2)

        with col_tacho:
            st.markdown('<div class="section-label">RR Tachogram</div>', unsafe_allow_html=True)
            fig_tacho = plot_rr_tachogram(wd, sample_rate)
            st.plotly_chart(fig_tacho, use_container_width=True, key="tacho_chart")

        with col_poincare:
            st.markdown('<div class="section-label">Poincare Plot</div>', unsafe_allow_html=True)
            fig_poincare = plot_poincare(wd, m)
            st.plotly_chart(fig_poincare, use_container_width=True, key="poincare_chart")

    except Exception as e:
        st.error(f"HRV analysis failed: {e}")
        st.caption("Try adjusting the bandpass filter or BPM range in the sidebar, "
                   "or widen the analysis window.")

# ── 4. Raw Data Table ────────────────────────────────────────────────────────

st.divider()
with st.expander("Raw Data Table (windowed)"):
    st.dataframe(df, use_container_width=True, height=500)
