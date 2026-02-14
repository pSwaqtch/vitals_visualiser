"""Data loading utilities for ADPD7000 sensor exports."""

import re
from pathlib import Path

import pandas as pd


# Patterns for auto-detecting column roles
_TIMESTAMP_PATTERNS = [r"^timestamp$", r"^time$", r"^t$", r"^ts$", r"^elapsed"]
_SIGNAL_PATTERNS = [r"slot", r"channel", r"signal", r"ch\d", r"adc", r"ppg", r"ecg", r"eeg", r"bioz", r"imu"]


def load_adpd_export(file) -> dict:
    """Load an ADPD7000 xlsx export file.

    Returns dict with keys:
        df: DataFrame from the Data sheet
        metadata: dict with board, version, export_start, export_stop, ref_sbp, ref_dbp
    """
    sheets = pd.read_excel(file, sheet_name=None, engine="openpyxl")

    df = sheets.get("Data", pd.DataFrame())

    metadata = {}

    if "Board" in sheets and not sheets["Board"].empty:
        metadata["board"] = str(sheets["Board"].iloc[0, 0])

    if "Version" in sheets and not sheets["Version"].empty:
        metadata["version"] = str(sheets["Version"].iloc[0, 0])

    if "ExportTime" in sheets and not sheets["ExportTime"].empty:
        et = sheets["ExportTime"]
        if "startExportTime" in et.columns:
            metadata["export_start"] = str(et["startExportTime"].iloc[0])
        if "stopExportTime" in et.columns:
            metadata["export_stop"] = str(et["stopExportTime"].iloc[0])

    if "Reference" in sheets and not sheets["Reference"].empty:
        ref = sheets["Reference"]
        if "ref_sbp" in ref.columns:
            metadata["ref_sbp"] = ref["ref_sbp"].iloc[0]
        if "ref_dbp" in ref.columns:
            metadata["ref_dbp"] = ref["ref_dbp"].iloc[0]

    return {"df": df, "metadata": metadata}


def load_generic(file, filename: str = "") -> dict:
    """Load a generic CSV or Excel file.

    Returns dict with keys:
        df: DataFrame
        metadata: empty dict (no metadata for generic files)
    """
    name = filename or (file.name if hasattr(file, "name") else str(file))
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, engine="openpyxl")
    return {"df": df, "metadata": {}}


def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect timestamp and signal columns by name patterns.

    Returns dict:
        timestamp: str or None - name of the timestamp column
        signals: list[str] - names of detected signal columns
    """
    cols = list(df.columns)
    timestamp_col = None
    signal_cols = []

    for col in cols:
        col_lower = col.lower().strip()

        # Check timestamp patterns
        if timestamp_col is None:
            for pattern in _TIMESTAMP_PATTERNS:
                if re.search(pattern, col_lower):
                    timestamp_col = col
                    break

        # Check signal patterns
        for pattern in _SIGNAL_PATTERNS:
            if re.search(pattern, col_lower):
                signal_cols.append(col)
                break

    # If no signal columns detected, use all numeric columns except timestamp
    if not signal_cols:
        for col in cols:
            if col != timestamp_col and pd.api.types.is_numeric_dtype(df[col]):
                signal_cols.append(col)

    return {"timestamp": timestamp_col, "signals": signal_cols}
