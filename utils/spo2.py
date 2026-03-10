"""SpO2 estimation utilities using ratio-of-ratios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal


@dataclass(frozen=True)
class SpO2Calibration:
    """Polynomial calibration in sensor R-space: SpO2 = a*R^2 + b*R + c."""

    name: str
    a: float
    b: float
    c: float

    def equation(self) -> str:
        """Human-readable equation for UI captions."""
        return f"SpO2 = {self.a:+.6f}*R² {self.b:+.6f}*R {self.c:+.3f}"


# Sensor-specific defaults derived from slotA=IR, slotB=Red calibration run.
SENSOR_QUADRATIC = SpO2Calibration(
    name="Sensor-Specific Quadratic (Recommended)",
    a=-0.0088722746,
    b=0.4259296021,
    c=94.845,
)

# Explicit ADPD constants/metadata for our current hardware setup.
# Channel mapping used during calibration: slotA=IR, slotB=Red.
# R definition: (AC_red/DC_red) / (AC_ir/DC_ir)
ADPD_SENSOR_CALIBRATION = {
    "channel_mapping": "slotA=IR, slotB=Red",
    "reference_spo2_pct": 99.0,
    "reference_r_median": 34.3885207842,
    "quadratic": {
        "a": SENSOR_QUADRATIC.a,
        "b": SENSOR_QUADRATIC.b,
        "c": SENSOR_QUADRATIC.c,
        "equation": "SpO2 = -0.0088722746*R^2 + 0.4259296021*R + 94.845",
    },
}


def get_calibration_presets() -> dict[str, SpO2Calibration]:
    """Return ADPD-only calibration preset(s) for the app."""
    return {
        SENSOR_QUADRATIC.name: SENSOR_QUADRATIC,
    }


def extract_ac_dc(ppg_signal: np.ndarray, sample_rate: float) -> tuple[float, float]:
    """Extract AC/DC components from one PPG channel."""
    if len(ppg_signal) < 10:
        return 0.0, 0.0

    ppg = np.asarray(ppg_signal, dtype=float)
    dc_mean = float(np.mean(ppg))
    if sample_rate <= 1:
        return 0.0, dc_mean

    b, a = signal.butter(3, [0.5, 4.0], btype="band", fs=sample_rate)
    try:
        ac_component = signal.filtfilt(b, a, ppg)
    except Exception:
        ac_component = ppg - np.mean(ppg)

    ac_rms = float(np.sqrt(np.mean(ac_component**2)))
    return ac_rms, dc_mean


def calculate_r_value(
    red_signal: np.ndarray, ir_signal: np.ndarray, sample_rate: float
) -> float:
    """Calculate R = (AC_red / DC_red) / (AC_ir / DC_ir)."""
    ac_red, dc_red = extract_ac_dc(red_signal, sample_rate)
    ac_ir, dc_ir = extract_ac_dc(ir_signal, sample_rate)

    if dc_red <= 0 or dc_ir <= 0:
        return 0.0

    ratio_red = ac_red / dc_red
    ratio_ir = ac_ir / dc_ir
    if ratio_ir <= 0:
        return 0.0

    r_value = ratio_red / ratio_ir
    return float(r_value if np.isfinite(r_value) and r_value > 0 else 0.0)


def estimate_spo2(
    r_value: float,
    calibration: SpO2Calibration,
    *,
    clamp_min: float = 70.0,
    clamp_max: float = 100.0,
) -> float:
    """Estimate SpO2 with a polynomial calibration."""
    spo2 = (calibration.a * r_value * r_value) + (calibration.b * r_value) + calibration.c
    return float(np.clip(spo2, clamp_min, clamp_max))


def analyze_spo2(
    df: pd.DataFrame,
    red_col: str,
    ir_col: str,
    sample_rate: float,
    calibration: SpO2Calibration,
    timestamp_col: str | None = "timestamp",
    window_size: int = 256,
    step_size: int = 128,
) -> dict:
    """Analyze SpO2 over time using a sliding-window R calculation."""
    if red_col not in df.columns or ir_col not in df.columns:
        return {"timestamps": [], "spo2_values": [], "r_values": []}

    timestamps: list[float] = []
    spo2_values: list[float] = []
    r_values: list[float] = []

    n = len(df)
    i = 0
    while i + window_size <= n:
        window_df = df.iloc[i : i + window_size]
        red_signal = window_df[red_col].values
        ir_signal = window_df[ir_col].values

        r_value = calculate_r_value(red_signal, ir_signal, sample_rate)
        spo2 = estimate_spo2(r_value, calibration=calibration)

        if len(df) > i + window_size // 2:
            ts = (
                df[timestamp_col].iloc[i + window_size // 2]
                if timestamp_col and timestamp_col in df.columns
                else i + window_size // 2
            )
            timestamps.append(float(ts))

        r_values.append(r_value)
        spo2_values.append(spo2)
        i += step_size

    return {"timestamps": timestamps, "spo2_values": spo2_values, "r_values": r_values}
