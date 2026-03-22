"""SpO2 estimation using ratio-of-ratios method."""

import numpy as np
import pandas as pd
from scipy import signal


def extract_ac_dc(ppg_signal: np.ndarray, sample_rate: float) -> tuple[float, float]:
    """Extract AC and DC components from PPG signal.

    AC: pulsatile component (heartbeat-synchronized, ~0.5-4 Hz)
    DC: quasi-static component (tissue, venous blood)

    Returns:
        tuple: (AC_rms, DC_mean)
    """
    if len(ppg_signal) < 10:
        return 0.0, 0.0

    ppg = np.asarray(ppg_signal)

    b, a = signal.butter(3, [0.5, 4.0], btype="band", fs=sample_rate)
    try:
        ac_component = signal.filtfilt(b, a, ppg)
    except Exception:
        ac_component = ppg - np.mean(ppg)

    ac_rms = np.sqrt(np.mean(ac_component**2))
    dc_mean = np.mean(ppg)

    return float(ac_rms), float(dc_mean)


def calculate_r_value(
    red_signal: np.ndarray, ir_signal: np.ndarray, sample_rate: float
) -> float:
    """Calculate ratio of ratios (R) for SpO2 estimation.

    R = (AC_red / DC_red) / (AC_ir / DC_ir)
    """
    ac_red, dc_red = extract_ac_dc(red_signal, sample_rate)
    ac_ir, dc_ir = extract_ac_dc(ir_signal, sample_rate)

    if dc_red <= 0 or dc_ir <= 0:
        return 0.0

    ratio_red = ac_red / dc_red if dc_red > 0 else 0
    ratio_ir = ac_ir / dc_ir if dc_ir > 0 else 0

    if ratio_ir <= 0:
        return 0.0

    return ratio_red / ratio_ir


def estimate_spo2(r_value: float, offset: float = 0.0, scale: float = 1.0) -> float:
    """Estimate SpO2 from R value using calibration curve.

    Default: SpO2 = 110 - 25 * R (generic formula)

    Args:
        r_value: Ratio of ratios
        offset: Calibration offset adjustment
        scale: Calibration scale adjustment

    Returns:
        SpO2 percentage (clamped to 70-100%)
    """
    r_adjusted = r_value * scale + offset
    spo2 = 110.0 - 25.0 * r_adjusted
    return float(np.clip(spo2, 70.0, 100.0))


def analyze_spo2(
    df: pd.DataFrame,
    red_col: str,
    ir_col: str,
    sample_rate: float,
    window_size: int = 256,
    step_size: int = 128,
) -> dict:
    """Analyze SpO2 over time using sliding window.

    Args:
        df: DataFrame with PPG data
        red_col: Column name for red channel
        ir_col: Column name for IR channel
        sample_rate: Sampling rate in Hz
        window_size: Window size for each calculation
        step_size: Step size for sliding window

    Returns:
        dict with arrays: timestamps, spo2_values, r_values
    """
    if red_col not in df.columns or ir_col not in df.columns:
        return {"timestamps": [], "spo2_values": [], "r_values": []}

    timestamps = []
    spo2_values = []
    r_values = []

    n = len(df)
    i = 0
    while i + window_size <= n:
        window_df = df.iloc[i : i + window_size]

        red_signal = window_df[red_col].values
        ir_signal = window_df[ir_col].values

        r = calculate_r_value(red_signal, ir_signal, sample_rate)
        spo2 = estimate_spo2(r)

        if len(df) > i + window_size // 2:
            if "timestamp" in df.columns:
                ts = df["timestamp"].iloc[i + window_size // 2]
            else:
                ts = i + window_size // 2
            timestamps.append(ts)

        r_values.append(r)
        spo2_values.append(spo2)

        i += step_size

    return {
        "timestamps": timestamps,
        "spo2_values": spo2_values,
        "r_values": r_values,
    }
