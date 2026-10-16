import numpy as np
import statsmodels.api as sm
import pandas as pd

"""
This module contains functions to remove noise from solar data.
It outputs solar data to be used in solar_math.py to calculate performance metrics.

Test 2 methods to remove noise: 3V bin averaging and LOESS
"""

def bin_avg(df, step):
    """Groups the data into voltage bins and calculates the mean of each bin"""
    voltage_bins = np.arange(0, df["VOLTS"].max() + step, step) # Get bin labels
    temp_df = df.copy() # Avoid shallow copy issues
    temp_df["VOLT_BIN"] = pd.cut(temp_df["VOLTS"], bins=voltage_bins) # Create bins

    # Calculate the mean of each bin for VOLTS, AMPS, and WATTS
    bin_averages_df = (
        temp_df.groupby("VOLT_BIN", observed=False)[["VOLTS", "AMPS"]]
        .mean()
        .dropna()
        .reset_index()
    )

    bin_averages_df["WATTS"] = (
        bin_averages_df["VOLTS"] * bin_averages_df["AMPS"]
    )

    return bin_averages_df.dropna().reset_index(drop=True)

def loess(df, frac=0.15):
    """Applies LOESS smoothing to AMPS and WATTS based on the VOLTS column"""
    smoothed_df = df.sort_values(by="VOLTS").copy()

    amps_smoothed = sm.nonparametric.lowess(
        endog=smoothed_df["AMPS"], exog=smoothed_df["VOLTS"], frac=frac
    )
    smoothed_df["AMPS"] = amps_smoothed[:, 1]
    smoothed_df["WATTS"] = smoothed_df["VOLTS"] * smoothed_df["AMPS"]
    return smoothed_df

def sliding_window_avg(df, M):
    """Applies a sliding window average to the AMPS columns"""
    smoothed_df = df.copy()

    kernel = np.ones(M) / M

    smoothed_df["AMPS"] = np.convolve(
        df["AMPS"].to_numpy(),
        kernel,
        mode="full"
    )[:len(df)]

    smoothed_df["WATTS"] = (
        smoothed_df["VOLTS"] * smoothed_df["AMPS"]
    )

    return smoothed_df

def moving_average_with_edges(
    df,
    M=3,
    edge_mode="partial",
    pad_width=None,
    pad_value=0.0,
):
    """
    Apply a centered moving average to AMPS while preserving
    the original DataFrame length.

    Edge modes:
        partial  - average only available samples
        edge     - extend endpoint values
        reflect  - reflect around endpoint samples
        symmetric - reflect including endpoint samples
        constant - pad with pad_value
    """
    if M < 1:
        raise ValueError("M must be at least 1.")

    if M % 2 == 0:
        raise ValueError("M must be odd for this centered implementation.")

    if edge_mode not in {
        "partial", "edge", "reflect", "symmetric", "constant"
    }:
        raise ValueError(f"Unsupported edge mode: {edge_mode}")

    result = df.copy()
    current = result["AMPS"].to_numpy(dtype=float)
    n = len(current)

    if n == 0:
        return result

    # M=1 leaves the signal unchanged.
    if M == 1:
        result["WATTS"] = (
            result["VOLTS"].to_numpy(dtype=float) * current
        )
        return result

    half = M // 2
    kernel = np.ones(M, dtype=float) / M

    if edge_mode == "partial":
        # Normalize each output by the number of real samples
        # contributing to that window.
        sums = np.convolve(current, np.ones(M), mode="same")
        counts = np.convolve(np.ones(n), np.ones(M), mode="same")
        smoothed = sums / counts

    else:
        if edge_mode == "edge":
            padded = np.pad(current, half, mode="edge")
        elif edge_mode == "reflect":
            pad_mode = "reflect" if n > 1 else "edge"
            padded = np.pad(current, half, mode=pad_mode)
        elif edge_mode == "symmetric":
            padded = np.pad(current, half, mode="symmetric")
        else:  # constant
            padded = np.pad(
                current,
                half,
                mode="constant",
                constant_values=pad_value,
            )

        # 'valid' returns exactly n samples after padding by half
        # the window width on both sides.
        smoothed = np.convolve(padded, kernel, mode="valid")

    if len(smoothed) != n:
        raise RuntimeError(
            f"Expected {n} smoothed samples, got {len(smoothed)}."
        )

    result["AMPS"] = smoothed
    result["WATTS"] = (
        result["VOLTS"].to_numpy(dtype=float) * smoothed
    )

    return result