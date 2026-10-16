import pandas as pd
import numpy as np

"""
This module does PV curve processing and parameter extraction.
All functions in this module assume that the input DataFrame has been cleaned of noise using remove_noise.py.

Calculate the following parameters: Voc, Isc, Max Power Point (MPP), Pmax
Calculate reference-conditions (1000 W/m^2)
"""

def get_power(df):
    """
    Calculates power (W) from voltage (V) and current (A): P = V * I
    Returns an array of power values corresponding to each row in the input DataFrame
    """
    return df["VOLTS"] * df["AMPS"]

def get_voc_isc(df):
    """
    Estimate Voc and Isc from the I-V samples.

    If a zero crossing is bracketed by samples, use linear interpolation.
    Otherwise, use the sample nearest the relevant zero and label it as an
    approximation. These are estimates from the sampled curve, not verified
    datasheet/STC ratings.
    """
    data = df[["VOLTS", "AMPS"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError("Cannot estimate Voc/Isc from an empty valid curve.")

    # Isc: current at V = 0. Sort by voltage for adjacent-point interpolation.
    by_voltage = data.sort_values("VOLTS").reset_index(drop=True)
    v = by_voltage["VOLTS"].to_numpy(dtype=float)
    i = by_voltage["AMPS"].to_numpy(dtype=float)

    isc_candidates = []
    exact_v_zero = np.flatnonzero(v == 0)
    if len(exact_v_zero):
        idx = int(exact_v_zero[0])
        isc_candidates.append((float(i[idx]), "sample at V=0"))
    else:
        for k in range(len(v) - 1):
            if (v[k] < 0 < v[k + 1]) or (v[k + 1] < 0 < v[k]):
                fraction = (0.0 - v[k]) / (v[k + 1] - v[k])
                current_at_zero = i[k] + fraction * (i[k + 1] - i[k])
                isc_candidates.append((float(current_at_zero), "linear interpolation"))
        if not isc_candidates:
            idx = int(np.argmin(np.abs(v)))
            isc_candidates.append((float(i[idx]), "nearest sample (approximation)"))
    isc, isc_method = isc_candidates[0]

    # Voc: voltage at I = 0. Use the highest-voltage crossing if noise creates
    # more than one crossing, which is the physically relevant end of a PV curve.
    voc_candidates = []
    exact_i_zero = np.flatnonzero(i == 0)
    if len(exact_i_zero):
        voc_candidates.extend((float(v[idx]), "sample at I=0") for idx in exact_i_zero)
    else:
        for k in range(len(i) - 1):
            if (i[k] < 0 < i[k + 1]) or (i[k + 1] < 0 < i[k]):
                fraction = (0.0 - i[k]) / (i[k + 1] - i[k])
                voltage_at_zero = v[k] + fraction * (v[k + 1] - v[k])
                voc_candidates.append((float(voltage_at_zero), "linear interpolation"))

    if voc_candidates:
        voc, voc_method = max(voc_candidates, key=lambda item: item[0])
    else:
        idx = int(np.argmin(np.abs(i)))
        voc = float(v[idx])
        voc_method = "nearest sample (approximation)"

    return {
        "Voc_V": voc,
        "Voc_method": voc_method,
        "Isc_A": isc,
        "Isc_method": isc_method,
    }

def get_mpp(df):
    """
    Finds the maximum power point (MPP): finds the largest calculated power value from each row in the input DataFrame
    Since the panels are imperfect, max power is not necessarily at a 'knee' point, so we find the max power value 
    and return the corresponding voltage and current values

    Returns a tuple of (V_mpp, I_mpp, P_mpp)
    """
    power_lst = get_power(df)
    max_power_idx = power_lst.idxmax()
    v_mpp = df.loc[max_power_idx, "VOLTS"]
    i_mpp = df.loc[max_power_idx, "AMPS"]
    return (v_mpp, i_mpp, power_lst.loc[max_power_idx])

def irradiance_scaling(df, measured_irradiance, target_irradiance=1000):
    """
    Estimate an I-V curve at a target irradiance by scaling current.

    Irradiance units: W/m^2.
    This approximation scales current only; voltage is unchanged.
    """
    if measured_irradiance <= 0 or target_irradiance <= 0:
        raise ValueError("Irradiance values must be positive.")

    scaled_df = df.copy()

    scale_factor = target_irradiance / measured_irradiance

    scaled_df["AMPS"] = scaled_df["AMPS"] * scale_factor
    scaled_df["WATTS"] = scaled_df["VOLTS"] * scaled_df["AMPS"]

    return scaled_df

