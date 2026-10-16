import pandas as pd

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

def get_voc(df):
    """Calculates the open-circuit voltage (Voc): max voltage when current is zero"""
    pass

def get_isc(df):
    """Calculates the short-circuit current (Isc): max current when voltage is zero"""
    pass

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

