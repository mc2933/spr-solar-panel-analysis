
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from remove_noise import bin_avg, loess, sliding_window_avg, moving_average_with_edges
from solar_math import get_power, get_mpp, irradiance_scaling

"""
This module contains experiments to validate smoothing and scaling methods for solar data.
"""


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FOLDER = Path("data")
WINDOW_SIZES = (1,3,5,7,10,15,20)
BIN_SIZE = 3.0

# Set this to a measured/documented value before testing scaling.
MEASURED_IRRADIANCE = 281.1713867 # PANEL 1
TARGET_IRRADIANCE = 1000


# ============================================================
# 2. GENERAL HELPER FUNCTIONS
# ============================================================

def percent_diff(x1, x2):
    """Calculate symmetric percent difference between two values."""
    denominator = abs(x1) + abs(x2)

    if denominator == 0:
        return 0.0

    return abs(x1 - x2) / (denominator / 2) * 100


# ============================================================
# 3. EXPERIMENT FUNCTIONS
# ============================================================

def test_moving_average(df, window_sizes=WINDOW_SIZES):
    """Compare MPP estimates across moving-average window sizes."""

    raw_vmp, raw_imp, raw_pmax = get_mpp(df)

    print("\n" + "=" * 70)
    print("MOVING-AVERAGE WINDOW EXPERIMENT")
    print("=" * 70)
    print(f"{'Method':<20} {'Vmp (V)':>10} {'Imp (A)':>10} "
          f"{'Pmax (W)':>12} {'Pmax diff':>12}")
    print("-" * 70)

    print(f"{'Raw':<20} {raw_vmp:>10.3f} {raw_imp:>10.3f} "
          f"{raw_pmax:>12.3f} {'--':>12}")

    results = []

    for M in window_sizes:
        smoothed_df = sliding_window_avg(df, M=M)
        vmp, imp, pmax = get_mpp(smoothed_df)

        result = {
            "Method": f"Moving average M={M}",
            "M": M,
            "Vmp": vmp,
            "Imp": imp,
            "Pmax": pmax,
            "Pmax_percent_diff_from_raw": percent_diff(raw_pmax, pmax),
        }
        results.append(result)

        print(f"{result['Method']:<20} {vmp:>10.3f} {imp:>10.3f} "
              f"{pmax:>12.3f} "
              f"{result['Pmax_percent_diff_from_raw']:>11.2f}%")

    return results


def test_irradiance_order(
    df,
    measured_irradiance,
    target_irradiance=TARGET_IRRADIANCE,
    M=3,
):
    """Compare scaling-then-smoothing with smoothing-then-scaling."""

    # A: Scale first, then smooth.
    scaled_first = irradiance_scaling(
        df, measured_irradiance, target_irradiance
    )
    result_A = sliding_window_avg(scaled_first, M=M)

    # B: Smooth first, then scale.
    smoothed_first = sliding_window_avg(df, M=M)
    result_B = irradiance_scaling(
        smoothed_first, measured_irradiance, target_irradiance
    )

    vmp_A, imp_A, pmax_A = get_mpp(result_A)
    vmp_B, imp_B, pmax_B = get_mpp(result_B)

    current_diff = (
        result_A["AMPS"].to_numpy() - result_B["AMPS"].to_numpy()
    )
    power_diff = get_power(result_A).to_numpy() - get_power(result_B).to_numpy()

    print("\n" + "=" * 70)
    print("IRRADIANCE-SCALING ORDER EXPERIMENT")
    print("=" * 70)
    print(f"Measured irradiance: {measured_irradiance} W/m^2")
    print(f"Target irradiance:   {target_irradiance} W/m^2")
    print(f"Moving-average M:    {M}")

    print(f"\n{'Parameter':<15} {'Scale → Smooth':>18} "
          f"{'Smooth → Scale':>18}")
    print("-" * 55)
    print(f"{'Vmp (V)':<15} {vmp_A:>18.6f} {vmp_B:>18.6f}")
    print(f"{'Imp (A)':<15} {imp_A:>18.6f} {imp_B:>18.6f}")
    print(f"{'Pmax (W)':<15} {pmax_A:>18.6f} {pmax_B:>18.6f}")

    print("\nCurve agreement")
    print(f"Maximum current difference: {abs(current_diff).max():.10g} A")
    print(f"Maximum power difference:   {abs(power_diff).max():.10g} W")
    print(f"Pmax percent difference:    "
          f"{percent_diff(pmax_A, pmax_B):.10g}%")

    return result_A, result_B

def analyze_irradiance_sensitivity(
    df,
    measured_irradiance,
    target_irradiance=1000,
    M=3,
    relative_uncertainties=(0.02, 0.05, 0.10),
):
    """
    Examine how uncertainty in measured irradiance affects
    the scaled MPP estimate.

    Relative uncertainties are scenarios, not assumed sensor specs.
    """
    rows = []

    for uncertainty in relative_uncertainties:
        for direction in (-1, 0, 1):
            G = measured_irradiance * (
                1 + direction * uncertainty
            )

            scaled = irradiance_scaling(
                df, G, target_irradiance
            )
            processed = sliding_window_avg(scaled, M=M)
            vmp, imp, pmax = get_mpp(processed)

            rows.append({
                "Irradiance scenario": (
                    f"{direction * uncertainty:+.0%}"
                    if direction != 0 else "Baseline"
                ),
                "Measured G": G,
                "Scale factor": target_irradiance / G,
                "Vmp": vmp,
                "Imp": imp,
                "Pmax": pmax,
            })

    results = pd.DataFrame(rows)
    print("\nIrradiance-input sensitivity")
    print(results.round(3).to_string(index=False))
    return results

def compare_curve_deviation(df, window_sizes=(3, 5, 10)):
    """Quantify current deviations from the raw curve."""

    raw_current = df["AMPS"].to_numpy(dtype=float)
    rows = []

    for M in window_sizes:
        smooth = sliding_window_avg(df, M=M)
        residual = raw_current - smooth["AMPS"].to_numpy()

        rows.append({
            "M": M,
            "Residual MAE (A)": abs(residual).mean(),
            "Residual RMSE (A)": (
                (residual ** 2).mean() ** 0.5
            ),
            "Maximum abs. residual (A)": abs(residual).max(),
        })

    results = pd.DataFrame(rows)
    print("\nCurve-deviation analysis")
    print(results.round(4).to_string(index=False))
    return results

def inspect_endpoint_coverage(df):
    """Check whether raw samples approach Isc and Voc conditions."""

    v = df["VOLTS"].to_numpy(dtype=float)
    i = df["AMPS"].to_numpy(dtype=float)

    isc_idx = abs(v).argmin()
    voc_idx = abs(i).argmin()

    print("\nRaw endpoint coverage")
    print(
        f"Closest sample to V=0: "
        f"V={v[isc_idx]:.4f} V, I={i[isc_idx]:.4f} A"
    )
    print(
        f"Closest sample to I=0: "
        f"V={v[voc_idx]:.4f} V, I={i[voc_idx]:.4f} A"
    )
    print(
        f"Measured voltage range: "
        f"{v.min():.4f} to {v.max():.4f} V"
    )
    print(
        f"Measured current range: "
        f"{i.min():.4f} to {i.max():.4f} A"
    )


# ============================================================
# 4. PRINTING FUNCTIONS
# ============================================================

def print_mpp_summary(panel_name, raw_df, processed_curves):
    """Print MPP parameters and data-point counts for each method."""

    print("\n" + "=" * 65)
    print(f"Panel: {panel_name}")
    print("=" * 65)

    print(f"{'Method':<20} {'Points':>8} {'Vmp (V)':>10} "
          f"{'Imp (A)':>10} {'Pmax (W)':>12}")
    print("-" * 65)

    all_curves = {"Raw": raw_df, **processed_curves}
    results = {}

    for name, curve in all_curves.items():
        vmp, imp, pmax = get_mpp(curve)
        results[name] = (vmp, imp, pmax)

        print(f"{name:<20} {len(curve):>8} {vmp:>10.3f} "
              f"{imp:>10.3f} {pmax:>12.3f}")

    print("\nSymmetric percent difference in Pmax")
    print("-" * 45)

    raw_pmax = results["Raw"][2]

    for name, (_, _, pmax) in results.items():
        if name != "Raw":
            print(f"Raw vs {name:<20}: {percent_diff(raw_pmax, pmax):.2f}%")

    return results

def compare_edge_handling(df, M=9):
    """Compare MPP estimates and curves for different edge rules."""

    modes = ["partial", "edge", "reflect", "symmetric", "constant"]
    curves = {}

    print(f"\nEdge-handling comparison: M={M}")
    print(f"{'Mode':<12} {'Vmp (V)':>10} {'Imp (A)':>10} {'Pmax (W)':>12}")
    print("-" * 48)

    for mode in modes:
        curve = moving_average_with_edges(
            df,
            M=M,
            edge_mode=mode,
            pad_value=0.0,
        )
        curves[mode] = curve

        vmp, imp, pmax = get_mpp(curve)
        print(f"{mode:<12} {vmp:>10.3f} {imp:>10.3f} {pmax:>12.3f}")

    plt.figure(figsize=(9, 5))
    plt.scatter(df["VOLTS"], df["AMPS"], label="Raw", s=10, alpha=0.5)

    for mode, curve in curves.items():
        plt.plot(curve["VOLTS"], curve["AMPS"], label=mode)

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Edge Handling Comparison (M={M})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return curves


# ============================================================
# 5. PLOTTING FUNCTIONS
# ============================================================

def plot_iv_curves(raw_df, processed_curves, panel_name):
    """Plot raw I-V measurements and processed curves together."""

    plt.figure()

    plt.scatter(
        raw_df["VOLTS"],
        raw_df["AMPS"],
        label="Raw",
        s=10,
    )

    for name, curve in processed_curves.items():
        plt.plot(curve["VOLTS"], curve["AMPS"], label=name)

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"I-V Curve: {panel_name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_moving_average_windows(df, window_sizes=WINDOW_SIZES):
    """Plot raw current and several moving-average curves."""

    plt.figure()
    plt.scatter(df["VOLTS"], df["AMPS"], label="Raw", s=10)

    for M in window_sizes:
        smoothed_df = sliding_window_avg(df, M=M)

        plt.plot(
            smoothed_df["VOLTS"],
            smoothed_df["AMPS"],
            label=f"Moving average M={M}",
        )

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("Moving-Average Window Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_smoothing_comparison(df, window_sizes=WINDOW_SIZES):
    """Compare raw and smoothed I-V curves, including the MPP region."""

    raw_vmp, _, _ = get_mpp(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(
        df["VOLTS"], df["AMPS"],
        label="Raw", s=10, alpha=0.6
    )

    axes[1].scatter(
        df["VOLTS"], df["AMPS"],
        label="Raw", s=10, alpha=0.6
    )

    for M in window_sizes:
        smoothed = sliding_window_avg(df, M=M)

        for ax in axes:
            ax.plot(
                smoothed["VOLTS"],
                smoothed["AMPS"],
                label=f"M={M}"
            )

    axes[0].set_title("Full I-V Curve")
    axes[1].set_title("Zoom Around Raw MPP")
    axes[1].set_xlim(raw_vmp - 4, raw_vmp + 4)

    for ax in axes:
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Effect of Moving-Average Window Size")
    fig.tight_layout()
    plt.show()

def plot_smoothing_residuals(df, window_sizes=WINDOW_SIZES):
    """Plot the current removed by each moving-average filter."""

    sample_index = range(len(df))

    plt.figure(figsize=(10, 5))

    for M in window_sizes:
        smoothed = sliding_window_avg(df, M=M)

        residual = (
            df["AMPS"].to_numpy()
            - smoothed["AMPS"].to_numpy()
        )

        plt.plot(
            sample_index,
            residual,
            label=f"M={M}",
            alpha=0.8
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Sample Index (acquisition order)")
    plt.ylabel("Raw Current - Smoothed Current (A)")
    plt.title("Moving-Average Residuals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_moving_average_sensitivity(
    df,
    window_sizes=(1, 3, 5, 7, 10, 15, 20),
):
    """Calculate and plot MPP metrics across window sizes."""

    results = []

    for M in window_sizes:
        smoothed = df.copy() if M == 1 else sliding_window_avg(df, M=M)
        vmp, imp, pmax = get_mpp(smoothed)

        results.append({
            "M": M,
            "Vmp": vmp,
            "Imp": imp,
            "Pmax": pmax,
        })

    results_df = pd.DataFrame(results)

    raw_pmax = results_df.loc[
        results_df["M"] == 1, "Pmax"
    ].iloc[0]

    results_df["Pmax_change_pct"] = (
        (results_df["Pmax"] - raw_pmax) / raw_pmax * 100
    )

    print("\nMoving-average sensitivity analysis")
    print(results_df.round(3).to_string(index=False))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    metrics = [
        ("Vmp", "Vmp (V)"),
        ("Imp", "Imp (A)"),
        ("Pmax", "Pmax (W)"),
    ]

    for ax, (column, label) in zip(axes, metrics):
        ax.plot(
            results_df["M"],
            results_df[column],
            marker="o"
        )
        ax.set_ylabel(label)
        ax.grid(True)

    axes[-1].set_xlabel("Moving-Average Window Size M")
    fig.suptitle("Sensitivity of MPP Estimates to Smoothing")
    fig.tight_layout()
    plt.show()

    return results_df


# ============================================================
# 6. MAIN WORKFLOW
# ============================================================

def main(csv_file):
    """Run all experiments to validate smoothing and scaling methods."""

    raw_df = pd.read_csv(csv_file)
    panel_name = csv_file.stem

    # --- Process the raw data ---
    window_df = sliding_window_avg(raw_df, M=3)
    bin_df = bin_avg(raw_df, BIN_SIZE)
    loess_df = loess(raw_df)

    # --- Choose which curves to compare ---
    processed_curves = {
        "Moving average M=3": window_df,
        "3-V Bin": bin_df,
        "LOESS": loess_df,
    }

    # --- Print results ---
    print_mpp_summary(panel_name, raw_df, processed_curves)

    # --- Plot results ---
    plot_iv_curves(raw_df, processed_curves, panel_name)

    # --- Other experiments: uncomment to run ---
    # TEST MOVING AVERAGE WINDOW SIZES
    test_moving_average(raw_df, WINDOW_SIZES)
    sensitivity_df = analyze_moving_average_sensitivity(
                        raw_df,
                        window_sizes=(1, 3, 5, 7, 10, 15, 20),
                    )
    plot_moving_average_windows(raw_df, WINDOW_SIZES)
    plot_smoothing_comparison(raw_df, WINDOW_SIZES)
    plot_smoothing_residuals(raw_df, WINDOW_SIZES)
    compare_curve_deviation(raw_df, WINDOW_SIZES)

    # TEST EDGE-HANDLING MODES
    compare_edge_handling(raw_df, M=9)
    inspect_endpoint_coverage(raw_df)

    # TEST IRRADIANCE METHODS
    analyze_irradiance_sensitivity(
        raw_df,
        measured_irradiance=MEASURED_IRRADIANCE,
        target_irradiance=TARGET_IRRADIANCE,
        M=3,
    )
    # if MEASURED_IRRADIANCE is not None: # Only run this after entering a measured irradiance
    #     test_irradiance_order(
    #         raw_df,
    #         measured_irradiance=MEASURED_IRRADIANCE,
    #         target_irradiance=TARGET_IRRADIANCE,
    #         M=3,
    #     )


if __name__ == "__main__":
    main()
