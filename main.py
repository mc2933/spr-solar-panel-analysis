from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from remove_noise import moving_average_with_edges
from solar_math import get_mpp, get_power, irradiance_scaling, get_voc_isc


"""
This module contains the main production workflow for processing solar panel CSVs.
"""


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FOLDER = Path("data")
IRRADIANCE_FILE = Path("measured_irradiance.csv")

OUTPUT_FOLDER = Path("output")
PLOT_FOLDER = OUTPUT_FOLDER / "plots"
RESULTS_FILE = OUTPUT_FOLDER / "results.txt"

TARGET_IRRADIANCE = 1000.0  # W/m^2; reference irradiance for the estimate
SMOOTHING_WINDOW = 3
EDGE_MODE = "edge"  # Extend endpoint values when applying the moving average


# ============================================================
# 2. INPUT CHECKS AND METRIC HELPERS
# ============================================================

def validate_raw_df(raw_df, panel_name):
    """
    Validate a raw panel CSV before smoothing.

    Invalid/non-finite voltage or current values raise an error rather than
    being silently removed from the measurement data.
    """
    if raw_df.empty:
        raise ValueError(f"{panel_name}: CSV contains no data rows.")

    # Tolerate whitespace/case differences in column names.
    raw_df = raw_df.copy()
    raw_df.columns = [str(column).strip().upper() for column in raw_df.columns]

    required = {"VOLTS", "AMPS"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(
            f"{panel_name}: missing required column(s): {', '.join(sorted(missing))}."
        )

    raw_df["VOLTS"] = pd.to_numeric(raw_df["VOLTS"], errors="coerce")
    raw_df["AMPS"] = pd.to_numeric(raw_df["AMPS"], errors="coerce")

    valid = (
        np.isfinite(raw_df["VOLTS"].to_numpy())
        & np.isfinite(raw_df["AMPS"].to_numpy())
    )
    if not valid.all():
        bad_rows = (np.flatnonzero(~valid) + 2).tolist()  # account for CSV header
        preview = ", ".join(map(str, bad_rows[:10]))
        suffix = " ..." if len(bad_rows) > 10 else ""
        raise ValueError(
            f"{panel_name}: found missing, non-numeric, or non-finite VOLTS/AMPS "
            f"value(s) on CSV row(s) {preview}{suffix}."
        )

    if len(raw_df) < 3:
        raise ValueError(
            f"{panel_name}: only {len(raw_df)} valid sample(s); at least 3 are required."
        )

    # Recalculate power consistently from voltage and current.
    raw_df["WATTS"] = get_power(raw_df)
    return raw_df.reset_index(drop=True)


def load_irradiance_table(irradiance_file):
    """Load and validate the panel-name-to-irradiance lookup table."""
    if not irradiance_file.exists():
        raise FileNotFoundError(
            f"Irradiance lookup file not found: {irradiance_file.resolve()}"
        )

    table = pd.read_csv(irradiance_file)
    table.columns = [str(column).strip().upper() for column in table.columns]

    required = {"PANEL NAME", "MEASURED IRRADIANCE"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{irradiance_file}: missing required column(s): "
            f"{', '.join(sorted(missing))}."
        )

    table = table[["PANEL NAME", "MEASURED IRRADIANCE"]].copy()
    table["PANEL NAME"] = table["PANEL NAME"].astype("string").str.strip()
    table["MEASURED IRRADIANCE"] = pd.to_numeric(
        table["MEASURED IRRADIANCE"], errors="coerce"
    )

    if table["PANEL NAME"].isna().any() or (table["PANEL NAME"] == "").any():
        raise ValueError(f"{irradiance_file}: one or more panel names are blank.")

    if table["PANEL NAME"].duplicated().any():
        duplicates = table.loc[
            table["PANEL NAME"].duplicated(keep=False), "PANEL NAME"
        ].tolist()
        raise ValueError(
            f"{irradiance_file}: duplicate panel name(s): {', '.join(duplicates)}."
        )

    irradiance = table["MEASURED IRRADIANCE"].to_numpy(dtype=float)
    valid_irradiance = np.isfinite(irradiance) & (irradiance > 0)
    if not valid_irradiance.all():
        bad_names = table.loc[~valid_irradiance, "PANEL NAME"].tolist()
        raise ValueError(
            f"{irradiance_file}: measured irradiance must be a finite positive "
            f"number for panel(s): {', '.join(bad_names)}."
        )

    return dict(zip(table["PANEL NAME"], table["MEASURED IRRADIANCE"]))


def curve_metrics(df):
    """Return MPP, Voc, and Isc estimates for a curve."""
    vmp, imp, pmax = get_mpp(df)
    endpoints = get_voc_isc(df)
    return {
        "Vmp_V": float(vmp),
        "Imp_A": float(imp),
        "Pmax_W": float(pmax),
        **endpoints,
    }


def format_metrics(label, metrics):
    return (
        f"{label}\n"
        f"  Vmp: {metrics['Vmp_V']:.4f} V\n"
        f"  Imp: {metrics['Imp_A']:.4f} A\n"
        f"  Pmax: {metrics['Pmax_W']:.4f} W\n"
        f"  Voc: {metrics['Voc_V']:.4f} V "
        f"({metrics['Voc_method']})\n"
        f"  Isc: {metrics['Isc_A']:.4f} A "
        f"({metrics['Isc_method']})\n"
    )


# ============================================================
# 3. OUTPUT HELPERS
# ============================================================

def save_iv_plot(panel_name, raw_df, smoothed_df, scaled_df, measured_irradiance):
    """Save one figure per panel, with measured and scaled curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        raw_df["VOLTS"], raw_df["AMPS"],
        label="Raw measured samples", s=10, alpha=0.45
    )
    axes[0].plot(
        smoothed_df["VOLTS"], smoothed_df["AMPS"],
        label=f"Smoothed measured curve (M={SMOOTHING_WINDOW})", linewidth=2
    )
    axes[0].set_title("Measured irradiance")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Current (A)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        scaled_df["VOLTS"], scaled_df["AMPS"],
        label=f"Linearly scaled estimate ({TARGET_IRRADIANCE:.0f} W/m²)",
        linewidth=2
    )
    axes[1].set_title(
        f"Scaled estimate\nMeasured G = {measured_irradiance:.2f} W/m²"
    )
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Current (A)")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"I-V Characterization: {panel_name}")
    fig.tight_layout()
    output_path = PLOT_FOLDER / f"{panel_name}_iv.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================
# 4. PRODUCTION WORKFLOW
# ============================================================

def process_panel(csv_file, irradiance_lookup):
    """Validate, smooth, scale, calculate metrics, and save a panel plot."""
    panel_name = csv_file.stem

    # Input checks happen immediately after loading raw_df, before smoothing.
    raw_df = pd.read_csv(csv_file)
    raw_df = validate_raw_df(raw_df, panel_name)

    if panel_name not in irradiance_lookup:
        raise ValueError(
            f"{panel_name}: no measured irradiance entry in "
            f"{IRRADIANCE_FILE}."
        )

    measured_irradiance = float(irradiance_lookup[panel_name])
    scale_factor = TARGET_IRRADIANCE / measured_irradiance

    # Smoothing method: centered/edge-extended moving average, M=3.
    smoothed_df = moving_average_with_edges(
        raw_df, M=SMOOTHING_WINDOW, edge_mode=EDGE_MODE
    )
    smoothed_df = smoothed_df.copy()
    smoothed_df["WATTS"] = get_power(smoothed_df)

    # Scale the smoothed measured curve. This is a linear irradiance estimate, not a verified STC measurement.
    scaled_df = irradiance_scaling(
        smoothed_df, measured_irradiance, TARGET_IRRADIANCE
    )
    scaled_df = scaled_df.copy()
    scaled_df["WATTS"] = get_power(scaled_df)

    raw_metrics = curve_metrics(raw_df)
    measured_metrics = curve_metrics(smoothed_df)
    scaled_metrics = curve_metrics(scaled_df)

    plot_path = save_iv_plot(
        panel_name, raw_df, smoothed_df, scaled_df, measured_irradiance
    )

    # Uniformly scaling current should leave the zero-current voltage crossing
    # (Voc) unchanged. Report the difference as a diagnostic.
    voc_difference = scaled_metrics["Voc_V"] - measured_metrics["Voc_V"]

    result_text = (
        f"{'=' * 72}\n"
        f"Panel: {panel_name}\n"
        f"Input file: {csv_file}\n"
        f"Raw sample count: {len(raw_df)}\n"
        f"Measured irradiance: {measured_irradiance:.4f} W/m²\n"
        f"Target irradiance: {TARGET_IRRADIANCE:.1f} W/m²\n"
        f"Linear scale factor applied to current: {scale_factor:.6f}\n"
        f"Smoothing: moving average M={SMOOTHING_WINDOW}, edge mode='{EDGE_MODE}'\n"
        f"Plot: {plot_path}\n\n"
        f"{format_metrics('RAW MEASURED CURVE', raw_metrics)}\n"
        f"{format_metrics('SMOOTHED MEASURED CURVE', measured_metrics)}\n"
        f"{format_metrics('SCALED CURVE ESTIMATE', scaled_metrics)}\n"
        f"Voc difference (scaled - smoothed measured): {voc_difference:.8f} V\n"
        f"Interpretation: scaled values are first-order irradiance estimates, "
        f"not verified STC ratings. STC also specifies cell temperature and "
        f"reference spectrum, which are not established by irradiance scaling alone.\n"
    )
    return result_text


def main():
    """Process every panel CSV and write one text report plus one plot per panel."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    panel_files = sorted(DATA_FOLDER.glob("*.csv"))
    if not panel_files:
        print(f"No panel CSV files found in {DATA_FOLDER.resolve()}")
        return

    try:
        irradiance_lookup = load_irradiance_table(IRRADIANCE_FILE)
    except (OSError, ValueError) as error:
        print(f"Could not load irradiance lookup: {error}")
        return

    panel_names = {file.stem for file in panel_files}
    missing_irradiance = sorted(panel_names - set(irradiance_lookup))
    extra_irradiance = sorted(set(irradiance_lookup) - panel_names)

    if missing_irradiance:
        print(
            "ERROR: no measured irradiance entry for panel(s): "
            + ", ".join(missing_irradiance)
        )
    if extra_irradiance:
        print(
            "WARNING: irradiance entries without matching panel CSV: "
            + ", ".join(extra_irradiance)
        )

    report_sections = [
        "SOLAR PANEL REBOOT — PANEL CHARACTERIZATION RESULTS\n",
        f"Data folder: {DATA_FOLDER.resolve()}\n",
        f"Irradiance lookup: {IRRADIANCE_FILE.resolve()}\n",
        f"Target irradiance: {TARGET_IRRADIANCE:.1f} W/m²\n",
        f"Production smoothing: M={SMOOTHING_WINDOW}, edge mode='{EDGE_MODE}'\n",
        "\nIMPORTANT: scaled results are first-order irradiance estimates, "
        "not verified STC ratings.\n",
    ]

    if missing_irradiance:
        report_sections.append(
            "\nPANELS SKIPPED — MISSING IRRADIANCE\n"
            + "\n".join(missing_irradiance)
            + "\n"
        )

    successful = 0
    failed = []

    for csv_file in panel_files:
        panel_name = csv_file.stem
        # Do not process a panel with no lookup value; this is also recorded above.
        if panel_name not in irradiance_lookup:
            failed.append((panel_name, "missing measured irradiance"))
            continue

        try:
            section = process_panel(csv_file, irradiance_lookup)
            report_sections.append("\n" + section)
            successful += 1
            print(f"Processed {panel_name}")
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failed.append((panel_name, message))
            report_sections.append(
                f"\n{'=' * 72}\nPANEL PROCESSING ERROR: {panel_name}\n{message}\n"
            )
            print(f"Failed {panel_name}: {message}")

    report_sections.append(
        f"\nSUMMARY\nSuccessful panels: {successful}\nFailed/skipped panels: {len(failed)}\n"
    )
    if failed:
        report_sections.append("\nFailure details:\n")
        report_sections.extend(f"- {name}: {message}\n" for name, message in failed)

    RESULTS_FILE.write_text("\n".join(report_sections), encoding="utf-8")
    print(f"\nResults written to: {RESULTS_FILE.resolve()}")
    print(f"Plots saved in: {PLOT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()
