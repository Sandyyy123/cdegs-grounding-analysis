"""
Electromagnetic interference decomposition for pipeline/telecom corridor studies.

Separates inductive and conductive coupling contributions from MALZ output.
Computes AC corrosion risk per EN 15280 / NACE SP0177.

Used in:
  - Pipeline-transmission line parallel corridor studies
  - Telecom cable interference assessments
  - Railroad track circuit interference
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# EN 15280:2013 / NACE SP0177 AC interference thresholds
AC_VOLTAGE_THRESHOLD_V = 15.0       # Touch voltage safety limit on metallic structures
AC_DENSITY_CORROSION_A_M2 = 30.0    # AC current density threshold for corrosion risk (A/m2)


def decompose_interference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose total induced voltage into inductive and conductive components.

    MALZ exports both components separately. This function:
      - Computes the ratio of each component to total
      - Flags points where AC voltage exceeds EN 15280 limit
      - Calculates inductive dominance (inductive/total ratio)

    Args:
        df: DataFrame with columns:
            induced_voltage_v, conductive_v, inductive_v, distance_m

    Returns:
        DataFrame with decomposition columns added
    """
    result = df.copy()

    # Avoid division by zero
    total = result["induced_voltage_v"].replace(0, np.nan)

    result["inductive_fraction"] = result["inductive_v"] / total
    result["conductive_fraction"] = result["conductive_v"] / total
    result["dominant_mechanism"] = np.where(
        result["inductive_fraction"] >= 0.5, "INDUCTIVE", "CONDUCTIVE"
    )

    result["exceeds_en15280"] = result["induced_voltage_v"] > AC_VOLTAGE_THRESHOLD_V

    return result


def ac_corrosion_risk(
    induced_voltage_v: float,
    coating_resistance_ohm_m2: float,
    pipe_diameter_m: float = 0.3,
) -> dict:
    """
    Estimate AC current density on a coated pipeline for corrosion risk assessment.

    Per EN 15280, AC current density > 30 A/m2 indicates elevated corrosion risk.

    i_ac = V_ac / (R_coat * A_holiday)

    For a standard 1 cm2 (1e-4 m2) coating holiday:

    Args:
        induced_voltage_v: Total induced AC voltage on pipe (V)
        coating_resistance_ohm_m2: Coating resistance (ohm-m2), typical 10k-100k
        pipe_diameter_m: Pipe outer diameter (m)

    Returns:
        dict with current density, risk flag, and margin
    """
    holiday_area_m2 = 1e-4  # 1 cm2 standard holiday per EN 15280

    # Spreading resistance at holiday (approximation)
    spreading_resistance = coating_resistance_ohm_m2 / holiday_area_m2

    i_ac = induced_voltage_v / spreading_resistance
    current_density = i_ac / holiday_area_m2

    return {
        "induced_voltage_v": induced_voltage_v,
        "coating_resistance_ohm_m2": coating_resistance_ohm_m2,
        "ac_current_density_a_m2": round(current_density, 2),
        "corrosion_risk": current_density > AC_DENSITY_CORROSION_A_M2,
        "margin_a_m2": round(AC_DENSITY_CORROSION_A_M2 - current_density, 2),
    }


def plot_interference_decomposition(
    df: pd.DataFrame,
    distance_col: str = "distance_m",
    output_path: str = None,
):
    """
    Stacked area plot of inductive vs conductive interference along corridor.
    """
    df_sorted = df.sort_values(distance_col)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#12121a")

    # Top: stacked area (inductive + conductive)
    ax1.stackplot(
        df_sorted[distance_col],
        df_sorted["inductive_v"].abs(),
        df_sorted["conductive_v"].abs(),
        labels=["Inductive coupling", "Conductive coupling"],
        colors=["#6c5ce7", "#00b894"],
        alpha=0.8,
    )
    ax1.axhline(y=AC_VOLTAGE_THRESHOLD_V, color="#e17055", linewidth=1.5,
                linestyle="--", label=f"EN 15280 limit: {AC_VOLTAGE_THRESHOLD_V} V")
    ax1.set_ylabel("Induced Voltage (V)", color="#a29bfe")
    ax1.set_title("AC Interference Decomposition - Inductive vs Conductive",
                  color="white", fontweight="bold", fontsize=13)
    ax1.legend(facecolor="#1a1a28", edgecolor="#2a2a3d", labelcolor="white", fontsize=9)
    ax1.tick_params(colors="#8888a8")
    ax1.grid(True, color="#2a2a3d", linewidth=0.5, alpha=0.6)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#2a2a3d")

    # Bottom: dominant mechanism
    is_inductive = (df_sorted["dominant_mechanism"] == "INDUCTIVE").astype(int)
    ax2.fill_between(df_sorted[distance_col], 0, is_inductive,
                     color="#6c5ce7", alpha=0.7, label="Inductive dominant")
    ax2.fill_between(df_sorted[distance_col], 0, 1 - is_inductive,
                     color="#00b894", alpha=0.5, label="Conductive dominant")
    ax2.set_ylabel("Dominant Mechanism", color="#a29bfe")
    ax2.set_xlabel("Distance along corridor (m)", color="#a29bfe")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Conductive", "Inductive"], color="#8888a8")
    ax2.tick_params(colors="#8888a8")
    ax2.legend(facecolor="#1a1a28", edgecolor="#2a2a3d", labelcolor="white", fontsize=9)
    ax2.grid(True, color="#2a2a3d", linewidth=0.5, alpha=0.6)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2a2a3d")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Interference decomposition plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="AC interference decomposition analysis.")
    parser.add_argument("--data", required=True, help="CSV with MALZ interference data")
    parser.add_argument("--output-csv", help="Save flagged dataset to CSV")
    parser.add_argument("--output-plot", help="Save decomposition plot to PNG")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    result = decompose_interference(df)

    n_exceed = int(result["exceeds_en15280"].sum())
    print(f"\nAC Interference Summary (EN 15280 threshold: {AC_VOLTAGE_THRESHOLD_V} V)")
    print(f"  Total points:          {len(result)}")
    print(f"  Exceeding EN 15280:    {n_exceed} ({100*n_exceed/len(result):.1f}%)")
    print(f"  Inductive dominant:    {int((result['dominant_mechanism']=='INDUCTIVE').sum())}")
    print(f"  Conductive dominant:   {int((result['dominant_mechanism']=='CONDUCTIVE').sum())}")
    print(f"  Max induced voltage:   {result['induced_voltage_v'].max():.2f} V")

    if args.output_csv:
        result.to_csv(args.output_csv, index=False)
        print(f"  Flagged dataset saved to {args.output_csv}")

    if args.output_plot:
        plot_interference_decomposition(result, output_path=args.output_plot)


if __name__ == "__main__":
    main()
