"""
IEEE 80-2013 touch and step voltage threshold analysis.

Calculates permissible touch and step voltage limits for a given:
  - fault duration (seconds)
  - surface layer resistivity (ohm-m)
  - body weight (50 kg default per IEEE 80 Table 1)

Flags all analysis points where simulated GPR, touch, or step voltage
exceeds the calculated safety threshold.

Reference: IEEE Std 80-2013, Clause 8.3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# IEEE 80-2013, Table 1 - fibrillation threshold constants
# 50 kg body: Ib = 0.116 / sqrt(t)   (A)
# 70 kg body: Ib = 0.157 / sqrt(t)   (A)
BODY_WEIGHT_CONSTANTS = {50: 0.116, 70: 0.157}

# IEEE 80-2013, Eq 32/33 - body resistance (internal, ohms)
BODY_RESISTANCE_OHM = 1000


def permissible_touch_voltage(
    fault_duration_s: float,
    surface_resistivity_ohm_m: float,
    body_weight_kg: int = 50,
) -> float:
    """
    IEEE 80-2013 Eq. 32: permissible touch voltage (V).

    E_touch50 = (Rb + Rf/2) * Ib
              = (1000 + rho_s * C_s / 2) * (0.116 / sqrt(t))

    For a 0.08 m crushed rock surface layer:
      Rf = 1.5 * rho_s  (IEEE 80, simplified, assuming C_s = 1)

    Args:
        fault_duration_s: Clearing time in seconds
        surface_resistivity_ohm_m: Surface material resistivity (rho_s)
        body_weight_kg: 50 or 70 kg (default 50 per IEEE 80)

    Returns:
        Permissible touch voltage in volts
    """
    k = BODY_WEIGHT_CONSTANTS.get(body_weight_kg, 0.116)
    ib = k / np.sqrt(fault_duration_s)
    # Foot resistance for touch: 1.5 * rho_s (single foot, series with body)
    rf_touch = 1.5 * surface_resistivity_ohm_m
    e_touch = (BODY_RESISTANCE_OHM + rf_touch) * ib
    return round(e_touch, 2)


def permissible_step_voltage(
    fault_duration_s: float,
    surface_resistivity_ohm_m: float,
    body_weight_kg: int = 50,
) -> float:
    """
    IEEE 80-2013 Eq. 33: permissible step voltage (V).

    E_step50 = (Rb + 2*Rf) * Ib
             = (1000 + 6 * rho_s) * (0.116 / sqrt(t))

    Two feet in series with body, both on ground surface.

    Args:
        fault_duration_s: Clearing time in seconds
        surface_resistivity_ohm_m: Surface material resistivity (rho_s)
        body_weight_kg: 50 or 70 kg

    Returns:
        Permissible step voltage in volts
    """
    k = BODY_WEIGHT_CONSTANTS.get(body_weight_kg, 0.116)
    ib = k / np.sqrt(fault_duration_s)
    # Foot resistance for step: 3 * rho_s per foot, two feet = 6 * rho_s
    rf_step = 6.0 * surface_resistivity_ohm_m
    e_step = (BODY_RESISTANCE_OHM + rf_step) * ib
    return round(e_step, 2)


def analyze(
    df: pd.DataFrame,
    fault_duration_s: float,
    surface_resistivity_ohm_m: float,
    body_weight_kg: int = 50,
    gpr_col: str = "gpr_v",
    touch_col: str = "touch_voltage_v",
    step_col: str = "step_voltage_v",
) -> pd.DataFrame:
    """
    Flag exceedances of IEEE 80 touch and step voltage thresholds.

    Adds columns:
      - e_touch_limit_v: permissible touch voltage
      - e_step_limit_v: permissible step voltage
      - touch_exceeds: bool
      - step_exceeds: bool
      - risk_level: 'SAFE' | 'TOUCH_RISK' | 'STEP_RISK' | 'CRITICAL'

    Args:
        df: DataFrame with simulation results (one row per observation point)
        fault_duration_s: Fault clearing time (seconds)
        surface_resistivity_ohm_m: Surface layer resistivity (ohm-m)
        body_weight_kg: Body weight for threshold calculation (50 or 70 kg)
        gpr_col: Column name for GPR values
        touch_col: Column name for touch voltage values
        step_col: Column name for step voltage values

    Returns:
        DataFrame with exceedance flags added
    """
    e_touch = permissible_touch_voltage(fault_duration_s, surface_resistivity_ohm_m, body_weight_kg)
    e_step = permissible_step_voltage(fault_duration_s, surface_resistivity_ohm_m, body_weight_kg)

    result = df.copy()
    result["e_touch_limit_v"] = e_touch
    result["e_step_limit_v"] = e_step
    result["fault_duration_s"] = fault_duration_s
    result["surface_resistivity_ohm_m"] = surface_resistivity_ohm_m

    # Flag exceedances
    if touch_col in result.columns:
        result["touch_exceeds"] = result[touch_col] > e_touch
    if step_col in result.columns:
        result["step_exceeds"] = result[step_col] > e_step

    # Risk classification
    def risk_level(row):
        touch_risk = row.get("touch_exceeds", False)
        step_risk = row.get("step_exceeds", False)
        if touch_risk and step_risk:
            return "CRITICAL"
        elif touch_risk:
            return "TOUCH_RISK"
        elif step_risk:
            return "STEP_RISK"
        return "SAFE"

    result["risk_level"] = result.apply(risk_level, axis=1)

    return result


def summarize(df: pd.DataFrame) -> dict:
    """
    Generate exceedance summary statistics.

    Returns dict with:
      - total_points: total observation points
      - safe_count: points within limits
      - touch_risk_count
      - step_risk_count
      - critical_count: both touch and step exceed
      - max_touch_v: maximum simulated touch voltage
      - max_step_v: maximum simulated step voltage
      - e_touch_limit_v: threshold used
      - e_step_limit_v: threshold used
    """
    summary = {
        "total_points": len(df),
        "safe_count": int((df["risk_level"] == "SAFE").sum()),
        "touch_risk_count": int((df["risk_level"] == "TOUCH_RISK").sum()),
        "step_risk_count": int((df["risk_level"] == "STEP_RISK").sum()),
        "critical_count": int((df["risk_level"] == "CRITICAL").sum()),
        "e_touch_limit_v": df["e_touch_limit_v"].iloc[0],
        "e_step_limit_v": df["e_step_limit_v"].iloc[0],
    }

    if "touch_voltage_v" in df.columns:
        summary["max_touch_v"] = round(df["touch_voltage_v"].max(), 2)
    if "step_voltage_v" in df.columns:
        summary["max_step_v"] = round(df["step_voltage_v"].max(), 2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="IEEE 80-2013 touch/step voltage threshold analysis."
    )
    parser.add_argument("--data", required=True, help="CSV with simulation results")
    parser.add_argument("--fault-duration", type=float, required=True,
                        help="Fault clearing time in seconds (e.g. 0.5)")
    parser.add_argument("--surface-resistivity", type=float, required=True,
                        help="Surface layer resistivity in ohm-m (e.g. 3000 for crushed rock)")
    parser.add_argument("--body-weight", type=int, default=50, choices=[50, 70],
                        help="Body weight for threshold calculation (default 50 kg)")
    parser.add_argument("--output", help="Output CSV path for flagged dataset")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    e_touch = permissible_touch_voltage(args.fault_duration, args.surface_resistivity, args.body_weight)
    e_step = permissible_step_voltage(args.fault_duration, args.surface_resistivity, args.body_weight)

    print(f"\nIEEE 80-2013 Threshold Analysis")
    print(f"  Fault duration:       {args.fault_duration} s")
    print(f"  Surface resistivity:  {args.surface_resistivity} ohm-m")
    print(f"  Body weight:          {args.body_weight} kg")
    print(f"  Permissible touch:    {e_touch} V")
    print(f"  Permissible step:     {e_step} V")

    result = analyze(df, args.fault_duration, args.surface_resistivity, args.body_weight)
    summary = summarize(result)

    print(f"\nExceedance Summary:")
    print(f"  Total points:   {summary['total_points']}")
    print(f"  Safe:           {summary['safe_count']}")
    print(f"  Touch risk:     {summary['touch_risk_count']}")
    print(f"  Step risk:      {summary['step_risk_count']}")
    print(f"  Critical:       {summary['critical_count']}")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nFlagged dataset saved to {args.output}")


if __name__ == "__main__":
    main()
