"""
Parse CDEGS Master Engineering Suite ASCII output files.

Supports: MALT, MALZ, HIFREQ, RESAP module exports.
Normalizes into structured pandas DataFrames for downstream analysis.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_MODULES = {"MALT", "MALZ", "HIFREQ", "RESAP", "FCDIST"}


def parse_malt_output(filepath: Path) -> pd.DataFrame:
    """
    Parse MALT module ASCII export.

    MALT computes grounding impedance, GPR, and current distribution
    for grounding electrodes in a multi-layer soil model.

    Expected columns in output:
      - conductor_id: CDEGS conductor segment identifier
      - x, y, z: coordinates (m)
      - current_magnitude (A)
      - current_phase (deg)
      - gpr_magnitude (V)
      - gpr_phase (deg)
    """
    records = []
    header_found = False

    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()

            # MALT output sections start with "CONDUCTOR" blocks
            if re.match(r"^\s*CONDUCTOR\s+\d+", line, re.IGNORECASE):
                header_found = True
                continue

            if not header_found:
                continue

            # Data rows: fixed-width or space-separated numeric columns
            parts = line.split()
            if len(parts) >= 7 and _all_numeric(parts[:7]):
                records.append({
                    "conductor_id": int(parts[0]),
                    "x_m": float(parts[1]),
                    "y_m": float(parts[2]),
                    "z_m": float(parts[3]),
                    "current_a": float(parts[4]),
                    "current_phase_deg": float(parts[5]),
                    "gpr_v": float(parts[6]),
                    "gpr_phase_deg": float(parts[7]) if len(parts) > 7 else np.nan,
                })

    if not records:
        raise ValueError(
            f"No MALT data records found in {filepath}. "
            "Check that the file is a MALT ASCII export and the CONDUCTOR block is present."
        )

    df = pd.DataFrame(records)
    df["module"] = "MALT"
    df["source_file"] = filepath.name
    return df


def parse_malz_output(filepath: Path) -> pd.DataFrame:
    """
    Parse MALZ module ASCII export.

    MALZ extends MALT to handle frequency-dependent soil parameters
    and complex grounding systems. Outputs interference levels on
    neighboring conductors (pipelines, cables, telecom).

    Key output fields:
      - observation_point: pipeline/cable segment index
      - distance_m: distance along the corridor
      - induced_voltage_v: total induced voltage magnitude
      - conductive_v: conductive coupling component
      - inductive_v: inductive coupling component
    """
    records = []

    with open(filepath, "r", errors="replace") as fh:
        content = fh.read()

    # MALZ exports observation point blocks
    block_pattern = re.compile(
        r"OBSERVATION\s+POINT\s+(\d+).*?DISTANCE[:\s]+([\d.E+\-]+).*?"
        r"INDUCED VOLTAGE[:\s]+([\d.E+\-]+).*?"
        r"CONDUCTIVE[:\s]+([\d.E+\-]+).*?"
        r"INDUCTIVE[:\s]+([\d.E+\-]+)",
        re.DOTALL | re.IGNORECASE,
    )

    for match in block_pattern.finditer(content):
        records.append({
            "observation_point": int(match.group(1)),
            "distance_m": float(match.group(2)),
            "induced_voltage_v": float(match.group(3)),
            "conductive_v": float(match.group(4)),
            "inductive_v": float(match.group(5)),
        })

    if not records:
        # Fallback: try tabular format (some MALZ versions export as table)
        df = _parse_tabular_fallback(filepath, expected_cols=5)
        df.columns = ["observation_point", "distance_m", "induced_voltage_v",
                      "conductive_v", "inductive_v"]
    else:
        df = pd.DataFrame(records)

    df["module"] = "MALZ"
    df["source_file"] = filepath.name
    return df


def parse_resap_output(filepath: Path) -> pd.DataFrame:
    """
    Parse RESAP module ASCII export.

    RESAP performs soil resistivity interpretation from Wenner or
    Schlumberger field measurements, producing a layered soil model.

    Output fields:
      - layer: soil layer index (1 = top)
      - resistivity_ohm_m: layer resistivity
      - thickness_m: layer thickness (last layer = infinite)
    """
    records = []

    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 3 and _all_numeric(parts):
                records.append({
                    "layer": int(parts[0]),
                    "resistivity_ohm_m": float(parts[1]),
                    "thickness_m": float(parts[2]),
                })

    if not records:
        raise ValueError(f"No RESAP layer data found in {filepath}.")

    df = pd.DataFrame(records)
    df["module"] = "RESAP"
    df["source_file"] = filepath.name
    return df


def parse_hifreq_output(filepath: Path) -> pd.DataFrame:
    """
    Parse HIFREQ module ASCII export.

    HIFREQ computes electromagnetic fields and induced voltages at
    high frequency, used for lightning and switching transient studies.

    Output fields:
      - frequency_hz
      - observation_x, observation_y, observation_z
      - e_field_v_per_m: electric field magnitude
      - h_field_a_per_m: magnetic field magnitude
    """
    records = []

    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 6 and _all_numeric(parts[:6]):
                records.append({
                    "frequency_hz": float(parts[0]),
                    "observation_x": float(parts[1]),
                    "observation_y": float(parts[2]),
                    "observation_z": float(parts[3]),
                    "e_field_v_per_m": float(parts[4]),
                    "h_field_a_per_m": float(parts[5]),
                })

    df = pd.DataFrame(records)
    df["module"] = "HIFREQ"
    df["source_file"] = filepath.name
    return df


def _all_numeric(parts: list) -> bool:
    try:
        [float(p) for p in parts]
        return True
    except ValueError:
        return False


def _parse_tabular_fallback(filepath: Path, expected_cols: int) -> pd.DataFrame:
    rows = []
    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == expected_cols and _all_numeric(parts):
                rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"Could not parse {filepath} in tabular fallback mode.")
    return pd.DataFrame(rows)


def parse(filepath: str, module: str) -> pd.DataFrame:
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(fp)

    module = module.upper()
    if module not in SUPPORTED_MODULES:
        raise ValueError(f"Module '{module}' not supported. Choose from {SUPPORTED_MODULES}.")

    parsers = {
        "MALT": parse_malt_output,
        "MALZ": parse_malz_output,
        "RESAP": parse_resap_output,
        "HIFREQ": parse_hifreq_output,
    }

    if module not in parsers:
        raise NotImplementedError(f"Parser for {module} not yet implemented.")

    return parsers[module](fp)


def main():
    parser = argparse.ArgumentParser(
        description="Parse CDEGS module ASCII output into structured CSV."
    )
    parser.add_argument("--input", required=True, help="Path to CDEGS output file")
    parser.add_argument("--module", required=True, choices=list(SUPPORTED_MODULES),
                        help="CDEGS module that produced the output")
    parser.add_argument("--output", help="Output CSV path (default: <input>.csv)")
    args = parser.parse_args()

    df = parse(args.input, args.module)

    out_path = args.output or str(Path(args.input).with_suffix(".csv"))
    df.to_csv(out_path, index=False)
    print(f"Parsed {len(df)} records from {args.module} -> {out_path}")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
