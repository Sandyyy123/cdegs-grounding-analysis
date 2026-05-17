# CDEGS Grounding Analysis Toolkit

Python post-processing suite for electromagnetic and grounding analysis of transmission line studies using CDEGS Master Engineering Suite outputs.

## What this toolkit does

CDEGS (MALT, MALZ, HIFREQ, RESAP modules) produces dense simulation outputs - ground potential rise (GPR) profiles, touch/step voltages, inductive interference levels, soil structure parameters. This toolkit extracts those outputs, applies IEEE 80 / IEC 60479 safety thresholds, and generates structured datasets and engineering reports.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/parse_cdegs.py` | Parse CDEGS ASCII/CSV exports into structured pandas DataFrames |
| `scripts/ieee80_analysis.py` | IEEE 80 touch and step voltage threshold calculations with exceedance flagging |
| `scripts/gpr_profile.py` | Ground Potential Rise profile visualization by tower/segment |
| `scripts/interference_analysis.py` | Inductive vs conductive interference decomposition |
| `scripts/grounding_report.py` | Full PDF engineering report generator |
| `scripts/sas_bridge.py` | Export analysis results as SAS-ready datasets (`.sas7bdat` or proc-ready CSV) |

## Stack

- Python 3.10+
- pandas, numpy, scipy, statsmodels
- matplotlib, seaborn
- reportlab (PDF generation)
- pyreadstat (SAS export bridge)

## IEEE 80 Threshold Reference

Touch voltage limit (50 kg body weight, t seconds fault duration):

```
E_touch = (1000 + 1.5 * rho_s) * 0.116 / sqrt(t)
E_step  = (1000 + 6.0 * rho_s) * 0.116 / sqrt(t)
```

Where `rho_s` is surface layer resistivity (ohm-m).

## Usage

```bash
pip install -r requirements.txt

# Parse CDEGS output
python scripts/parse_cdegs.py --input data/sample/malt_output.txt --module MALT

# Run IEEE 80 threshold analysis
python scripts/ieee80_analysis.py --data data/processed/gpr_dataset.csv --fault-duration 0.5 --surface-resistivity 3000

# Generate full report
python scripts/grounding_report.py --dataset data/processed/ --output reports/transmission_line_grounding_report.pdf
```

## Author

Dr. Sandeep Grover - https://www.upwork.com/freelancers/sandeepgrover1
