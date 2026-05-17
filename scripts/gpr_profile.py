"""
Ground Potential Rise (GPR) profile visualization.

Generates per-tower and per-segment GPR plots from MALT/MALZ output,
with IEEE 80 touch voltage threshold overlay and risk zone shading.
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieee80_analysis import permissible_touch_voltage


def plot_gpr_profile(
    df: pd.DataFrame,
    fault_duration_s: float,
    surface_resistivity_ohm_m: float,
    distance_col: str = "distance_m",
    gpr_col: str = "gpr_v",
    segment_col: str = None,
    output_path: str = None,
    title: str = "Ground Potential Rise Profile",
):
    """
    Plot GPR magnitude along a transmission line corridor.

    Overlays the IEEE 80 permissible touch voltage limit as a horizontal
    threshold line. Shades regions where GPR exceeds the limit in red.

    Args:
        df: DataFrame with distance and GPR columns
        fault_duration_s: Fault clearing time for threshold calculation
        surface_resistivity_ohm_m: Surface resistivity for threshold calculation
        distance_col: Column name for distance along corridor (m)
        gpr_col: Column name for GPR magnitude (V)
        segment_col: Optional column to color-code by line segment
        output_path: Save figure to this path (PNG/PDF). If None, shows interactively.
        title: Plot title
    """
    e_touch = permissible_touch_voltage(fault_duration_s, surface_resistivity_ohm_m)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#12121a")

    # Color by segment if provided
    if segment_col and segment_col in df.columns:
        segments = df[segment_col].unique()
        cmap = plt.cm.get_cmap("tab10", len(segments))
        for i, seg in enumerate(sorted(segments)):
            seg_df = df[df[segment_col] == seg].sort_values(distance_col)
            ax.plot(
                seg_df[distance_col],
                seg_df[gpr_col],
                color=cmap(i),
                linewidth=2,
                label=f"Segment {seg}",
            )
    else:
        ax.plot(
            df[distance_col],
            df[gpr_col],
            color="#74b9ff",
            linewidth=2,
            label="GPR",
        )

    # IEEE 80 threshold line
    ax.axhline(
        y=e_touch,
        color="#e17055",
        linewidth=1.5,
        linestyle="--",
        label=f"IEEE 80 Touch Limit ({e_touch:.0f} V)",
    )

    # Shade exceedance zones
    x = df[distance_col].values
    y = df[gpr_col].values
    ax.fill_between(
        x,
        e_touch,
        y,
        where=(y > e_touch),
        color="#e17055",
        alpha=0.25,
        label="Exceedance zone",
    )

    # Annotation: max GPR point
    max_idx = df[gpr_col].idxmax()
    max_dist = df.loc[max_idx, distance_col]
    max_gpr = df.loc[max_idx, gpr_col]
    ax.annotate(
        f"Peak: {max_gpr:.0f} V\n@ {max_dist:.0f} m",
        xy=(max_dist, max_gpr),
        xytext=(max_dist + 0.03 * (x.max() - x.min()), max_gpr * 1.05),
        color="#fdcb6e",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#fdcb6e", lw=1),
    )

    ax.set_xlabel("Distance along corridor (m)", color="#a29bfe", fontsize=11)
    ax.set_ylabel("Ground Potential Rise (V)", color="#a29bfe", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=16)
    ax.tick_params(colors="#8888a8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3d")
    ax.legend(facecolor="#1a1a28", edgecolor="#2a2a3d", labelcolor="white", fontsize=9)
    ax.grid(True, color="#2a2a3d", linewidth=0.5, alpha=0.7)

    # Stats box
    n_exceed = int((df[gpr_col] > e_touch).sum())
    stats_text = (
        f"Fault duration: {fault_duration_s} s\n"
        f"Surface resistivity: {surface_resistivity_ohm_m} ohm-m\n"
        f"Points exceeding limit: {n_exceed}/{len(df)}"
    )
    ax.text(
        0.02, 0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        color="#8888a8",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#12121a", edgecolor="#2a2a3d"),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"GPR profile saved to {output_path}")
    else:
        plt.show()


def plot_tower_gpr_bar(
    df: pd.DataFrame,
    e_touch_limit: float,
    tower_col: str = "conductor_id",
    gpr_col: str = "gpr_v",
    output_path: str = None,
):
    """
    Bar chart of peak GPR per tower with risk threshold overlay.

    Bars colored green (safe) or red (exceeds IEEE 80 limit).
    """
    tower_gpr = df.groupby(tower_col)[gpr_col].max().reset_index()
    tower_gpr["exceeds"] = tower_gpr[gpr_col] > e_touch_limit
    colors = ["#e17055" if exc else "#00b894" for exc in tower_gpr["exceeds"]]

    fig, ax = plt.subplots(figsize=(max(10, len(tower_gpr) * 0.6), 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#12121a")

    ax.bar(tower_gpr[tower_col].astype(str), tower_gpr[gpr_col], color=colors, width=0.7)
    ax.axhline(y=e_touch_limit, color="#fdcb6e", linewidth=1.5, linestyle="--",
               label=f"IEEE 80 limit: {e_touch_limit:.0f} V")

    safe_patch = mpatches.Patch(color="#00b894", label="Safe")
    risk_patch = mpatches.Patch(color="#e17055", label="Exceeds limit")
    ax.legend(handles=[safe_patch, risk_patch,
                        mpatches.Patch(color="#fdcb6e", label=f"Limit: {e_touch_limit:.0f} V")],
              facecolor="#1a1a28", edgecolor="#2a2a3d", labelcolor="white", fontsize=9)

    ax.set_xlabel("Tower / Conductor ID", color="#a29bfe", fontsize=11)
    ax.set_ylabel("Peak GPR (V)", color="#a29bfe", fontsize=11)
    ax.set_title("Peak GPR by Tower - IEEE 80 Exceedance", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8888a8", axis="both")
    ax.tick_params(axis="x", rotation=45)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3d")
    ax.grid(True, axis="y", color="#2a2a3d", linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Tower GPR bar chart saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="GPR profile visualization.")
    parser.add_argument("--data", required=True, help="CSV with GPR simulation results")
    parser.add_argument("--fault-duration", type=float, required=True)
    parser.add_argument("--surface-resistivity", type=float, required=True)
    parser.add_argument("--distance-col", default="distance_m")
    parser.add_argument("--gpr-col", default="gpr_v")
    parser.add_argument("--segment-col", default=None)
    parser.add_argument("--output", help="Output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    plot_gpr_profile(
        df,
        args.fault_duration,
        args.surface_resistivity,
        distance_col=args.distance_col,
        gpr_col=args.gpr_col,
        segment_col=args.segment_col,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
