#!/usr/bin/env python3
"""Plot Sunspot ACE2 2-cycle canary metrics from the checked-in CSV data."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_PNG = ROOT / "ace2_sunspot_2cycle_10it.png"
OUT_SVG = ROOT / "ace2_sunspot_2cycle_10it.svg"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Libertinus Serif", "DejaVu Serif", "STIX Two Text"],
        "font.sans-serif": ["Libertinus Sans", "DejaVu Sans"],
        "font.monospace": ["JetBrains Mono", "Libertinus Mono", "DejaVu Sans Mono"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "lw4dvar-ace2-canary-v1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "font.size": 9,
        "legend.fontsize": 8,
    }
)

COLORS = {"2015-01-01T00": "#0072B2", "2015-01-01T06": "#D55E00"}
LABELS = {"2015-01-01T00": "Cycle 1 (00Z)", "2015-01-01T06": "Cycle 2 (06Z)"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_figure() -> None:
    loss_rows = read_csv(DATA / "ace2_sunspot_2cycle_10it_loss.csv")
    z500_rows = read_csv(DATA / "ace2_sunspot_2cycle_10it_z500.csv")
    losses: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in loss_rows:
        losses[row["cycle_start"]].append((int(row["epoch"]), float(row["loss"])))

    fig, (ax_loss, ax_rmse) = plt.subplots(
        1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    fig.suptitle("ACE2 on Sunspot: two-cycle real-data canary", y=0.98, fontsize=12)
    fig.text(
        0.5,
        0.925,
        "PBS 12483476 · 2 cycles · 10 epochs · 120 h windows · exit 0",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )

    for cycle in ("2015-01-01T00", "2015-01-01T06"):
        points = sorted(losses[cycle])
        epochs, values = zip(*points)
        pct = 100 * (values[-1] / values[0] - 1)
        ax_loss.plot(
            epochs,
            values,
            color=COLORS[cycle],
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            label=f"{LABELS[cycle]} ({pct:.1f}%)",
        )
    ax_loss.set_title("A  Saved objective by optimizer epoch", loc="left", pad=10)
    ax_loss.set_xlabel("Optimizer epoch")
    ax_loss.set_ylabel("Objective loss (×10³)")
    ax_loss.set_xlim(0.7, 10.3)
    ax_loss.set_xticks(range(1, 11))
    ax_loss.set_ylim(480_000, 530_000)
    ax_loss.set_yticks([480_000, 490_000, 500_000, 510_000, 520_000, 530_000])
    ax_loss.set_yticklabels(["480", "490", "500", "510", "520", "530"])
    ax_loss.grid(axis="y", color="#d8dee8", linewidth=0.7)
    ax_loss.legend(frameon=False, loc="lower left")

    positions = list(range(len(z500_rows)))
    deltas = [float(r["analysis_rmse_m"]) - float(r["control_rmse_m"]) for r in z500_rows]
    for x, row, delta in zip(positions, z500_rows, deltas):
        ax_rmse.plot([0, delta], [x, x], color="#6b7280", linewidth=1.2, zorder=1)
        ax_rmse.scatter(delta, x, s=44, color="#009E73", edgecolor="white", linewidth=0.7, zorder=3)
        ax_rmse.annotate(
            f'{float(row["control_rmse_m"]):.2f} -> {float(row["analysis_rmse_m"]):.2f} m',
            (delta, x),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            color="#374151",
        )
    ax_rmse.axvline(0, color="#6b7280", linestyle="--", linewidth=0.9, zorder=0)
    ax_rmse.set_title("B  Change in +6 h global Z500 RMSE", loc="left", pad=10)
    ax_rmse.set_xlabel("Analysis − control (m; negative is lower error)")
    ax_rmse.set_yticks(positions)
    ax_rmse.set_yticklabels(["Valid 06Z", "Valid 12Z"])
    ax_rmse.set_xlim(-0.07, 0.035)
    ax_rmse.set_xticks([-0.06, -0.04, -0.02, 0.00, 0.02])
    ax_rmse.grid(axis="x", color="#d8dee8", linewidth=0.7)

    fig.text(
        0.5,
        0.015,
        "Lower RMSE is better. Ten epochs are an early optimization check, not a convergence result.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.89), w_pad=2.3)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    svg = OUT_SVG.read_text(encoding="utf-8")
    OUT_SVG.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    make_figure()
