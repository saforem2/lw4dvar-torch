#!/usr/bin/env python3
"""Plot Sunspot ACE2 canary metrics from its original output files and log."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "ace2_sunspot_2cycle_10it.png"
OUT_SVG = ROOT / "ace2_sunspot_2cycle_10it.svg"
RUN_DIR = Path("output/test_ace2_2cycle_10it_sunspot")
JOB_LOG = Path("lw4dvar-ace2-2x10-serial.o12483476")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR, help="Sunspot experiment output directory")
    parser.add_argument("--job-log", type=Path, default=JOB_LOG, help="PBS stdout log containing Z500 scores")
    return parser.parse_args()


def load_loss_traces(output_dir: Path) -> dict[str, list[float]]:
    traces = {}
    for cycle in ("2015-01-01T00", "2015-01-01T06"):
        path = output_dir / f"{cycle}_loss_120h_10it.txt"
        traces[cycle] = [float(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(traces[cycle]) != 10:
            raise ValueError(f"Expected 10 loss values in {path}, found {len(traces[cycle])}")
    return traces


def load_z500_scores(job_log: Path) -> list[dict[str, float | str]]:
    pattern = re.compile(
        r"z500err (before|after) (\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for line in job_log.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            phase, valid_time = match.group(1, 2)
            scores[valid_time][phase] = float(match.group(6))
    rows = []
    for valid_time in ("2015-01-01T06", "2015-01-01T12"):
        values = scores.get(valid_time, {})
        if not {"before", "after"}.issubset(values):
            raise ValueError(f"Missing before/after Z500 score for {valid_time} in {job_log}")
        rows.append(
            {
                "valid_time": valid_time,
                "control_rmse_m": values["before"],
                "analysis_rmse_m": values["after"],
            }
        )
    return rows


def make_figure() -> None:
    args = parse_args()
    losses = load_loss_traces(args.output_dir)
    z500_rows = load_z500_scores(args.job_log)

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
        values = losses[cycle]
        epochs = range(1, len(values) + 1)
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
    print(f"Read losses from {args.output_dir}")
    print(f"Read Z500 scores from {args.job_log}")


if __name__ == "__main__":
    make_figure()
