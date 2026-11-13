#!/usr/bin/env python3
"""Plot completed-cycle diagnostics from the partial Sunspot ACE2 run.

The two source logs are kept on Sunspot. Copy them locally, then pass both
paths with ``--job-logs``. This script writes only the PNG and SVG figures;
it does not create or track intermediate CSV files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


FIGURES = Path(__file__).resolve().parent
OUT_PNG = FIGURES / "ace2_sunspot_240cycle_partial.png"
OUT_SVG = FIGURES / "ace2_sunspot_240cycle_partial.svg"

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
        "svg.hashsalt": "lw4dvar-ace2-240cycle-partial-v1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "font.size": 9,
        "legend.fontsize": 8,
    }
)

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GRID = "#d8dee8"

EPOCH_RE = re.compile(r"INFO - epoch=(\d+), lr=[^,]+, loss=([-+0-9.eE]+)")
START_RE = re.compile(r"z500err bg\(t0\) (\S+)")
SCORE_RE = re.compile(
    r"z500err (before|after) (\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-logs", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-cycles", type=int, default=120)
    return parser.parse_args()


def parse_job_log(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    traces: list[list[float]] = []
    for match in EPOCH_RE.finditer(text):
        epoch, loss = int(match.group(1)), float(match.group(2))
        if epoch == 1:
            traces.append([])
        if not traces or epoch != len(traces[-1]) + 1:
            raise ValueError(f"Nonsequential optimizer epochs in {path} near epoch {epoch}")
        traces[-1].append(loss)

    starts = [
        datetime.strptime(match.group(1), "%Y-%m-%dT%H")
        for line in text.splitlines()
        if (match := START_RE.search(line))
    ]
    scores: dict[datetime, dict[str, float]] = {}
    for line in text.splitlines():
        match = SCORE_RE.search(line)
        if not match:
            continue
        phase, valid_time = match.group(1, 2)
        cycle_start = datetime.strptime(valid_time, "%Y-%m-%dT%H") - timedelta(hours=6)
        # The logger prints NH, tropical, SH, then global; the last value is
        # the global area-weighted 500 hPa height RMSE in metres.
        scores.setdefault(cycle_start, {})[phase] = float(match.group(6))

    if not (len(traces) == len(starts) == len(scores)):
        raise ValueError(
            f"Inconsistent cycle coverage in {path}: {len(traces)} optimizer traces, "
            f"{len(starts)} starts, {len(scores)} paired score times"
        )

    rows = []
    for start, losses in zip(starts, traces, strict=True):
        if len(losses) != 50:
            raise ValueError(f"Expected 50 epochs for {start:%Y-%m-%d %H} in {path}, found {len(losses)}")
        pair = scores.get(start)
        if not pair or not {"before", "after"}.issubset(pair):
            raise ValueError(f"Missing paired +6 h Z500 scores for {start:%Y-%m-%d %H} in {path}")
        rows.append(
            {
                "start": start,
                "objective_change_pct": 100.0 * (losses[-1] / losses[0] - 1.0),
                "objective_trace_pct": [100.0 * (loss / losses[0] - 1.0) for loss in losses],
                "control_rmse_m": pair["before"],
                "analysis_rmse_m": pair["after"],
            }
        )
    return rows


def rolling_median(values: np.ndarray, window: int = 12) -> np.ndarray:
    half = window // 2
    return np.asarray(
        [np.median(values[max(0, i - half) : min(len(values), i + half + 1)]) for i in range(len(values))]
    )


def make_figure() -> None:
    args = parse_args()
    rows = sorted((row for path in args.job_logs for row in parse_job_log(path)), key=lambda row: row["start"])
    if len(rows) != args.expected_cycles:
        raise ValueError(f"Expected {args.expected_cycles} completed cycles, found {len(rows)}")
    starts = [row["start"] for row in rows]
    objective = np.asarray([row["objective_change_pct"] for row in rows], dtype=float)
    objective_traces = np.asarray([row["objective_trace_pct"] for row in rows], dtype=float)
    epoch_median = np.median(objective_traces, axis=0)
    epoch_p10, epoch_p90 = np.percentile(objective_traces, [10, 90], axis=0)
    control = np.asarray([row["control_rmse_m"] for row in rows], dtype=float)
    analysis = np.asarray([row["analysis_rmse_m"] for row in rows], dtype=float)
    error_change = analysis - control

    improved = int(np.count_nonzero(error_change < 0))
    worsened = int(np.count_nonzero(error_change > 0))
    unchanged = len(rows) - improved - worsened
    median_objective = float(np.median(objective))
    median_error_change = float(np.median(error_change))

    fig, (ax_obj, ax_rmse) = plt.subplots(2, 1, figsize=(10.2, 6.5))
    fig.suptitle("ACE2 4D-Var on Sunspot: first 120 of 240 cycles", y=0.985, fontsize=12)
    fig.text(
        0.5,
        0.945,
        "Jan 1–Jan 30, 2015 · 120 six-hour cycles · 5-day windows · 50 epochs per cycle · PBS 12483400 + 12483402",
        ha="center",
        fontsize=8.3,
        color="#4b5563",
    )

    epochs = np.arange(1, objective_traces.shape[1] + 1)
    ax_obj.axhline(0, color="#6b7280", linestyle="--", linewidth=0.9, zorder=0)
    ax_obj.fill_between(epochs, epoch_p10, epoch_p90, color=BLUE, alpha=0.16, label="10th–90th percentile")
    ax_obj.plot(epochs, epoch_median, color=BLUE, linewidth=1.8, label="Median across cycles")
    ax_obj.set_title("A  Within-cycle objective change across 50 optimizer epochs", loc="left", pad=8)
    ax_obj.set_xlabel("Optimizer epoch")
    ax_obj.set_ylabel("Change from epoch 1 (%)")
    ax_obj.set_xlim(1, 50)
    ax_obj.set_xticks([1, 10, 20, 30, 40, 50])
    ax_obj.grid(axis="y", color=GRID, linewidth=0.7)
    ax_obj.legend(frameon=False, loc="best")
    ax_obj.text(
        0.99,
        0.93,
        f"median at epoch 50: {median_objective:.2f}%",
        transform=ax_obj.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=BLUE,
    )

    ax_rmse.axhline(0, color="#6b7280", linestyle="--", linewidth=0.9, zorder=0)
    ax_rmse.scatter(starts, error_change, s=13, color=GREEN, alpha=0.48, linewidths=0, label="Per-cycle change")
    ax_rmse.plot(
        starts,
        rolling_median(error_change),
        color=GREEN,
        linewidth=1.8,
        label="12-cycle rolling median",
    )
    ax_rmse.set_title("B  +6 h global 500 hPa height RMSE: analysis − control", loc="left", pad=8)
    ax_rmse.set_ylabel("Global Z500 RMSE Δ (m)")
    ax_rmse.grid(axis="y", color=GRID, linewidth=0.7)
    ax_rmse.legend(frameon=False, loc="best")
    ax_rmse.text(
        0.99,
        0.93,
        f"median {median_error_change:+.2f} m · lower in {improved}/{len(rows)} cycles · "
        f"higher in {worsened}/{len(rows)} · unchanged {unchanged}",
        transform=ax_rmse.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=GREEN,
    )
    ax_rmse.set_xlim(starts[0] - timedelta(hours=12), starts[-1] + timedelta(hours=12))
    ax_rmse.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax_rmse.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax_rmse.set_xlabel("Cycle start date (UTC)")

    fig.text(
        0.5,
        0.015,
        "Objective reduction shows optimizer progress, not forecast skill. Z500 compares both forecasts at the same +6 h valid time; this is a partial run.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.04, 0.055, 0.99, 0.915), h_pad=1.6)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    svg = OUT_SVG.read_text(encoding="utf-8")
    OUT_SVG.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")
    print(f"Cycles: {len(rows)} ({starts[0]:%Y-%m-%d %H} through {starts[-1]:%Y-%m-%d %H} UTC)")
    print(f"Epochs: {sum(50 for _ in rows)}")
    print(f"Median within-cycle objective change: {median_objective:.4f}%")
    print(
        f"Median +6 h global Z500 RMSE change: {median_error_change:+.4f} m "
        f"(lower {improved}, higher {worsened}, unchanged {unchanged})"
    )


if __name__ == "__main__":
    make_figure()
