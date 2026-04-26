#!/usr/bin/env python3
"""
=============================================================================
  MTTP (Manufacturing Throughput Time Per Part) Simulator
  Based on: Johnson, D.J. (2003). A Framework for Reducing Manufacturing
            Throughput Time. Journal of Manufacturing Systems, Vol. 22/No. 4
  
  IIT Kharagpur — Breadth Course Project
=============================================================================
"""

import math
import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─── colour palette ──────────────────────────────────────────────────────────
DARK   = "#0d1117"
SURF   = "#161b22"
BORDER = "#21262d"
ACC1   = "#f97316"   # orange  – setup
ACC2   = "#38bdf8"   # sky     – processing
ACC3   = "#a3e635"   # lime    – move
ACC4   = "#e879f9"   # fuchsia – wait-for-lot
RED    = "#f85149"   # red     – queue
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
GREEN  = "#3fb950"

OUTPUT_DIR = "mttp_outputs"

# ─── data structures ─────────────────────────────────────────────────────────
@dataclass
class SimParams:
    processing_time: float = 10.0    # min per part
    setup_time:      float = 40.0    # min per batch
    move_time:       float = 15.0    # min per batch
    batch_size:      int   = 10      # parts per batch
    cv:              float = 0.8     # coefficient of variation
    utilization:     float = 0.75    # workstation utilization (0–1)
    num_workstations: int  = 2       # number of workstations in routing

@dataclass
class MTTPResult:
    setup_time:     float
    processing_time: float
    move_time:      float
    wait_for_lot:   float
    queue_time:     float
    waiting_time:   float
    total_mttp:     float
    utilization:    float

# ─── core computation ─────────────────────────────────────────────────────────
def compute_mttp(p: SimParams) -> MTTPResult:
    """
    MTTP = S + P + M + W
    W    = wait_for_lot + queue_time
    wait_for_lot = (B-1) * P          [linear in batch size]
    queue_time   = CV² * P * U / (1-U) [GI/G/M queuing formula, Whitt 1983]
    """
    wait_for_lot = (p.batch_size - 1) * p.processing_time
    
    u = min(p.utilization, 0.999)   # avoid division by zero
    queue_time = (p.cv ** 2) * p.processing_time * u / (1.0 - u)
    
    # Scale by number of workstations (each adds its own setup/move/wait)
    total_setup    = p.setup_time * p.num_workstations
    total_process  = p.processing_time * p.num_workstations
    total_move     = p.move_time * (p.num_workstations - 1)
    total_wfl      = wait_for_lot * p.num_workstations
    total_queue    = queue_time * p.num_workstations
    total_waiting  = total_wfl + total_queue

    total_mttp = total_setup + total_process + total_move + total_waiting
    
    return MTTPResult(
        setup_time      = total_setup,
        processing_time = total_process,
        move_time       = total_move,
        wait_for_lot    = total_wfl,
        queue_time      = total_queue,
        waiting_time    = total_waiting,
        total_mttp      = total_mttp,
        utilization     = p.utilization
    )

# ─── terminal display helpers ─────────────────────────────────────────────────
def _bar(value, total, width=30, color=""):
    filled = int(round((value / max(total, 0.001)) * width))
    return "█" * filled + "░" * (width - filled)

def _hline(char="─", width=65):
    return char * width

def print_result(p: SimParams, r: MTTPResult, label: str = "Current Scenario"):
    print()
    print(_hline("═"))
    print(f"  {label}")
    print(_hline("═"))
    print(f"  {'Parameter':<28} {'Value':>10}")
    print(_hline())
    print(f"  {'Processing Time/Part':<28} {p.processing_time:>9.1f} min")
    print(f"  {'Setup Time/Batch':<28} {p.setup_time:>9.1f} min")
    print(f"  {'Move Time':<28} {p.move_time:>9.1f} min")
    print(f"  {'Batch Size':<28} {p.batch_size:>9d} parts")
    print(f"  {'Variability (CV)':<28} {p.cv:>9.2f}")
    print(f"  {'Utilization':<28} {p.utilization*100:>9.1f} %")
    print(f"  {'Workstations':<28} {p.num_workstations:>9d}")
    print(_hline())
    print(f"  {'MTTP COMPONENT':<28} {'Minutes':>8}   {'%':>5}   {'BAR'}")
    print(_hline())

    comps = [
        ("Setup Time",       r.setup_time,      ACC1),
        ("Processing Time",  r.processing_time, ACC2),
        ("Move Time",        r.move_time,       ACC3),
        ("  → Wait-for-Lot", r.wait_for_lot,    ACC4),
        ("  → Queue Time",   r.queue_time,      RED),
    ]
    t = r.total_mttp
    for name, val, _ in comps:
        pct = 100 * val / t if t > 0 else 0
        bar = _bar(val, t, width=20)
        print(f"  {name:<28} {val:>8.1f}   {pct:>4.1f}%  [{bar}]")
    print(_hline())
    pct_wait = 100 * r.waiting_time / t if t > 0 else 0
    print(f"  {'Waiting Time (total)':<28} {r.waiting_time:>8.1f}   {pct_wait:>4.1f}%")
    print(_hline("═"))
    print(f"  ► TOTAL MTTP  =  {r.total_mttp:.1f} min")
    warn = "  ⚠ HIGH UTIL — queue time exploding!" if p.utilization > 0.85 else "  ✓ Utilization in safe range"
    print(warn)
    print(_hline("═"))
    print()

# ─── scenario comparison ─────────────────────────────────────────────────────
def compare_scenarios(scenarios: List[Tuple[str, SimParams]]):
    """Print a comparison table for multiple scenarios."""
    print()
    print(_hline("═"))
    print("  SCENARIO COMPARISON")
    print(_hline("═"))
    header = f"  {'Scenario':<22} {'Setup':>7} {'Proc':>7} {'Move':>7} {'Wait':>8} {'TOTAL':>8}"
    print(header)
    print(_hline())
    base_mttp = None
    for name, p in scenarios:
        r = compute_mttp(p)
        if base_mttp is None:
            base_mttp = r.total_mttp
        delta = r.total_mttp - base_mttp
        delta_str = f"({delta:+.0f})" if delta != 0 else "(base)"
        print(f"  {name:<22} {r.setup_time:>7.1f} {r.processing_time:>7.1f} "
              f"{r.move_time:>7.1f} {r.waiting_time:>8.1f} {r.total_mttp:>8.1f} {delta_str}")
    print(_hline("═"))
    print()

# ─── chart 1: component breakdown bar chart ──────────────────────────────────
def plot_component_breakdown(scenarios: List[Tuple[str, SimParams]], filename: str):
    labels = [s[0] for s in scenarios]
    results = [compute_mttp(s[1]) for s in scenarios]

    setup_vals  = [r.setup_time      for r in results]
    proc_vals   = [r.processing_time for r in results]
    move_vals   = [r.move_time       for r in results]
    wfl_vals    = [r.wait_for_lot    for r in results]
    queue_vals  = [r.queue_time      for r in results]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(SURF)

    bars = [
        ax.bar(x, setup_vals,                                            color=ACC1, label="Setup Time"),
        ax.bar(x, proc_vals,  bottom=setup_vals,                         color=ACC2, label="Processing Time"),
        ax.bar(x, move_vals,  bottom=np.array(setup_vals)+proc_vals,     color=ACC3, label="Move Time"),
        ax.bar(x, wfl_vals,   bottom=np.array(setup_vals)+proc_vals+move_vals, color=ACC4, label="Wait-for-Lot"),
        ax.bar(x, queue_vals, bottom=np.array(setup_vals)+proc_vals+move_vals+wfl_vals, color=RED, label="Queue Time"),
    ]

    # Total labels on top
    for i, r in enumerate(results):
        ax.text(i, r.total_mttp + 4, f"{r.total_mttp:.0f}", ha="center", va="bottom",
                color=TEXT, fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=9)
    ax.set_ylabel("MTTP (minutes)", color=TEXT)
    ax.set_title("MTTP Component Breakdown by Scenario", color=TEXT, fontsize=13, pad=14)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    ax.yaxis.label.set_color(TEXT)
    ax.legend(facecolor=SURF, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_ylim(0, max(r.total_mttp for r in results) * 1.15)
    ax.grid(axis="y", color=BORDER, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ─── chart 2: queue time vs utilization ──────────────────────────────────────
def plot_queue_vs_utilization(base_params: SimParams, filename: str):
    utils = np.linspace(0.05, 0.98, 200)
    
    cv_values = [0.3, 0.8, 1.5]
    colors    = [GREEN, ACC1, RED]
    labels    = ["Low Variability (CV=0.3)", f"Current (CV={base_params.cv:.1f})", "High Variability (CV=1.5)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK)

    for ax in [ax1, ax2]:
        ax.set_facecolor(SURF)
        ax.tick_params(colors=TEXT)
        ax.spines[:].set_color(BORDER)
        ax.grid(color=BORDER, linestyle="--", alpha=0.5)

    # Left: Queue time only
    for cv, col, lbl in zip(cv_values, colors, labels):
        p2 = SimParams(**{**base_params.__dict__, "cv": cv})
        queue_times = []
        for u in utils:
            p2.utilization = u
            r = compute_mttp(p2)
            queue_times.append(r.queue_time)
        ax1.plot(utils * 100, queue_times, color=col, lw=2.2, label=lbl)

    # Mark current utilization
    ax1.axvline(base_params.utilization * 100, color=ACC2, linestyle=":", lw=1.5, alpha=0.8, label=f"Current U={base_params.utilization*100:.0f}%")
    ax1.set_xlabel("Utilization (%)", color=TEXT)
    ax1.set_ylabel("Queue Time (min)", color=TEXT)
    ax1.set_title("Queue Time vs. Utilization\n(Fig. 4 equivalent)", color=TEXT, fontsize=11)
    ax1.legend(facecolor=SURF, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax1.yaxis.label.set_color(TEXT)
    ax1.xaxis.label.set_color(TEXT)

    # Right: Total MTTP
    for cv, col, lbl in zip(cv_values, colors, labels):
        p2 = SimParams(**{**base_params.__dict__, "cv": cv})
        mttps = []
        for u in utils:
            p2.utilization = u
            r = compute_mttp(p2)
            mttps.append(r.total_mttp)
        ax2.plot(utils * 100, mttps, color=col, lw=2.2, label=lbl)

    ax2.axvline(base_params.utilization * 100, color=ACC2, linestyle=":", lw=1.5, alpha=0.8)
    ax2.set_xlabel("Utilization (%)", color=TEXT)
    ax2.set_ylabel("Total MTTP (min)", color=TEXT)
    ax2.set_title("Total MTTP vs. Utilization", color=TEXT, fontsize=11)
    ax2.legend(facecolor=SURF, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax2.yaxis.label.set_color(TEXT)
    ax2.xaxis.label.set_color(TEXT)

    fig.suptitle("Impact of Utilization & Variability on MTTP", color=TEXT, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ─── chart 3: mttp vs batch size ─────────────────────────────────────────────
def plot_mttp_vs_batch(base_params: SimParams, filename: str):
    batch_sizes = range(1, 51)
    
    setup_scenarios = [
        (base_params.setup_time,        ACC1, f"Setup={base_params.setup_time:.0f} min (original)"),
        (base_params.setup_time * 0.5,  ACC3, f"Setup={base_params.setup_time*0.5:.0f} min (50% reduction)"),
        (base_params.setup_time * 0.25, ACC2, f"Setup={base_params.setup_time*0.25:.0f} min (75% reduction)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(SURF)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    ax.grid(color=BORDER, linestyle="--", alpha=0.5)

    for setup, col, lbl in setup_scenarios:
        mttps = []
        for b in batch_sizes:
            p2 = SimParams(**{**base_params.__dict__, "batch_size": b, "setup_time": setup})
            r = compute_mttp(p2)
            mttps.append(r.total_mttp)
        ax.plot(list(batch_sizes), mttps, color=col, lw=2.2, label=lbl)

    # Mark current batch size
    ax.axvline(base_params.batch_size, color=ACC4, linestyle=":", lw=1.5, alpha=0.8,
               label=f"Current Batch={base_params.batch_size}")
    ax.set_xlabel("Batch Size (parts)", color=TEXT)
    ax.set_ylabel("Total MTTP (min)", color=TEXT)
    ax.set_title("MTTP vs. Batch Size\n(Fig. 8 equivalent — effect of setup time reduction)", color=TEXT, fontsize=12)
    ax.legend(facecolor=SURF, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ─── chart 4: sensitivity tornado chart ──────────────────────────────────────
def plot_sensitivity(base_params: SimParams, filename: str):
    base_r = compute_mttp(base_params)
    base_mttp = base_r.total_mttp

    factors = [
        ("Processing Time",  "processing_time", base_params.processing_time,  0.5, 2.0),
        ("Setup Time",       "setup_time",       base_params.setup_time,       0.0, 2.0),
        ("Move Time",        "move_time",        base_params.move_time,        0.0, 2.0),
        ("Batch Size",       "batch_size",       base_params.batch_size,       1,   3.0),
        ("Variability (CV)", "cv",               base_params.cv,               0.1, 2.5),
        ("Utilization",      "utilization",      base_params.utilization,      0.3, 0.95),
    ]

    low_deltas  = []
    high_deltas = []
    labels      = []

    for name, attr, base_val, low_mult, high_mult in factors:
        if attr == "utilization":
            low_val  = low_mult
            high_val = high_mult
        elif attr == "batch_size":
            low_val  = max(1, int(base_val * low_mult))
            high_val = int(base_val * high_mult)
        else:
            low_val  = base_val * low_mult
            high_val = base_val * high_mult

        p_low  = SimParams(**{**base_params.__dict__, attr: low_val})
        p_high = SimParams(**{**base_params.__dict__, attr: high_val})
        r_low  = compute_mttp(p_low)
        r_high = compute_mttp(p_high)

        low_deltas.append(r_low.total_mttp  - base_mttp)
        high_deltas.append(r_high.total_mttp - base_mttp)
        labels.append(name)

    # Sort by absolute range
    ranges = [abs(h - l) for h, l in zip(high_deltas, low_deltas)]
    order  = np.argsort(ranges)
    labels      = [labels[i] for i in order]
    low_deltas  = [low_deltas[i] for i in order]
    high_deltas = [high_deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(SURF)

    y_pos = np.arange(len(labels))
    for i, (lo, hi, lbl) in enumerate(zip(low_deltas, high_deltas, labels)):
        ax.barh(i, lo, left=0, color=GREEN, alpha=0.85, height=0.55)
        ax.barh(i, hi, left=0, color=RED,   alpha=0.85, height=0.55)
        ax.text(lo - 2,  i, f"{lo:+.0f}", va="center", ha="right",  color=TEXT, fontsize=9)
        ax.text(hi + 2,  i, f"{hi:+.0f}", va="center", ha="left",   color=TEXT, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    ax.axvline(0, color=TEXT, lw=1.2)
    ax.set_xlabel("Change in MTTP (min) from Baseline", color=TEXT)
    ax.set_title("Sensitivity / Tornado Chart\n(Impact of each parameter on MTTP)", color=TEXT, fontsize=12)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.5)
    ax.xaxis.label.set_color(TEXT)

    green_p = mpatches.Patch(color=GREEN, alpha=0.85, label="Low value → MTTP change")
    red_p   = mpatches.Patch(color=RED,   alpha=0.85, label="High value → MTTP change")
    ax.legend(handles=[green_p, red_p], facecolor=SURF, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ─── chart 5: gantt diagram ───────────────────────────────────────────────────
def plot_gantt(p: SimParams, filename: str):
    r = compute_mttp(p)

    # Build timeline for 2 workstations
    seg_colors = {"Setup": ACC1, "Process": ACC2, "Move": ACC3, "Queue": RED, "Wait-Lot": ACC4, "Idle": BORDER}

    # WS-1: setup → process (batch) → move
    ws1_end = p.setup_time + p.processing_time * p.batch_size + p.move_time
    # WS-2 starts after queue delay
    ws2_start = ws1_end + r.queue_time / p.num_workstations

    # Timeline for Part 1 (first part in batch) — escapes early on transfer
    timelines = {
        "WS-1": [
            (0,              p.setup_time,                                     "Setup",   ACC1),
            (p.setup_time,   p.setup_time + p.processing_time * p.batch_size,  "Process", ACC2),
            (p.setup_time + p.processing_time * p.batch_size,
             p.setup_time + p.processing_time * p.batch_size + p.move_time,    "Move",    ACC3),
        ],
        "WS-2": [
            (ws2_start,               ws2_start + p.setup_time,                                    "Setup",   ACC1),
            (ws2_start + p.setup_time, ws2_start + p.setup_time + p.processing_time * p.batch_size, "Process", ACC2),
        ],
    }

    # Queue shown as shaded gap
    queue_start = p.setup_time + p.processing_time * p.batch_size + p.move_time
    queue_end   = ws2_start

    total_time = max(ws2_start + p.setup_time + p.processing_time * p.batch_size, ws1_end) * 1.05

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(SURF)

    ws_labels = list(timelines.keys())
    for yi, (ws, segs) in enumerate(timelines.items()):
        for (start, end, label, col) in segs:
            dur = end - start
            rect = mpatches.FancyBboxPatch((start, yi + 0.1), dur, 0.75,
                                           boxstyle="round,pad=0.01", facecolor=col,
                                           edgecolor=DARK, linewidth=1, alpha=0.9)
            ax.add_patch(rect)
            if dur > total_time * 0.04:
                ax.text(start + dur / 2, yi + 0.48, label, ha="center", va="center",
                        fontsize=8, color="#000", fontweight="bold")

    # Draw queue gap
    if queue_end > queue_start + 1:
        rect_q = mpatches.FancyBboxPatch((queue_start, 0.85), queue_end - queue_start, 0.3,
                                         boxstyle="round,pad=0.01", facecolor=RED,
                                         edgecolor=DARK, linewidth=1, alpha=0.5)
        ax.add_patch(rect_q)
        ax.text((queue_start + queue_end) / 2, 1.0, "Queue", ha="center", va="center",
                fontsize=8, color=TEXT)

    ax.set_yticks([0.48, 1.48])
    ax.set_yticklabels(ws_labels, color=TEXT, fontsize=10)
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.1, len(ws_labels) + 0.1)
    ax.set_xlabel("Time (minutes)", color=TEXT)
    ax.set_title(f"Gantt-Style Workstation Timeline\n"
                 f"Batch={p.batch_size}, Setup={p.setup_time}min, CV={p.cv}, U={p.utilization*100:.0f}%  →  MTTP={r.total_mttp:.0f}min",
                 color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.5)
    ax.xaxis.label.set_color(TEXT)

    legend_items = [mpatches.Patch(facecolor=c, label=l, edgecolor=DARK)
                    for l, c in [("Setup", ACC1), ("Processing", ACC2), ("Move", ACC3), ("Queue", RED)]]
    ax.legend(handles=legend_items, facecolor=SURF, edgecolor=BORDER, labelcolor=TEXT,
              fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ─── chart 6: 3D surface — MTTP vs batch & utilization ───────────────────────
def plot_3d_surface(base_params: SimParams, filename: str):
    from mpl_toolkits.mplot3d import Axes3D

    batches = np.arange(1, 26, 1)
    utils   = np.linspace(0.3, 0.95, 30)
    B, U    = np.meshgrid(batches, utils)
    Z       = np.zeros_like(B, dtype=float)

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            p2 = SimParams(**{**base_params.__dict__,
                              "batch_size": int(B[i, j]),
                              "utilization": float(U[i, j])})
            Z[i, j] = compute_mttp(p2).total_mttp

    fig = plt.figure(figsize=(11, 7))
    fig.patch.set_facecolor(DARK)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURF)

    surf = ax.plot_surface(B, U * 100, Z, cmap="plasma", alpha=0.85, linewidth=0)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="MTTP (min)", pad=0.1)

    ax.set_xlabel("Batch Size",     color=TEXT, labelpad=10)
    ax.set_ylabel("Utilization (%)", color=TEXT, labelpad=10)
    ax.set_zlabel("MTTP (min)",     color=TEXT, labelpad=10)
    ax.set_title("3D Surface: MTTP vs. Batch Size & Utilization", color=TEXT, fontsize=12, pad=16)
    ax.tick_params(colors=TEXT)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(BORDER)
    ax.yaxis.pane.set_edgecolor(BORDER)
    ax.zaxis.pane.set_edgecolor(BORDER)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ─── interactive menu ─────────────────────────────────────────────────────────
def get_float(prompt, default, lo=None, hi=None):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if lo is not None and v < lo:
                print(f"  ✗ Must be ≥ {lo}")
                continue
            if hi is not None and v > hi:
                print(f"  ✗ Must be ≤ {hi}")
                continue
            return v
        except ValueError:
            print("  ✗ Please enter a number")

def get_int(prompt, default, lo=None, hi=None):
    v = get_float(prompt, default, lo, hi)
    return int(round(v))

def edit_params(p: SimParams) -> SimParams:
    print()
    print("  Enter new values (press Enter to keep current):")
    p.processing_time  = get_float("Processing Time/Part (min)", p.processing_time, 0.1, 300)
    p.setup_time       = get_float("Setup Time/Batch (min)",     p.setup_time,      0,   600)
    p.move_time        = get_float("Move Time (min)",            p.move_time,       0,   300)
    p.batch_size       = get_int  ("Batch Size (parts)",         p.batch_size,      1,   1000)
    p.cv               = get_float("Variability CV (0–3)",       p.cv,              0.0, 3.0)
    p.utilization      = get_float("Utilization (0.01–0.99)",    p.utilization,     0.01, 0.99)
    p.num_workstations = get_int  ("Number of Workstations",     p.num_workstations, 1,  20)
    return p

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          MTTP Simulator — Johnson (2003) Framework              ║")
    print("║          IIT Kharagpur · Breadth Course Project                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    p = SimParams()

    while True:
        print()
        print("  ┌─ MAIN MENU ─────────────────────────────────────────────┐")
        print("  │  1. View current MTTP breakdown                         │")
        print("  │  2. Edit parameters                                     │")
        print("  │  3. Run scenario comparison (paper examples)            │")
        print("  │  4. Generate all charts → saved to mttp_outputs/        │")
        print("  │  5. Run sensitivity analysis                            │")
        print("  │  6. Exit                                                │")
        print("  └─────────────────────────────────────────────────────────┘")
        choice = input("  Choice [1–6]: ").strip()

        if choice == "1":
            r = compute_mttp(p)
            print_result(p, r, "Current Parameters")

        elif choice == "2":
            p = edit_params(p)
            r = compute_mttp(p)
            print_result(p, r, "Updated Parameters")

        elif choice == "3":
            # Recreate the paper's examples (Fig 1a through 1e progression)
            scenarios = [
                ("Fig1a: No Var, B=1",      SimParams(10, 0,  0,  1,  0.0,  0.60, 2)),
                ("Fig1b: No Var, B=10",     SimParams(10, 0,  0,  10, 0.0,  0.60, 2)),
                ("Fig1c: Setup+Move, B=10", SimParams(10, 40, 15, 10, 0.0,  0.75, 2)),
                ("Fig1d: +Arrival Var",     SimParams(10, 40, 15, 10, 0.6,  0.80, 2)),
                ("Fig1e: High Var",         SimParams(10, 40, 15, 10, 1.0,  0.85, 2)),
                ("Optimal: B=2, U=0.65",   SimParams(10, 20, 10, 2,  0.4,  0.65, 2)),
            ]
            compare_scenarios(scenarios)

        elif choice == "4":
            print()
            print("  Generating all charts ...")

            # Define paper-example scenarios for comparison chart
            scenarios = [
                ("Baseline\n(B=10, U=75%)",    SimParams(10, 40, 15, 10, 0.8, 0.75, 2)),
                ("↓ Batch Size\n(B=3)",         SimParams(10, 40, 15, 3,  0.8, 0.75, 2)),
                ("↓ Setup\n(S=15min)",          SimParams(10, 15, 15, 10, 0.8, 0.75, 2)),
                ("↓ Utilization\n(U=65%)",      SimParams(10, 40, 15, 10, 0.8, 0.65, 2)),
                ("↓ Variability\n(CV=0.3)",     SimParams(10, 40, 15, 10, 0.3, 0.75, 2)),
                ("All Combined\n(Best)",        SimParams(8,  15, 10, 3,  0.3, 0.65, 2)),
            ]

            plot_component_breakdown(scenarios,        f"{OUTPUT_DIR}/01_component_breakdown.png")
            plot_queue_vs_utilization(p,               f"{OUTPUT_DIR}/02_queue_vs_utilization.png")
            plot_mttp_vs_batch(p,                      f"{OUTPUT_DIR}/03_mttp_vs_batch.png")
            plot_sensitivity(p,                        f"{OUTPUT_DIR}/04_sensitivity_tornado.png")
            plot_gantt(p,                              f"{OUTPUT_DIR}/05_gantt_timeline.png")
            plot_3d_surface(p,                         f"{OUTPUT_DIR}/06_3d_surface.png")

            print()
            print(f"  ✓ All charts saved to ./{OUTPUT_DIR}/")

        elif choice == "5":
            # Quick sensitivity print
            r_base = compute_mttp(p)
            print()
            print(_hline("═"))
            print("  SENSITIVITY ANALYSIS — effect of ±50% change on each parameter")
            print(_hline("═"))
            print(f"  {'Parameter':<25} {'−50% MTTP':>12} {'Base MTTP':>12} {'+50% MTTP':>12}")
            print(_hline())
            
            sens_params = [
                ("Processing Time",  "processing_time", p.processing_time,  0.5, 1.5),
                ("Setup Time",       "setup_time",       p.setup_time,       0.5, 1.5),
                ("Move Time",        "move_time",        p.move_time,        0.5, 1.5),
                ("Batch Size",       "batch_size",       p.batch_size,       0.5, 1.5),
                ("Variability (CV)", "cv",               p.cv,               0.5, 1.5),
            ]
            for name, attr, base_val, lo_m, hi_m in sens_params:
                low_val = max(0.01, base_val * lo_m)
                high_val = base_val * hi_m
                if attr == "batch_size":
                    low_val = max(1, int(low_val))
                    high_val = int(high_val)
                p_lo = SimParams(**{**p.__dict__, attr: low_val})
                p_hi = SimParams(**{**p.__dict__, attr: high_val})
                r_lo = compute_mttp(p_lo)
                r_hi = compute_mttp(p_hi)
                print(f"  {name:<25} {r_lo.total_mttp:>12.1f} {r_base.total_mttp:>12.1f} {r_hi.total_mttp:>12.1f}")
            print(_hline("═"))

        elif choice == "6":
            print()
            print("  Exiting MTTP Simulator. Goodbye!")
            print()
            break
        else:
            print("  ✗ Invalid choice. Enter 1–6.")

if __name__ == "__main__":
    main()
