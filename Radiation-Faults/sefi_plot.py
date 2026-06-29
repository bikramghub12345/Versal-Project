#!/usr/bin/env python3
"""
sefi_plot.py — Recursive SEFI Results Plotter
===============================================
Place at FaultResults/ level and run:
    python3 sefi_plot.py

Recursively walks every subdirectory of FaultResults/, finds any numbered
mode folder (e.g. "01. SEFI-row", "02. transient-SEFI-row"), and plots
every target subfolder (weights / input_tensor / buffers) inside it.

Works automatically with any results folder name the user chose at runtime
(sefi_results, sefi_transient_results, custom_run, etc.).

Saves two PNG plots per target folder:
    plot_accuracy_<mode>_<target>.png
    plot_prob_drop_<mode>_<target>.png

Requirements: matplotlib only.
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

# FaultResults/ — the directory this script lives in
FAULT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def avg_by_image(rows, field):
    bucket = defaultdict(list)
    for r in rows:
        try:
            bucket[r["image_name"]].append(float(r[field]))
        except (KeyError, ValueError):
            pass
    names = sorted(bucket.keys())
    avgs  = [sum(bucket[n]) / len(bucket[n]) for n in names]
    return names, avgs


def short_name(full_name, n=25):
    return full_name[-n:]


# ---------------------------------------------------------------------------
# PLOT FUNCTIONS
# ---------------------------------------------------------------------------
def plot_accuracy(acc_path, mode, target, out_dir):
    rows = read_csv(acc_path)
    if not rows:
        print(f"    [Skip] Empty accuracy_summary.csv")
        return

    row        = rows[0]
    base_acc   = float(row.get("baseline_accuracy_pct", 0))
    faulty_acc = float(row.get("faulty_accuracy_pct",  0))
    recov_acc  = float(row.get("recovery_accuracy_pct", 0))

    has_recovery = any(int(r.get("msefi_recovered", 0)) for r in rows) or recov_acc > 0
    if has_recovery:
        labels = ["Baseline", "Post-SEFI", "Post-Recovery"]
        vals   = [base_acc, faulty_acc, recov_acc]
        colors = ["forestgreen", "tomato", "steelblue"]
    else:
        labels = ["Baseline", "Post-SEFI"]
        vals   = [base_acc, faulty_acc]
        colors = ["forestgreen", "tomato"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=colors, edgecolor="black", width=0.5)
    ax.axhline(base_acc, color="forestgreen", linestyle="--",
               label=f"Baseline {base_acc:.1f}%")
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(f"SEFI: {mode} | Target: {target}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"plot_accuracy_{mode}_{target}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"    [Plot] {out_png}")


def plot_prob_drop(res_path, mode, target, out_dir):
    rows = [r for r in read_csv(res_path) if r.get("crash", "0") == "0"]
    if not rows:
        print(f"    [Skip] No valid rows for prob_drop")
        return

    names, avgs = avg_by_image(rows, "prob_drop")
    if not names:
        return

    short  = [short_name(n) for n in names]
    colors = ["tomato" if v > 0.05 else "steelblue" for v in avgs]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    ax.bar(short, avgs, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Prob Drop | {mode} | {target}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Image")
    ax.set_ylabel("Baseline prob − Faulty prob")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"plot_prob_drop_{mode}_{target}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"    [Plot] {out_png}")


# ---------------------------------------------------------------------------
# RECURSIVE SCAN
# ---------------------------------------------------------------------------
def process_mode_folder(mode_path, mode_folder_name, results_folder_name):
    """Plot all target subfolders inside one numbered mode folder."""
    mode_name = mode_folder_name.split(". ", 1)[-1] if ". " in mode_folder_name else mode_folder_name

    target_dirs = sorted(
        d for d in os.listdir(mode_path)
        if os.path.isdir(os.path.join(mode_path, d))
    )
    if not target_dirs:
        print(f"  [Skip] No target subfolders in {mode_path}")
        return 0

    total = 0
    for target_name in target_dirs:
        target_path = os.path.join(mode_path, target_name)

        acc_csv = os.path.join(target_path, "accuracy_summary.csv")
        res_csv = next(
            (os.path.join(target_path, fn)
             for fn in os.listdir(target_path)
             if fn.startswith("results_") and fn.endswith(".csv")),
            None
        )

        print(f"  [Target] {target_name}")

        if os.path.exists(acc_csv):
            plot_accuracy(acc_csv, mode_name, target_name, target_path)
            total += 1
        else:
            print(f"    [Skip] accuracy_summary.csv not found")

        if res_csv:
            plot_prob_drop(res_csv, mode_name, target_name, target_path)
            total += 2
        else:
            print(f"    [Skip] results_*.csv not found")

    return total


def main():
    print(f"[sefi_plot] FaultResults root: {FAULT_ROOT}")
    print(f"[sefi_plot] Scanning recursively for SEFI result folders...\n")

    total_plots = 0

    # Walk direct children of FaultResults/ (each is a results folder like
    # sefi_results, sefi_transient_results, or any custom name the user chose)
    result_dirs = sorted(
        d for d in os.listdir(FAULT_ROOT)
        if os.path.isdir(os.path.join(FAULT_ROOT, d))
        and not d.startswith(".")
        and d != "__pycache__"
    )

    if not result_dirs:
        print("[sefi_plot] No subdirectories found under FaultResults/. Nothing to plot.")
        return

    for results_folder in result_dirs:
        results_path = os.path.join(FAULT_ROOT, results_folder)

        # Find all numbered mode folders inside this results folder
        mode_folders = sorted(
            d for d in os.listdir(results_path)
            if os.path.isdir(os.path.join(results_path, d))
            and d[0].isdigit()
        )

        if not mode_folders:
            # No numbered folders here — skip silently (could be a non-results dir)
            continue

        print(f"[Results folder] {results_folder}/")
        for mf in mode_folders:
            mode_path = os.path.join(results_path, mf)
            print(f"  [Mode] {mf}")
            total_plots += process_mode_folder(mode_path, mf, results_folder)
            print()

    print(f"[sefi_plot] Done. {total_plots} total plots generated.")


if __name__ == "__main__":
    main()
